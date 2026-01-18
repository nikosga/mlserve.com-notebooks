"""
pricing_oracle.py

This module implements:
1) A hidden "true" demand oracle (logistic DGP with interactions).
2) Synthetic train/test dataset generation (students call these functions).
3) A scoring function that returns purchase outcomes and realized revenue.

Design goals:
- Numericals + binary only (no categoricals).
- Train set includes: features + price + purchase label.
- Test set includes: features only (+ user_id).
- Students choose a price per user and call score_prices().
- Oracle can score deterministically (recommended for fair leaderboard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)  # numerical stability
    return 1.0 / (1.0 + np.exp(-z))


def _coerce_price_grid(price_grid: Sequence[float]) -> np.ndarray:
    g = np.array(list(price_grid), dtype=float)
    if g.ndim != 1 or g.size < 2:
        raise ValueError("price_grid must be a 1D sequence with at least 2 prices.")
    if np.any(g <= 0):
        raise ValueError("price_grid must contain only positive prices.")
    return np.unique(g)


def _snap_to_grid(prices: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Snap arbitrary prices to nearest price in a discrete grid.
    """
    # broadcast: (n,1) - (1,k) -> (n,k)
    diffs = np.abs(prices.reshape(-1, 1) - grid.reshape(1, -1))
    idx = np.argmin(diffs, axis=1)
    return grid[idx]


def _seeded_uniform_0_1(secret_seed: int, user_ids: np.ndarray) -> np.ndarray:
    """
    Deterministic per-user pseudo-random u in (0,1), used to generate deterministic
    binary outcomes. This makes leaderboard stable (no "luck" arguments).

    We combine a global secret seed + user_id into a SeedSequence for each user.
    """
    u = np.empty(user_ids.shape[0], dtype=float)
    # Vectorizing SeedSequence per user is awkward; loop is fine at webinar scale.
    for i, uid in enumerate(user_ids.astype(np.int64)):
        ss = np.random.SeedSequence([int(secret_seed), int(uid)])
        rng = np.random.default_rng(ss)
        u[i] = rng.random()
    return u


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class OracleConfig:
    """
    Controls the synthetic market and the hidden demand function.

    All coefficients below are hidden from students.
    """
    # Global randomness (only used for data generation; scoring can be deterministic)
    seed: int = 123

    # Allowed discrete prices for the exercise
    price_grid: Tuple[int, ...] = tuple(range(80, 401, 20))  # 80, 100, ..., 400

    # Score mode:
    # - deterministic=True: purchase_i = 1[p_true_i > u_i] where u_i is per-user fixed.
    # - deterministic=False: purchase_i ~ Bernoulli(p_true_i) with fresh randomness.
    deterministic_scoring: bool = True

    # If deterministic scoring is enabled, u_i depends on this secret seed:
    deterministic_secret_seed: int = 999_001

    # Optional margin/cost: revenue = (price - unit_cost) * purchase
    unit_cost: float = 0.0

    # --- Hidden demand model coefficients (logit) ---
    # We model purchase probability as:
    # logit(P) = b0 + b'x + bp*price + interactions + noise_term(=0 for scoring)
    b0: float = -1.2

    # Main effects on propensity to buy
    b_income: float = 0.55          # higher income -> higher base purchase
    b_urgency: float = 1.15         # urgency -> higher purchase
    b_days_to_dep: float = -0.35    # farther from departure -> lower purchase (less urgency)
    b_business: float = 0.70        # business travelers buy more
    b_loyalty: float = 0.45         # loyalty increases purchase
    b_prev_purch: float = 0.25      # previous purchases increases purchase

    # Price effect (negative): higher price -> lower purchase
    b_price: float = -0.018

    # Interactions (key for "pricing optimization" to be non-trivial)
    b_price_x_income: float = 0.006     # higher income reduces price sensitivity
    b_price_x_business: float = 0.010   # business travelers less price sensitive
    b_price_x_urgency: float = 0.008    # urgency reduces price sensitivity

    # Nonlinear effect: slight curvature in price
    b_price_sq: float = -0.000015       # tiny concavity (keeps probs reasonable)

    # Feature noise levels for data generation (not scoring)
    feature_noise_scale: float = 0.05
    logit_noise_scale: float = 0.35     # adds irreducible noise in train labels

    # Logged pricing policy (for train data)
    # If True: price correlates with features (realistic, but still learnable).
    # If False: random uniform over price grid (clean identification).
    logged_policy_correlated: bool = True


# -----------------------------
# Oracle implementation
# -----------------------------

class PricingOracle:
    """
    Hidden oracle that generates datasets and scores chosen prices.
    """

    def __init__(self, config: Optional[OracleConfig] = None):
        self.cfg = config or OracleConfig()
        self._grid = _coerce_price_grid(self.cfg.price_grid)
        self._rng = np.random.default_rng(self.cfg.seed)

    # ---------- Public API (for students) ----------

    def get_train(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Returns a synthetic training dataset with columns:
            user_id, income, urgency, days_to_departure, is_business, loyalty, prev_purchases,
            price, purchase

        - All features are numeric or binary.
        - Price is drawn from a logged pricing policy.
        """
        if n <= 0:
            raise ValueError("n must be positive")

        rng = np.random.default_rng(seed) if seed is not None else self._rng
        X = self._sample_users(n=n, rng=rng, include_user_id=True)

        # Logged policy chooses observed price in train
        price = self._logged_price_policy(X, rng=rng)
        X["price"] = price

        # Generate purchase labels with noise (train should be noisy)
        p_true = self._true_purchase_prob(X, include_logit_noise=True, rng=rng)
        y = rng.binomial(1, p_true).astype(int)

        df = X.copy()
        df["purchase"] = y
        return df

    def get_test(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Returns a synthetic test dataset with columns:
            user_id, income, urgency, days_to_departure, is_business, loyalty, prev_purchases

        (No price, no purchase label.)
        """
        if n <= 0:
            raise ValueError("n must be positive")

        rng = np.random.default_rng(seed) if seed is not None else self._rng
        X = self._sample_users(n=n, rng=rng, include_user_id=True)
        return X

    def score_prices(
        self,
        users: pd.DataFrame,
        chosen_prices: pd.DataFrame,
        *,
        user_id_col: str = "user_id",
        price_col: str = "price",
        deterministic: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Score chosen prices for each user.

        Inputs:
          users: DataFrame from get_test() (features + user_id)
          chosen_prices: DataFrame with [user_id, price] chosen by the student/system

        Returns:
          DataFrame with columns:
            user_id, purchase (0/1), revenue (float)

        Scoring:
          - If deterministic: purchase = 1[p_true > u(user_id)]
          - Else: purchase ~ Bernoulli(p_true)

        revenue = (price - unit_cost) * purchase
        """
        det = self.cfg.deterministic_scoring if deterministic is None else deterministic

        if user_id_col not in users.columns:
            raise ValueError(f"users must include '{user_id_col}'")
        if user_id_col not in chosen_prices.columns:
            raise ValueError(f"chosen_prices must include '{user_id_col}'")
        if price_col not in chosen_prices.columns:
            raise ValueError(f"chosen_prices must include '{price_col}'")

        # Merge to align features with chosen price
        df = users.merge(chosen_prices[[user_id_col, price_col]], on=user_id_col, how="inner")
        if df.shape[0] != users.shape[0]:
            # If a student forgets some users, we only score the intersection.
            # You may prefer to raise; up to you.
            pass

        # Snap prices to grid (enforce allowed actions)
        snapped = _snap_to_grid(df[price_col].to_numpy(dtype=float), self._grid)
        df[price_col] = snapped

        # True purchase probabilities (NO logit noise at scoring time)
        p_true = self._true_purchase_prob(df, include_logit_noise=False, rng=None)

        if det:
            u = _seeded_uniform_0_1(self.cfg.deterministic_secret_seed, df[user_id_col].to_numpy())
            purchase = (p_true > u).astype(int)
        else:
            # stochastic scoring (less fair for leaderboard)
            rng = np.random.default_rng(self.cfg.seed + 777)
            purchase = rng.binomial(1, p_true).astype(int)

        revenue = (df[price_col].to_numpy(dtype=float) - float(self.cfg.unit_cost)) * purchase

        out = pd.DataFrame(
            {
                user_id_col: df[user_id_col].to_numpy(),
                "purchase": purchase.astype(int),
                "revenue": revenue.astype(float),
            }
        )
        return out

    def allowed_price_grid(self) -> List[float]:
        """
        Convenience for notebooks: returns the discrete grid of allowable prices.
        """
        return self._grid.tolist()

    # ---------- Internal methods (hidden) ----------

    def _sample_users(self, n: int, rng: np.random.Generator, include_user_id: bool) -> pd.DataFrame:
        """
        Sample user features. All numerical or binary.
        """
        # Income: lognormal-ish scaled to [0, 1.5]
        income = np.clip(rng.lognormal(mean=0.0, sigma=0.55, size=n) / 2.5, 0.0, 1.5)

        # Urgency: [0,1], skewed high for some users
        urgency = np.clip(rng.beta(a=2.2, b=1.8, size=n), 0.0, 1.0)

        # Days to departure: scaled numeric in [0,1], where 0=far, 1=near
        # We'll represent as "days_to_departure_scaled" with higher meaning closer to departure.
        # Example: sample raw days 1..60 then scale to [0,1] by (60 - days)/59.
        days = rng.integers(low=1, high=61, size=n)
        days_to_departure = (60.0 - days) / 59.0  # near departure => closer to 1

        # Business trip: binary, correlated with income
        p_business = _sigmoid(-0.4 + 1.2 * (income - 0.5))
        is_business = rng.binomial(1, p_business).astype(int)

        # Loyalty: binary, correlated with previous purchases and business
        # We'll sample prev_purchases first
        prev_purchases = rng.poisson(lam=1.2 + 1.0 * is_business + 0.6 * (income > 0.8), size=n)
        prev_purchases = np.clip(prev_purchases, 0, 10).astype(int)
        p_loyalty = _sigmoid(-1.0 + 0.5 * is_business + 0.25 * prev_purchases)
        loyalty = rng.binomial(1, p_loyalty).astype(int)

        # Optional small feature noise to avoid perfect separations
        noise = self.cfg.feature_noise_scale
        income = np.clip(income + rng.normal(0, noise, size=n), 0.0, 1.5)
        urgency = np.clip(urgency + rng.normal(0, noise, size=n), 0.0, 1.0)
        days_to_departure = np.clip(days_to_departure + rng.normal(0, noise, size=n), 0.0, 1.0)

        df = pd.DataFrame(
            {
                "income": income.astype(float),
                "urgency": urgency.astype(float),
                "days_to_departure": days_to_departure.astype(float),
                "is_business": is_business.astype(int),
                "loyalty": loyalty.astype(int),
                "prev_purchases": prev_purchases.astype(int),
            }
        )

        if include_user_id:
            # user_id should be stable and not "meaningful"
            # Using a random int range is fine; deterministic scoring uses this ID.
            df.insert(0, "user_id", rng.integers(1_000_000, 9_999_999, size=n, dtype=np.int64))

        return df

    def _logged_price_policy(self, users_with_features: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        """
        Generate observed prices for the TRAIN set.

        Two options:
        - correlated logged policy (more realistic): price depends on business, urgency, income
        - randomized policy: uniform random price from grid (clean identification)

        Prices are always snapped to the allowed grid.
        """
        if not self.cfg.logged_policy_correlated:
            price = rng.choice(self._grid, size=len(users_with_features), replace=True)
            return price.astype(float)

        # Correlated policy: higher prices for business, urgent, higher income
        income = users_with_features["income"].to_numpy(dtype=float)
        urgency = users_with_features["urgency"].to_numpy(dtype=float)
        business = users_with_features["is_business"].to_numpy(dtype=float)

        base = 160.0
        # push up for business/urgency/income; add noise; then snap to grid
        raw = (
            base
            + 80.0 * business
            + 70.0 * urgency
            + 60.0 * np.clip(income, 0, 1.5)
            + rng.normal(0, 35.0, size=len(users_with_features))
        )
        raw = np.clip(raw, self._grid.min(), self._grid.max())
        return _snap_to_grid(raw, self._grid).astype(float)

    def _true_purchase_prob(
        self,
        df_with_price: pd.DataFrame,
        *,
        include_logit_noise: bool,
        rng: Optional[np.random.Generator],
    ) -> np.ndarray:
        """
        Compute true purchase probability given features and price.

        include_logit_noise=True is used for TRAIN label generation to make the
        exercise realistic; include_logit_noise=False is used at scoring time.
        """
        cfg = self.cfg

        income = df_with_price["income"].to_numpy(dtype=float)
        urgency = df_with_price["urgency"].to_numpy(dtype=float)
        dtd = df_with_price["days_to_departure"].to_numpy(dtype=float)
        business = df_with_price["is_business"].to_numpy(dtype=float)
        loyalty = df_with_price["loyalty"].to_numpy(dtype=float)
        prev = df_with_price["prev_purchases"].to_numpy(dtype=float)

        price = df_with_price["price"].to_numpy(dtype=float)

        # Center/scale price for stability (still interpretable)
        # Typical price range ~ [80,400], center at 200.
        p = (price - 200.0) / 100.0

        # logit
        logit = (
            cfg.b0
            + cfg.b_income * income
            + cfg.b_urgency * urgency
            + cfg.b_days_to_dep * dtd
            + cfg.b_business * business
            + cfg.b_loyalty * loyalty
            + cfg.b_prev_purch * prev
            + cfg.b_price * price  # strong negative in raw-price units
            + cfg.b_price_sq * (price ** 2)
            + cfg.b_price_x_income * price * income
            + cfg.b_price_x_business * price * business
            + cfg.b_price_x_urgency * price * urgency
        )

        if include_logit_noise:
            if rng is None:
                raise ValueError("rng must be provided when include_logit_noise=True")
            logit = logit + rng.normal(0.0, cfg.logit_noise_scale, size=logit.shape[0])

        return _sigmoid(logit)