---

# 🧠 MLServe.com Python SDK

Official Python SDK for interacting with the **MLServe.com API** — a cloud platform for serving, monitoring, and collaborating on machine learning models.

This SDK provides a simple and secure interface to manage your models, users, datasets, and experiments — directly from Python or integrated applications.

---

## 🚀 Installation

Install via **pip**:

```bash
pip install mlserve-sdk
```

Or from source:

```bash
git clone https://github.com/nikosga/mlserve-sdk
cd mlserve-sdk
pip install -e .
```

---

## ⚙️ Setup & Authentication

The MLServe.com SDK requires an **API token** for authenticated requests.

You can:

* Obtain a token after **logging in** with your email and password, or
* Use the **Google OAuth** login flow (for SDK integrations).

### Example: Login and set token

```python
from mlserve import MLServeClient

client = MLServeClient()

# Login using your credentials
response = client.login(email="user@example.com", password="YourPassword123")

# Store your token automatically
print(response)
# → {"access_token": "...", "token_type": "bearer"}
```

You can also **set your token manually**:

```python
client.set_token("your-jwt-token")
```

---

## 🧑‍💻 User Management

### 🔹 Register a new account

```python
client.register(
    user_name="Alice Example",
    email="alice@example.com",
    password="SecurePass123!"
)
```

After registration, MLServe.com will send you a verification email. Once verified, you can log in using your credentials.

### 🔹 Request a password reset

```python
client.request_password_reset(
    email="alice@example.com",
    new_password="MyNewPassword123!"
)
```

You’ll receive an email with a link to confirm your password change.

### 🔹 Login

```python
response = client.login(
    email="alice@example.com",
    password="MyNewPassword123!"
)
print(response["access_token"])
```

### 🔹 Logout

```python
client.logout()
```

### 🔹 Check token validity

```python
profile = client.check_token()
print(profile["user_email"])
```

---

## 👥 Team Management

### 🔹 Invite a new team member

```python
client.invite_user("new.member@example.com")
```

The invitee will receive a verification link to join your organization.

### 🔹 List all team members

```python
team = client.list_team()
for member in team:
    print(member["user_name"], "-", member["role"])
```

### 🔹 Update a team member’s role

```python
client.update_user_role(user_id=42, role="admin")
```

### 🔹 Remove a team member

```python
client.remove_team_member(user_id=42)
```

This will disable their access (soft delete).

---

## 🧠 ML Model Serving & Deployment

### 🔹 Deploy a model

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
# Train your model here

response = client.deploy_model(
    model=model,
    name="my_model",
    version="v1",
    features=["feature1", "feature2"],
    background_df=df.sample(100)
)
deployment_id = resp["deployment_id"]
final_status = client.wait_for_deployment(deployment_id)
```

### 🔹 Make predictions

```python
data = {"inputs": [{"feature1": 1.2, "feature2": 3.4}]}

predictions = client.predict(
    name="my_model",
    version="v1",
    data=data
)

print(predictions)
```

### 🔹 Weighted predictions across versions (A/B testing)

```python
weighted_preds = client.predict_weighted(
    name="my_model",
    data=data
)
```

### 🔹 Configure A/B test weights

```python
client.configure_abtest("my_model", weights={"v1": 0.7, "v2": 0.3})
```

### 🔹 List deployed models

```python
models = client.list_models()
print(models)
```

### 🔹 Get latest model version

```python
latest = client.get_latest_version("my_model")
print(latest)
```
---

## 🧊 Materialized Inference (NEW)

Materialized inference supports a common production pattern where inference happens asynchronously (e.g., event-driven pipelines), and results are stored for fast retrieval by identifier.

Instead of returning predictions directly, MLServe.com can store them keyed by `entity_id` (e.g., user_id, customer_id), allowing downstream systems to fetch predictions later without re-sending features.

### ✅ Materialize predictions (store-by-entity)

Use `materialize=True` to store predictions in the online store.

**Important:** you must provide `entity_ids` aligned with `inputs`.

```python
payload = {
    "inputs": [
        {"feature1": 1.2, "feature2": 3.4},
        {"feature1": 0.9, "feature2": 2.1}
    ],
    "entity_ids": ["user_123", "user_456"]
}

ack = client.predict(
    name="my_model",
    version="v1",
    data=payload,
    materialize=True
)

print(ack)
# → {"materialized": true, "model": "my_model", "stored_at": "...", "keys": [...]}
```

The same is supported for weighted inference:

```python
ack = client.predict_weighted(
    name="my_model",
    data=payload,
    materialize=True
)
```

### ✅ Fetch materialized prediction later (NEW)

Once stored, any downstream service can fetch the latest stored prediction for an entity:

```python
pred = client.fetch_materialized(
    name="my_model",
    entity_id="user_123",
    max_age_seconds=300  # optional freshness constraint
)

print(pred)
```

If the prediction does not exist or is too old, the server may return 404 (or an error depending on configuration).

---

## 🔔 Webhooks for Materialized Inference (NEW)

Webhooks allow downstream services (e.g., your backend) to get notified when predictions are ready — avoiding polling.

Typical pipeline:

1) Data/event triggers inference + materialization  
2) MLServe.com stores predictions  
3) MLServe.com sends webhook: `prediction.materialized`  
4) Backend fetches with `fetch_materialized()` and serves the frontend

### ✅ Register the webhook endpoint (MVP)

```python
client.set_webhook(
    url="https://your-backend.com/mlserve/webhook",
    secret="optional_shared_secret",
    is_active=True
)
```

When materialization completes, MLServe.com will POST a small payload like:

```json
{
  "event": "prediction.materialized",
  "model": "my_model",
  "entity_id": "user_123",
  "ts": "2025-12-25T10:12:33Z"
}
```

> For MVP, delivery is best-effort. You can start with this and later add retries / delivery logs as needed.

---


## 🧩 Supported Model Types

MLServe.com currently supports deployment for models built using the following frameworks:

| Framework / Library        | Supported Objects                                                                             | Notes                                                                                                 |
| -------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **scikit-learn**           | `BaseEstimator`, `Pipeline`                                                                   | Full support for all classifiers, regressors, transformers, and pipelines.                            |
| **XGBoost**                | `XGBClassifier`, `XGBRegressor`                                                               | Includes automatic conversion and serialization for efficient serving.                                |
| **LightGBM (sklearn API)** | `LGBMClassifier`, `LGBMRegressor`                                                             | Supports sklearn API models for classification and regression tasks.                                  |
| **CatBoost (sklearn API)** | `CatBoostClassifier`, `CatBoostRegressor`                                                     | Supports sklearn API models.                                                                          |
| **Custom Causal Models**   | Any ML model with covariates and actions                                                      | Deploy causal models built using standard ML libraries — no need for specialized causal frameworks.   |
| **Recommender Systems**    | Any sklearn-compatible model returning `predict_proba` scores                                 | Designed for ranking candidate items per user. Supports hybrid user–item tabular pipelines.           |
| **Outlier Detection**      | `IsolationForest`, `OneClassSVM`, `EllipticEnvelope`, or any sklearn-compatible anomaly model | Supports unsupervised anomaly detection in tabular data, including fraud, drift, and quality control. |


### 🧠 Causal Model Support

MLServe.com also supports deployment of **causal models** — even when they are built using standard ML estimators instead of specialized causal libraries like `econml` or `causalml`.

Simply train your model to estimate outcomes as a function of covariates and actions/interventions:

$$
\hat{y} = f(X, A)
$$

where:

* ( X ) = features / covariates
* ( A ) = treatment, action, or intervention

Once deployed, **MLServe.com automatically estimates treatment effects and determines the next best action** internally.
At inference time, you only need to provide the input features — and MLServe.com will return both the treatment effects and the recommended action.

### 🎯 Recommender Model Support

MLServe.com supports deployment of **ranking and recommendation models** trained on user–item interactions.  
These models typically return engagement likelihoods (`predict_proba`) for each user–item pair and can include both structured and text features.

Use cases include:
- Personalized product or content recommendations  
- Ranking candidate items for a user  
- Hybrid (tabular + text) recommenders with TF-IDF or embeddings  

At inference, the user sends a JSON payload with one user profile and a list of candidate items; MLServe.com returns ranked recommendations.



### ⚡ Outlier Detection Support

MLServe.com natively supports **unsupervised anomaly detection models** such as `IsolationForest`, `OneClassSVM`, and other sklearn-compatible estimators.  
These models can detect abnormal patterns in numeric or categorical tabular data — useful for:
- Fraud and transaction monitoring  
- Sensor or equipment fault detection  
- Data drift and quality control  

---

## 📊 Model Monitoring & Performance Tracking

MLServe.com makes it easy to **monitor deployed models in production**, track **performance over time**, and **detect data quality issues** — all through the SDK.

### 🔹 Retrieve recent online metrics

Get recent model metrics (e.g., accuracy, rewards) aggregated over a time window.

```python
metrics = client.get_online_metrics(
    name="my_model",
    version="v2",
    window_hours=168,  # past 7 days
    as_dataframe=True
)
print(metrics)
```

Returns a single-row **pandas DataFrame** (if `as_dataframe=True`) or a dictionary with unpacked metrics.

---

### 🔹 Track model evolution across versions

Compare metrics and deltas between model versions to see performance improvements or regressions over time.

```python
evolution = client.get_model_evolution(
    name="my_model",
    as_dataframe=True
)
evolution.head()
```

Returns a DataFrame with:
- Each row representing a model version  
- Columns for `metrics`, `deltas`, and `deployed_at` timestamps  

---

### 🔹 Get hourly metrics for a specific version

Fetch fine-grained endpoint performance data like requests, predictions, latency percentiles and throughput for a given model version.

```python
hourly = client.get_metrics(
    name="my_model",
    version="v2",
    hours=48,
    as_dataframe=True
)
hourly.tail()
```

Useful for **trend visualization** and **alerting pipelines**.

---

### 🔹 Check data quality (drift, missingness, outliers)

Monitor input data to ensure model stability and detect upstream data issues.

```python
dq = client.get_data_quality(
    name="my_model",
    version="v2",
    hours=24,
    as_dataframe=True
)
```

Returns a dictionary of DataFrames for:
- `missingness`: feature-wise missing value ratios  
- `drift`: distribution shifts vs. training data  
- `outliers`: detected anomalies in input features  

---

## 🎯 Reinforcement Learning (Policy) Models (Experimental)

MLServe.com makes it possible to build **self-learning decision systems** — without any ML training code.  
These **online policy models** learn directly from inference-time feedback, so there’s **no need for separate training pipelines**, datasets, or feature stores.

---

### ⚡ Why Use Policy Models

| Benefit                                | Description                                                                  |
| -------------------------------------- | ---------------------------------------------------------------------------- |
| 🧩 **No ML expertise needed**           | You don’t train or fine-tune anything — MLServe handles learning internally. |
| 🔄 **No data drift or schema mismatch** | The same data you send at prediction time is used for learning.              |
| 🚀 **Continuously improving**           | Models adapt to feedback automatically — no retraining required.             |
| 💡 **Developer-first workflow**         | Just send inputs → get action → send feedback — that’s it.                   |

---

### ⚙️ Configure an RL Policy Model

```python
from mlserve import MLServeClient
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

client = MLServeClient()

# 1️⃣ Configure RL model (support for numerical-only at the moment)
feature_contract = {
    "features": [
        {"name": "age", "type": "continuous"},
        {"name": "income", "type": "continuous"},
        {"name": "loyalty_score", "type": "continuous"}
    ],
    "actions": ["show_discount", "show_premium", "no_action"]
}

resp = client.configure_online_model(
    name="promo-policy",
    version="v5",
    task_type="policy_next",
    feature_contract=feature_contract
)

print("✅ RL Model configured:", resp)
deployment_id = resp["deployment_id"]

# 2️⃣ Wait for background deployment to finish
final_status = client.wait_for_deployment(deployment_id)
print("✅ Final deployment status:", final_status)
```

### 🔁 Simulate Online Learning

Once deployed, your RL policy model can predict actions and receive feedback continuously:

```python
# --- Simulation Setup ---
model_name = "promo-policy"
model_version = "v5"
BATCH_FEEDBACK_SIZE = 10
n_steps = 500
actions_map = {"show_discount": 0, "show_premium": 1, "no_action": 2}

feedback_buffer, rewards, avg_rewards = [], [], []

for t in trange(n_steps):
    # 1️⃣ Simulate a user context
    user_features = {
        "age": np.random.randint(18, 65),
        "income": np.random.uniform(20000, 100000),
        "loyalty_score": np.random.uniform(0, 1)
    }

    # 2️⃣ Get next action from RL policy
    pred = client.online_predict(model_name, model_version, inputs=[user_features])
    result = pred["predictions"][0]
    action_name = result["action"]
    policy_id = result["policy_id"]

    # 3️⃣ Simulate reward (example environment)
    reward = np.random.choice([0, 1], p=[0.7, 0.3])

    # 4️⃣ Add feedback to batch buffer
    feedback_buffer.append({
        "features": user_features,
        "action": action_name,
        "reward": reward,
        "policy_id": policy_id
    })

    # 5️⃣ Send feedback in batches
    if len(feedback_buffer) >= BATCH_FEEDBACK_SIZE or t == n_steps - 1:
        client.online_feedback(model_name, model_version, feedback_buffer)
        feedback_buffer.clear()

    rewards.append(reward)
    avg_rewards.append(np.mean(rewards[-50:]))

# --- 🔹 Visualization ---
plt.figure(figsize=(10, 5))
plt.plot(avg_rewards, color="C0")
plt.title("Online RL Policy: Average Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Reward (avg last 50)")
plt.grid(True)
plt.show()
```

### 🧩 How It Works

1. Model Configuration
Define your feature space and actions using a JSON `feature_contract`.
MLServe automatically initializes a multi-policy RL agent.

2. Action Selection (`online_predict`)
The model returns the next action given current user features, plus metadata:

* `action`: selected action name
* `policy_id`: policy index used
* `mode`: `"explore"` or `"exploit"`
* `epsilon`: current exploration rate

3. Feedback Loop (`online_feedback`)
After each action, send back a reward (e.g. conversion = 1, no conversion = 0).
The RL agent updates its internal weights to improve over time.

4. Continuous Learning
Each call updates the model’s policy.

### 📈 Typical Use Cases

| Scenario                    | Description                                                    |
| --------------------------- | -------------------------------------------------------------- |
| **Marketing optimization**  | Test and adapt campaign strategies dynamically.                |
| **Pricing decisions**       | Adjust discounts or promotions based on real-time performance. |
| **Personalization**         | Learn user preferences across products, ads, or notifications. |
| **Multi-policy evaluation** | Compare several policy networks simultaneously.                |


---

## 🔐 Google OAuth Authentication (Optional)

```python
auth_url = client.get_google_auth_url()
print("Visit this URL to authenticate:", auth_url)
```

After the user grants access, MLServe.com will handle the token exchange.

---

## ⚡ SDK Reference

| Method                                          | Description                                                     |
| ----------------------------------------------- | --------------------------------------------------------------- |
| `register(user_name, email, password)`          | Register a new account                                          |
| `login(email, password)`                        | Login and obtain an access token                                |
| `logout()`                                      | Logout the current session                                      |
| `check_token()`                                 | Verify token and return current user info                       |
| `invite_user(email)`                            | Invite a new user to your team                                  |
| `list_team()`                                   | List all users in the organization                              |
| `update_user_role(user_id, role)`               | Change user role (admin/user)                                   |
| `remove_team_member(user_id)`                   | Disable a user account                                          |
| `request_password_reset(email, new_password)`   | Send password reset email                                       |
| `deploy_model(...)`                             | Deploy a trained ML model                                       |
| `configure_online_model(...)`                   | Deploy a RL agent                                               |
| `wait_for_deployment(...)`                      | Check deployment progress                                       |
| `predict(name, version, data, materialize=...)` | Make predictions (or materialize outputs by entity_id)          |
| `predict_weighted(name, data, materialize=...)` | Weighted predictions (or materialize outputs by entity_id)      |
| `fetch_materialized(name, entity_id, ...)`      | Fetch a stored prediction by entity_id (materialized inference) |
| `set_webhook(url, secret=None, is_active=True)` | Register/update webhook for materialization events (MVP)        |
| `configure_abtest(name, weights)`               | Configure A/B test weights                                      |
| `list_models()`                                 | List all deployed models                                        |
| `get_latest_version(model_name)`                | Get the latest deployed version                                 |
| `google_login()`                                | Login with Google OAuth                                         |
| `get_online_metrics(name, version)`             | Retrieve recent performance metrics                             |
| `get_model_evolution(name)`                     | Retrieve performance evolution                                  |
| `get_metrics(name, version, hours)`             | Fetch hourly metrics for a given ML model                       |
| `get_data_quality(name, version)`.              | Retrieve data quality metrics                                   |

---

## 🧱 Example Workflow

```python
from mlserve import MLServeClient

client = MLServeClient()

# Step 1: Register a new account
client.register("Bob", "bob@example.com", "Secure123!")

# Step 2: Verify via email
# (User clicks link in email)

# Step 3: Login
login_data = client.login("bob@example.com", "Secure123!")

# Step 4: Invite teammates
client.invite_user("teammate@example.com")

# Step 5: Deploy a model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
client.deploy_model(model=model, name="my_model", version="v1", features=["f1", "f2"], background_df=df.sample(100))

# Step 6: Make predictions
data = {"inputs": [{"f1": 1, "f2": 2}]}
preds = client.predict("my_model", "v1", data)
print(preds)
```

---

## 🔒 Data & Privacy Policy (Updated)

**1 Data Processing and Storage**

MLServe.com processes data transmitted through the SDK solely for the purpose of providing and improving its services, including model prediction, monitoring, and user management functionalities.

By default, MLServe.com does not permanently store user input data submitted for prediction. Such data may be temporarily cached (for up to five minutes) in an in-memory store (Redis) to optimize performance and prevent redundant processing.

**Materialized inference exception:**  
If users enable `materialize=True`, MLServe.com stores the **prediction output** (and optionally metadata such as timestamps and explanations) keyed by `entity_id` in an online store (e.g., Redis) with a configurable **time-to-live (TTL)**. Materialized outputs expire automatically after TTL and are not intended for permanent storage.

**2 Request and Performance Logging**

For operational purposes, MLServe.com maintains limited system logs containing aggregate request information such as request counts, processing latency, and error metrics. These logs do not contain user inputs or personally identifiable prediction data.

**3 Feedback Data**

If users provide feedback (e.g., true labels, performance scores, or reward values), MLServe.com may store this information to evaluate and improve model accuracy. Feedback data are not linked to the original prediction inputs, ensuring that they cannot be used to reconstruct user data.

**4 Security and Encryption**

All communication between the SDK and the MLServe.com API occurs over encrypted HTTPS (TLS) connections.

**5 Data Retention**

MLServe.com retains only data necessary for:
- Account and authentication management
- Performance and usage analytics
- Feedback analysis for model improvement

Materialized outputs (when enabled) expire automatically according to TTL.

**6 User Responsibilities and Compliance**

Users are responsible for ensuring that their use of MLServe.com complies with all applicable privacy and data protection regulations, including but not limited to GDPR, HIPAA, and relevant local laws.

**7 Liability and Disclaimer**

MLServe.com employs reasonable technical and organizational safeguards to protect user data. However, no system can be guaranteed to be completely secure.

---

## 💬 Support

- 📧 Email: [support@mlserve.com](mailto:support@mlserve.com)

---

## 🧾 License

This SDK is licensed under the **Apache Software License**.  
© 2025 MLServe.com — All rights reserved.

---