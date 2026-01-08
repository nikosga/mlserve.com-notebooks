# Live Tutorial: Pricing Optimization with Predictive Models and Online Evaluation

## Overview

This tutorial is a **hands-on, end-to-end pricing optimization exercise** designed for a 1-hour live session.  
Students build, deploy, and evaluate a predictive pricing model in a realistic simulated business environment.

The goal is **not** to teach new machine learning models, but to demonstrate how:
- statistical models are used in decision-making,
- predictions translate into economic objectives,
- deployment and feedback close the learning loop.

The exercise emphasizes **applied econometrics and decision theory**, rather than algorithmic complexity.

---

## Learning Objectives

By the end of the session, students will be able to:

1. Model demand using logistic regression.
2. Translate predicted probabilities into an optimal pricing decision.
3. Deploy a model as a live service.
4. Evaluate performance using **business metrics**, not just statistical accuracy.
5. Understand the distinction between **prediction quality** and **decision quality**.

---

## Economic Setting

We simulate a simplified airline pricing problem.

Each user represents a potential customer searching for a ticket.  
For each user \( i \), we observe a feature vector \( x_i \) (e.g. urgency, income proxy, trip type).

The firm chooses a price \( p \in \mathcal{P} \) from a fixed price grid.

The customer then makes a purchase decision:
\[
y_i \sim \text{Bernoulli}\left( \Pr(y_i = 1 \mid x_i, p) \right)
\]

The **true demand function** is unknown to students and fixed throughout the exercise.

---

## Data Provided to Students

### Training Dataset

The training dataset consists of historical observations generated under a logged pricing policy.

Each row contains:
- user features \( x_i \)
- price \( p_i \)
- purchase outcome \( y_i \in \{0,1\} \)

This dataset is used to estimate demand.

### Test Dataset

The test dataset contains:
- user features only
- no prices
- no purchase outcomes

Students must choose prices for these users.

---

## Modeling Demand

Students estimate a logistic regression model:

\[
\Pr(y = 1 \mid x, p) = \sigma\left( \beta_0 + \beta_x^\top x + \beta_p p + \beta_{xp}^\top (x \cdot p) \right)
\]

where:
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Students are free to:
- include or exclude interactions,
- use regularization,
- evaluate goodness-of-fit on validation data.

Model choice is deliberately kept simple to focus on decision-making.

---

## From Prediction to Pricing

Prediction alone is not the objective.

For each user \( i \), students compute **expected revenue**:

\[
\mathbb{E}[R_i(p)] = p \cdot \Pr(y_i = 1 \mid x_i, p)
\]

Given a finite price grid \( \mathcal{P} \), the pricing rule is:

\[
p_i^* = \arg\max_{p \in \mathcal{P}} \; p \cdot \Pr(y_i = 1 \mid x_i, p)
\]

This step explicitly links econometric modeling to an optimization problem.

---

## Deployment

The trained model is deployed using **MLServe.com** as a live prediction service.

Deployment exposes a function:

\[
(x_i, p) \rightarrow \hat{\Pr}(y_i = 1)
\]

Students interact with the deployed model via API calls, mirroring real-world production workflows:
- batch inference
- request identifiers
- logging and monitoring

---

## Online Evaluation and Feedback

Chosen prices are evaluated using a **hidden demand oracle**:
- the oracle uses the true data-generating process,
- outcomes are not observable to students ex ante.

For each priced user, the system returns:
- realized purchase outcome
- realized revenue

These outcomes are registered as feedback to the deployed model.

---

## Performance Metrics

Models are compared using **economic performance**, not predictive accuracy alone.

Primary metric:
\[
\text{Total Revenue} = \sum_i p_i^* \cdot y_i
\]

Secondary diagnostics:
- conversion rate
- average price
- classification accuracy (for reference)

This highlights the distinction between:
- *statistical fit* and
- *decision quality under uncertainty*.

---

## Competition and Discussion

Students are ranked by total revenue.

This typically leads to discussion around:
- price sensitivity vs. margin tradeoffs,
- overfitting vs. robustness,
- why higher accuracy does not guarantee higher revenue,
- the economic interpretation of regression coefficients.

---

## Summary

This tutorial provides a compact, realistic demonstration of:

- applied demand estimation,
- optimization under uncertainty,
- the operational role of deployed models,
- evaluation using business-relevant metrics.

It bridges econometrics, machine learning, and decision-making in a single, coherent exercise suitable for advanced undergraduate or graduate courses.
