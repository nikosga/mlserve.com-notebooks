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

## 🧠 Model Serving & Deployment

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
print(response)
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

## 🔐 Google OAuth Authentication (Optional)

```python
auth_url = client.get_google_auth_url()
print("Visit this URL to authenticate:", auth_url)
```

After the user grants access, MLServe.com will handle the token exchange.

---

## ⚡ SDK Reference

| Method                                        | Description                               |
| --------------------------------------------- | ----------------------------------------- |
| `register(user_name, email, password)`        | Register a new account                    |
| `login(email, password)`                      | Login and obtain an access token          |
| `logout()`                                    | Logout the current session                |
| `check_token()`                               | Verify token and return current user info |
| `invite_user(email)`                          | Invite a new user to your team            |
| `list_team()`                                 | List all users in the organization        |
| `update_user_role(user_id, role)`             | Change user role (admin/user)             |
| `remove_team_member(user_id)`                 | Disable a user account                    |
| `request_password_reset(email, new_password)` | Send password reset email                 |
| `deploy_model(...)`                           | Deploy a trained ML model                 |
| `predict(name, version, data)`                | Make predictions with a deployed model    |
| `predict_weighted(name, data)`                | Weighted predictions across versions      |
| `configure_abtest(name, weights)`             | Configure A/B test weights                |
| `list_models()`                               | List all deployed models                  |
| `get_latest_version(model_name)`              | Get the latest deployed version           |
| `google_login()`                              | Login with Google OAuth                   |
| `get_online_metrics(name, version)`           | Retrieve recent performance metrics       |
| `get_model_evolution(name)`                   | Retrieve performance evolution            |
| `get_metrics(name, version, hours)`           | Fetch hourly metrics for a given model    |
| `get_data_quality(name, version)`.            | Retrieve data quality metrics             |

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

## Data & Privacy Disclaimer

- The MLServe.com SDK sends data to the MLServe.com API for predictions, registration, feedback, and other services.
- Data transmitted may include input features, user identifiers (emails, IDs), and feedback information.
- MLServe.com may store, process, and log any data sent via the SDK for analytics, model improvement, or operational purposes.
- Users are responsible for ensuring compliance with applicable privacy laws and regulations (e.g., GDPR, HIPAA).
- By using this SDK, you acknowledge that MLServe.com **does not guarantee the privacy or confidentiality** of transmitted data.
- All actions using the SDK are performed **at your own risk**, and MLServe.com is **not liable** for any misuse, data loss, or unintended exposure.
- It is recommended to anonymize sensitive data before sending it through the SDK.

---

## 💬 Support

* 📧 Email: [support@mlserve.com](mailto:support@mlserve.com)
---

## 🧾 License

This SDK is licensed under the **Apache Software License**.
© 2025 MLServe.com — All rights reserved.

---