# Prerequisites & Setup Guide  
**Pricing Optimization Tutorial (MLServe.com + Synthetic Oracle)**

This document explains everything required to run the pricing optimization notebook end-to-end.

---

## 1) System Requirements

- **Python**: 3.10+ (3.11 recommended)
- **Operating System**: macOS / Linux / Windows (WSL supported)
- **Internet Access**: required for MLServe.com API calls
- **Jupyter**: JupyterLab or Jupyter Notebook

---

## 2) Project Structure

Your working directory should look like this:
.
├── airline.ipynb
├── setup.md
├── .env # MLServe.com credentials
└── oracle.py # PricingOracle implementation
└── README.md

## 3) Create and Activate a Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (PowerShell)

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 4) Install Python Dependencies

Install all required packages with:

```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn python-dotenv tqdm jupyterlab mlserve-sdk
```

## 5) Environment Variables (Required)

```base
touch .env
```

Add your MLServe.com credentials:

```env
USERNAME=your_mlserve_username
TOKEN=your_mlserve_password
```

The notebook loads these using:

```python
from dotenv import load_dotenv
load_dotenv()
```

If either value is None, the .env file is missing or misconfigured.

## 6) Running the Notebook

```bash
jupyter lab
```
Open notebook.ipynb and run all cells top to bottom.

## 7) Important Implementation Notes

### Prediction batching

* The MLServe.com prediction endpoint accepts up to 500 records per request.
* The notebook already handles batching internally.
* Removing batching will cause API errors.

### Feature consistency

* The deployed feature list must exactly match:
  * the training dataframe columns
  * the candidate dataframe columns sent to `predict`
* The `price` column must be included as a model feature.

### Identifiers

* `user_id` is an identifier and must not be used as a model feature.
* It is required only for grouping predictions and scoring outcomes.

## 8) Expected Outputs

When run successfully, the notebook will:

1. Train and deploy a demand model.
2. Predict purchase probabilities for each (user, price) pair.
3. Select revenue-maximizing prices.
4. Evaluate revenue using the oracle.
5. Send feedback to MLServe.com.
6. Retrieve online metrics for all submitted model versions.