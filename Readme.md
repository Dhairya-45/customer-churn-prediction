# Customer Churn Prediction

A Streamlit web app to predict customer churn using the Online Retail II dataset.

## Folder Structure

```
project/
├── pkl/
│   ├── churn_model.pkl
│   └── churn_scaler.pkl
├── application.py
├── requirements.txt
└── online_retail_II.csv
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate pickle files
Run these cells in your notebook:
```python
import pickle

with open('pkl/churn_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('pkl/churn_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

### 3. Run the app
```bash
streamlit run application.py
```

## Features

- **Single Prediction** — Enter Invoice, Quantity, Price to get churn prediction
- **Batch Prediction** — Upload a CSV and score multiple customers at once
- **Download Results** — Export batch predictions as CSV

## Model Details

| Item | Detail |
|---|---|
| Algorithm | Logistic Regression |
| Scaler | StandardScaler |
| Features | Invoice, Quantity, Price |
| Churn Definition | No purchase in last 90 days |
| Dataset | Online Retail II (UCI) |

## Dataset

Download from: https://archive.ics.uci.edu/dataset/502/online+retail+ii