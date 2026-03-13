# Customer Churn Prediction

A Streamlit web app to predict customer churn using the Online Retail II dataset.

## Live Demo
🔗 [customer-churn-prediction.streamlit.app](https://customer-churn-prediction-jzhdappppcwmuzsrs5b6lqnm.streamlit.app)

## Folder Structure

```
customer-churn-prediction/
├── pkl/
│   ├── churn_model.pkl
│   └── churn_scaler.pkl
├── Model/
│   └── customer_churn_analysis.ipynb
├── application.py
├── requirements.txt
├── runtime.txt
└── README.md
```

## Features

- **Single Prediction** — Enter Invoice, Quantity, Price to get instant churn prediction
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

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Dhairya-45/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run application.py
```

## Dataset

Download from: https://archive.ics.uci.edu/dataset/502/online+retail+ii

Place it inside the `Model/` folder before running the notebook.