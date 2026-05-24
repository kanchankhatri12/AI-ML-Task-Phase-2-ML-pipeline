#  Customer Churn Prediction Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Project Overview

An end-to-end machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset. This production-ready solution achieves **81.5% accuracy** in identifying customers likely to churn.

### Key Features
-  Complete preprocessing pipeline (scaling + encoding)
-  Multiple models (Logistic Regression & Random Forest)
-  Hyperparameter tuning with GridSearchCV
-  Model export using joblib
-  Production-ready predictions

##  Dataset

**Telco Customer Churn Dataset** - 7,043 customer records with 21 features

| Feature Type | Examples |
|-------------|----------|
| Demographics | Gender, Senior Citizen, Partner |
| Account Info | Tenure, Contract Type, Payment Method |
| Services | Internet, Phone, Streaming, Security |
| Charges | Monthly Charges, Total Charges |

**Target**: Customer churn (Yes/No) - 26.5% churn rate

##  Quick Start

### 1. Clone & Install

# Clone repository
git clone https://github.com/kanchankhatri12/AI-ML-Task-Phase-2-ML-pipeline.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt
2. Run Training
python
# Option A: Run Python script
python train.py

# Option B: Use Jupyter notebook
jupyter notebook churn_prediction.ipynb
3. Make Predictions
python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('churn_prediction_pipeline.pkl')

# Predict new customer
new_customer = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['No'],
    'tenure': [24],
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['Yes'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['Yes'],
    'StreamingMovies': ['No'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [85.5],
    'TotalCharges': [1500.0]
})

prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

print(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Churn Probability: {probability[0][1]:.2%}")
 Model Performance
Results Comparison
Model	Accuracy	ROC-AUC	CV Score
Logistic Regression (Base)	80.12%	83.45%	83.12%
Random Forest (Base)	79.23%	82.67%	82.45%
Logistic Regression (Tuned)	81.56%	84.89%	84.56%
Random Forest (Tuned)	81.02%	84.23%	83.98%
Best Model: Tuned Logistic Regression
Test Accuracy: 81.56%

ROC-AUC: 0.8489

5-fold CV Score: 0.8456

Top 5 Important Features
Tenure (customer loyalty) - Most important

Monthly Charges - Higher charges = higher churn

Contract Type - Month-to-month has 3x churn rate

Total Charges - Lifetime value indicator

Internet Service Type - Fiber optic users churn more
