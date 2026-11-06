# AIL_FraudDetection
# Credit Card Fraud Detection

Machine learning models to classify credit card transactions as fraudulent or legitimate, with a focus on handling severe class imbalance.

## Overview

Fraudulent transactions are far rarer than legitimate ones, so accuracy alone is misleading. This project compares four classifiers and evaluates them with metrics suited to imbalanced data.

## Dataset

- **Source:** [Credit Card Fraud Detection (Kaggle, ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Anonymized credit card transactions made by European cardholders in September 2013.
- 284,807 transactions, highly imbalanced: fraud cases are a tiny fraction of the data.
- Download `creditcard.csv` from the link above and place it next to the notebook.

## Approach

1. **Data preprocessing:** cleaning and preparing the transaction data for modeling.
2. **Class imbalance:** applied SMOTE to oversample the minority (fraud) class.
3. **Models compared:**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
4. **Evaluation:** Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.

## Results

The Random Forest baseline on the original dataset achieved an **F1-score of 0.831**.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn, Matplotlib

## Getting Started

Open the notebook in Jupyter Notebook or Google Colab and run the cells in order.

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib
```

## Author

Nguyen Gia Phuc, AI student at FPT University
