# Credit Card Fraud Detection Project

## Overview

This project implements a machine learning model to detect fraudulent credit card transactions. Using XGBoost classification and advanced data preprocessing techniques, the model achieves high accuracy in identifying suspicious transactions while maintaining a low false alarm rate.

## Project Structure

fraud-detection/
├── data/ # Data files (not tracked in git)
├── notebooks/ # Jupyter notebooks
│ └── fraud_detection.ipynb
├── scripts/
│ ├── eda_and_preprocessing.py
│ └── model_training.py
├── outputs/ # Saved models and plots (not tracked in git)
├── .gitignore
└── README.md


## About the Data

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains credit card transactions made by European cardholders over a two-day period in September 2013. It includes 284,807 transactions, of which 492 are fraudulent. The dataset is highly imbalanced, with only 0.172% of transactions being fraudulent.


## Key Features
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- XGBoost model implementation
- Performance evaluation and visualization
- Business metrics analysis

## Results
- AUC-ROC Score: 0.98
- Fraud Detection Rate: 85.71%
- False Alarm Rate: 0.13%

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn



## Model Performance
- High precision in fraud detection
- Low false positive rate
- Robust performance on imbalanced data
- Real-time scoring capability

## Detailed Performance Metrics

### Model Performance at Different Thresholds

| Threshold | Detection Rate (%) | False Alarm Rate (%) |
|-----------|-------------------|---------------------|
| 0.1       | 90.82            | 0.95               |
| 0.3       | 86.73            | 0.23               |
| 0.5       | 85.71            | 0.13               |
| 0.7       | 84.69            | 0.07               |
| 0.9       | 82.65            | 0.05               |

### Key Performance Indicators

| Metric                  | Value    |
|------------------------|----------|
| AUC-ROC Score          | 0.98     |
| Total Transactions     | 56,962   |
| Fraud Cases Detected   | 84       |
| False Alarms           | 74       |
| Detection Rate         | 85.71%   |
| False Alarm Rate       | 0.13%    |

These results demonstrate the model's strong performance in detecting fraudulent transactions while maintaining a low false alarm rate. The threshold analysis shows the trade-off between detection rate and false alarms, allowing for flexible deployment based on business requirements.



