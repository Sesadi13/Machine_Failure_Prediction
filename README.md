# Machine_Failure_Prediction

A machine learning project that predicts whether a machine failure
will occur within the next 24 hours using sensor measurements and
operating conditions.

## Problem

Unexpected machine failures can lead to production downtime,
maintenance costs, and operational disruptions.

The goal of this project is to predict:

- `0` → No failure expected within 24 hours
- `1` → Failure expected within 24 hours

Dataset: https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance

## Features

The model uses:

- Vibration RMS
- Motor temperature
- Average phase current
- Pressure level
- RPM
- Hours since maintenance
- Ambient temperature
- Machine type
- Operating mode

## Machine Learning Models

Three classification models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
