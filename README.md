# Predict Customer Churn

A basic MLOps system that follows coding (PEP8) and engineering best practices.

## Purpose

The purpose of the project is to implement modular, documented, and tested software to predict which customers are most likely to churn at a bank.

## Directory Overview

This is what the directory will look like after running churn_script_logging_and_tests.py 

* data
    - bank_data.csv - data used for predictions
* images
    - eda
        - churn_distribution.png - bar plot of the distribution of the target variable: Churn
        - customer_age_distribution.png - bar plot of the distribution of Age
        - heatmap.png - correlation heatmap
        - marital_status_distribution.png - bar plot of the distribution of Marital_Status
        - total_transaction_distribution.png - bar plot of the distribution of total_transactions
    - results
        - feature_importance.png - plot of feature importance for the random forest model
        - logistics_results.png - table of resulting metrics for the logistic regression model
        - rf_results.png - table of resulting metrics for the random forest model
        - roc_curve_result.png - ROC curve for the logistic and random forest models
    - logs
        - churn_library.log - logged outputs from tests in churn_scipt_logging_and_tests.py
    - models
        - logistic_model.pkl - saved logistic regression model from the train_models function in churn library
        - rfc_model.pkl - saved random forest model from the train_models function in churn library
* churn_library.py - functions for EDA, feature engineering, and model training
* churn_notebook.ipynb - jupyter notebook for doing ad-hoc analyses
* churn_script_logging_and_tests.py - tests and executes the functions in churn_library.py

## Instructions

Run `ipython churn_script_logging_and_tests.py` to test each function in churn_library.py. This should produce successfull tests in the logs folder.

## Dependencies
joblib to save trained models

pandas for data organizing and processing

matplotlib for visualizations

seaborn for heatmap visualiation

scikit-learn for training and diagnosing the models
