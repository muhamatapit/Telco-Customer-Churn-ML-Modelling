# Telco-Customer-Churn-ML-Modelling

## Background

In the highly competitive mobile telecommunications industry, understanding customer churn is crucial. Customer churn, or attrition, occurs when clients switch to a different service provider. This is often due to dissatisfaction with service, competitive offers, or personal changes.

The "Telco Customer Churn" dataset provides insights into the profiles of customers who have left a telecom company. It includes data on **'Dependents', 'OnlineSecurity', 'OnlineBackup', 'InternetService', 'DeviceProtection', 'TechSupport', 'Contract', 'PaperlessBilling', 'tenure', 'monthly charges' and 'Churn'**. The main focus is on the 'Churn' attribute, indicating whether a customer has ended their service.

A telecom provider aims to reduce churn by predicting which customers are likely to leave using machine learning. This predictive model will help the provider to develop strategies to retain customers and reduce churn.
<br>

> The cost of acquiring new customers is five times higher than the cost of retaining existing customers. While acquisition allows you to increase the amount of customers you have, customer retention allows you to maximize the value of customers you have already captured. https://www.optimove.com/resources/learning-center/customer-acquisition-vs-retention-costs#:~:text=The%20cost%20of%20acquiring%20new,customers%20you%20have%20already%20captured.

> Thus, we can assume the retention cost is **$1** and Acquisition Cost is **$5**

## Problem Statement

 Our main goal is to figure out why customers are leaving a telecom company and to build a model that can predict which customers might leave next. The big question we try to answer is, `What makes a customer want to leave, and how can we spot them before they go?` By doing this, we aim to help companies keep their customers for a longer time and save company's money. 

## Project Objective

To maximize cost efficiency, the company aims to focus on targeted customers (those potentially at risk of churning) by increasing `Recall Score`. The approach involves:

1. The company wants to **predict which customers are likely to churn**. This will enable them to concentrate their efforts on targeted customers who are at risk of churning.

2. The company wants to **identify the factors that influence customer retention**. This knowledge will help them develop more targeted programs to reduce the number of customers who churn.

3. The company wants to know **how much savings can be achieved** with the implementation of a machine learning model.

## Defining Target and Matrix

Target:

0 : Not Churn (Retained)

1 : Churn (Turnover)

- True Positives (TP) = Prediction: `churn` and actual: `churned` 
- True Negatives (TN) = Prediction: `not churn` and actual: `not churned`
- `False Positives (FP)` =  Prediction: `churn` and actual: `not churned`
- `False Negatives (FN)`=   Prediction: `not churn` and actual: `churned`

*In the context of customer churn, both types of errors can be costly for the business:*

- False Positives (FP): These are customers mistakenly identified as likely to churn. Although resources spent on them might seem wasted, this can unexpectedly boost their satisfaction and loyalty. **The company loss  $ 1 for regularly retention cost**

- False Negatives (FN): These are customers who actually churn but were not identified as at-risk. Missing these customers is more problematic as it leads to direct revenue loss and missed opportunities for service improvement. **The company will loss their CAC as much as  $ 5 for acquiring**

> Reducing False Negatives (FN) is crucial in business, as it's more cost-effective to retain existing customers than acquire new ones. Focusing on accurately identifying customers at risk of churning aligns with a cost-efficient Customer Retention Strategy, ultimately saving resources and enhancing business sustainability.

> We will utilize `Recall` as the primary metric for model evaluation, and the effectiveness of the **Customer Retention Cost Strategy** will guide our selection of the most suitable model for this project.

## **Project Flowchart**
1. Introduction

1. Data Understanding

1. Data Wrangling

1. Data Preprocessing
    - MinMaxScaler : We apply MinMaxScaler to scale numerical data because the distribution is not normal and there are no outliers.
    - One-Hot Encoding : We use one-hot encoding for categorical data to convert categories into a numerical format that can be used in our models.

1. Machine Learning Model Benchmark<br>
    a. Model Banchmarking
    - Logistic Regression
    - K-Nearest Neighbors
    - Decision Tree
    - XGBoost
    - Gradient Boosting
    - Random Forest
    - LGBMClassifier
    - CatBoostClassifier
    - AdaBoostClassifier

    b. Addressing Imbalance with Resample benchmarking<br>
    - RandomOverSampler
    - RandomUnderSampler
    - SMOTE
    - Near Miss
    - SMOTEENN 
    - EditedNearestNeighbours

1. Applying to Test Data

    After identifying the top five performers from the modeling and resampling phase, we apply "the top 5" from each model and resampling method to predict outcomes on the test data and evaluate the prediction scores.

1. Hyperparameter Tuning with Grid Search

    We conduct hyperparameter tuning using Grid Search on the **best-performing model and resampling method** based on the test data results.

1. Final Model Explanation (in this project is Logistic Regression With SMOTEENN)
    - Prediction And Actual Analysis
    - Feature Importance Analysis
    - SHAP Analaysis
    - Lime Analaysis
    - Cost Calculation Analysis

1. Conclusion
1. Recomendation
