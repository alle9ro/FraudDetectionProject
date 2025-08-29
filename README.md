# Fraud Detection Project

This project implements a **Fraud Detection System** using Machine Learning and Deep Learning techniques on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. The goal is to accurately identify fraudulent transactions in a highly imbalanced dataset, showcasing expertise in data preprocessing, model training, and performance evaluation.

## Problem Statement
The Credit Card Fraud Detection dataset contains anonymized credit card transactions, with only 0.17% labeled as fraudulent (Class 1). This extreme class imbalance presents a challenge, as models may overfit to the majority class (non-fraudulent transactions, Class 0) and fail to detect frauds. The primary objective is to maximize the detection of fraudulent transactions (high Recall) while minimizing false positives (high Precision) to ensure an effective fraud detection system.

## Methodology
To address the problem, the following approach was implemented:

- **Data Preprocessing**:
  - Features were normalized using `StandardScaler` to ensure consistent scaling across variables.
  - Synthetic Minority Oversampling Technique (SMOTE) was applied to balance the training data by generating synthetic fraud cases, mitigating the class imbalance issue.
- **Model Selection**:
  - **Random Forest**: An ensemble model combining multiple decision trees to capture complex patterns. It is well-suited for imbalanced datasets due to its robustness and ability to handle feature interactions.
  - **Neural Network**: A deep learning model with three hidden layers (64, 32, and 16 neurons) and 30% dropout layers to prevent overfitting. It is designed to learn intricate relationships in the data.
- **Training**:
  - Random Forest was trained with 100 estimators and parallel processing enabled.
  - Neural Network was trained for 50 epochs with a batch size of 32, using binary cross-entropy loss and the Adam optimizer.
- **Evaluation**:
  - Models were evaluated using Precision, Recall, F1-Score, and Area Under the ROC Curve (AUC), focusing on fraud detection performance.
  - Visualizations (Confusion Matrix and ROC Curve) were generated to provide insights into model behavior.

## Results Analysis
The models were evaluated on a test set containing 56,962 transactions (98 frauds, 56,864 non-frauds). Key metrics for the fraud class (Class 1) are summarized below:

- **Random Forest**:
  - **Precision**: 0.87 (87% of transactions predicted as fraud were actually fraudulent).
  - **Recall**: 0.83 (83% of actual frauds were correctly detected).
  - **F1-Score**: 0.85 (balanced measure of Precision and Recall).
  - **AUC**: 0.98 (excellent ability to distinguish frauds from non-frauds).
  - The Random Forest model achieved high Precision, reducing false positives, which is critical to avoid flagging legitimate transactions as fraudulent.

- **Neural Network**:
  - **Precision**: 0.74 (74% of transactions predicted as fraud were actually fraudulent).
  - **Recall**: 0.86 (86% of actual frauds were correctly detected).
  - **F1-Score**: 0.80 (slightly lower than Random Forest due to lower Precision).
  - **AUC**: 0.96 (strong performance, slightly below Random Forest).
  - The Neural Network prioritized Recall, detecting a higher percentage of frauds, but produced more false positives compared to Random Forest.

- **Why Accuracy is Less Relevant**:
  - Due to the dataset's imbalance (99.83% non-fraudulent transactions), both models achieved near 100% accuracy. However, accuracy is misleading in this context, as a naive model predicting all transactions as non-fraudulent would also achieve high accuracy but fail to detect frauds. Instead, Recall, Precision, F1-Score, and AUC were prioritized to evaluate fraud detection performance effectively.

- **Visualizations**:
  - **Confusion Matrix**: Illustrates the distribution of true positives, false positives, true negatives, and false negatives.
  - **ROC Curve**: Shows the trade-off between True Positive Rate (Recall) and False Positive Rate, with AUC values indicating excellent model performance.
