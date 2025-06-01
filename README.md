# EL-Task-4
# Logistic Regression on Breast Cancer Dataset

This code demonstrates a complete machine learning workflow using Logistic Regression for binary classification. We use a breast cancer dataset (loaded from "data.csv") to classify tumors as malignant (M) or benign (B).

# Objectives

1. Choose a binary classification dataset.
2. Split data into training and testing sets and standardize features.
3. Fit a logistic regression model.
4. Evaluate the model using:
   - Confusion matrix
   - Precision, Recall, F1-score
   - ROC-AUC score and curve
5. Tune the decision threshold and explain the sigmoid function.

# Dataset

The dataset contains features derived from digitized images of breast mass cell nuclei. Key features include:
- Radius, texture, perimeter, area, smoothness, etc.
- Diagnosis ("M" = malignant, "B" = benign)

# Libraries Used

- pandas, numpy: Data handling
- matplotlib, seaborn: Visualization
- scikit-learn: ML models and evaluation tools

# Procedure

1.Import and Preprocess
- Load CSV using pandas
- Drop irrelevant columns like `id` and 'Unnamed: 32' (if they exist)
- Convert 'diagnosis' column to binary (1 = M, 0 = B)

2. Train-Test Split and Scaling
- Split data into train (80%) and test (20%) sets
- Standardize features using 'StandardScaler'

3. Model Training
- Train a Logistic Regression model using `sklearn.linear_model.LogisticRegression`

4. Model Evaluation
- Use:
  - confusion_matrix
  - classification_report
  - roc_auc_score & roc_curve
- Plot ROC Curve to visualize classifier performance

5. Threshold Tuning
- Adjust default threshold (0.5) to a custom one (e.g., 0.3)
- Show how predictions and confusion matrix change

6. Sigmoid Function
- Visualize the sigmoid activation function to show how it converts linear output into probability


