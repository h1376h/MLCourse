# Regression vs Classification

## Overview
In supervised machine learning, there are two primary types of tasks: regression and classification. Understanding the difference between these tasks is essential for selecting appropriate models and evaluation metrics.

## Regression
Regression predicts continuous, numerical outputs.

### Key Characteristics of Regression:
- **Output Type**: Continuous numerical values (real numbers)
- **Goal**: Predict a quantity or value
- **Example Problems**: 
  - House price prediction
  - Temperature forecasting
  - Stock price prediction
  - Age estimation
  - Sales forecasting

### Common Regression Models:
- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression
- Decision Tree Regression
- Random Forest Regression
- Neural Network Regression

### Evaluation Metrics for Regression:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (Coefficient of Determination)
- Adjusted R-squared
- Mean Absolute Percentage Error (MAPE)

## Classification
Classification predicts discrete, categorical outputs or class labels.

### Key Characteristics of Classification:
- **Output Type**: Discrete categorical values (classes or labels)
- **Goal**: Predict a category or class membership
- **Example Problems**:
  - Email spam detection (spam/not spam)
  - Disease diagnosis (positive/negative)
  - Image classification (cat/dog/other)
  - Sentiment analysis (positive/negative/neutral)
  - Customer churn prediction (will churn/won't churn)

### Types of Classification Problems:
- **Binary Classification**: Two possible output classes
- **Multi-class Classification**: More than two mutually exclusive classes
- **Multi-label Classification**: Multiple non-exclusive classes can be assigned simultaneously

### Common Classification Models:
- Logistic Regression
- Support Vector Machines
- Decision Trees
- Random Forests
- Naive Bayes
- K-Nearest Neighbors
- Neural Networks

### Evaluation Metrics for Classification:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- Area Under ROC Curve (AUC-ROC)
- Confusion Matrix
- Cohen's Kappa

## Key Differences

### Mathematical Formulation:
- **Regression**: $y = f(X) + \epsilon$, where $y$ is a continuous value
- **Classification**: $P(y = c | X)$, predicting the probability of belonging to class $c$

### Loss Functions:
- **Regression**: Often uses squared loss, absolute loss
- **Classification**: Often uses log loss (cross-entropy), hinge loss

### Decision Boundaries:
- **Regression**: No explicit decision boundaries
- **Classification**: Models learn decision boundaries to separate classes

### Output Interpretation:
- **Regression**: Direct prediction of quantities
- **Classification**: Often outputs probabilities that are thresholded into class predictions

## Relationships Between Regression and Classification
- Classification can be formulated as regression to a discrete value
- Regression can be viewed as classification with infinitely many classes
- Some models can be adapted for both tasks (e.g., decision trees, neural networks)
- Logistic regression, despite its name, is a classification algorithm

## Considerations for Model Selection
- Nature of the target variable (continuous vs categorical)
- Distribution of the data
- Interpretability requirements
- Computational constraints
- Required performance metrics 