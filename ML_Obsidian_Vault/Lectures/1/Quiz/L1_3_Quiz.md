# Lecture 1.3: Generalization Concepts Quiz

## Overview
This quiz contains 4 questions from different topics covered in section 1.3 of the lectures on Generalization Concepts.

## Question 1

### Problem Statement
Consider the following learning scenarios and identify whether each is more likely to suffer from underfitting or overfitting:

#### Task
1. A linear regression model used to predict housing prices based on 20 features, trained on 10 data points
2. A decision tree with maximum depth of 2 used to classify emails as spam or not spam
3. A neural network with 5 hidden layers and 1000 neurons per layer used to predict stock prices based on 100 data points
4. A polynomial regression of degree 1 (i.e., linear) used to model a clearly non-linear relationship

For each case, explain your reasoning and suggest an approach to improve the model's performance.

## Question 2

### Problem Statement
The bias-variance tradeoff is a fundamental concept in machine learning.

#### Task
1. Define bias and variance in the context of machine learning models
2. Explain how the bias-variance tradeoff relates to model complexity
3. Describe how increasing the model complexity typically affects bias and variance
4. For each of the following models, indicate whether it typically has high bias, high variance, or a balance of both:
   a. Linear regression
   b. Decision tree with no maximum depth
   c. k-nearest neighbors with k=1
   d. Support vector machine with a linear kernel

## Question 3

### Problem Statement
Regularization is a key technique to prevent overfitting in machine learning models.

#### Task
1. Explain the purpose of regularization in machine learning
2. Compare and contrast L1 (Lasso) and L2 (Ridge) regularization techniques
3. Describe how increasing the regularization parameter affects the model's complexity and performance
4. For a linear regression model with 100 features where many features are suspected to be irrelevant, would you recommend L1 or L2 regularization? Explain your reasoning

## Question 4

### Problem Statement
Consider a classification problem where you are trying to predict whether a customer will churn (leave) a subscription service.

#### Task
1. Describe how you would split your data into training, validation, and test sets, and explain the purpose of each
2. Explain how k-fold cross-validation works and why it might be preferred over a simple train-test split
3. If your model achieves 95% accuracy on the training data but only 75% on the test data, what might be happening? What strategies would you employ to address this issue?
4. If your dataset is imbalanced (e.g., only 5% of customers churn), discuss how this might affect your evaluation metrics and what techniques you could use to address this challenge 