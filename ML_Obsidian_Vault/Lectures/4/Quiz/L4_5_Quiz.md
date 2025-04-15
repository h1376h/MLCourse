# Lecture 4.5: Advanced Topics and Applications Quiz

## Overview
This quiz contains 4 questions from different topics covered in section 4.5 of the lectures on Advanced Topics and Applications in Linear Classification.

## Question 1

### Problem Statement
Consider the problem of regularization in linear classifiers.

#### Task
1. Explain the purpose of regularization in the context of linear classifiers
2. Compare L1 and L2 regularization for linear classifiers in terms of:
   a. Effect on model weights
   b. Feature selection properties
   c. Geometric interpretation
   d. Resulting decision boundaries
3. If you suspect many features in your dataset are irrelevant for classification, which type of regularization would you prefer and why?
4. Derive the optimization objective for a logistic regression model with L2 regularization

## Question 2

### Problem Statement
Consider the kernel trick for extending linear classifiers to non-linear decision boundaries.

#### Task
1. Explain the key insight behind the kernel trick
2. Define the following kernel functions and describe the type of decision boundaries they can create:
   a. Linear kernel: K(x, y) = x^T y
   b. Polynomial kernel: K(x, y) = (x^T y + c)^d
   c. Radial Basis Function (RBF) kernel: K(x, y) = exp(-γ||x-y||²)
3. How does the choice of kernel affect model complexity and the risk of overfitting?
4. For a dataset with features in high-dimensional space but with a relatively simple decision boundary, which kernel would you recommend and why?

## Question 3

### Problem Statement
Consider a real-world email spam classification application.

#### Task
1. Describe appropriate features for this classification task
2. Discuss the challenges of class imbalance (typically few spam emails compared to legitimate ones) and approaches to address it
3. Compare the following classifier options for this application:
   a. Logistic regression with L1 regularization
   b. Support Vector Machine with RBF kernel
   c. Multi-layer perceptron
4. How would you evaluate and compare these classifiers in a practical setting?

## Question 4

### Problem Statement
Consider the following advanced linear classification techniques:
- Maximum Entropy Classifiers
- Confidence-weighted Linear Classification
- Passive-Aggressive Algorithms
- Averaged Perceptron

#### Task
1. Briefly describe each of these techniques and their key characteristics
2. For each technique, identify scenarios where it might outperform standard logistic regression or SVMs
3. Discuss how online learning capability differs among these methods
4. If you were designing a classification system that needs to continuously learn from a stream of data, which of these approaches would you recommend and why? 