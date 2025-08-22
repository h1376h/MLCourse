# Lecture 5.5: SVM Regression Quiz

## Overview
This quiz contains 10 questions covering different topics from section 5.5 of the lectures on Support Vector Regression (SVR), ε-insensitive Loss, Linear and Nonlinear SVR, ν-SVR, and SVR Implementation.

## Question 1

### Problem Statement
Consider the fundamental difference between SVM classification and Support Vector Regression (SVR).

#### Task
1. [🔍] In SVM classification, we try to maximize the margin between classes. What does SVR try to optimize instead?
2. [📚] Instead of predicting discrete class labels, what does SVR predict?
3. [📚] In classification, support vectors are points on the margin boundary. What are support vectors in SVR?
4. [🔍] How does the concept of "margin" translate from classification to regression?

For a detailed explanation of this problem, see [Question 1: SVR vs SVM Classification](L5_5_1_explanation.md).

## Question 2

### Problem Statement
Consider the ε-insensitive loss function used in SVR: $L_ε(y, f(x)) = \max(0, |y - f(x)| - ε)$.

#### Task
1. [📚] For a prediction $f(x) = 3.2$ and true value $y = 3.5$ with $ε = 0.5$, what is the ε-insensitive loss?
2. [📚] For a prediction $f(x) = 2.8$ and true value $y = 3.5$ with $ε = 0.5$, what is the ε-insensitive loss?
3. [📚] For a prediction $f(x) = 4.2$ and true value $y = 3.5$ with $ε = 0.5$, what is the ε-insensitive loss?
4. [🔍] Explain why this loss function creates an "ε-tube" around the regression function
5. [📚] How does ε-insensitive loss compare to squared loss $(y - f(x))^2$ in terms of robustness to outliers?

For a detailed explanation of this problem, see [Question 2: ε-insensitive Loss Function](L5_5_2_explanation.md).

## Question 3

### Problem Statement
Consider the SVR optimization problem with slack variables:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^n (\xi_i + \xi_i^*)$$
$$\text{subject to: } y_i - \mathbf{w}^T\mathbf{x}_i - b \leq ε + \xi_i$$
$$\mathbf{w}^T\mathbf{x}_i + b - y_i \leq ε + \xi_i^*$$
$$\xi_i, \xi_i^* \geq 0$$

#### Task
1. [📚] Why do we need both $\xi_i$ and $\xi_i^*$ slack variables (instead of just one)?
2. [📚] What do the constraints $y_i - \mathbf{w}^T\mathbf{x}_i - b \leq ε + \xi_i$ and $\mathbf{w}^T\mathbf{x}_i + b - y_i \leq ε + \xi_i^*$ represent?
3. [🔍] For a point that lies exactly on the upper boundary of the ε-tube, what are the values of $\xi_i$ and $\xi_i^*$?
4. [📚] For a point that lies outside the ε-tube, which slack variable(s) will be non-zero?

For a detailed explanation of this problem, see [Question 3: SVR Optimization Formulation](L5_5_3_explanation.md).

## Question 4

### Problem Statement
Consider the role of support vectors in SVR.

#### Task
1. [📚] In SVR, what are the three types of points based on their relationship to the ε-tube?
2. [📚] Which points become support vectors in SVR?
3. [🔍] For a point lying inside the ε-tube, what are its corresponding Lagrange multipliers $\alpha_i$ and $\alpha_i^*$?
4. [📚] For a point lying exactly on the boundary of the ε-tube, what can you say about its Lagrange multipliers?
5. [🔍] How does the number of support vectors in SVR relate to the choice of $ε$ parameter?

For a detailed explanation of this problem, see [Question 4: Support Vectors in SVR](L5_5_4_explanation.md).

## Question 5

### Problem Statement
Consider the effect of the ε parameter on SVR behavior.

#### Task
1. [📚] When ε is very small (close to 0), what happens to the SVR model?
2. [📚] When ε is very large, what happens to the SVR model?
3. [🔍] How does changing ε affect the number of support vectors?
4. [🔍] How does changing ε affect the bias-variance tradeoff?
5. [📚] For noisy data, would you generally prefer a smaller or larger ε value, and why?

For a detailed explanation of this problem, see [Question 5: ε Parameter Effects](L5_5_5_explanation.md).

## Question 6

### Problem Statement
Consider the ν-SVR formulation as an alternative to ε-SVR.

#### Task
1. [🔍] What is the main difference between ε-SVR and ν-SVR in terms of parameter specification?
2. [📚] In ν-SVR, what does the parameter ν control (what are its bounds and interpretation)?
3. [🔍] What is the advantage of ν-SVR over ε-SVR in terms of parameter selection?
4. [📚] How is the ε value automatically determined in ν-SVR?
5. [🔍] When might you prefer ν-SVR over ε-SVR in practice?

For a detailed explanation of this problem, see [Question 6: ν-SVR Formulation](L5_5_6_explanation.md).

## Question 7

### Problem Statement
Consider applying kernels to SVR for nonlinear regression.

#### Task
1. [📚] How do you extend linear SVR to handle nonlinear relationships using kernels?
2. [📚] What are the most commonly used kernels for regression tasks?
3. [🔍] For time series data with periodic patterns, what type of kernel might be most appropriate?
4. [📚] How does the RBF kernel parameter γ affect the flexibility of the regression function in SVR?
5. [🔍] What happens to the computational complexity when using kernels in SVR compared to linear SVR?

For a detailed explanation of this problem, see [Question 7: Nonlinear SVR with Kernels](L5_5_7_explanation.md).

## Question 8

### Problem Statement
Compare SVR with other regression methods.

#### Task
1. [🔍] **vs. Linear Regression**: In what scenarios might SVR be preferred over ordinary least squares linear regression?
2. [🔍] **vs. Ridge Regression**: How does the regularization in SVR compare to L2 regularization in Ridge regression?
3. [🔍] **vs. Neural Networks**: What are the advantages and disadvantages of SVR compared to neural network regression?
4. [📚] **Robustness**: Which regression method is generally more robust to outliers: SVR or linear regression?
5. [🔍] **Interpretability**: How does the interpretability of SVR compare to linear regression?

For a detailed explanation of this problem, see [Question 8: SVR Comparison with Other Methods](L5_5_8_explanation.md).

## Question 9

### Problem Statement
Consider practical hyperparameter tuning for SVR.

#### Task
1. [📚] For an RBF kernel SVR, what are the main hyperparameters you need to tune?
2. [🔍] Describe a systematic approach for tuning C, ε, and γ parameters using cross-validation
3. [📚] What ranges would you typically search for these parameters (C, ε, γ)?
4. [🔍] How can you detect overfitting vs. underfitting during SVR parameter tuning?
5. [📚] What validation metrics are most appropriate for evaluating SVR performance (MSE, MAE, R²)?

For a detailed explanation of this problem, see [Question 9: SVR Hyperparameter Tuning](L5_5_9_explanation.md).

## Question 10

### Problem Statement
Consider practical implementation considerations for SVR.

#### Task
1. [📚] What preprocessing steps are crucial for SVR (feature scaling, outlier handling, etc.)?
2. [🔍] How should you handle missing values in the input features for SVR?
3. [📚] For very large datasets, what strategies can you use to make SVR training more efficient?
4. [🔍] How do you interpret and explain SVR predictions to non-technical stakeholders?
5. [📚] In what real-world scenarios would you recommend SVR over simpler regression methods?

For a detailed explanation of this problem, see [Question 10: SVR Implementation Considerations](L5_5_10_explanation.md).
