# Lecture 3.7: Regularization in Linear Models Quiz

## Overview
This quiz contains 10 questions covering different topics from section 3.7 of the lectures on Regularization in Linear Models, including bias-variance tradeoff, ridge regression, lasso regression, elastic net, and methods for selecting regularization parameters.

## Question 1

### Problem Statement
Consider a 9th degree polynomial model fitting a dataset with just 10 data points. The model has extremely large coefficient values and fits the training data perfectly.

In this problem:
- The polynomial model is $f(x) = w_0 + w_1x + w_2x^2 + ... + w_9x^9$
- The coefficients have very large magnitudes (e.g., $w_5 = 640042.26$, $w_9 = 125201.43$)
- The training error is nearly zero

#### Task
1. Explain what phenomenon this model is likely experiencing
2. Describe how this problem would affect the model's performance on new, unseen data
3. Explain the relationship between the large coefficient values and the model's generalization ability
4. Suggest a technique to address this issue without changing the degree of the polynomial

For a detailed explanation of this problem, including step-by-step analysis, see [Question 1: Detecting Overfitting in Polynomial Models](L3_7_1_explanation.md).

## Question 2

### Problem Statement
The bias-variance tradeoff is a fundamental concept in machine learning. Consider a linear model fitting data generated from a sine function.

In this problem:
- A linear model without regularization has bias = 0.21, variance = 1.69
- A linear model with regularization has bias = 0.23, variance = 0.33

#### Task
1. Calculate the total expected error (bias² + variance) for both models
2. Explain why regularization increased the bias but decreased the variance
3. Describe the geometric interpretation of the regularization effect on the parameter space
4. Explain why, despite having higher bias, the regularized model has better overall performance

For a detailed explanation of this problem, including the mathematical analysis of bias-variance tradeoff, see [Question 2: Bias-Variance Tradeoff Analysis](L3_7_2_explanation.md).

## Question 3

### Problem Statement
In ridge regression (L2 regularization), we modify the cost function by adding a penalty term:

$$J_{\text{ridge}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_2^2$$

#### Task
1. Derive the closed-form solution for the ridge regression weights
2. Explain why ridge regression always has a unique solution even when $\boldsymbol{X}^T\boldsymbol{X}$ is not invertible
3. Describe how the regularization parameter $\lambda$ affects the solution as it approaches:
   a. $\lambda \rightarrow 0$
   b. $\lambda \rightarrow \infty$
4. Explain why the regularization effect is stronger for directions with smaller eigenvalues

For a detailed explanation of this problem, including the full mathematical derivation, see [Question 3: Ridge Regression Mathematics](L3_7_3_explanation.md).

## Question 4

### Problem Statement
Consider a polynomial regression model with degree 5 fitted to a dataset. The unregularized model has the following coefficients:
$w_0 = 1.2$, $w_1 = 15.7$, $w_2 = -45.3$, $w_3 = 86.9$, $w_4 = -67.2$, $w_5 = 21.5$

When L2 regularization is applied with parameter $\lambda = 10$, the coefficients change to:
$w_0 = 1.1$, $w_1 = 5.2$, $w_2 = -8.4$, $w_3 = 12.1$, $w_4 = -7.5$, $w_5 = 2.3$

#### Task
1. Calculate the L2 norm of the coefficient vector (excluding $w_0$) for both models
2. Explain why the intercept term $w_0$ is typically not regularized
3. Describe how the coefficients change as a result of regularization and why
4. Sketch how you would expect the fitted curves to differ between the two models

For a detailed explanation of this problem, including coefficient analysis, see [Question 4: Polynomial Coefficient Analysis](L3_7_4_explanation.md).

## Question 5

### Problem Statement
Lasso regression (L1 regularization) uses the following cost function:

$$J_{\text{lasso}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_1$$

#### Task
1. Explain why lasso regression tends to produce sparse solutions (many weights exactly zero)
2. Describe the geometric intuition behind why L1 regularization leads to sparsity
3. Compare the effect of lasso regularization to ridge regularization when features are correlated
4. Explain a practical scenario where lasso regression would be preferred over ridge regression

For a detailed explanation of this problem, including geometric interpretation, see [Question 5: Lasso Regression and Sparsity](L3_7_5_explanation.md).

## Question 6

### Problem Statement
Elastic Net combines both L1 and L2 penalties:

$$J_{\text{elastic}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda_1\|\boldsymbol{w}\|_1 + \lambda_2\|\boldsymbol{w}\|_2^2$$

#### Task
1. Describe the advantage of Elastic Net over Lasso when features are highly correlated
2. Explain how Elastic Net balances the properties of Ridge and Lasso regression
3. In what scenario would you prefer Elastic Net over both Ridge and Lasso?
4. Describe how you would approach selecting both regularization parameters $\lambda_1$ and $\lambda_2$

For a detailed explanation of this problem, including the benefits of Elastic Net, see [Question 6: Elastic Net Regularization](L3_7_6_explanation.md).

## Question 7

### Problem Statement
Consider the regularization path for a linear regression model with 5 features as the regularization parameter $\lambda$ varies from very large to very small values.

#### Task
1. Describe what a regularization path is and how it visualizes the effect of regularization
2. Explain the typical behavior of coefficient values in the regularization path for:
   a. Ridge regression
   b. Lasso regression
3. How would you use the regularization path to select an appropriate value of $\lambda$?
4. What does it mean when different features enter the model at different points along the lasso regularization path?

For a detailed explanation of this problem, including regularization path analysis, see [Question 7: Regularization Path Analysis](L3_7_7_explanation.md).

## Question 8

### Problem Statement
From a Bayesian perspective, regularization can be interpreted as imposing a prior distribution on the model parameters.

#### Task
1. Describe the prior distribution on weights that corresponds to L2 regularization
2. Describe the prior distribution on weights that corresponds to L1 regularization
3. Derive the connection between maximum a posteriori (MAP) estimation with a Gaussian prior and ridge regression
4. Explain how the regularization parameter $\lambda$ relates to the variance of the prior distribution

For a detailed explanation of this problem, including the Bayesian interpretation, see [Question 8: Bayesian Interpretation of Regularization](L3_7_8_explanation.md).

## Question 9

### Problem Statement
Cross-validation is commonly used to select the optimal regularization parameter $\lambda$.

#### Task
1. Describe the K-fold cross-validation procedure for selecting $\lambda$
2. Explain why using only training error to select $\lambda$ would lead to a poor choice
3. In a regularization context, describe what the validation curve typically looks like as $\lambda$ increases from very small to very large values
4. Describe a practical strategy for efficiently testing multiple values of $\lambda$

For a detailed explanation of this problem, including validation strategies, see [Question 9: Selecting Regularization Parameters](L3_7_9_explanation.md).

## Question 10

### Problem Statement
Early stopping in iterative optimization methods (like gradient descent) can be viewed as a form of implicit regularization.

#### Task
1. Explain how early stopping acts as a regularization technique
2. Compare early stopping to explicit regularization methods like Ridge and Lasso
3. Describe the relationship between the number of iterations and the effective model complexity
4. Explain how you would determine the optimal stopping point in practice

For a detailed explanation of this problem, including the regularization effect of early stopping, see [Question 10: Early Stopping as Regularization](L3_7_10_explanation.md). 