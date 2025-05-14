# Lecture 3.1: Linear Modeling Fundamentals Quiz

## Overview
This quiz contains 12 questions covering different topics from section 3.1 of the lectures on Linear Model Theory, Matrix Properties, Gauss-Markov Theorem, and Statistical Properties.

## Question 1

### Problem Statement
Consider a $2 \times 2$ hat matrix $\mathbf{H}$ in linear regression. You are given that one of its eigenvalues is 1.

#### Task
1. What must be the other eigenvalue of $\mathbf{H}$?
2. Explain why these eigenvalues make sense given the properties of the hat matrix
3. What does the hat matrix represent geometrically in linear regression?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Hat Matrix Properties](L3_1_1_explanation.md).

## Question 2

### Problem Statement
Consider the Gauss-Markov assumptions in linear regression. You are given a dataset where the errors have constant variance $\sigma^2 = 4$.

#### Task
1. State what homoscedasticity means in simple terms
2. Explain why constant variance is important for linear regression
3. What would be the variance of the errors if $\sigma = 2$?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Gauss-Markov Assumptions](L3_1_2_explanation.md).

## Question 3

### Problem Statement
Consider the geometric interpretation of least squares in linear regression from a mathematical foundations perspective.

#### Task
1. The regression line can be viewed as a projection of the target vector $\mathbf{y}$ onto the column space of the design matrix $\mathbf{X}$. Explain what this means mathematically.
2. Why is the residual vector orthogonal to the column space of $\mathbf{X}$ in least squares regression?
3. What mathematical property ensures this orthogonality?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Geometric Interpretation of Regression](L3_1_3_explanation.md).

## Question 4

### Problem Statement
Consider a linear regression model with design matrix $\mathbf{X}$ and response vector $\mathbf{y}$. The hat matrix $\mathbf{H}$ is defined as $\mathbf{H} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$.

#### Task
1. State two key properties of the hat matrix $\mathbf{H}$
2. Explain why $\mathbf{H}$ is called a projection matrix
3. What is the relationship between $\mathbf{H}$ and the fitted values $\hat{\mathbf{y}}$?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Hat Matrix in Regression](L3_1_4_explanation.md).

## Question 5

### Problem Statement
In the context of the Gauss-Markov theorem, consider the Best Linear Unbiased Estimator (BLUE).

#### Task
1. List three key assumptions of the Gauss-Markov theorem
2. Explain what "unbiased" means in the context of BLUE
3. Why is the OLS estimator considered "best" among all linear unbiased estimators?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Gauss-Markov Theorem](L3_1_5_explanation.md).

## Question 6

### Problem Statement
Consider the statistical properties of linear regression estimators. You are given that the variance of the error term ($\sigma^2$) is 25.

#### Task
1. [📚] What is the standard error of the error term?
2. How does the variance of the error term affect the variance of the coefficient estimators?
3. Explain why we want estimators with minimum variance.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Statistical Properties of Estimators](L3_1_6_explanation.md).

## Question 7

### Problem Statement
Linear algebra concepts form the foundation of linear models. In this problem, you'll explore the connection between vector spaces and linear regression.

#### Task
1. Explain what it means when we say "the column space of $\mathbf{X}$ contains the fitted values $\hat{\mathbf{y}}$."
2. If $\mathbf{X}$ is an $n \times 2$ matrix (one column for the intercept, one for a single predictor), what is the dimension of the column space of $\mathbf{X}$? What does this tell us about the flexibility of our model?
3. How can we geometrically interpret the projection of vector $\mathbf{y}$ onto the column space of $\mathbf{X}$ in a linear regression context?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Vector Spaces in Regression](L3_1_7_explanation.md).

## Question 8

### Problem Statement
Consider the bias-variance decomposition in the context of model complexity. You have two models:

- Model A: A simple model with bias = 0.50 and variance = 0.25
- Model B: A more complex model with bias = 0.21 and variance = 1.69

#### Task
1. Calculate the total expected error (bias² + variance) for both models
2. Which model would you select based on total expected error?
3. Explain the bias-variance trade-off and why increasing model complexity can lead to higher variance
4. In which scenarios might you prefer a model with higher bias but lower variance?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Bias-Variance Trade-off](L3_1_8_explanation.md).

## Question 9

### Problem Statement
Consider the following visualization from the lecture showing two model families applied to the same dataset:

- Model Family A: A constant model $f(x) = b$ (horizontal line)
- Model Family B: A linear model $f(x) = ax + b$ (sloped line)

The true underlying function is a sine curve, and we have a limited number of training examples.

#### Task
1. [📚] If you have only 2 training examples, explain why the simpler constant model might generalize better despite having higher bias
2. [📚] As the number of training examples increases, how would you expect the relative performance of the two model families to change? Explain your reasoning.
3. [📚] Sketch how you would expect the training error and test error curves to behave as a function of the number of training examples for both model families
4. [📚] How does regularization help address the bias-variance trade-off without changing the model family? Provide an example.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Model Complexity and Sample Size](L3_1_9_explanation.md).

## Question 10

### Problem Statement
Consider the bias-variance tradeoff in the context of model complexity. You are given two models:

1. A constant model $\mathcal{H}_0: f(x) = b$ with bias = 0.50 and variance = 0.25
2. A linear model $\mathcal{H}_1: f(x) = ax + b$ with bias = 0.21 and variance = 1.69

Both models are being used to approximate a sine function with a limited training dataset.

#### Task
1. Calculate the total expected error (bias² + variance) for both models
2. Which model would perform better in terms of overall error?
3. If you were to apply regularization to the linear model, resulting in a model with bias = 0.23 and variance = 0.33, what would be the new total expected error?
4. Explain why the regularized linear model performs better than both the unregularized constant and linear models, despite having slightly higher bias than the unregularized linear model

For a detailed explanation of this problem, including step-by-step calculations and analysis of the bias-variance tradeoff, see [Question 10: Bias-Variance Tradeoff](L3_1_10_explanation.md).

## Question 11

### Problem Statement
Consider the geometric interpretation of linear regression, where we project the target vector $\boldsymbol{y}$ onto the column space of the design matrix $\boldsymbol{X}$.

#### Task
1. [📚] Write down the formula for the projection matrix $\boldsymbol{P}$ in terms of the design matrix $\boldsymbol{X}$
2. [📚] Prove that the projection matrix $\boldsymbol{P}$ is symmetric (i.e., $\boldsymbol{P}^T = \boldsymbol{P}$)
3. [📚] Prove that the projection matrix $\boldsymbol{P}$ is idempotent (i.e., $\boldsymbol{P}^2 = \boldsymbol{P}$)
4. [📚] If $\hat{\boldsymbol{y}} = \boldsymbol{P}\boldsymbol{y}$ is the projection of $\boldsymbol{y}$ onto the column space of $\boldsymbol{X}$, show that the residual vector $\boldsymbol{y} - \hat{\boldsymbol{y}}$ is orthogonal to the column space of $\boldsymbol{X}$

For a detailed explanation of this problem, including step-by-step proofs and geometric intuition, see [Question 11: Projection Matrix Properties](L3_1_11_explanation.md).

## Question 12

### Problem Statement
Consider the concept of hypothesis spaces in linear modeling as discussed in the lecture notes.

#### Task
1. [📚] Define what a hypothesis space is in the context of machine learning
2. [📚] Compare and contrast the following hypothesis spaces in terms of their complexity:
   - $\mathcal{H}_0$: Constant functions $f(x) = b$
   - $\mathcal{H}_1$: Linear functions $f(x) = ax + b$
   - $\mathcal{H}_2$: Quadratic functions $f(x) = ax^2 + bx + c$
3. [📚] Explain the "approximation-generalization trade-off" when selecting a hypothesis space
4. [📚] For a target function that is a sine curve and only two training examples, explain which hypothesis space ($\mathcal{H}_0$ or $\mathcal{H}_1$) might generalize better and why

For a detailed explanation of this problem, including step-by-step analysis and key insights, see [Question 12: Hypothesis Spaces and Model Complexity](L3_1_12_explanation.md).

