# Lecture 3.1: Linear Modeling Fundamentals Quiz

## Overview
This quiz contains 7 questions covering different topics from section 3.1 of the lectures on Linear Model Theory, Matrix Properties, Gauss-Markov Theorem, and Statistical Properties.

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
2. [📚] How does the variance of the error term affect the variance of the coefficient estimators?
3. Explain why we want estimators with minimum variance.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Statistical Properties of Estimators](L3_1_6_explanation.md).

## Question 7

### Problem Statement
Linear algebra concepts form the foundation of linear models. In this problem, you'll explore the connection between vector spaces and linear regression.

#### Task
1. [📚] Explain what it means when we say "the column space of $\mathbf{X}$ contains the fitted values $\hat{\mathbf{y}}$."
2. [📚] If $\mathbf{X}$ is an $n \times 2$ matrix (one column for the intercept, one for a single predictor), what is the dimension of the column space of $\mathbf{X}$? What does this tell us about the flexibility of our model?
3. [📚] How can we geometrically interpret the projection of vector $\mathbf{y}$ onto the column space of $\mathbf{X}$ in a linear regression context?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Vector Spaces in Regression](L3_1_7_explanation.md).

