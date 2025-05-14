# Lecture 3.5: Optimization Techniques for Linear Regression Quiz

## Overview
This quiz contains 8 questions from different topics covered in section 3.5 of the lectures on Optimization Techniques for Linear Regression.

## Question 1

### Problem Statement
Consider a linear regression model with the cost function $J(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|^2$. You want to implement batch gradient descent to find the optimal parameters.

In this problem:
- The training data consists of 5 examples
- The feature matrix $\boldsymbol{X} \in \mathbb{R}^{5 \times 3}$ (including a column of ones for the bias term)
- The initial parameter vector is $\boldsymbol{w}^{(0)} = [0, 0, 0]^T$
- The learning rate is $\alpha = 0.1$

#### Task
1. Write down the gradient descent update rule for linear regression with batch gradient descent
2. Give the formula for computing the gradient $\nabla_{\boldsymbol{w}}J(\boldsymbol{w})$ for linear regression
3. Explain how many training examples are used in each iteration of batch gradient descent
4. Describe the convergence criteria you would use to determine when to stop the gradient descent algorithm

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Batch Gradient Descent Implementation](L3_5_1_explanation.md).

## Question 2

### Problem Statement
You are implementing stochastic gradient descent (SGD) for linear regression with the following dataset:

| $x_1$ | $x_2$ | $y$ |
|-------|-------|-----|
| 1     | 2     | 5   |
| 2     | 1     | 4   |
| 3     | 3     | 9   |
| 4     | 2     | 8   |

In this problem:
- The model is $h(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 x_1 + w_2 x_2$
- Your initial parameters are $\boldsymbol{w}^{(0)} = [0, 0, 0]^T$
- The learning rate is $\alpha = 0.1$

#### Task
1. Write down the update rule for stochastic gradient descent
2. Calculate the gradient for the first training example
3. Perform one parameter update using this gradient
4. Explain the key differences between stochastic gradient descent and batch gradient descent

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Stochastic Gradient Descent](L3_5_2_explanation.md).

## Question 3

### Problem Statement
You're training a linear regression model on a large dataset with 1 million examples using mini-batch gradient descent.

In this problem:
- You have a feature matrix $\boldsymbol{X} \in \mathbb{R}^{1,000,000 \times 10}$
- You use mini-batches of size 64
- The cost function is the standard squared error loss
- The learning rate is fixed at $\alpha = 0.01$

#### Task
1. Write down the update rule for mini-batch gradient descent
2. Calculate how many gradient updates would be performed in one full pass through the dataset (one epoch)
3. Explain the advantages of mini-batch gradient descent compared to both batch and stochastic gradient descent
4. Describe how you would implement mini-batch gradient descent to ensure random sampling without replacement

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Mini-batch Gradient Descent](L3_5_3_explanation.md).

## Question 4

### Problem Statement
Consider the choice between using normal equations and gradient descent for linear regression.

In this problem:
- You need to train a linear regression model on a dataset with 100,000 examples
- The feature dimension is 1,000 (after one-hot encoding categorical variables)
- You have limited computational resources
- The matrix $\boldsymbol{X}^T\boldsymbol{X}$ is not singular

#### Task
1. Analyze the computational complexity of solving linear regression using normal equations
2. Analyze the computational complexity of one iteration of batch gradient descent
3. Explain which method would be more efficient in this scenario and why
4. Describe a scenario where your recommendation would be different

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Normal Equations vs Gradient Descent](L3_5_4_explanation.md).

## Question 5

### Problem Statement
You are training a linear regression model using gradient descent, but you notice that the algorithm is not converging properly.

In this problem:
- The features have very different scales:
  - $x_1$ ranges from 0 to 1
  - $x_2$ ranges from 0 to 10,000
  - $x_3$ ranges from -100 to 100
- You're using batch gradient descent with a learning rate of $\alpha = 0.01$

#### Task
1. Explain why the different feature scales can cause problems for gradient descent
2. Describe two common feature scaling techniques to address this issue
3. For each technique, show how you would transform the features in this example
4. Explain how feature scaling affects the interpretation of the learned parameters

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Feature Scaling for Optimization](L3_5_5_explanation.md).

## Question 6

### Problem Statement
Consider the convergence properties of gradient descent for linear regression when the cost function is the squared error.

In this problem:
- The squared error cost function $J(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|^2$ is used
- The learning rate is a critical parameter for convergence
- You know that the eigenvalues of $\boldsymbol{X}^T\boldsymbol{X}$ range from $\lambda_{\min} = 0.1$ to $\lambda_{\max} = 10$

#### Task
1. Derive the condition on the learning rate $\alpha$ that ensures convergence of gradient descent
2. Calculate the range of valid learning rates for this specific problem
3. If you choose a learning rate that's too large, describe what would happen to the optimization process
4. Explain the relationship between the condition number of $\boldsymbol{X}^T\boldsymbol{X}$ and the convergence rate

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Convergence Analysis](L3_5_6_explanation.md).

## Question 7

### Problem Statement
You're implementing the Least Mean Squares (LMS) algorithm for an online learning scenario, where data arrives sequentially.

In this problem:
- New training examples arrive one at a time
- You need to update your model parameters after seeing each example
- The model is a linear regression model $h(\boldsymbol{x}; \boldsymbol{w}) = \boldsymbol{w}^T\boldsymbol{x}$
- You're using a fixed learning rate $\alpha = 0.1$

#### Task
1. Write down the update rule for the LMS algorithm
2. Explain why LMS is well-suited for online learning scenarios
3. Describe the relationship between LMS and stochastic gradient descent
4. Discuss the trade-offs between immediately updating parameters with each new example versus collecting batches of examples

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Least Mean Squares Algorithm](L3_5_7_explanation.md).

## Question 8

### Problem Statement
You're implementing advanced optimization techniques for linear regression to improve convergence speed.

In this problem:
- You've already implemented batch gradient descent
- The cost function exhibits high curvature in some directions and low curvature in others
- The current convergence is slow with a fixed learning rate

#### Task
1. Explain how momentum can accelerate gradient descent, and write down the update rules
2. Describe how RMSprop adapts the learning rate for each parameter, and provide its update rules
3. Explain how Adam combines the benefits of momentum and RMSprop
4. For a parameter that consistently receives large gradients in the same direction, explain how each of these optimizers would behave differently from standard gradient descent

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Advanced Optimizers](L3_5_8_explanation.md). 