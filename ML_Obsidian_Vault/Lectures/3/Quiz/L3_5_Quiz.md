# Lecture 3.5: Optimization Techniques for Linear Regression Quiz

## Overview
This quiz contains 22 questions from different topics covered in section 3.5 of the lectures on Optimization Techniques for Linear Regression.

## Question 1

### Problem Statement
Consider a linear regression model with the cost function $$J(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|^2$$. You want to implement batch gradient descent to find the optimal parameters.

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
- The model is $$h(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 x_1 + w_2 x_2$$
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
- The squared error cost function $$J(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|^2$$ is used
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
- The model is a linear regression model $$h(\boldsymbol{x}; \boldsymbol{w}) = \boldsymbol{w}^T\boldsymbol{x}$$
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

## Question 9

### Problem Statement
Consider a univariate linear regression problem with the following three data points: (1,2), (2,4), (3,5).

In this problem:
- You want to fit a simple linear model $$h(x) = w_0 + w_1x$$
- You'll use stochastic gradient descent with learning rate $\alpha = 0.1$
- Initial parameters are $w_0 = 0, w_1 = 0$

#### Task
1. Calculate the prediction and loss for the first data point $(1,2)$ with the initial parameters
2. Calculate the gradient of the loss with respect to $w_0$ and $w_1$ for this data point
3. Update the parameters using this gradient
4. Calculate the new prediction for the same data point with the updated parameters

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Simple SGD Calculation](L3_5_9_explanation.md).

## Question 10

### Problem Statement
You're implementing different learning rate schedules for gradient descent in linear regression.

In this problem:
- Your initial learning rate is $\alpha_0 = 0.1$
- You'll compare three common learning rate schedules
- You want to calculate the learning rate at iteration $t = 10$

#### Task
1. Calculate the learning rate at $t = 10$ for a step decay schedule where the learning rate is halved every 5 iterations
2. Calculate the learning rate at $t = 10$ for an exponential decay schedule with $$\alpha_t = \alpha_0 \cdot e^{-0.1t}$$
3. Calculate the learning rate at $t = 10$ for a $1/t$ decay schedule with $$\alpha_t = \frac{\alpha_0}{1 + t}$$
4. Which schedule would you recommend for a problem where the cost function is highly non-uniform with steep valleys?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Learning Rate Schedules](L3_5_10_explanation.md).

## Question 11

### Problem Statement
Consider comparing the number of floating-point operations required for different linear regression optimization methods.

In this problem:
- You have $n = 1000$ training examples
- Each example has $d = 20$ features (including the bias term)
- You're considering both normal equations and gradient descent

#### Task
1. What is the approximate number of floating-point operations needed to compute $\boldsymbol{X}^T\boldsymbol{X}$ for the normal equations?
2. What is the approximate number of floating-point operations needed to compute $\boldsymbol{X}^T\boldsymbol{y}$ for the normal equations?
3. How many floating-point operations are needed for one iteration of batch gradient descent?
4. If you need approximately 100 iterations for gradient descent to converge, which method would be computationally more efficient?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: Computational Complexity](L3_5_11_explanation.md).

## Question 12

### Problem Statement
You're implementing online learning for linear regression where data arrives sequentially, and you need to decide how to process it.

In this problem:
- New data points arrive every minute
- You need to make predictions in real-time
- You've trained an initial model on historical data
- You have limited computational resources

#### Task
1. Write down the LMS update rule that you would use for online learning
2. A new data point $(x=3, y=7)$ arrives. If your current model is $$h(x) = 1 + 1.5x$$, calculate the prediction error
3. If you use learning rate $\alpha = 0.1$, calculate the updated parameters after seeing this data point
4. Compare this online learning approach with batch retraining in terms of computational efficiency and model quality

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: Online Learning Implementation](L3_5_12_explanation.md).

## Question 13

### Problem Statement
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

#### Task
1. The computational complexity of solving linear regression using normal equations is O($n^3$), where $n$ is the number of features.

2. Stochastic gradient descent uses all training examples to compute the gradient in each iteration.

3. Mini-batch gradient descent combines the advantages of both batch and stochastic gradient descent.

4. A learning rate that is too small in gradient descent will always result in divergence (i.e., the parameters moving away from the optimum).

5. Feature scaling is generally unnecessary when using the normal equations method to solve linear regression.

For detailed explanations and solutions, see [Question 13: TRUE/FALSE Questions](L3_5_13_explanation.md).

## Question 14

### Problem Statement
For each question, choose the best answer among the given options.

#### Task
1. Which optimization method is most suitable for training a linear regression model with millions of examples?
   - A) Normal equations
   - B) Batch gradient descent
   - C) Mini-batch gradient descent
   - D) Analytical solution with matrix inversion
   
2. If the gradient of the cost function is calculated as $$\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) = -2\boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w})$$, what is the gradient descent update rule?
   - A) $$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t - \alpha \boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}^t)$$
   - B) $$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t + \alpha \boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}^t)$$
   - C) $$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t - 2\alpha \boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}^t)$$
   - D) $$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t + 2\alpha \boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}^t)$$
   
3. When features have very different scales, which of the following is likely to happen during gradient descent?
   - A) The algorithm will converge faster
   - B) The cost function becomes non-convex
   - C) The algorithm will take a zigzag path toward the minimum
   - D) The normal equations solution becomes invalid
   
4. Which of the following optimizers adds a fraction of the previous update to the current update to accelerate convergence?
   - A) RMSprop
   - B) Momentum
   - C) Adam
   - D) Stochastic gradient descent

For detailed explanations and solutions, see [Question 14: Multiple Choice Questions](L3_5_14_explanation.md).

## Question 15

### Problem Statement
Complete each statement with the appropriate term or mathematical expression.

#### Task
1. In the Least Mean Squares (LMS) algorithm, the update rule is $$\boldsymbol{w} := \boldsymbol{w} + \alpha \times \_\_\_\_\_\_\_\_\_$$ (fill in the expression).
2. For gradient descent to converge in linear regression, the learning rate must satisfy $$0 < \alpha < \_\_\_\_\_\_\_\_\_$$ (fill in the bound in terms of eigenvalues).
3. The computational complexity of one iteration of batch gradient descent is $\_\_\_\_\_\_\_\_\_$ (fill in the big-O notation in terms of $n$ and $d$).
4. In online learning, parameters are updated after $\_\_\_\_\_\_\_\_\_ new training example(s).

For detailed explanations and solutions, see [Question 15: Fill in the Blank Questions](L3_5_15_explanation.md).

## Question 16

### Problem Statement
Match each optimization technique on the left with its most appropriate characteristic on the right.

#### Task
Match the following:

**Column A:**
1. Batch Gradient Descent
2. Stochastic Gradient Descent
3. Normal Equations
4. Mini-batch Gradient Descent
5. RMSprop

**Column B:**
a) Updates parameters using one randomly selected example
b) Adapts learning rates for each parameter individually
c) Uses small batches of examples for each update
d) Computes exact solution in one step
e) Uses all examples for each parameter update

For detailed explanations and solutions, see [Question 16: Matching Questions](L3_5_16_explanation.md).

## Question 17

### Problem Statement
Provide brief answers to the following questions about optimization techniques for linear regression.

#### Task
1. What is the main advantage of using stochastic gradient descent over batch gradient descent for very large datasets?
2. Explain why feature scaling is important for gradient descent but not for normal equations.
3. Why does gradient descent sometimes oscillate in narrow valleys of the cost function, and how can momentum help?
4. Name two scenarios where online learning would be particularly useful for linear regression.

For detailed explanations and solutions, see [Question 17: Short Answer Questions](L3_5_17_explanation.md).

## Question 18

### Problem Statement
Consider a simple one-dimensional linear regression problem with squared error loss.

In this problem:
- The model is $$h(x; w) = wx$$ (no bias term for simplicity)
- The initial parameter value is $w^{(0)} = 1$
- There are two training examples: $(x^{(1)}=2, y^{(1)}=5)$ and $(x^{(2)}=3, y^{(2)}=6)$
- The learning rate is $\alpha = 0.02$

#### Task
1. Calculate the gradient of the cost function at the initial parameter value
2. Update the parameter using batch gradient descent for one iteration
3. Calculate the gradient using only the first training example
4. Update the parameter using stochastic gradient descent with the first example

For detailed explanations and solutions, see [Question 18: Numerical Calculation](L3_5_18_explanation.md).

## Question 19

### Problem Statement
You need to decide whether to use the normal equations or gradient descent for a linear regression problem with the following characteristics:

- The training set has $n = 10,000$ examples
- You have $d = 1,000$ features after one-hot encoding categorical variables
- The matrix $\boldsymbol{X}^T\boldsymbol{X}$ is non-singular
- Your computational resources are limited

#### Task
1. [📚] Write down the closed-form solution for linear regression using normal equations
2. [📚] Write down the update rule for batch gradient descent in linear regression
3. [📚] Compare the computational complexity of both methods in terms of $n$ and $d$
4. [📚] Based on the given problem characteristics, which method would you recommend and why?
5. [📚] How would your recommendation change if $n = 10$ million and $d = 100$?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 19: Normal Equations vs. Gradient Descent Tradeoffs](L3_5_19_explanation.md).

## Question 20

### Problem Statement
You're developing a real-time recommendation system for an e-commerce platform where user interaction data arrives continuously, and you need to update your linear regression model as new data becomes available.

#### Task
1. [📚] Explain what online learning is and why it's particularly suitable for this scenario compared to batch learning
2. [📚] Write down the stochastic gradient descent (SGD) update rule for linear regression with squared error loss, and explain how it enables online learning
3. [📚] If a new data point $(x^{(new)}, y^{(new)})$ arrives, where $x^{(new)}$ is a feature vector representing user behavior and $y^{(new)}$ is a purchase amount, show the exact mathematical steps to update your model parameters
4. [📚] Compare the computational and memory requirements of online learning with SGD versus retraining the entire model using normal equations each time new data arrives
5. [📚] Describe a potential issue with simple SGD for online learning and suggest one technique to address this issue (such as using adaptive learning rates)

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 20: Online Learning for Real-time Systems](L3_5_20_explanation.md).

## Question 21

### Problem Statement
Consider a linear regression model with the sum of squared errors (SSE) cost function:

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - \boldsymbol{w}^T \boldsymbol{x}^{(i)})^2$$

You want to optimize this cost function using batch gradient descent.

#### Task
1. [📚] Derive the gradient of the cost function with respect to the parameter vector $\boldsymbol{w}$
2. [📚] Write down the update rule for batch gradient descent in both mathematical notation and as a simple algorithm (pseudocode)
3. [📚] For gradient descent to converge, the learning rate $\alpha$ must be chosen carefully. Derive a bound on $\alpha$ in terms of the eigenvalues of $\boldsymbol{X}^T\boldsymbol{X}$
4. [📚] Explain what happens when $\alpha$ is too small and when it is too large
5. [📚] Describe a simple learning rate scheduling strategy that can improve convergence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 21: Batch Gradient Descent Analysis](L3_5_21_explanation.md).

## Question 22

### Problem Statement
Consider a scenario where data arrives sequentially in a stream, and you need to update your linear regression model in real-time.

#### Task
1. [📚] Explain what online learning is and how it differs from batch learning
2. [📚] Write down the Least Mean Squares (LMS) update rule for online learning of linear regression
3. [📚] A new data point arrives with features $\boldsymbol{x}^{(new)} = [1, 2, 3]^T$ and target $y^{(new)} = 14$. If your current model parameters are $\boldsymbol{w} = [1, 2, 1]^T$ and you use a learning rate of $\alpha = 0.1$, calculate the updated parameters after processing this data point
4. [📚] Discuss the trade-offs between:
   a) A large learning rate vs. a small learning rate
   b) Online learning vs. batch learning
5. [📚] Describe a real-world scenario where online learning would be particularly valuable

For a detailed explanation of this problem, including step-by-step calculations and key insights, see [Question 22: Online Learning and LMS Algorithm](L3_5_22_explanation.md). 