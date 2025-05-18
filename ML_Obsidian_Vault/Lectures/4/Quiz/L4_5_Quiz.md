# Lecture 4.5: Optimization for Linear Classifiers Quiz

## Overview
This quiz contains 10 questions covering different topics from section 4.5 of the lectures on Optimization for Linear Classifiers, including batch learning, online learning, stochastic vs. batch methods, and various optimization techniques for linear classifiers.

## Question 1

### Problem Statement
Compare and contrast batch learning and online learning approaches for optimizing linear classifiers.

#### Task
1. [🔍] Define batch learning and online learning in one sentence each
2. [🔍] List two advantages of online learning over batch learning
3. [🔍] List two advantages of batch learning over online learning
4. Name one algorithm that is inherently online and one that is inherently batch

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Batch vs Online Learning](L4_5_1_explanation.md).

## Question 2

### Problem Statement
Consider the following objective function for a logistic regression model:

$$J(w) = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(h_w(x_i)) + (1-y_i) \log(1-h_w(x_i))] + \lambda \|w\|^2$$

where $h_w(x_i) = \frac{1}{1 + e^{-w^T x_i}}$ is the sigmoid function.

#### Task
1. [📚] Compute the gradient $\nabla J(w)$ with respect to the weights $w$
2. [📚] Write the update rule for gradient descent optimization
3. [📚] Write the update rule for stochastic gradient descent optimization
4. [🔍] What is the role of $\lambda$ in this objective function? Answer in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Logistic Regression Optimization](L4_5_2_explanation.md).

## Question 3

### Problem Statement
Consider training a linear classifier on a dataset with 10,000 examples using different optimization methods.

#### Task
1. [🔍] If each epoch of batch gradient descent takes 2 seconds, and stochastic gradient descent processes examples one at a time at a rate of 0.0005 seconds per example, how long would one epoch take for each method?
2. [🔍] If SGD requires 5 epochs to converge and batch gradient descent requires 100 epochs, calculate the total time to convergence for each method
3. [🔍] If mini-batch gradient descent with a batch size of 100 examples takes 0.03 seconds per mini-batch, how long would one epoch take?
4. [🔍] What is the trade-off between batch size and convergence speed? Answer in 1-2 sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Optimization Methods Comparison](L4_5_3_explanation.md).

## Question 4

### Problem Statement
Consider the following dataset for binary classification:

| $x_1$ | $x_2$ | $y$ (target) |
|-------|-------|--------------|
| 1     | 2     | 1            |
| 2     | 1     | 1            |
| 3     | 3     | 1            |
| 6     | 4     | 0            |
| 5     | 6     | 0            |
| 7     | 5     | 0            |

You are training a logistic regression model with the following update rule:

$$w_{t+1} = w_t - \eta \nabla J_i(w_t)$$

where $\nabla J_i(w_t) = (h_{w_t}(x_i) - y_i) x_i$ and $h_w(x) = \frac{1}{1 + e^{-w^T x}}$.

#### Task
1. [📚] Initialize $w = [w_0, w_1, w_2]^T = [0, 0, 0]^T$ and $\eta = 0.1$, where $w_0$ is the bias term
2. [📚] Perform the first update of stochastic gradient descent using the first data point
3. [📚] Calculate the predicted probability for the second data point using the updated weights
4. [📚] Compare the computational complexity of performing one complete pass (epoch) through this dataset using batch gradient descent versus stochastic gradient descent

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: SGD for Logistic Regression](L4_5_4_explanation.md).

## Question 5

### Problem Statement
Consider the perceptron learning algorithm with the following update rule:

$$w_{t+1} = w_t + \eta y_i x_i$$

if $y_i (w_t^T x_i) \leq 0$ (misclassification), and $w_{t+1} = w_t$ otherwise.

#### Task
1. [🔍] How does this update rule differ from the standard gradient descent update for logistic regression? Answer in one sentence
2. [🔍] Why might this update rule converge faster than gradient descent for linearly separable data? Answer in one sentence
3. [📚] For a misclassified point with $x = [1, 2, 1]^T$ (including bias term), $y = 1$, and current weights $w = [0, 1, -1]^T$, calculate the updated weights using $\eta = 0.5$
4. [🔍] Will this update always reduce the number of misclassified points? Explain why or why not in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Perceptron Optimization](L4_5_5_explanation.md).

## Question 6

### Problem Statement
Consider the Voted Perceptron algorithm, an ensemble approach to perceptron learning.

#### Task
1. Explain the main idea behind the Voted Perceptron algorithm in one sentence
2. How does the Voted Perceptron address the limitations of the standard perceptron for non-separable data? Answer in one sentence
3. Given a sequence of weight vectors $w_1, w_2, w_3, w_4$ with survival times (counts) $c_1 = 3, c_2 = 2, c_3 = 5, c_4 = 1$, how would the Voted Perceptron make a prediction for a new point $x$?
4. How does the Voted Perceptron differ from the Averaged Perceptron? Answer in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Voted Perceptron](L4_5_6_explanation.md).

## Question 7

### Problem Statement
Consider the Passive-Aggressive algorithm for online learning.

#### Task
1. What is the main idea behind Passive-Aggressive algorithms? Answer in one sentence
2. How does the Passive-Aggressive algorithm differ from the perceptron? Answer in one sentence
3. When is the algorithm "passive" and when is it "aggressive"? Explain in one sentence each
4. For a point $x = [2, 1]^T$ with true label $y = 1$, current weights $w = [1, 0]^T$, and hinge loss margin parameter $\gamma = 1$, determine whether the algorithm will be passive or aggressive, and calculate the new weights if an update is needed

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Passive-Aggressive Learning](L4_5_7_explanation.md).

## Question 8

### Problem Statement
Consider the Coordinate Descent optimization approach for training linear classifiers.

#### Task
1. Explain the main idea behind Coordinate Descent in one sentence
2. How does Coordinate Descent differ from Gradient Descent? Answer in one sentence
3. What are the advantages of Coordinate Descent for sparse features? Answer in 1-2 sentences
4. For a simple squared error loss function $J(w) = (w_1 - 3)^2 + (w_2 + 2)^2$, perform two iterations of coordinate descent starting from $w = [0, 0]^T$, optimizing $w_1$ first

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Coordinate Descent](L4_5_8_explanation.md).

## Question 9

### Problem Statement
Consider the Averaged Perceptron algorithm, a modification of the standard perceptron for improved generalization.

#### Task
1. How does the Averaged Perceptron differ from the standard perceptron? Answer in one sentence
2. What problem does the Averaged Perceptron address? Answer in one sentence
3. If a perceptron algorithm generates weight vectors $w_1 = [1, 0]^T$, $w_2 = [1, 1]^T$, $w_3 = [2, 1]^T$, and $w_4 = [2, 2]^T$ in sequence during training, what would be the averaged weight vector?
4. When would you prefer using an Averaged Perceptron over a standard perceptron? Answer in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Averaged Perceptron](L4_5_9_explanation.md).

## Question 10

### Problem Statement
Consider different optimization objectives for linear classifiers.

#### Task
1. [🔍] Compare the optimization objectives of the perceptron algorithm and logistic regression in one sentence each
2. [🔍] How does adding L1 regularization affect the optimization landscape? Answer in one sentence
3. [🔍] How does adding L2 regularization affect the optimization landscape? Answer in one sentence
4. [🔍] Why might coordinate descent be preferable to gradient descent for L1-regularized objectives? Answer in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Optimization Objectives](L4_5_10_explanation.md). 