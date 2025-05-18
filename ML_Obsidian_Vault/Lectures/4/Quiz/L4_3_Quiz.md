# Lecture 4.3: Probabilistic Linear Classifiers Quiz

## Overview
This quiz contains 13 questions from different topics covered in section 4.3 of the lectures on Probabilistic Linear Classifiers, including discriminative vs. generative approaches, logistic regression, sigmoid functions, MLE and MAP estimation, and regularization techniques.

## Question 1

### Problem Statement
Consider a binary classification problem where the class-conditional densities are Gaussian. Assume that $P(y = 0) = P(y = 1) = \frac{1}{2}$ (equal prior probabilities). The class-conditional densities are Gaussian with mean $\mu_0$ and covariance $\Sigma_0$ under class 0, and mean $\mu_1$ and covariance $\Sigma_1$ under class 1. Further, assume that $\mu_0 = \mu_1$ (the means are equal).

The covariance matrices for the two classes are:

$$\Sigma_0 = \begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix}, \Sigma_1 = \begin{bmatrix} 4 & 0 \\ 0 & 1 \end{bmatrix}$$

#### Task
1. Draw the contours of the level sets of $p(x|y = 0)$ and $p(x|y = 1)$
2. Identify the decision boundary for the Bayes optimal classifier in this scenario
3. Indicate the regions where the classifier will predict class 0 and where it will predict class 1
4. Explain why the decision boundary has this shape despite equal means
5. Describe how the decision boundary would change if the prior probabilities were not equal

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Bayes Optimal Classifier with Equal Means](L4_3_1_explanation.md).

## Question 2

### Problem Statement
Compare the discriminative and generative approaches to classification.

#### Task
1. Define discriminative and generative models in one sentence each
2. Explain how Logistic Regression and LDA differ in their approach to the same classification problem
3. List one advantage and one disadvantage of generative models compared to discriminative models
4. When would you prefer to use a generative model over a discriminative model? Give one specific scenario

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Discriminative vs Generative Models](L4_3_2_explanation.md).

## Question 3

### Problem Statement
Consider logistic regression for binary classification.

#### Task
1. Write the logistic regression model equation for predicting the probability $P(y=1|x)$
2. Explain why the cross-entropy loss is used for logistic regression instead of squared error loss
3. Derive the gradient of the logistic regression loss function with respect to the model parameters
4. Explain how regularization can be applied to logistic regression and why it might be necessary

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Logistic Regression Fundamentals](L4_3_3_explanation.md).

## Question 4

### Problem Statement
Consider the sigmoid function $\sigma(z) = \frac{1}{1+e^{-z}}$ used in logistic regression.

#### Task
1. Sketch the sigmoid function for values of $z$ in the range $[-5, 5]$
2. Calculate the value of $\sigma(0)$, $\sigma(-2)$, and $\sigma(3)$
3. Show that the derivative of the sigmoid function can be written as $\sigma'(z) = \sigma(z)(1-\sigma(z))$
4. Explain in one sentence why the sigmoid function is appropriate for binary classification

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Sigmoid Function Properties](L4_3_4_explanation.md).

## Question 5

### Problem Statement
Consider a simple logistic regression model for binary classification with a single feature $x$:
$$P(y=1|x) = \sigma(w_0 + w_1x)$$

The following table shows 5 training examples:

| $x$ | $y$ |
|-----|-----|
| 1   | 0   |
| 2   | 0   |
| 3   | 1   |
| 4   | 1   |
| 5   | 1   |

#### Task
1. Write down the likelihood function for this dataset
2. Write down the log-likelihood function
3. Calculate the gradient of the log-likelihood with respect to $w_0$ and $w_1$
4. If after training we get $w_0 = -6$ and $w_1 = 2$, what is the probability $P(y=1|x=2.5)$?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: MLE for Logistic Regression](L4_3_5_explanation.md).

## Question 6

### Problem Statement
Consider regularized logistic regression with $L2$ regularization:
$$J(\mathbf{w}) = -\frac{1}{N}\sum_{i=1}^{N}[y_i\log(\sigma(\mathbf{w}^T\mathbf{x}_i)) + (1-y_i)\log(1-\sigma(\mathbf{w}^T\mathbf{x}_i))] + \frac{\lambda}{2}||\mathbf{w}||^2$$

#### Task
1. Explain in one sentence why regularization is used in logistic regression
2. Derive the gradient of $J(\mathbf{w})$ with respect to $\mathbf{w}$
3. How does increasing the regularization parameter $\lambda$ affect the model complexity?
4. If a dataset has 100 features but only 30 training examples, would you recommend using regularization? Explain why in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Regularized Logistic Regression](L4_3_6_explanation.md).

## Question 7

### Problem Statement
Compare Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) approaches for logistic regression.

#### Task
1. Write the objective function for MLE in logistic regression
2. Write the objective function for MAP with a Gaussian prior on weights
3. How does MAP estimation relate to regularized logistic regression? Answer in one sentence
4. In what scenarios would you prefer MAP over MLE? Give one specific example

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: MLE vs MAP Estimation](L4_3_7_explanation.md).

## Question 8

### Problem Statement
Consider the following dataset for binary classification:

| $x_1$ | $x_2$ | $y$ |
|-------|-------|-----|
| 1     | 1     | 0   |
| 2     | 1     | 0   |
| 1     | 2     | 0   |
| 3     | 3     | 1   |
| 4     | 3     | 1   |
| 3     | 4     | 1   |

#### Task
1. Sketch these points in a 2D coordinate system and indicate the two classes
2. Write down the form of the decision boundary for logistic regression in this 2D space
3. For a logistic regression model with parameters $w_0 = -5$, $w_1 = 1$, and $w_2 = 1$, draw the decision boundary on your sketch
4. Calculate the predicted probability $P(y=1|x)$ for the point $(x_1,x_2) = (2,2)$ using this model

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Decision Boundaries in Logistic Regression](L4_3_8_explanation.md).

## Question 9

### Problem Statement
Consider Newton's method and gradient descent for optimizing logistic regression.

#### Task
1. Write the update rule for gradient descent in logistic regression
2. Write the update rule for Newton's method in logistic regression
3. Explain one advantage and one disadvantage of Newton's method compared to gradient descent
4. Under what circumstances would you prefer gradient descent over Newton's method?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Optimization Methods for Logistic Regression](L4_3_9_explanation.md).

## Question 10

### Problem Statement
Consider a logistic regression model with two features:
$$P(y=1|x) = \sigma(w_0 + w_1x_1 + w_2x_2)$$

If the fitted coefficients are $w_0 = -3$, $w_1 = 2$, and $w_2 = -1$:

#### Task
1. What is the effect of increasing $x_1$ by one unit on the log-odds ratio?
2. What combination of $x_1$ and $x_2$ values would give a predicted probability of exactly 0.5?
3. Sketch the decision boundary in the feature space
4. If a data point has $x_1 = 2$ and $x_2 = 1$, calculate the predicted probability $P(y=1|x)$

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Interpreting Logistic Regression Coefficients](L4_3_10_explanation.md).

## Question 11

### Problem Statement
The sigmoid function is defined as $\sigma(z) = \frac{1}{1+e^{-z}}$.

#### Task
1. Calculate $\sigma(0)$, $\sigma(1)$, and $\sigma(-2)$ by hand
2. If $\sigma(z) = 0.8$, what is the value of $z$?
3. Show that $\frac{d\sigma(z)}{dz} = \sigma(z)(1-\sigma(z))$
4. If $\sigma(w^Tx) = 0.7$, what is the value of $\sigma(-w^Tx)$?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: Sigmoid Function Mathematics](L4_3_11_explanation.md).

## Question 12

### Problem Statement
Consider L1 and L2 regularization in logistic regression:

$$J_{L1}(w) = -\sum_{i=1}^{n}[y_i\log(\sigma(w^Tx_i)) + (1-y_i)\log(1-\sigma(w^Tx_i))] + \lambda\sum_{j=1}^{d}|w_j|$$

$$J_{L2}(w) = -\sum_{i=1}^{n}[y_i\log(\sigma(w^Tx_i)) + (1-y_i)\log(1-\sigma(w^Tx_i))] + \lambda\sum_{j=1}^{d}w_j^2$$

#### Task
1. Draw a contour plot of the L1 penalty term in 2D (for $w_1$ and $w_2$)
2. Draw a contour plot of the L2 penalty term in 2D (for $w_1$ and $w_2$)
3. Which regularization is more likely to produce sparse models (many zero weights)?
4. If you know that only a few features are relevant for your classification task, would you prefer L1 or L2 regularization?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: L1 vs L2 Regularization](L4_3_12_explanation.md).

## Question 13

### Problem Statement
Consider a binary classification problem with three features. You're comparing two approaches:
- A generative approach using LDA with Gaussian class-conditional densities
- A discriminative approach using logistic regression

#### Task
1. What parameters need to be estimated for the LDA model?
2. What parameters need to be estimated for the logistic regression model?
3. If the true class-conditional distributions are not Gaussian, which model is likely to perform better?
4. If you have very few training samples but know the data is approximately Gaussian, which approach would you recommend?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 13: Generative vs Discriminative Models in Practice](L4_3_13_explanation.md). 