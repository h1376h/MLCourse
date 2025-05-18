# Lecture 4.3: Probabilistic Linear Classifiers Quiz

## Overview
This quiz contains 8 questions from different topics covered in section 4.3 of the lectures on Probabilistic Linear Classifiers, including discriminative vs. generative approaches, logistic regression, sigmoid functions, MLE and MAP estimation, and regularization techniques.

## Question 1

### Problem Statement
Now consider the regular 0/1 loss $\ell$, and assume that $P(y = 0) = P(y = 1) = 1/2$. Also, assume that the class-conditional densities are Gaussian with mean $\mu_0$ and co-variance $\Sigma_0$ under class 0, and mean $\mu_1$ and co-variance $\Sigma_1$ under class 1. Further, assume that $\mu_0 = \mu_1$.

For the following case, draw contours of the level sets of the class conditional densities and label them with $p(x|y = 0)$ and $p(x|y = 1)$. Also, draw the decision boundaries obtained using the Bayes optimal classifier in each case and indicate the regions where the classifier will predict class 0 and where it will predict class 1.

$$\Sigma_0 = \begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix}, \Sigma_1 = \begin{bmatrix} 4 & 0 \\ 0 & 1 \end{bmatrix}$$

#### Task
1. Draw the contours of the level sets of $p(x|y = 0)$ and $p(x|y = 1)$
2. Identify the decision boundary for the Bayes optimal classifier in this scenario
3. Explain why the decision boundary has this shape despite equal means
4. Describe how the decision boundary would change if the prior probabilities were not equal

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
1. [📚] Write down the likelihood function for this dataset
2. [📚] Write down the log-likelihood function
3. [📚] Calculate the gradient of the log-likelihood with respect to $w_0$ and $w_1$
4. [🔍] If after training we get $w_0 = -6$ and $w_1 = 2$, what is the probability $P(y=1|x=2.5)$?

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