# Lecture 4.3: Probabilistic Linear Classifiers Quiz

## Overview
This quiz contains questions from different topics covered in section 4.3 of the lectures on Probabilistic Linear Classifiers, including discriminative vs. generative approaches, logistic regression, and Bayesian classification.

## Question 1

### Problem Statement
Now consider the regular 0/1 loss $\ell$, and assume that $P(y = 0) = P(y = 1) = 1/2$. Also, assume that the class-conditional densities are Gaussian with mean $\mu_0$ and co-variance $\Sigma_0$ under class 0, and mean $\mu_1$ and co-variance $\Sigma_1$ under class 1. Further, assume that $\mu_0 = \mu_1$.

For the following case, draw contours of the level sets of the class conditional densities and label them with $p(x|y = 0)$ and $p(x|y = 1)$. Also, draw the decision boundaries obtained using the Bayes optimal classifier in each case and indicate the regions where the classifier will predict class 0 and where it will predict class 1.

$$\Sigma_0 = \begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix}, \Sigma_1 = \begin{bmatrix} 4 & 0 \\ 0 & 1 \end{bmatrix}$$

#### Task
1. [📚] Draw the contours of the level sets of $p(x|y = 0)$ and $p(x|y = 1)$
2. [📚] Identify the decision boundary for the Bayes optimal classifier in this scenario
3. [📚] Explain why the decision boundary has this shape despite equal means
4. [📚] Describe how the decision boundary would change if the prior probabilities were not equal

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