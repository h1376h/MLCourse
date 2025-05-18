# Lecture 4.3: Probabilistic Linear Classifiers Quiz

## Overview
This quiz contains 10 questions covering different topics from section 4.3 of the lectures on Probabilistic Linear Classifiers, including discriminative vs. generative approaches, decision boundaries, and probabilistic classification concepts.

## [⭐] Question 1

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
1. [🔍] Define discriminative and generative models in one sentence each
2. Explain how Logistic Regression and LDA differ in their approach to the same classification problem
3. [🔍] List one advantage and one disadvantage of generative models compared to discriminative models
4. [🔍] When would you prefer to use a generative model over a discriminative model? Give one specific scenario

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Discriminative vs Generative Models](L4_3_2_explanation.md).

## Question 3

### Problem Statement
Consider a binary classification problem with two features $x_1$ and $x_2$. You're given that the posterior probability can be written as:

$$P(y=1|x) = \frac{1}{1 + \exp(-w_0 - w_1x_1 - w_2x_2)}$$

#### Task
1. If $w_0 = -3$, $w_1 = 2$, and $w_2 = -1$, write the equation of the decision boundary where $P(y=1|x) = P(y=0|x) = 0.5$
2. Sketch this decision boundary in the $(x_1, x_2)$ plane
3. For a point $(x_1, x_2) = (2, 1)$, determine which class it belongs to and calculate the posterior probability
4. If we change the threshold from 0.5 to 0.7 (i.e., predict class 1 if $P(y=1|x) > 0.7$), how would the decision boundary change?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Decision Boundaries in Probabilistic Classification](L4_3_3_explanation.md).

## Question 4

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
2. Write down the form of the decision boundary for a linear probabilistic classifier in this 2D space
3. For a model with parameters $w_0 = -5$, $w_1 = 1$, and $w_2 = 1$, draw the decision boundary on your sketch
4. Calculate the log-odds ratio for the point $(x_1,x_2) = (2,2)$ using this model

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Decision Boundaries in 2D Space](L4_3_4_explanation.md).

## Question 5

### Problem Statement
Consider Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) approaches for probabilistic linear classifiers.

#### Task
1. Write the objective function for MLE in a probabilistic linear classifier
2. Write the objective function for MAP with a Gaussian prior on weights
3. How does MAP estimation relate to regularized classification? Answer in one sentence
4. In what scenarios would you prefer MAP over MLE? Give one specific example

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: MLE vs MAP Estimation](L4_3_5_explanation.md).

## Question 6

### Problem Statement
Consider a generative approach to binary classification where we model the class-conditional densities as Gaussians with different means but the same covariance matrix:

$$p(x|y=0) = \mathcal{N}(x|\mu_0, \Sigma)$$
$$p(x|y=1) = \mathcal{N}(x|\mu_1, \Sigma)$$

Assume equal prior probabilities: $P(y=0) = P(y=1) = 0.5$.

#### Task
1. Derive the form of the posterior probability $P(y=1|x)$
2. Show that this posterior probability has the same form as a linear classifier with a sigmoid function
3. Express the weights of the equivalent linear classifier in terms of $\mu_0$, $\mu_1$, and $\Sigma$
4. What happens to the decision boundary if $\mu_0 = [1, 1]^T$, $\mu_1 = [3, 3]^T$, and $\Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Gaussian Generative Models](L4_3_6_explanation.md).

## Question 7

### Problem Statement
Consider Newton's method and gradient descent for optimizing probabilistic linear classifiers.

#### Task
1. Write the update rule for gradient descent in a probabilistic linear classifier
2. Write the update rule for Newton's method in a probabilistic linear classifier
3. Explain one advantage and one disadvantage of Newton's method compared to gradient descent
4. Under what circumstances would you prefer gradient descent over Newton's method?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Optimization Methods for Probabilistic Classifiers](L4_3_7_explanation.md).

## Question 8

### Problem Statement
Consider a binary classification problem with Gaussian class-conditional densities. The prior probabilities are $P(y = 0) = 0.7$ and $P(y = 1) = 0.3$. The class-conditional densities have the following parameters:

$$\mu_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \mu_1 = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$

$$\Sigma_0 = \Sigma_1 = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

#### Task
1. Draw the contours of the level sets for both classes
2. Derive and sketch the decision boundary 
3. How does the unequal prior shift the boundary compared to the case where priors are equal?
4. If a new data point is located at $(1, 1)$, which class would it be assigned to?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Gaussian Mixtures with Shifted Means](L4_3_8_explanation.md).

## Question 9

### Problem Statement
Consider a three-class classification problem with equal priors $P(y=0) = P(y=1) = P(y=2) = \frac{1}{3}$ and Gaussian class-conditional densities with parameters:

$$\mu_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \mu_1 = \begin{bmatrix} 4 \\ 0 \end{bmatrix}, \mu_2 = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$$

$$\Sigma_0 = \Sigma_1 = \Sigma_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

#### Task
1. Sketch the three Gaussian distributions in the 2D feature space
2. Derive the form of the decision boundaries between classes
3. Draw the regions where each class would be predicted
4. If we change $\Sigma_2$ to $\begin{bmatrix} 3 & 0 \\ 0 & 3 \end{bmatrix}$, how would the decision boundaries change?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Three-Class Gaussian Classification](L4_3_9_explanation.md).

## Question 10

### Problem Statement
Consider a binary classification problem where the class-conditional densities are Gaussian with equal priors. The parameters are:

$$\mu_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \mu_1 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$\Sigma_0 = \begin{bmatrix} 1 & 0.8 \\ 0.8 & 1 \end{bmatrix}, \Sigma_1 = \begin{bmatrix} 1 & -0.8 \\ -0.8 & 1 \end{bmatrix}$$

#### Task
1. Draw the contours of the level sets for both classes
2. Derive the decision boundary equation
3. Explain how the correlation between features influences the decision boundary shape
4. Would a linear classifier be able to separate these classes effectively? Why or why not?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Gaussian Classification with Correlated Features](L4_3_10_explanation.md).