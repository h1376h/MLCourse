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

#### Solution Approaches
This problem can be tackled using multiple approaches:

For a detailed explanation with extensive mathematical derivations and visualizations, including 3D probability density plots and heatmaps, see [Question 1: Bayes Optimal Classifier with Equal Means](L4_3_1_explanation.md).

For an alternative approach with more concise mathematical explanations focused on the core concepts and decision boundary derivation, see [Alternative Solution: Concise Mathematical Approach](L4_3_1_explanation_alternative.md).

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
1. [📚] If $w_0 = -3$, $w_1 = 2$, and $w_2 = -1$, write the equation of the decision boundary where $P(y=1|x) = P(y=0|x) = 0.5$
2. [📚] Sketch this decision boundary in the $(x_1, x_2)$ plane
3. [📚] For a point $(x_1, x_2) = (2, 1)$, determine which class it belongs to and calculate the posterior probability
4. [📚] If we change the threshold from 0.5 to 0.7 (i.e., predict class 1 if $P(y=1|x) > 0.7$), how would the decision boundary change?

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
1. [📚] Sketch these points in a 2D coordinate system and indicate the two classes
2. [📚] Write down the form of the decision boundary for a linear probabilistic classifier in this 2D space
3. [📚] For a model with parameters $w_0 = -5$, $w_1 = 1$, and $w_2 = 1$, draw the decision boundary on your sketch
4. [📚] Calculate the log-odds ratio for the point $(x_1,x_2) = (2,2)$ using this model

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
1. [📚] Derive the form of the posterior probability $P(y=1|x)$
2. [🔍] Show that this posterior probability has the same form as a linear classifier with a sigmoid function
3. [📚] Express the weights of the equivalent linear classifier in terms of $\mu_0$, $\mu_1$, and $\Sigma$
4. [📚] What happens to the decision boundary if $\mu_0 = [1, 1]^T$, $\mu_1 = [3, 3]^T$, and $\Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$?

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
1. [📚] Draw the contours of the level sets for both classes
2. [📚] Derive and sketch the decision boundary 
3. [📚] How does the unequal prior shift the boundary compared to the case where priors are equal?
4. [📚] If a new data point is located at $(1, 1)$, which class would it be assigned to?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Gaussian Mixtures with Shifted Means](L4_3_8_explanation.md).

## [📕] Question 9

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
1. [📚] Draw the contours of the level sets for both classes
2. [📚] Derive the decision boundary equation
3. [📚] Explain how the correlation between features influences the decision boundary shape
4. [📚] Would a linear classifier be able to separate these classes effectively? Why or why not?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Gaussian Classification with Correlated Features](L4_3_10_explanation.md).

## Question 11

### Problem Statement
You are given the following dataset for a binary classification problem with two features $x_1$ and $x_2$:

| $x_1$ | $x_2$ | $y$ (class) |
|-------|-------|-------------|
| 1.2   | 2.3   | 0           |
| 2.1   | 1.0   | 0           |
| 3.5   | 3.2   | 1           |
| 2.8   | 2.7   | 1           |
| 1.5   | 3.0   | 0           |
| 3.2   | 1.8   | 1           |

You want to fit a logistic regression model to this data.

In this problem:
- The logistic function is defined as $\sigma(z) = \frac{1}{1+e^{-z}}$
- The model is $h(x) = \sigma(\theta_0 + \theta_1 x_1 + \theta_2 x_2)$
- The log-likelihood function is $L(\theta) = \sum_{i=1}^{n} [y_i \log(h(x_i)) + (1-y_i)\log(1-h(x_i))]$

#### Task
1. Write out the complete log-likelihood function for this dataset.
2. Derive the gradient of the log-likelihood with respect to $\theta_0$, $\theta_1$, and $\theta_2$.
3. Explain why the log-likelihood function for logistic regression is convex.
4. If we were to add L2 regularization to this problem, how would it affect the log-likelihood function?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: Log-Likelihood Derivation](L4_3_11_explanation.md).

## Question 12

### Problem Statement
Consider a logistic regression model for binary classification where we have $n$ data points with $p$ features. The model uses the standard sigmoid function.

In this problem:
- We define $y_i$ as the true label (0 or 1) for the i-th data point
- We define $h_\theta(x_i)$ as the predicted probability that $y_i = 1$
- We use maximum likelihood estimation to find optimal parameters

#### Task
1. Explain the connection between cross-entropy loss and maximum likelihood estimation for logistic regression.
2. Derive the cross-entropy loss function from the likelihood function.
3. Explain why minimizing cross-entropy loss is equivalent to maximizing likelihood.
4. How would you interpret the magnitude of the coefficients ($\theta_1$, $\theta_2$, etc.) in a logistic regression model?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: Cross Entropy and MLE](L4_3_12_explanation.md).

## Question 13

### Problem Statement
A researcher is studying the relationship between exam scores and pass/fail outcomes. They collect data from 100 students, recording their hours of study ($x_1$) and previous GPA ($x_2$), along with whether they passed ($y=1$) or failed ($y=0$) the exam.

In this problem:
- The researcher wants to use maximum likelihood estimation to fit a logistic regression model
- They are considering both analytical and numerical approaches
- The model will be used to predict the probability of passing based on study hours and GPA

#### Task
1. Explain why finding the MLE for logistic regression requires numerical methods, unlike linear regression which has a closed-form solution.
2. Describe the Hessian matrix for logistic regression and explain its role in optimization.
3. Compare gradient descent and Newton's method for finding the MLE in logistic regression.
4. If the researcher finds that $\theta_0 = -3.5$, $\theta_1 = 0.8$, and $\theta_2 = 2.1$, interpret these coefficients in context.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 13: MLE Optimization Techniques](L4_3_13_explanation.md).

## Question 14

### Problem Statement
A medical researcher wants to classify tumor samples as either benign or malignant based on the size of the tumor and the age of the patient. They collect a dataset of 8 tumor samples, 4 of which are benign and 4 are malignant (see the table below). For each tumor, the researcher records the size of the tumor in centimeters and the age of the patient in years.

| Tumor Size (cm) | Age (years) | Label (0=benign, 1=malignant) |
|-----------------|-------------|-------------------------------|
| 3               | 50          | 0                             |
| 2               | 40          | 0                             |
| 3               | 20          | 0                             |
| 4               | 70          | 0                             |
| 6               | 80          | 1                             |
| 5               | 75          | 1                             |
| 3               | 55          | 1                             |
| 7               | 85          | 1                             |

Using logistic regression, the researcher builds a binary classification model to predict whether a tumor sample is benign or malignant based on the tumor size and patient age features. The logistic regression model is given by:

$$h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$$

where $\theta$ is the vector of parameters and $x$ is the vector of features.
You can assume a learning rate of 0.1 for this whole problem.

#### Task
1. Find the optimal parameter values for $\theta_0$, $\theta_1$, and $\theta_2$.
2. Using the parameter values above, calculate the predicted probability of malignancy for a tumor sample with size 4 cm and age 60 years. Round your answer to two decimal places.
3. Conceptually, how would increasing and decreasing the learning rate affect the training process?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 14: Logistic Regression for Tumor Classification](L4_3_14_explanation.md).

## [⭐] Question 15

### Problem Statement
Consider a medical dataset with tumor features and diagnostic outcomes. Each patient has data on age (years) and tumor size (mm), with the target variable $y$ indicating whether the tumor is malignant (1) or benign (0).

| Age (years) | Tumor Size (mm) | $y$ (Malignant) |
|-------------|-----------------|-----------------|
| 15          | 20              | 0               |
| 65          | 30              | 0               |
| 30          | 50              | 1               |
| 90          | 20              | 1               |
| 44          | 35              | 0               |
| 20          | 70              | 1               |
| 50          | 40              | 1               |
| 36          | 25              | 0               |

A logistic regression model is being trained on this dataset to predict whether tumors are malignant or benign based on age and tumor size.

The model uses the sigmoid function:
$$g(z) = \frac{1}{1+e^{-z}}$$

And the hypothesis function:
$$h_\theta(x) = g(\theta^T x)$$

The cost function used for training is:
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m} [y^{(i)}\log h_\theta(x^{(i)}) + (1 - y^{(i)})\log(1 - h_\theta(x^{(i)}))]$$

#### Task
1. Starting with initial parameters $\theta_0 = 0$, $\theta_1 = 0$, and $\theta_2 = 0$, calculate the initial cost $J(\theta)$ for this dataset.
2. Calculate the first two iterations of gradient descent using the following update rule and a learning rate $\alpha = 0.01$:
   $$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$
   Where:
   $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$
3. For the same initial parameters, calculate the first two iterations of stochastic gradient descent using a single randomly selected training example at each step with learning rate $\alpha = 0.1$. Show all calculations.
4. Explain the decision boundary equation $\theta^T x = 0$ in the context of logistic regression. What does it represent geometrically?
5. Using the final optimized parameters $\theta_0 = -136.95$, $\theta_1 = 1.1$, and $\theta_2 = 2.2$, derive the equation of the decision boundary for this model.
6. The final optimized parameters for this model are $\theta_0 = -136.95$, $\theta_1 = 1.1$, and $\theta_2 = 2.2$. For a new patient with age 50 years and tumor size 30mm, calculate the predicted probability of the tumor being malignant and provide the classification.
7. Explain how the coefficients $\theta_1 = 1.1$ and $\theta_2 = 2.2$ can be interpreted in this medical context.
8. Conceptually, how would increasing and decreasing the learning rate affect the training process?

#### Solution Approaches
This problem can be tackled using multiple approaches:

For a detailed standard explanation, see [Question 15: Logistic Regression for Tumor Classification](L4_3_15_explanation.md).

For an alternative approach that was used in our class lectures, see [Alternative Solution: Class-based Approach](L4_3_15_explanation_alternative.md).

While both solutions address the same core concepts, they may differ in some numerical calculations or specific methodologies. The fundamental principles of logistic regression and gradient descent remain consistent across both approaches. 