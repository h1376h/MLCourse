# Lecture 5.2: Maximum Likelihood for Logistic Regression Quiz

## Overview
This quiz contains 5 questions from different topics covered in section 5.2 of the lectures on Maximum Likelihood for Logistic Regression.

## Question 1

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

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Log-Likelihood Derivation](L5_2_1_explanation.md).

## Question 2

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

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Cross Entropy and MLE](L5_2_2_explanation.md).

## Question 3

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

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: MLE Optimization Techniques](L5_2_3_explanation.md).

## Question 4

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

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Logistic Regression for Tumor Classification](L5_2_4_explanation.md).

## [⭐] Question 5

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

For a detailed standard explanation, see [Question 5: Logistic Regression for Tumor Classification](L5_2_5_explanation.md).

For an alternative approach that was used in our class lectures, see [Alternative Solution: Class-based Approach](L5_2_5_explanation_alternative.md).

While both solutions address the same core concepts, they may differ in some numerical calculations or specific methodologies. The fundamental principles of logistic regression and gradient descent remain consistent across both approaches. 