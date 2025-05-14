# Lecture 3.3: Probabilistic View of Linear Regression Quiz

## Overview
This quiz contains 12 questions from different topics covered in section 3.3 of the lectures on the Probabilistic View of Linear Regression.

## Question 1

### Problem Statement
Consider a linear regression model with a probabilistic interpretation:
$$y = w_0 + w_1x + \epsilon$$

where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ represents Gaussian noise with zero mean and variance $\sigma^2$.

For the following dataset with 4 observations:

| $x$ | $y$ |
|-----|-----|
| 1   | 3   |
| 2   | 5   |
| 3   | 4   |
| 4   | 7   |

#### Task
1. Write down the likelihood function for this model
2. Derive the log-likelihood function
3. Show that maximizing the log-likelihood is equivalent to minimizing the sum of squared errors
4. Based on this, calculate the maximum likelihood estimates for $w_0$ and $w_1$

For a detailed explanation of this problem, including step-by-step solutions and derivations, see [Question 1: Maximum Likelihood for Linear Regression](L3_3_1_explanation.md).

## Question 2

### Problem Statement
In probabilistic linear regression, we assume that observations follow a Gaussian distribution around the predicted values.

#### Task
1. Write the probability density function (PDF) for observing a specific target value $y^{(i)}$ given input $x^{(i)}$ and parameters $\boldsymbol{w}$ and $\sigma^2$
2. Using this PDF, construct the likelihood function for a dataset with $n$ observations
3. Explain why taking the logarithm of the likelihood simplifies the optimization problem
4. Identify the connection between the negative log-likelihood and the sum of squared errors cost function

For a detailed explanation of this problem, including full derivations and important probabilistic concepts, see [Question 2: Gaussian Likelihood in Linear Regression](L3_3_2_explanation.md).

## Question 3

### Problem Statement
Consider a simple linear regression model with a probabilistic interpretation. You have collected the following data:

| $x$ | $y$ |
|-----|-----|
| 2   | 5   |
| 4   | 9   |
| 6   | 11  |
| 8   | 15  |

The model assumes that observations follow $$y^{(i)} = w_0 + w_1x^{(i)} + \epsilon^{(i)}$$ where $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$.

#### Task
1. Calculate the maximum likelihood estimates for parameters $w_0$ and $w_1$
2. Estimate the noise variance $\sigma^2$ using the maximum likelihood approach
3. Write down the predictive distribution for a new input $x_{\text{new}} = 5$
4. Calculate the probability that $y_{\text{new}} > 12$ for this new input

For a detailed explanation of this problem, including maximum likelihood estimation and predictive distributions, see [Question 3: Predictive Distribution in Linear Regression](L3_3_3_explanation.md).

## Question 4

### Problem Statement
In a probabilistic linear regression model, the likelihood function plays a crucial role in parameter estimation. Consider a simple model $$y = w_0 + w_1x + \epsilon$$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

#### Task
1. Derive the maximum likelihood estimators for $w_0$ and $w_1$ by setting the derivatives of the log-likelihood to zero
2. Show that these estimators are identical to the least squares estimators
3. Derive the maximum likelihood estimator for the noise variance $\sigma^2$
4. Explain what happens to the likelihood when we increase or decrease $\sigma^2$ while keeping $w_0$ and $w_1$ fixed

For a detailed explanation of this problem, including the full mathematical derivation, see [Question 4: Maximum Likelihood Estimation](L3_3_4_explanation.md).

## Question 5

### Problem Statement
In probabilistic linear regression, we are often interested in the uncertainty associated with our parameter estimates. Consider a simple linear regression model with Gaussian noise.

#### Task
1. Explain why parameter estimates in linear regression have a distribution rather than being single fixed values
2. Describe the distribution of the parameter estimates $\hat{\boldsymbol{w}}$ when derived using maximum likelihood estimation
3. What factors affect the uncertainty (variance) of these parameter estimates?
4. How does increasing the number of training examples affect this uncertainty?

For a detailed explanation of this problem, including the distribution of parameter estimates, see [Question 5: Parameter Uncertainty in Linear Regression](L3_3_5_explanation.md).

## Question 6

### Problem Statement
Error decomposition is an important concept in understanding model performance. In probabilistic linear regression, total error can be broken down into different components.

#### Task
1. Define and explain the concept of "structural error" in probabilistic linear regression
2. Define and explain the concept of "approximation error" in probabilistic linear regression
3. Write down the mathematical decomposition of the expected error in terms of these components
4. Explain how these error components relate to bias and variance

For a detailed explanation of this problem, including error decomposition analysis, see [Question 6: Error Decomposition in Probabilistic Framework](L3_3_6_explanation.md).

## Question 7

### Problem Statement
Consider a probabilistic linear regression model where we want to predict a student's final exam score based on their midterm score. We assume:

$$y = w_0 + w_1x + \epsilon$$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$

After fitting the model, we obtain:
- $\hat{w}_0 = 20$
- $\hat{w}_1 = 0.8$
- $\hat{\sigma}^2 = 25$

#### Task
1. Write down the predictive distribution for a new student with midterm score $x_{\text{new}} = 75$
2. Calculate the mean and standard deviation of this predictive distribution
3. Construct a 95% prediction interval for the final exam score
4. Calculate the probability that this student will score above 85 on the final exam

For a detailed explanation of this problem, including probability calculations, see [Question 7: Making Probabilistic Predictions](L3_3_7_explanation.md).

## Question 8

### Problem Statement
The Gaussian error assumption in linear regression has important implications for parameter estimation and inference.

#### Task
1. State the probability density function for Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$
2. Explain why the Gaussian assumption leads to the least squares objective function
3. How would the optimization objective change if we assumed Laplace-distributed errors instead?
4. What are the advantages and limitations of the Gaussian error assumption?

For a detailed explanation of this problem, including the implications of different noise distributions, see [Question 8: Noise Models in Linear Regression](L3_3_8_explanation.md).

## Question 9

### Problem Statement
Consider a dataset with 3 observations: $(1,2)$, $(2,3)$, and $(3,5)$. We want to fit a simple linear regression model using maximum likelihood estimation with the assumption that errors are normally distributed.

#### Task
1. Write down the log-likelihood function for this dataset
2. Find the partial derivatives of the log-likelihood with respect to $w_0$ and $w_1$
3. Set these derivatives to zero and solve for the maximum likelihood estimates
4. Calculate the maximum likelihood estimate for the noise variance $\sigma^2$

For a detailed explanation of this problem, including complete calculations, see [Question 9: Manual Calculation of Maximum Likelihood Estimates](L3_3_9_explanation.md).

## Question 10

### Problem Statement
In probabilistic linear regression, the optimization of the log-likelihood can be approached in different ways.

#### Task
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

1. The negative log-likelihood with Gaussian noise is proportional to the sum of squared errors plus a constant term.
2. Maximum likelihood estimation with Gaussian noise always results in the same parameter estimates as least squares regression.
3. If we use maximum likelihood estimation, increasing the noise variance $\sigma^2$ will always increase the log-likelihood.
4. The log-likelihood function for linear regression with Gaussian noise is always concave with respect to the parameters $\boldsymbol{w}$.

For a detailed explanation of this problem, including proofs for each statement, see [Question 10: Log-Likelihood Optimization Properties](L3_3_10_explanation.md).

## Question 11

### Problem Statement
Consider the error decomposition in linear regression. Given:

- $\boldsymbol{w}^*$ is the optimal parameter vector with infinite training data
- $\hat{\boldsymbol{w}}$ is the parameter vector estimated from a finite training set
- $\boldsymbol{y}$ is the vector of true target values
- $\boldsymbol{x}$ is the feature vector

#### Task
1. [📚] Write down the mathematical expression for the structural error in linear regression
2. [📚] Write down the mathematical expression for the approximation error in linear regression
3. [📚] Prove that the expected error can be decomposed into the sum of structural and approximation errors:
   $$E_{\boldsymbol{x},y}[(y - \hat{\boldsymbol{w}}^T \boldsymbol{x})^2] = E_{\boldsymbol{x},y}[(y - \boldsymbol{w}^{*T} \boldsymbol{x})^2] + E_{\boldsymbol{x}}[(\boldsymbol{w}^{*T} \boldsymbol{x} - \hat{\boldsymbol{w}}^T \boldsymbol{x})^2]$$
4. [📚] Explain the practical significance of this error decomposition for model selection

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: Error Decomposition in Linear Regression](L3_3_11_explanation.md).

## Question 12

### Problem Statement
In linear regression, the total error can be decomposed into different components. Consider a scenario where we have:

- A true underlying function generating the data: $y = f(\boldsymbol{x}) + \epsilon$ where $\epsilon$ is random noise
- A linear model family: $\hat{y} = \boldsymbol{w}^T \boldsymbol{x}$
- The optimal linear parameter vector $\boldsymbol{w}^*$ (with infinite data)
- An estimated parameter vector $\hat{\boldsymbol{w}}$ from a finite training set

#### Task
1. [📚] Define structural error and explain what it represents conceptually in the context of linear regression
2. [📚] Define approximation error and explain why it depends on the specific training dataset used
3. [📚] For a dataset where the true underlying function is non-linear (e.g., $f(\boldsymbol{x}) = \sin(\boldsymbol{x})$), explain why the structural error cannot be eliminated even with infinite training data
4. [📚] How does increasing the number of training examples affect structural error and approximation error? Explain with mathematical reasoning.
5. [📚] Draw a diagram illustrating how the mean squared error (MSE) decomposes into structural error and approximation error, and how these relate to bias and variance

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: Understanding Error Components](L3_3_12_explanation.md).

## Question 13

### Problem Statement
Consider a linear regression model from a probabilistic perspective:

$$y = \boldsymbol{w}^T \boldsymbol{x} + \epsilon$$

where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is Gaussian noise with zero mean and variance $\sigma^2$.

#### Task
1. [📚] Write down the probability density function for observing a target value $y^{(i)}$ given input $\boldsymbol{x}^{(i)}$ and parameters $\boldsymbol{w}$ and $\sigma^2$
2. [📚] Construct the likelihood function for a dataset with $n$ observations
3. [📚] Derive the log-likelihood function and simplify it
4. [📚] Show mathematically that maximizing the log-likelihood is equivalent to minimizing the sum of squared errors
5. [📚] Derive the maximum likelihood estimator for the noise variance $\sigma^2$ after finding the optimal parameters $\boldsymbol{w}$

For a detailed explanation of this problem, including step-by-step derivations and key insights, see [Question 13: Maximum Likelihood in Linear Regression](L3_3_13_explanation.md). 