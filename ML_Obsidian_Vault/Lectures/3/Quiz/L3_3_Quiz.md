# Lecture 3.3: Probabilistic View of Linear Regression Quiz

## Overview
This quiz contains 26 questions from different topics covered in section 3.3 of the lectures on the Probabilistic View of Linear Regression.

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

## Question 14

### Problem Statement
Consider a scenario where the true underlying function is $f(x) = x^2$, but we're using a linear model $\hat{y} = w_0 + w_1x$. 

Given:
- The optimal linear model parameters (with infinite data) are $w_0^* = 2$ and $w_1^* = 2$
- From a finite training set, we estimated $\hat{w}_0 = 1.5$ and $\hat{w}_1 = 2.5$
- We want to evaluate the model at $x = 3$

#### Task
1. [📚] Calculate the true value $y = f(3)$
2. [📚] Calculate the prediction from the optimal linear model $y_{opt} = w_0^* + w_1^* \cdot 3$
3. [📚] Calculate the prediction from the estimated model $\hat{y} = \hat{w}_0 + \hat{w}_1 \cdot 3$
4. [📚] Compute the structural error $(y - y_{opt})^2$
5. [📚] Compute the approximation error $(y_{opt} - \hat{y})^2$
6. [📚] Verify that the total squared error $(y - \hat{y})^2$ equals the sum of structural and approximation errors

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 14: Calculating Error Components](L3_3_14_explanation.md).

## Question 15

### Problem Statement
Consider a simple dataset with 3 points where the true function is $f(x) = x^2$:

| $x$ | True $y = x^2$ |
|-----|----------------|
| 1   | 1              |
| 2   | 4              |
| 3   | 9              |

The best possible linear approximation for this data is $\hat{y} = -1 + 3x$, while an estimated model from a sample is $\hat{y} = -0.5 + 2.5x$.

#### Task
1. [📚] Calculate the predictions from both models for each $x$ value
2. [📚] Compute the structural error for each point $(y - y_{opt})^2$ and its average
3. [📚] Compute the approximation error for each point $(y_{opt} - \hat{y})^2$ and its average
4. [📚] Calculate the total squared error for each point and verify it equals the sum of the corresponding structural and approximation errors
5. [📚] What percentage of the average total error is due to structural error vs. approximation error?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 15: Error Decomposition with Multiple Points](L3_3_15_explanation.md).

## Question 16

### Problem Statement
You're a treasure hunter on Probability Island where treasures are buried along a straight path. The true location of a treasure is given by $y = 3x + 2 + \epsilon$ where $\epsilon \sim \mathcal{N}(0, 4)$ represents random displacement due to tides.

You've found 4 treasures at the following locations:

| $x$ (distance from shore) | $y$ (steps along coast) |
|---------------------------|--------------------------|
| 1                         | 6                        |
| 2                         | 9                        |
| 3                         | 11                       |
| 4                         | 16                       |

#### Task
1. [📚] Write the likelihood function for finding treasures at these locations
2. [📚] Calculate the maximum likelihood estimates for parameters $w_0$ and $w_1$
3. [📚] Estimate the noise variance $\sigma^2$ using MLE
4. [📚] A map indicates a treasure at $x=2.5$. Calculate the probability this treasure is located beyond $y>12$
5. Calculate the 90% prediction interval for where to dig at $x=2.5$

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 16: Probability Island Treasure Hunt](L3_3_16_explanation.md).

## Question 17

### Problem Statement
A social media company models user engagement (hours spent) based on number of connections. The true relationship is $f(x) = 0.5x^2$, but the company uses a linear model $\hat{y} = w_0 + w_1x$.

With infinite data, the optimal linear approximation would be $\hat{y} = 2 + 3x$. However, with limited data, the company estimated $\hat{y} = 1 + 3.5x$.

#### Task
1. [📚] For a user with 5 connections, calculate:
   - The true expected engagement hours
   - Prediction from the optimal linear model
   - Prediction from the estimated model
   - The structural error
   - The approximation error
   - Verify the total squared error equals the sum of structural and approximation errors
2. [📚] For users with 4, 6, and 8 connections, determine which error component (structural or approximation) contributes more to total prediction error and by what percentage.
3. [📚] If the company wants to reduce total error below 5 hours$^2$, should they collect more data or use a non-linear model? Justify mathematically.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 17: Error Decomposition Challenge](L3_3_17_explanation.md).

## Question 18

### Problem Statement
A medical researcher is developing a model to predict blood pressure response ($y$) to medication dosage ($x$). The researcher believes the relationship follows a linear model with Gaussian noise:

$$y = w_0 + w_1x + \epsilon$$

where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

Clinical trials produced the following data:

| Dosage ($x$, mg) | Blood Pressure Reduction ($y$, mmHg) |
|------------------|-------------------------------------|
| 10               | 5                                   |
| 20               | 8                                   |
| 30               | 13                                  |
| 40               | 15                                  |
| 50               | 21                                  |

#### Task
1. [📚] Calculate the maximum likelihood estimates for $w_0$ and $w_1$
2. [📚] Estimate the noise variance $\sigma^2$
3. [📚] Write the complete predictive distribution for a new patient receiving a 35mg dose
4. [📚] The FDA requires that a medication demonstrate at least 12mmHg reduction with 80% probability to be approved. Does a 35mg dose meet this requirement? Show calculations.
5. [📚] If the researcher suspects the true relationship is quadratic rather than linear, explain how this would affect the structural error and whether MLE would still be appropriate for parameter estimation.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 18: Medical Dosage Response Model](L3_3_18_explanation.md).

## Question 19

### Problem Statement
Consider a linear regression model with a probabilistic interpretation and a Bayesian approach using MAP estimation:
$$y = w_0 + w_1x + \epsilon$$

where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ and you have prior beliefs about the parameters: $w_0 \sim \mathcal{N}(0, \tau_0^2)$ and $w_1 \sim \mathcal{N}(0, \tau_1^2)$.

For the following dataset with 3 observations:

| $x$ | $y$ |
|-----|-----|
| 1   | 2   |
| 2   | 4   |
| 3   | 5   |

#### Task
1. Write down the posterior distribution for the parameters $w_0$ and $w_1$ given the data
2. Derive the logarithm of the posterior distribution
3. Show that MAP estimation with Gaussian priors is equivalent to ridge regression with specific regularization parameters
4. Calculate the MAP estimates for $w_0$ and $w_1$ assuming $\sigma^2 = 1$, $\tau_0^2 = 10$, and $\tau_1^2 = 2$

For a detailed explanation of this problem, including step-by-step solutions and derivations, see [Question 19: MAP Estimation for Linear Regression](L3_3_19_explanation.md).

## Question 20

### Problem Statement
Consider a Bayesian approach to linear regression where we use conjugate priors for efficient posterior computations. For a simple linear regression model:
$$y = w_0 + w_1x + \epsilon$$

where $\epsilon \sim \mathcal{N}(0, \sigma^2)$, and assuming $\sigma^2$ is known.

#### Task
1. Identify the conjugate prior distribution for the parameter vector $\boldsymbol{w} = [w_0, w_1]^T$ in linear regression with Gaussian noise
2. Given a prior $\boldsymbol{w} \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$ where $\boldsymbol{\mu}_0 = [0, 0]^T$ and $\boldsymbol{\Sigma}_0 = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$, derive the posterior distribution after observing the data points $(x^{(1)}, y^{(1)}) = (1, 3)$ and $(x^{(2)}, y^{(2)}) = (2, 5)$
3. Calculate the posterior predictive distribution for a new input $x_{\text{new}} = 1.5$
4. Explain how the posterior uncertainty in parameters affects prediction uncertainty compared to maximum likelihood estimation

For a detailed explanation of this problem, including step-by-step derivations and key insights, see [Question 20: Conjugate Priors in Bayesian Linear Regression](L3_3_20_explanation.md).

## Question 21

### Problem Statement
In probabilistic linear regression, the noise variance $\sigma^2$ is often treated as a known parameter, but in practice, it must be estimated from data. Consider a Bayesian approach to estimating both the regression coefficients and the noise variance.

#### Task
1. For a linear model $y = \boldsymbol{w}^T\boldsymbol{x} + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$, specify an appropriate conjugate prior for the unknown variance $\sigma^2$
2. Given the prior distribution, derive the joint posterior distribution for both $\boldsymbol{w}$ and $\sigma^2$
3. Explain the concept of the marginal likelihood (model evidence) $p(\boldsymbol{y}|\boldsymbol{X})$ in this context and why it's useful for model comparison
4. For a dataset with 5 observations and sum of squared residuals (using MLE estimates for $\boldsymbol{w}$) of 12, calculate the posterior mean for $\sigma^2$ assuming an Inverse-Gamma(2, 4) prior

For a detailed explanation of this problem, including key Bayesian concepts and derivations, see [Question 21: Bayesian Estimation of Regression Variance](L3_3_21_explanation.md).

## Question 22

### Problem Statement
You are analyzing housing price data and need to decide between different linear regression models with varying complexity:
- Model 1: $y = w_0 + w_1x_1 + \epsilon$ (price depends only on house size)
- Model 2: $y = w_0 + w_1x_1 + w_2x_2 + \epsilon$ (price depends on size and number of bedrooms)
- Model 3: $y = w_0 + w_1x_1 + w_2x_2 + w_3x_3 + \epsilon$ (price depends on size, bedrooms, and age)

Where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ for all models.

#### Task
1. Explain how to compute the marginal likelihood (model evidence) $p(\boldsymbol{y}|\mathcal{M}_i)$ for model comparison
2. Describe the role of Occam's razor in Bayesian model selection and how it's naturally incorporated in the marginal likelihood
3. Given the following (log) model evidences, compute the posterior probabilities for each model assuming equal prior probabilities:
   - $\log p(\boldsymbol{y}|\mathcal{M}_1) = -45.2$
   - $\log p(\boldsymbol{y}|\mathcal{M}_2) = -42.8$
   - $\log p(\boldsymbol{y}|\mathcal{M}_3) = -43.1$
4. Explain the relationship between Bayesian model selection and information criteria like BIC

For a detailed explanation of this problem, including key concepts in Bayesian model comparison, see [Question 22: Bayesian Model Selection for Linear Regression](L3_3_22_explanation.md).

## Question 23

### Problem Statement
Exact Bayesian inference can be computationally intensive, especially with complex models. The Laplace approximation provides a useful Gaussian approximation to the posterior distribution. Consider its application to Bayesian linear regression.

#### Task
1. Explain the key idea behind the Laplace approximation for approximating posterior distributions
2. For a linear regression model with the posterior distribution $p(\boldsymbol{w}|\boldsymbol{y},\boldsymbol{X}) \propto p(\boldsymbol{y}|\boldsymbol{X},\boldsymbol{w})p(\boldsymbol{w})$, outline the steps to derive the Laplace approximation
3. If the negative log posterior has the form $E(\boldsymbol{w}) = \frac{1}{2}(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w})^T(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w}) + \frac{\lambda}{2}\boldsymbol{w}^T\boldsymbol{w} + C$, compute the Hessian matrix needed for the Laplace approximation
4. Using the Laplace approximation, derive the approximate predictive distribution for a new input $\boldsymbol{x}_{\text{new}}$

For a detailed explanation of this problem, including the mathematical derivation of the Laplace approximation, see [Question 23: Laplace Approximation in Bayesian Linear Regression](L3_3_23_explanation.md).

## Question 24

### Problem Statement
Instead of selecting a single "best" linear regression model, Bayesian model averaging (BMA) combines predictions from multiple models, weighted by their posterior probabilities. Consider three competing linear regression models:

- $\mathcal{M}_1$: $y = w_0 + w_1x_1 + \epsilon$
- $\mathcal{M}_2$: $y = w_0 + w_1x_1 + w_2x_2 + \epsilon$
- $\mathcal{M}_3$: $y = w_0 + w_1x_1 + w_3x_3 + \epsilon$

Where all models assume $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

#### Task
1. Write down the formula for making predictions using Bayesian model averaging across these three models
2. For a new input point $\boldsymbol{x}_{\text{new}} = [1, 1.5, 2.0, 0.5]$, if the models predict:
   - $\mathcal{M}_1$: $\hat{y}_1 = 4.2$ with predictive variance $v_1 = 0.8$
   - $\mathcal{M}_2$: $\hat{y}_2 = 5.1$ with predictive variance $v_2 = 0.6$
   - $\mathcal{M}_3$: $\hat{y}_3 = 3.8$ with predictive variance $v_3 = 1.0$
   - And the posterior model probabilities are $p(\mathcal{M}_1|\boldsymbol{y}) = 0.2$, $p(\mathcal{M}_2|\boldsymbol{y}) = 0.5$, $p(\mathcal{M}_3|\boldsymbol{y}) = 0.3$
   - Calculate the BMA prediction and its total variance
3. Explain how BMA naturally accounts for model uncertainty unlike traditional model selection
4. Discuss the computational challenges of implementing BMA in practice and potential approximation methods

For a detailed explanation of this problem, including key concepts in Bayesian model averaging, see [Question 24: Bayesian Model Averaging for Linear Regression](L3_3_24_explanation.md).

## Question 25

### Problem Statement
Different noise distributions in linear regression lead to different loss functions. Consider the standard linear model:

$$y = \boldsymbol{w}^T\boldsymbol{x} + \epsilon$$

And three different assumptions about the noise distribution:

- Model A: $\epsilon \sim \mathcal{N}(0, \sigma^2)$ (Gaussian noise)
- Model B: $\epsilon \sim \text{Laplace}(0, b)$ (Laplace noise)
- Model C: $\epsilon \sim \text{Student-}t(\nu, 0, \sigma)$ (Student's t-noise with $\nu$ degrees of freedom)

#### Task
1. Derive the negative log-likelihood (loss function) for each noise model
2. Explain why Gaussian noise leads to squared error loss, Laplace noise leads to absolute error loss, and Student's t-noise leads to a robust loss function
3. For the following dataset with a potential outlier:

| $x$ | $y$ |
|-----|-----|
| 1   | 2   |
| 2   | 4   |
| 3   | 6   |
| 4   | 12  |
| 5   | 10  |

Calculate the optimal $w_0$ and $w_1$ for a simple linear model $y = w_0 + w_1x$ under both Gaussian and Laplace noise assumptions, and compare the results

4. Discuss the robustness properties of each model and scenarios where each would be preferred

For a detailed explanation of this problem, including derivations and comparisons of different noise models, see [Question 25: Noise Models and Loss Functions in Linear Regression](L3_3_25_explanation.md).

## Question 26

### Problem Statement
Empirical Bayes (EB) methods provide a practical approach to Bayesian inference by estimating hyperparameters from the data. Consider a Bayesian linear regression model with the following hierarchical structure:

$$y = \boldsymbol{w}^T\boldsymbol{x} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
$$\boldsymbol{w} \sim \mathcal{N}(\boldsymbol{0}, \alpha^{-1}\boldsymbol{I})$$

where $\alpha$ is a precision hyperparameter controlling the strength of the prior.

#### Task
1. Explain the key idea behind empirical Bayes and how it differs from fully Bayesian approaches
2. Derive the marginal likelihood $p(\boldsymbol{y}|\boldsymbol{X}, \alpha, \sigma^2)$ by integrating out the parameters $\boldsymbol{w}$
3. Show how to find the optimal value of $\alpha$ by maximizing the marginal likelihood
4. For a given dataset with design matrix $\boldsymbol{X}$ (10 examples, 3 features) and target vector $\boldsymbol{y}$, if the eigenvalues of $\boldsymbol{X}^T\boldsymbol{X}$ are $\{50, 30, 10\}$ and the sum of squared errors using the least squares solution is 20 with $\sigma^2 = 2$, estimate the optimal $\alpha$ using empirical Bayes

For a detailed explanation of this problem, including key concepts in empirical Bayes methods for linear regression, see [Question 26: Empirical Bayes in Linear Regression](L3_3_26_explanation.md). 