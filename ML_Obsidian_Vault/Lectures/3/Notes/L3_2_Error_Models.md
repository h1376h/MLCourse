# Error Models in Linear Regression

## Introduction
Error models in linear regression specify the distribution of the error terms (residuals) that account for the deviation between the model's predictions and the actual observations. The choice of error model has significant implications for parameter estimation, inference, and predictive performance.

## Classical Error Model
In the standard linear regression framework, the error term $\epsilon$ is typically assumed to follow a Gaussian (normal) distribution:

$$y = X\beta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

Where:
- $y$ is the vector of target values
- $X$ is the design matrix
- $\beta$ is the vector of coefficients
- $\epsilon$ is the vector of error terms
- $\sigma^2$ is the error variance
- $I$ is the identity matrix

### Key Assumptions of the Gaussian Error Model
1. **Zero Mean**: $E[\epsilon_i] = 0$
2. **Homoscedasticity**: $\text{Var}(\epsilon_i) = \sigma^2$ (constant variance)
3. **Independence**: $\text{Cov}(\epsilon_i, \epsilon_j) = 0$ for $i \neq j$
4. **Normality**: $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$

### Implications of the Gaussian Error Model
- Maximum Likelihood Estimation (MLE) is equivalent to Ordinary Least Squares (OLS)
- Coefficient estimates have the smallest variance among all unbiased linear estimators (BLUE)
- Facilitates straightforward statistical inference using t and F distributions
- Enables the construction of confidence and prediction intervals

## Alternative Error Models

### Heteroscedastic Error Model
When the variance of errors is not constant:

$$\epsilon_i \sim \mathcal{N}(0, \sigma_i^2)$$

Solutions include:
- Weighted Least Squares
- Robust standard errors
- Variance-stabilizing transformations

### Correlated Error Model
When errors are not independent:

$$\text{Cov}(\epsilon_i, \epsilon_j) \neq 0$$

Solutions include:
- Generalized Least Squares (GLS)
- Time series models (AR, MA, ARMA)
- Spatial models

### Non-Gaussian Error Models

#### Student's t-Distribution
More robust to outliers:

$$\epsilon_i \sim t_{\nu}(0, \sigma^2)$$

Where $\nu$ is the degrees of freedom parameter.

#### Laplace Distribution
Leads to Least Absolute Deviation (LAD) regression:

$$\epsilon_i \sim \text{Laplace}(0, b)$$

#### Mixture Models
For multi-modal error distributions:

$$\epsilon_i \sim \pi_1 \mathcal{N}(\mu_1, \sigma_1^2) + \pi_2 \mathcal{N}(\mu_2, \sigma_2^2) + \ldots$$

#### Skewed Distributions
For asymmetric errors:
- Skewed normal
- Gamma
- Log-normal

## Testing Error Model Assumptions

### Tests for Normality
- Shapiro-Wilk test
- Kolmogorov-Smirnov test
- Anderson-Darling test
- Q-Q plots

### Tests for Homoscedasticity
- Breusch-Pagan test
- White test
- Goldfeld-Quandt test
- Residual plots

### Tests for Independence
- Durbin-Watson test (autocorrelation)
- Runs test
- Autocorrelation and partial autocorrelation plots

## Addressing Violated Assumptions

### Transformations
- Log transformation
- Box-Cox transformation
- Yeo-Johnson transformation

### Robust Regression Methods
- M-estimation
- Huber loss
- Tukey's bisquare function

### Generalized Linear Models (GLMs)
Extension of linear regression to non-Gaussian error distributions:
- Binomial (logistic regression)
- Poisson (count data)
- Gamma (positive continuous data)
- Inverse Gaussian

## Error Models in Bayesian Linear Regression
In Bayesian regression, the error model is incorporated into the likelihood function:

$$p(y|X, \beta, \sigma^2) = \prod_{i=1}^n p(y_i|x_i, \beta, \sigma^2)$$

Common choices include:
- Gaussian likelihood (standard Bayesian linear regression)
- Student's t likelihood (robust Bayesian regression)
- Laplace likelihood (Bayesian LAD regression)

## Impact of Error Models on Model Performance

### When to Use Different Error Models
- Gaussian: When errors are approximately normally distributed
- Student's t: When data contains outliers
- Laplace: When minimizing absolute deviation is preferred
- Heteroscedastic: When error variance depends on predictors
- Autocorrelated: For time series and spatial data

### Trade-offs
- Model complexity vs. interpretability
- Robustness vs. efficiency
- Computational cost vs. statistical properties
