# Statistical Properties of Linear Regression

## Introduction
The statistical properties of linear regression estimators are essential for understanding their behavior, making valid inferences, and assessing the reliability of model predictions. These properties form the foundation for hypothesis testing, confidence intervals, and other statistical analyses in regression modeling.

## Gauss-Markov Assumptions
The classical linear regression model makes several key assumptions, collectively known as the Gauss-Markov assumptions:

1. **Linearity**: The relationship between predictors and the response is linear.
2. **Full Rank**: The design matrix $X$ has full column rank (no perfect multicollinearity).
3. **Exogeneity**: The errors have zero conditional mean: $E[\epsilon|X] = 0$.
4. **Homoscedasticity**: The errors have constant variance: $\text{Var}(\epsilon|X) = \sigma^2 I$.
5. **No Autocorrelation**: The errors are uncorrelated: $\text{Cov}(\epsilon_i, \epsilon_j|X) = 0$ for all $i \neq j$.
6. **Optional: Normality**: The errors are normally distributed: $\epsilon|X \sim \mathcal{N}(0, \sigma^2 I)$.

## Properties of OLS Estimators

### Unbiasedness
Under assumptions 1-3, the OLS estimator $\hat{\beta}$ is unbiased:

$$E[\hat{\beta}|X] = \beta$$

This means that on average, the estimator gives the true parameter value.

### Variance-Covariance Matrix
The variance-covariance matrix of the OLS estimator is:

$$\text{Var}(\hat{\beta}|X) = \sigma^2 (X^TX)^{-1}$$

This quantifies the uncertainty in our parameter estimates.

### Best Linear Unbiased Estimator (BLUE)
Under assumptions 1-5 (Gauss-Markov), the OLS estimator is BLUE, meaning it has the minimum variance among all linear unbiased estimators. This is known as the Gauss-Markov Theorem.

### Consistency
As the sample size increases, the OLS estimator converges in probability to the true parameter values:

$$\hat{\beta} \xrightarrow{p} \beta \text{ as } n \to \infty$$

This property ensures that with enough data, we can estimate the true parameters arbitrarily closely.

### Asymptotic Normality
Under certain conditions, as the sample size increases, the distribution of the OLS estimator approaches a normal distribution:

$$\sqrt{n}(\hat{\beta} - \beta) \xrightarrow{d} \mathcal{N}(0, \sigma^2 Q^{-1})$$

Where $Q = \lim_{n \to \infty} \frac{1}{n}X^TX$.

## Distribution of Estimators Under Normality
When the normality assumption (6) is added, we can derive the exact finite-sample distributions of various statistics:

### Distribution of Coefficients
The OLS estimator follows a multivariate normal distribution:

$$\hat{\beta}|X \sim \mathcal{N}(\beta, \sigma^2(X^TX)^{-1})$$

### Distribution of Error Variance Estimator
The unbiased estimator of error variance is:

$$\hat{\sigma}^2 = \frac{1}{n-p}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n-p}(y - X\hat{\beta})^T(y - X\hat{\beta})$$

This estimator follows a scaled chi-squared distribution:

$$\frac{(n-p)\hat{\sigma}^2}{\sigma^2} \sim \chi^2_{n-p}$$

### Distribution of t-Statistics
For testing individual coefficients, the t-statistic is:

$$t_j = \frac{\hat{\beta}_j - \beta_j}{\text{SE}(\hat{\beta}_j)} = \frac{\hat{\beta}_j - \beta_j}{\hat{\sigma}\sqrt{(X^TX)^{-1}_{jj}}}$$

Under the null hypothesis $\beta_j = 0$, this follows a t-distribution with $n-p$ degrees of freedom:

$$t_j \sim t_{n-p}$$

### Distribution of F-Statistic
For testing multiple coefficients simultaneously, the F-statistic is used:

$$F = \frac{(RSS_R - RSS_U)/q}{RSS_U/(n-p)}$$

Where $RSS_R$ and $RSS_U$ are the residual sum of squares for the restricted and unrestricted models, and $q$ is the number of restrictions. Under the null hypothesis, this follows an F-distribution:

$$F \sim F_{q, n-p}$$

## Properties Related to Prediction

### Prediction Variance
The variance of a prediction at a new point $x_0$ is:

$$\text{Var}(\hat{y}_0|X, x_0) = \sigma^2 \left(1 + x_0^T(X^TX)^{-1}x_0\right)$$

This variance is minimized at the mean of the predictors and increases as we move away from the mean.

### Prediction Intervals
A $(1-\alpha)$ prediction interval for a new observation at $x_0$ is:

$$\hat{y}_0 \pm t_{n-p, 1-\alpha/2} \cdot \hat{\sigma}\sqrt{1 + x_0^T(X^TX)^{-1}x_0}$$

### Confidence Intervals for Mean Response
A $(1-\alpha)$ confidence interval for the mean response at $x_0$ is:

$$\hat{y}_0 \pm t_{n-p, 1-\alpha/2} \cdot \hat{\sigma}\sqrt{x_0^T(X^TX)^{-1}x_0}$$

## Properties Under Violations of Assumptions

### Multicollinearity
When predictors are highly correlated:
- OLS estimates remain unbiased
- Variance of coefficients increases
- Estimators become sensitive to small changes in the data
- Coefficients may have the wrong sign

### Heteroscedasticity
When error variance is not constant:
- OLS estimates remain unbiased
- Standard errors are biased, leading to invalid hypothesis tests
- OLS is no longer BLUE
- Weighted Least Squares becomes more efficient

### Autocorrelation
When errors are correlated:
- OLS estimates remain unbiased
- Standard errors are biased
- OLS is no longer BLUE
- GLS methods are more appropriate

### Endogeneity
When $E[\epsilon|X] \neq 0$ (e.g., due to omitted variables or measurement error):
- OLS estimates are biased and inconsistent
- Instrumental variable methods may be needed

## Properties of Regularized Estimators

### Ridge Regression
The ridge estimator $\hat{\beta}_{ridge} = (X^TX + \lambda I)^{-1}X^Ty$ has:
- Biased estimates (bias increases with $\lambda$)
- Reduced variance compared to OLS
- Potentially lower mean squared error than OLS
- Shrinkage of all coefficients toward zero

### Lasso Regression
The lasso estimator, which minimizes $\|y - X\beta\|_2^2 + \lambda\|\beta\|_1$, has:
- Biased estimates
- Feature selection capability (sparse solutions)
- No closed-form expression (except in orthogonal design)
- Potential for exact zero coefficients

## Diagnostic Measures and Influence

### Leverage
The leverage of the $i$-th observation is the $i$-th diagonal element of the hat matrix $H = X(X^TX)^{-1}X^T$:

$$h_{ii} = [H]_{ii}$$

High leverage points can have a large influence on the regression estimates.

### Cook's Distance
Cook's distance measures the influence of each observation:

$$D_i = \frac{r_i^2}{p\hat{\sigma}^2} \cdot \frac{h_{ii}}{(1-h_{ii})^2}$$

Where $r_i$ is the residual for the $i$-th observation.

### DFBETA
DFBETA measures how each observation affects individual coefficient estimates:

$$\text{DFBETA}_{ij} = \frac{\hat{\beta}_j - \hat{\beta}_{j(i)}}{\text{SE}(\hat{\beta}_j)}$$

Where $\hat{\beta}_{j(i)}$ is the estimate of $\beta_j$ when the $i$-th observation is removed.
