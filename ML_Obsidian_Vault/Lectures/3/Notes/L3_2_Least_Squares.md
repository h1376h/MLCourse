# Least Squares Method

## Introduction
The Least Squares Method is a fundamental approach for estimating the parameters of a linear regression model. It finds the parameter values that minimize the sum of squared differences between observed and predicted values, providing an optimal solution under certain assumptions.

## Objective Function
In linear regression, we model the relationship between a dependent variable $y$ and independent variables $X$ as:

$$y = X\beta + \epsilon$$

The Least Squares objective is to find the parameter vector $\beta$ that minimizes the sum of squared residuals (SSR):

$$\text{SSR}(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - X_i\beta)^2$$

In matrix notation:

$$\text{SSR}(\beta) = (y - X\beta)^T(y - X\beta)$$

## Derivation of Normal Equations
To find the minimizer of the SSR, we compute its gradient with respect to $\beta$ and set it to zero:

$$\nabla_\beta \text{SSR}(\beta) = \nabla_\beta [(y - X\beta)^T(y - X\beta)]$$

Expanding the expression:

$$\nabla_\beta [y^Ty - y^TX\beta - \beta^TX^Ty + \beta^TX^TX\beta]$$

Taking the derivative:

$$\nabla_\beta \text{SSR}(\beta) = -2X^Ty + 2X^TX\beta = 0$$

This gives us the **normal equations**:

$$X^TX\beta = X^Ty$$

## Solving for the Parameters
If $X^TX$ is invertible (full rank), the unique solution is:

$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

This formula gives the Ordinary Least Squares (OLS) estimator.

## Geometric Interpretation
The least squares solution has several geometric interpretations:

1. **Orthogonal Projection**: $X\hat{\beta}$ is the orthogonal projection of $y$ onto the column space of $X$.
2. **Residual Orthogonality**: The residual vector $(y - X\hat{\beta})$ is orthogonal to the column space of $X$.
3. **Minimizing Distance**: The solution minimizes the Euclidean distance between $y$ and the hyperplane spanned by the columns of $X$.

This can be visualized as:
- Each column of $X$ represents a direction in n-dimensional space
- $y$ is a point in this space
- $X\hat{\beta}$ is the point in the subspace spanned by columns of $X$ that is closest to $y$

## Properties of the Least Squares Estimator

### Unbiasedness
Under standard assumptions, the OLS estimator is unbiased:

$$E[\hat{\beta}] = \beta$$

### Variance
The covariance matrix of the estimator is:

$$\text{Var}(\hat{\beta}) = \sigma^2 (X^TX)^{-1}$$

Where $\sigma^2$ is the variance of the error term.

### Gauss-Markov Theorem
Under the assumptions of the classical linear regression model, the OLS estimator is the Best Linear Unbiased Estimator (BLUE), meaning it has the lowest variance among all linear unbiased estimators.

## Alternative Formulations

### QR Decomposition
For numerical stability, the least squares problem can be solved using QR decomposition of $X$:

$$X = QR$$

Where $Q$ is orthogonal and $R$ is upper triangular. The solution becomes:

$$\hat{\beta} = R^{-1}Q^Ty$$

### Singular Value Decomposition (SVD)
When $X$ is ill-conditioned or rank-deficient, SVD provides a robust solution:

$$X = U\Sigma V^T$$

The solution is:

$$\hat{\beta} = V\Sigma^{+}U^Ty$$

Where $\Sigma^{+}$ is the pseudoinverse of $\Sigma$.

## Weighted Least Squares
When error variances are not equal (heteroscedasticity), we can use Weighted Least Squares (WLS):

$$\text{SSR}_W(\beta) = (y - X\beta)^T W (y - X\beta)$$

Where $W$ is a diagonal weight matrix. The solution is:

$$\hat{\beta}_{WLS} = (X^TWX)^{-1}X^TWy$$

## Regularized Least Squares
To combat overfitting or multicollinearity, regularized variants add penalty terms:

### Ridge Regression (L2 penalty)
$$\hat{\beta}_{ridge} = \arg\min_\beta \left\{ \|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2 \right\}$$

Solution:
$$\hat{\beta}_{ridge} = (X^TX + \lambda I)^{-1}X^Ty$$

### Lasso Regression (L1 penalty)
$$\hat{\beta}_{lasso} = \arg\min_\beta \left\{ \|y - X\beta\|_2^2 + \lambda\|\beta\|_1 \right\}$$

## Iteratively Reweighted Least Squares (IRLS)
For robust regression or generalized linear models, IRLS iteratively updates weights based on previous estimates:

1. Initialize weights $W^{(0)}$
2. Compute $\hat{\beta}^{(t)} = (X^TW^{(t-1)}X)^{-1}X^TW^{(t-1)}y$
3. Update weights $W^{(t)}$ based on residuals
4. Repeat until convergence

## Numerical Considerations
- Directly computing $(X^TX)^{-1}$ can be numerically unstable
- QR decomposition or SVD are more stable alternatives
- Ill-conditioned data may require regularization
- Large datasets may require iterative or stochastic methods

## Limitations
- Sensitive to outliers
- Assumes homoscedasticity and independence of errors
- May perform poorly with multicollinearity
- Can overfit with high-dimensional data
