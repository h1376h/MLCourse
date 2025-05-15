# Question 18: Normal Equations in Linear Regression

## Problem Statement
The normal equations provide an analytical solution to the linear regression problem by finding parameter values that minimize the sum of squared errors. Understanding their matrix formulation is crucial for theoretical analysis.

### Task
Given a simple linear regression model $y = \beta_0 + \beta_1x$ with $n$ data points:

1. Write down the formula for the normal equations in matrix form
2. What matrix property ensures that a unique solution exists?

## Understanding the Problem
This problem focuses on the mathematical foundation of linear regression - specifically how we can use matrix algebra to derive closed-form solutions for finding the best-fitting parameters. The normal equations are a set of equations derived from the principle of least squares, which provide the analytical solution to linear regression problems. 

In linear regression, we aim to find the parameters (coefficients) that minimize the sum of squared differences between observed and predicted values. The matrix form of the normal equations allows us to concisely represent and solve this minimization problem for any number of predictors.

## Solution

### Step 1: Deriving the Normal Equations in Matrix Form
To solve a linear regression problem, we first express the model in matrix form. For a simple linear regression model $y = \beta_0 + \beta_1x$ with $n$ data points, we can write:

$$y = X\beta + \varepsilon$$

Where:
- $y$ is an $n \times 1$ vector of response values $[y_1, y_2, \ldots, y_n]^T$
- $X$ is an $n \times 2$ design matrix where the first column contains all 1s (for the intercept) and the second column contains the $x$ values:

$$X = \begin{bmatrix} 
1 & x_1 \\
1 & x_2 \\
\vdots & \vdots \\
1 & x_n
\end{bmatrix}$$

- $\beta$ is a $2 \times 1$ vector $[\beta_0, \beta_1]^T$ of parameters
- $\varepsilon$ is an $n \times 1$ vector of error terms $[\varepsilon_1, \varepsilon_2, \ldots, \varepsilon_n]^T$

The objective in linear regression is to minimize the sum of squared errors (SSE):

$$\min_{\beta} S(\beta) = \varepsilon^T\varepsilon = (y - X\beta)^T(y - X\beta)$$

Let's expand this expression:

$$\begin{align}
S(\beta) &= (y - X\beta)^T(y - X\beta) \\
&= y^Ty - y^TX\beta - \beta^TX^Ty + \beta^TX^TX\beta \\
&= y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta
\end{align}$$

Note that $y^TX\beta = \beta^TX^Ty$ because both are scalar (1×1) values.

To find the values of $\beta$ that minimize $S(\beta)$, we take the derivative with respect to $\beta$ and set it equal to zero:

$$\begin{align}
\frac{\partial S}{\partial \beta} &= \frac{\partial}{\partial \beta}(y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta) \\
&= 0 - 2X^Ty + 2X^TX\beta \\
&= -2X^Ty + 2X^TX\beta \\
&= -2X^T(y - X\beta)
\end{align}$$

Setting the derivative equal to zero (necessary condition for minimum):

$$\frac{\partial S}{\partial \beta} = -2X^T(y - X\beta) = 0$$

Simplifying:

$$\begin{align}
X^T(y - X\beta) &= 0 \\
X^Ty - X^TX\beta &= 0 \\
X^TX\beta &= X^Ty
\end{align}$$

This final equation, $X^TX\beta = X^Ty$, is the formula for the normal equations in matrix form.

If $X^TX$ is invertible, the solution is:

$$\beta = (X^TX)^{-1}X^Ty$$

This is the least squares estimator for $\beta$.

For our simple linear regression model with two parameters, the normal equations can be written explicitly as:

$$\begin{bmatrix} 
\sum(1) & \sum(x_i) \\
\sum(x_i) & \sum(x_i^2)
\end{bmatrix} 
\begin{bmatrix} 
\beta_0 \\
\beta_1
\end{bmatrix} = 
\begin{bmatrix} 
\sum(y_i) \\
\sum(x_iy_i)
\end{bmatrix}$$

This gives the familiar formulas for simple linear regression:

$$\beta_1 = \frac{\sum(x_iy_i) - n^{-1}(\sum x_i)(\sum y_i)}{\sum(x_i^2) - n^{-1}(\sum x_i)^2}$$

$$\beta_0 = \bar{y} - \beta_1\bar{x}$$

### Step 2: Matrix Property for Unique Solution

The key matrix property that ensures a unique solution exists is that $X^TX$ must be invertible (non-singular).

For $X^TX$ to be invertible, the following conditions must be met:

1. The matrix $X$ must have full column rank.
   This means that the columns of $X$ must be linearly independent vectors.
   In mathematical terms: $\text{rank}(X) = p$, where $p$ is the number of columns in $X$.

2. There must be at least as many observations as parameters ($n \geq p$).
   This is because a matrix with more columns than rows cannot have full column rank.

3. No exact linear relationship can exist among the predictor variables.
   This means no column can be expressed as a linear combination of other columns.

$X^TX$ has several important mathematical properties:

1. $X^TX$ is always a square matrix with dimensions $p \times p$.
2. $X^TX$ is always symmetric: $(X^TX)^T = X^TX$
3. $X^TX$ is positive semi-definite, meaning:
   a. All eigenvalues are non-negative
   b. For any vector $v$, $v^T(X^TX)v \geq 0$
4. $X^TX$ is positive definite (and thus invertible) if and only if $X$ has full column rank.
   When positive definite:
   a. All eigenvalues are strictly positive
   b. For any non-zero vector $v$, $v^T(X^TX)v > 0$

In our simple linear regression case with $y = \beta_0 + \beta_1x$, the design matrix $X$ has 2 columns:
- First column: all 1s (for the intercept $\beta_0$)
- Second column: the $x$ values (for the slope $\beta_1$)

For $X^TX$ to be invertible in this case:
1. We need at least 2 data points ($n \geq 2$)
2. The $x$ values must not all be identical

If all $x$ values were identical (e.g., all equal to 5), then the second column would be a scalar multiple of the first: $[x_1, x_2, \ldots, x_n]^T = 5[1, 1, \ldots, 1]^T$. This would make the columns linearly dependent, causing $X^TX$ to be singular.

Testing for invertibility:
1. Check the determinant: $\det(X^TX) \neq 0$
2. Check the eigenvalues: all eigenvalues of $X^TX > 0$
3. Check the rank: $\text{rank}(X) = p$

When $X^TX$ is invertible, the normal equations have a unique solution $\beta = (X^TX)^{-1}X^Ty$. When $X^TX$ is not invertible, the system is rank-deficient, and either no solution exists or infinitely many solutions exist that minimize the sum of squared errors.

## Numerical Examples

### Example 1: Well-Conditioned Case
Let's demonstrate these concepts with a simple example using 10 data points where the true model is $y = 3 + 2x$ with some random noise.

First, we construct the design matrix $X$ with a column of 1s and a column of $x$ values:

```
Design matrix X:
  X[0] = [1.0, 0.00]
  X[1] = [1.0, 1.11]
  X[2] = [1.0, 2.22]
  ...
  X[9] = [1.0, 10.00]
```

Computing $X^TX$:
```
X^T X matrix:
  [10.00, 50.00]
  [50.00, 351.85]
```

The elements of $X^TX$ have specific interpretations:
- $X^TX[0,0] = 10.00 =$ number of observations ($n$)
- $X^TX[0,1] = X^TX[1,0] = 50.00 =$ sum of $x$ values
- $X^TX[1,1] = 351.85 =$ sum of squared $x$ values

Calculating the determinant:
```
Determinant of X^T X: 1018.52
```
Since the determinant is non-zero, $X^TX$ is invertible and a unique solution exists.

Checking the eigenvalues:
```
Eigenvalues of X^T X: 2.84, 359.01
```
All eigenvalues are positive, confirming $X^TX$ is positive definite.

Computing $X^Ty$:
```
X^T y vector:
  [134.48, 875.47]
```

The elements of $X^Ty$ represent:
- $X^Ty[0] = 134.48 =$ sum of $y$ values
- $X^Ty[1] = 875.47 =$ sum of $x \times y$ products

Calculating the inverse of $X^TX$:
```
(X^T X)^-1 matrix:
  [0.345455, -0.049091]
  [-0.049091, 0.009818]
```

Finally, solving for $\beta$ using the normal equations:
$$\begin{align}
\beta &= (X^TX)^{-1}X^Ty \\
&= \begin{bmatrix}
0.345455 & -0.049091 \\
-0.049091 & 0.009818
\end{bmatrix}
\begin{bmatrix}
134.48 \\
875.47
\end{bmatrix} \\
&= \begin{bmatrix}
3.4791 \\
1.9938
\end{bmatrix}
\end{align}$$

So our estimated parameters are:
- $\hat{\beta}_0 = 3.4791$ (true value: 3)
- $\hat{\beta}_1 = 1.9938$ (true value: 2)

The small differences are due to the random noise in our simulated data.

Evaluating the model:
```
Sum of Squared Errors (SSE): 4.7007
R-squared: 0.9885
```

The high R-squared value indicates that our model explains over 98% of the variance in the response variable.

### Example 2: Singular Case
To demonstrate what happens when $X^TX$ is not invertible, we can create a case where all $x$ values are identical (perfect collinearity):

```
Data points where all x values = 5:
X^T X matrix:
  [10.00, 50.00]
  [50.00, 250.00]
```

Examining this matrix, we can see that the second row is exactly 5 times the first row:
```
Ratio check for linear dependence in X^T X:
  Row 1 ratio: 50.0 / 10.0 = 5.0
  Row 2 ratio: 250.0 / 50.0 = 5.0
```

This confirms the linear dependence in the columns of $X$.

Calculating the determinant:
```
Determinant of X^T X: 0.0000000000
```

Since the determinant is zero, $X^TX$ is not invertible. Attempting to solve the normal equations directly results in an error:

```
Error: Singular matrix
```

Checking the eigenvalues:
```
Eigenvalues of X^T X: 0.000000, 260.000000
```
One eigenvalue is zero, confirming that $X^TX$ is singular.

In this case, we can use the pseudoinverse (Moore-Penrose) to find one of infinitely many solutions that minimize the sum of squared errors:

```
β₀ = 0.0885
β₁ = 0.4423
Sum of squared errors: 8.1000
```

We can verify that infinitely many solutions exist by testing an alternative solution:
```
Alternative solution:
β₀ = 1.0885
β₁ = 0.2423
Sum of squared errors: 8.1000
```

Both solutions give exactly the same SSE, confirming that when $X^TX$ is singular, there are infinitely many solutions that minimize the sum of squared errors.

## Visual Explanations

### Linear Regression Model and Fitted Line
![Linear Regression Data and Fitted Line](../Images/L3_2_Quiz_18/plot1_regression_line.png)

This figure shows the data points (blue), the fitted line (green), and the true underlying relationship (red dashed line). The vertical black lines represent the residuals between observed and predicted values, which the normal equations help minimize.

### Sum of Squared Errors Surface
![SSE Surface](../Images/L3_2_Quiz_18/plot2_sse_surface.png)

The 3D surface (left) illustrates how the sum of squared errors varies with different values of intercept and slope parameters. The red star marks the unique minimum found by solving the normal equations. The contour plot (right) provides a top-down view of this surface, with contour lines representing equal SSE values.

### Collinearity vs Non-Collinearity
![Collinearity Comparison](../Images/L3_2_Quiz_18/plot3_collinearity.png)

This comparison illustrates the difference between non-collinear data (left) and collinear data (right). In the non-collinear case, the x-values vary, making $X^TX$ invertible ($\det(X^TX) > 0$) and ensuring a unique solution exists. In the collinear case, all x-values are identical, making $X^TX$ singular ($\det(X^TX) = 0$) and resulting in no unique solution.

### Eigenvalues of X^T X Matrix
![Eigenvalues](../Images/L3_2_Quiz_18/plot4_eigenvalues.png)

This visualization compares the eigenvalues of $X^TX$ in three scenarios:
- **Well-conditioned case**: Both eigenvalues are significantly above zero, indicating a stable, invertible matrix
- **Ill-conditioned case**: The smallest eigenvalue approaches zero, resulting in a nearly singular matrix
- **Singular case**: One eigenvalue equals zero, making the matrix non-invertible

The condition number (ratio of largest to smallest eigenvalue) quantifies the numerical stability of solving the normal equations.

### Matrix Representation
![Matrix Representation](../Images/L3_2_Quiz_18/plot5_matrix_representation.png)

This visualization illustrates the actual matrices involved in the normal equations. The left panel shows the design matrix $X$ with a column of ones (intercept) and a column of x-values. The right panel shows the resulting $X^TX$ matrix, which must be invertible for a unique solution to exist. The determinant value in the title quantifies this invertibility.

### Geometric Interpretation
![Geometric Interpretation](../Images/L3_2_Quiz_18/plot6_geometric_interpretation.png)

This 3D visualization demonstrates the geometric interpretation of the normal equations. The blue plane represents the column space of $X$ (all possible linear combinations of the predictor variables). The red points are the original data points, and the green points are their projections onto the column space. The black lines show that these projections are orthogonal to the column space, illustrating that $X^T(y - X\beta) = 0$, which is equivalent to the normal equations.

## Key Insights

### Theoretical Foundations
- The normal equations provide a direct analytical solution to the linear regression problem: $X^TX\beta = X^Ty$
- The solution minimizes the sum of squared errors between observed and predicted values
- The solution is a global minimum when $X^TX$ is invertible
- The geometry of the sum of squared errors is a quadratic (paraboloid) surface with a unique minimum when $X^TX$ is invertible

### Properties and Conditions
- For a unique solution to exist, $X^TX$ must be invertible (non-singular)
- This requires that the columns of $X$ are linearly independent
- Perfect collinearity among predictors makes $X^TX$ singular, resulting in no unique solution
- Near collinearity (multicollinearity) makes $X^TX$ poorly conditioned, leading to unstable solutions
- The eigenvalues of $X^TX$ provide insight into its conditioning and invertibility

### Practical Implications
- The normal equations provide a computationally efficient solution for small to moderate-sized problems
- For numerical stability, it's better to use QR decomposition or Singular Value Decomposition rather than directly inverting $X^TX$
- Regularization techniques (like ridge regression) can help address multicollinearity by making $X^TX$ more stable
- Understanding the normal equations is fundamental to extensions like weighted least squares, generalized least squares, and ridge regression

## Conclusion
- The normal equations in matrix form are $X^TX\beta = X^Ty$, providing an analytical solution to linear regression.
- The key matrix property that ensures a unique solution exists is that $X^TX$ must be invertible (non-singular).
- This invertibility depends on the linear independence of the columns of $X$, which requires sufficient data points and no perfect collinearity among predictors.

Understanding the normal equations and the conditions for their solution is essential for theoretical analysis of linear regression and provides the foundation for more advanced regression techniques. 