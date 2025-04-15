# Least Squares Examples

This document provides practical examples of the least squares method for linear regression, illustrating how to derive optimal model parameters by minimizing the sum of squared errors.

## Key Concepts and Formulas

The least squares method finds the parameters of a linear model that minimize the sum of squared differences between observed and predicted values. It is the most common approach for estimating parameters in linear regression.

### The Least Squares Objective Function

$$S(\beta_0, \beta_1) = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2$$

### Normal Equations

$$\frac{\partial S}{\partial \beta_0} = -2\sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i) = 0$$

$$\frac{\partial S}{\partial \beta_1} = -2\sum_{i=1}^{n} x_i(y_i - \beta_0 - \beta_1 x_i) = 0$$

### Closed-form Solutions

$$\hat{\beta_1} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{Cov(X,Y)}{Var(X)}$$

$$\hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x}$$

Where:
- $y_i$ = Observed value of the dependent variable
- $x_i$ = Value of the independent variable
- $\bar{x}$ = Mean of the independent variable
- $\bar{y}$ = Mean of the dependent variable
- $\hat{\beta_0}$ = Estimated intercept
- $\hat{\beta_1}$ = Estimated slope

## Examples

The following examples demonstrate the least squares method:

- **Deriving OLS Estimators**: Step-by-step derivation of the optimal parameters
- **Geometric Interpretation**: Understanding least squares from a geometric perspective
- **Weighted Least Squares**: Accounting for varying precision in observations

### Example 1: Deriving Least Squares Estimators for a Simple Dataset

#### Problem Statement
Consider the following dataset relating advertising spend (in $1000s) to sales (in units):

| Advertising Spend ($x_i$) | Sales ($y_i$) |
|---------------------------|---------------|
| 1                         | 4             |
| 2                         | 7             |
| 3                         | 7             |
| 4                         | 9             |
| 5                         | 12            |

Use the least squares method to find the best-fitting linear model $\hat{y} = \beta_0 + \beta_1 x$.

#### Solution

We'll apply the least squares formulas to find the optimal parameters.

##### Step 1: Calculate the means
$\bar{x} = \frac{1+2+3+4+5}{5} = \frac{15}{5} = 3$

$\bar{y} = \frac{4+7+7+9+12}{5} = \frac{39}{5} = 7.8$

##### Step 2: Calculate components for the slope formula
We need to compute $\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$ and $\sum_{i=1}^{n} (x_i - \bar{x})^2$:

| $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i - \bar{x})(y_i - \bar{y})$ | $(x_i - \bar{x})^2$ |
|-------|-------|-----------------|-----------------|----------------------------------|----------------------|
| 1     | 4     | -2              | -3.8            | 7.6                              | 4                    |
| 2     | 7     | -1              | -0.8            | 0.8                              | 1                    |
| 3     | 7     | 0               | -0.8            | 0                                | 0                    |
| 4     | 9     | 1               | 1.2             | 1.2                              | 1                    |
| 5     | 12    | 2               | 4.2             | 8.4                              | 4                    |
| Sum   |       |                 |                 | 18                               | 10                   |

##### Step 3: Calculate the slope ($\hat{\beta_1}$)
$$\hat{\beta_1} = \frac{18}{10} = 1.8$$

##### Step 4: Calculate the intercept ($\hat{\beta_0}$)
$$\hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x} = 7.8 - 1.8 \cdot 3 = 7.8 - 5.4 = 2.4$$

##### Step 5: Write the regression equation
$$\hat{y} = 2.4 + 1.8x$$

This equation tells us that for each additional $1000 spent on advertising, sales increase by approximately 1.8 units, with a baseline of 2.4 units when there is no advertising.

##### Step 6: Verify that this minimizes the sum of squared errors
We can compute the fitted values and residuals to verify:

| $x_i$ | $y_i$ | $\hat{y}_i = 2.4 + 1.8x_i$ | $e_i = y_i - \hat{y}_i$ | $e_i^2$ |
|-------|-------|----------------------------|--------------------------|---------|
| 1     | 4     | 2.4 + 1.8(1) = 4.2         | 4 - 4.2 = -0.2          | 0.04    |
| 2     | 7     | 2.4 + 1.8(2) = 6.0         | 7 - 6.0 = 1             | 1       |
| 3     | 7     | 2.4 + 1.8(3) = 7.8         | 7 - 7.8 = -0.8          | 0.64    |
| 4     | 9     | 2.4 + 1.8(4) = 9.6         | 9 - 9.6 = -0.6          | 0.36    |
| 5     | 12    | 2.4 + 1.8(5) = 11.4        | 12 - 11.4 = 0.6         | 0.36    |
| Sum   |       |                            |                          | 2.4     |

The sum of squared errors is 2.4, which is the minimum possible value for any linear model on this dataset.

### Example 2: Geometric Interpretation of Least Squares

#### Problem Statement
To understand the geometric interpretation of least squares, consider a simple dataset with 3 points: (1, 2), (2, 4), and (3, 4). 

Visualize how the least squares solution represents the orthogonal projection of the response vector onto the column space of the design matrix.

#### Solution

##### Step 1: Set up the problem in vector-matrix form
For our dataset, we have:

$$y = \begin{bmatrix} 2 \\ 4 \\ 4 \end{bmatrix}$$

and our design matrix is:

$$X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}$$

where the first column represents the intercept (all 1s) and the second column represents our x values.

##### Step 2: Calculate the least squares solution
The least squares solution is given by:

$$\beta = (X^TX)^{-1}X^Ty$$

Calculating $X^TX$:

$$X^TX = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}$$

Calculating $(X^TX)^{-1}$:

$$det(X^TX) = 3 \cdot 14 - 6 \cdot 6 = 42 - 36 = 6$$

$$(X^TX)^{-1} = \frac{1}{6} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix} = \begin{bmatrix} \frac{14}{6} & -1 \\ -1 & \frac{1}{2} \end{bmatrix}$$

Calculating $X^Ty$:

$$X^Ty = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 2 \\ 4 \\ 4 \end{bmatrix} = \begin{bmatrix} 10 \\ 22 \end{bmatrix}$$

Now we can calculate $\beta$:

$$\beta = \begin{bmatrix} \frac{14}{6} & -1 \\ -1 & \frac{1}{2} \end{bmatrix} \begin{bmatrix} 10 \\ 22 \end{bmatrix} = \begin{bmatrix} \frac{14 \cdot 10 - 22}{6} \\ -10 + \frac{22}{2} \end{bmatrix} = \begin{bmatrix} \frac{118}{6} \approx 1.97 \\ 1 \end{bmatrix}$$

So our least squares solution is approximately $\beta_0 \approx 1.97$ and $\beta_1 = 1$, giving us the line $\hat{y} = 1.97 + 1x$.

##### Step 3: Geometric interpretation
The predicted values are:

$$\hat{y} = X\beta = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 1.97 \\ 1 \end{bmatrix} = \begin{bmatrix} 2.97 \\ 3.97 \\ 4.97 \end{bmatrix}$$

The residuals are:

$$e = y - \hat{y} = \begin{bmatrix} 2 \\ 4 \\ 4 \end{bmatrix} - \begin{bmatrix} 2.97 \\ 3.97 \\ 4.97 \end{bmatrix} = \begin{bmatrix} -0.97 \\ 0.03 \\ -0.97 \end{bmatrix}$$

Geometrically, $\hat{y}$ is the projection of $y$ onto the column space of $X$. The residual vector $e$ is orthogonal to this column space, meaning:

$$X^Te = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} -0.97 \\ 0.03 \\ -0.97 \end{bmatrix} = \begin{bmatrix} -0.97 + 0.03 - 0.97 \\ -0.97 + 0.06 - 2.91 \end{bmatrix} \approx \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

This confirms that the residual vector is orthogonal to the column space of $X$, which is a fundamental property of the least squares solution.

### Example 3: Weighted Least Squares

#### Problem Statement
Consider a dataset where observations have different levels of reliability. For example, in a study measuring reaction time, some measurements were based on more trials than others:

| Stimulus intensity ($x_i$) | Reaction time ($y_i$) | Number of trials (weight) |
|----------------------------|------------------------|---------------------------|
| 1                          | 1.5                    | 10                        |
| 2                          | 1.2                    | 5                         |
| 3                          | 0.9                    | 8                         |
| 4                          | 0.7                    | 15                        |
| 5                          | 0.6                    | 7                         |

Use weighted least squares to find a linear relationship, giving more importance to more reliable measurements.

#### Solution

In weighted least squares, we minimize:

$$S_w(\beta_0, \beta_1) = \sum_{i=1}^{n} w_i (y_i - \beta_0 - \beta_1 x_i)^2$$

where $w_i$ are the weights.

##### Step 1: Set up the weighted least squares formulas
The weighted least squares estimators are:

$$\hat{\beta_1} = \frac{\sum_{i=1}^{n} w_i(x_i - \bar{x}_w)(y_i - \bar{y}_w)}{\sum_{i=1}^{n} w_i(x_i - \bar{x}_w)^2}$$

$$\hat{\beta_0} = \bar{y}_w - \hat{\beta_1}\bar{x}_w$$

where $\bar{x}_w$ and $\bar{y}_w$ are the weighted means:

$$\bar{x}_w = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}$$

$$\bar{y}_w = \frac{\sum_{i=1}^{n} w_i y_i}{\sum_{i=1}^{n} w_i}$$

##### Step 2: Calculate the weighted means
First, we calculate the sum of weights:
$$\sum_{i=1}^{n} w_i = 10 + 5 + 8 + 15 + 7 = 45$$

For the weighted mean of $x$:
$$\bar{x}_w = \frac{10 \cdot 1 + 5 \cdot 2 + 8 \cdot 3 + 15 \cdot 4 + 7 \cdot 5}{45} = \frac{10 + 10 + 24 + 60 + 35}{45} = \frac{139}{45} \approx 3.09$$

For the weighted mean of $y$:
$$\bar{y}_w = \frac{10 \cdot 1.5 + 5 \cdot 1.2 + 8 \cdot 0.9 + 15 \cdot 0.7 + 7 \cdot 0.6}{45} = \frac{15 + 6 + 7.2 + 10.5 + 4.2}{45} = \frac{42.9}{45} \approx 0.95$$

##### Step 3: Calculate the components for the weighted slope formula

| $x_i$ | $y_i$ | $w_i$ | $x_i - \bar{x}_w$ | $y_i - \bar{y}_w$ | $w_i(x_i - \bar{x}_w)(y_i - \bar{y}_w)$ | $w_i(x_i - \bar{x}_w)^2$ |
|-------|-------|-------|-------------------|-------------------|------------------------------------------|---------------------------|
| 1     | 1.5   | 10    | -2.09             | 0.55              | 10 × (-2.09) × 0.55 = -11.495           | 10 × (-2.09)² = 43.681    |
| 2     | 1.2   | 5     | -1.09             | 0.25              | 5 × (-1.09) × 0.25 = -1.363             | 5 × (-1.09)² = 5.941      |
| 3     | 0.9   | 8     | -0.09             | -0.05             | 8 × (-0.09) × (-0.05) = 0.036           | 8 × (-0.09)² = 0.065      |
| 4     | 0.7   | 15    | 0.91              | -0.25             | 15 × 0.91 × (-0.25) = -3.413            | 15 × 0.91² = 12.437       |
| 5     | 0.6   | 7     | 1.91              | -0.35             | 7 × 1.91 × (-0.35) = -4.681             | 7 × 1.91² = 25.557        |
| Sum   |       | 45    |                   |                   | -20.916                                  | 87.681                    |

##### Step 4: Calculate the weighted slope
$$\hat{\beta_1} = \frac{-20.916}{87.681} \approx -0.238$$

##### Step 5: Calculate the weighted intercept
$$\hat{\beta_0} = \bar{y}_w - \hat{\beta_1}\bar{x}_w = 0.95 - (-0.238) \cdot 3.09 = 0.95 + 0.735 = 1.685$$

##### Step 6: Write the weighted regression equation
$$\hat{y} = 1.685 - 0.238x$$

This equation tells us that for each one-unit increase in stimulus intensity, the reaction time decreases by approximately 0.238 units, with an initial reaction time of about 1.685 units when the stimulus intensity is zero.

The weighted approach gives more influence to observations with higher weights (more trials), making our parameter estimates more reliable than if we had treated all observations equally.

## Key Insights

### Theoretical Insights
- The least squares solution is the orthogonal projection of the response vector onto the column space of the design matrix
- The residuals are orthogonal to the column space of the design matrix
- The least squares solution minimizes the Euclidean distance between the observed and predicted values
- Weighted least squares allows incorporating known reliability differences among observations

### Practical Applications
- Ordinary least squares provides the Best Linear Unbiased Estimator (BLUE) when errors are independent and have constant variance
- Weighted least squares is useful when observations have different levels of precision or reliability
- The least squares method forms the foundation for many advanced regression techniques

### Common Pitfalls
- Assuming homoscedasticity (constant error variance) when it's not present
- Not considering weights when some observations are known to be more reliable than others
- Applying least squares when the relationship is fundamentally non-linear
- Not checking for influential outliers that can drastically affect the least squares solution

## Related Topics

- [[L3_2_Simple_Linear_Regression|Simple Linear Regression]]: The model whose parameters are estimated using least squares
- [[L3_2_Cost_Function|Cost Function]]: The objective function that least squares minimizes
- [[L3_2_Analytical_Solution|Analytical Solution]]: The closed-form solution derived from the least squares method 