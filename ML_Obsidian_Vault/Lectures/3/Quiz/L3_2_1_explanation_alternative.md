# Question 1: Simple Linear Regression for House Prices

## Problem Statement
Consider a simple linear regression model for predicting house prices based on house size (in square feet). The following data points are observed:

| House Size (x) | Price (y) in $1000s |
|----------------|---------------------|
| 1000           | 150                 |
| 1500           | 200                 |
| 2000           | 250                 |
| 2500           | 300                 |
| 3000           | 350                 |

### Task
1. Find the least squares estimates for the slope ($\beta_1$) and intercept ($\beta_0$) of the linear regression model
2. Interpret the meaning of the slope coefficient in the context of this problem
3. Calculate the prediction for a house with 1800 square feet
4. Calculate the residuals for each data point and the residual sum of squares (RSS)

## Understanding the Problem
This problem asks us to build a simple linear regression model that relates house prices to their sizes. The goal is to find the line of best fit that minimizes the sum of squared differences between the observed prices and the prices predicted by our model. The linear regression model takes the form:

$$\text{Price} = \beta_0 + \beta_1 \times \text{Size} + \epsilon$$

where $\beta_0$ is the intercept, $\beta_1$ is the slope coefficient, and $\epsilon$ represents the error term. Once we find $\beta_0$ and $\beta_1$, we can use the model to predict house prices for houses of different sizes.

## Solution

### Step 1: Calculate the least squares estimates for slope and intercept using matrix approach

To find the least squares estimates, we can use the matrix formula:

$$\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T y$$

where $X$ is the design matrix with a column of 1s (for the intercept) and a column of house sizes, and $y$ is the vector of house prices.

#### Step 1.1: Calculate means (for reference)
$\bar{x} = \frac{1000 + 1500 + 2000 + 2500 + 3000}{5} = \frac{10000}{5} = 2000$ sq ft

$\bar{y} = \frac{150 + 200 + 250 + 300 + 350}{5} = \frac{1250}{5} = 250$ thousand dollars

#### Step 1.2: Create the design matrix X
We need to add a column of 1s for the intercept term:

$$X = \begin{bmatrix} 
1 & 1000 \\
1 & 1500 \\
1 & 2000 \\
1 & 2500 \\
1 & 3000
\end{bmatrix}$$

This matrix has rows for each data point, where the first column is 1 (for the intercept) and the second column is the house size.

#### Step 1.3: Calculate $X^T$ (transpose of X)

$$X^T = \begin{bmatrix} 
1 & 1 & 1 & 1 & 1 \\
1000 & 1500 & 2000 & 2500 & 3000
\end{bmatrix}$$

The transpose swaps rows and columns, giving us a 2×5 matrix.

#### Step 1.4: Calculate $X^T X$

$$X^T X = \begin{bmatrix} 
1 & 1 & 1 & 1 & 1 \\
1000 & 1500 & 2000 & 2500 & 3000
\end{bmatrix} \begin{bmatrix} 
1 & 1000 \\
1 & 1500 \\
1 & 2000 \\
1 & 2500 \\
1 & 3000
\end{bmatrix} = \begin{bmatrix} 
5 & 10000 \\
10000 & 22500000
\end{bmatrix}$$

Let's calculate each element of this 2×2 matrix:

- Top-left element (sum of 1²):
  $X^T X[0,0] = 1 \times 1 + 1 \times 1 + 1 \times 1 + 1 \times 1 + 1 \times 1 = 5$

- Top-right element (sum of 1 × each x-value):
  $X^T X[0,1] = 1 \times 1000 + 1 \times 1500 + 1 \times 2000 + 1 \times 2500 + 1 \times 3000 = 10000$

- Bottom-left element (same as top-right due to symmetry):
  $X^T X[1,0] = 1000 \times 1 + 1500 \times 1 + 2000 \times 1 + 2500 \times 1 + 3000 \times 1 = 10000$

- Bottom-right element (sum of each x-value squared):
  $X^T X[1,1] = 1000 \times 1000 + 1500 \times 1500 + 2000 \times 2000 + 2500 \times 2500 + 3000 \times 3000 = 22500000$

#### Step 1.5: Calculate $(X^T X)^{-1}$

For a 2×2 matrix $M = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse is:

$M^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$

With our values:
- $a = 5$
- $b = 10000$
- $c = 10000$
- $d = 22500000$

First, calculate the determinant:
$\det = ad - bc = 5 \times 22500000 - 10000 \times 10000 = 112500000 - 100000000 = 12500000$

Then, calculate each element of the inverse:
$X^T X^{-1}[0,0] = d/\det = 22500000/12500000 = 1.8$
$X^T X^{-1}[0,1] = -b/\det = -10000/12500000 = -0.0008$
$X^T X^{-1}[1,0] = -c/\det = -10000/12500000 = -0.0008$
$X^T X^{-1}[1,1] = a/\det = 5/12500000 = 0.0000004$

Thus:

$$\begin{bmatrix} 
5 & 10000 \\
10000 & 22500000
\end{bmatrix}^{-1} = \begin{bmatrix} 
1.8 & -0.0008 \\
-0.0008 & 0.0000004
\end{bmatrix}$$

#### Step 1.6: Calculate $X^T y$

$$X^T y = \begin{bmatrix} 
1 & 1 & 1 & 1 & 1 \\
1000 & 1500 & 2000 & 2500 & 3000
\end{bmatrix} \begin{bmatrix} 
150 \\
200 \\
250 \\
300 \\
350
\end{bmatrix}$$

Let's calculate each element of this 2×1 matrix:

- First element (sum of all y-values):
  $X^T y[0] = 1 \times 150 + 1 \times 200 + 1 \times 250 + 1 \times 300 + 1 \times 350 = 1250$

- Second element (sum of each x-value times its corresponding y-value):
  $X^T y[1] = 1000 \times 150 + 1500 \times 200 + 2000 \times 250 + 2500 \times 300 + 3000 \times 350 = 2750000$

So:

$$X^T y = \begin{bmatrix} 
1250 \\
2750000
\end{bmatrix}$$

#### Step 1.7: Calculate $\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T y$

$$\hat{\boldsymbol{\beta}} = \begin{bmatrix} 
1.8 & -0.0008 \\
-0.0008 & 0.0000004
\end{bmatrix} \begin{bmatrix} 
1250 \\
2750000
\end{bmatrix}$$

Let's calculate each element of the parameter vector:

- Intercept (β₀):
  $\beta_0 = 1.8 \times 1250 + (-0.0008) \times 2750000 = 2250 - 2200 = 50$

- Slope (β₁):
  $\beta_1 = (-0.0008) \times 1250 + 0.0000004 \times 2750000 = -1 + 1.1 = 0.1$

Therefore:
- $\beta_0 = 50$ (intercept)
- $\beta_1 = 0.1$ (slope)

#### Verification using traditional formulas

We can verify our results using the traditional formulas:

$\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$

Let's calculate the numerator:
$\text{Numerator} = (1000 - 2000)(150 - 250) + (1500 - 2000)(200 - 250) + (2000 - 2000)(250 - 250) + (2500 - 2000)(300 - 250) + (3000 - 2000)(350 - 250)$
$\text{Numerator} = (-1000)(-100) + (-500)(-50) + (0)(0) + (500)(50) + (1000)(100)$
$\text{Numerator} = 100000 + 25000 + 0 + 25000 + 100000 = 250000$

And the denominator:
$\text{Denominator} = (1000 - 2000)^2 + (1500 - 2000)^2 + (2000 - 2000)^2 + (2500 - 2000)^2 + (3000 - 2000)^2$
$\text{Denominator} = (-1000)^2 + (-500)^2 + (0)^2 + (500)^2 + (1000)^2$
$\text{Denominator} = 1000000 + 250000 + 0 + 250000 + 1000000 = 2500000$

Therefore:
$\beta_1 = \frac{250000}{2500000} = 0.1$

And for the intercept:
$\beta_0 = \bar{y} - \beta_1 \bar{x} = 250 - 0.1 \times 2000 = 250 - 200 = 50$

This confirms our matrix-based calculations: $\beta_0 = 50$ and $\beta_1 = 0.1$.

Our linear regression model is:
$$\text{Price} = 50 + 0.1 \times \text{Size}$$

### Step 2: Interpret the meaning of the slope coefficient

The slope coefficient $\beta_1 = 0.1$ represents the change in house price (in $1000s) for each additional square foot of house size. In other words, for every additional square foot, the house price increases by $0.1 \times 1000 = $100. This makes intuitive sense - larger houses generally cost more, and in this dataset, each additional square foot adds $100 to the price on average.

### Step 3: Calculate the prediction for a house with 1800 square feet

Using our regression model, we can predict the price of a house with 1800 square feet:

$$\text{Predicted Price} = 50 + 0.1 \times 1800 = 50 + 180 = 230$$

So, our model predicts that a house with 1800 square feet would cost $230,000.

### Step 4: Calculate residuals and RSS

#### Step 4.1: Calculate predicted values (ŷ = Xβ)

For each data point, the predicted price is:
- For house 1 (1000 sq ft): $ŷ_1 = 1 \times 50 + 1000 \times 0.1 = 50 + 100 = 150$
- For house 2 (1500 sq ft): $ŷ_2 = 1 \times 50 + 1500 \times 0.1 = 50 + 150 = 200$
- For house 3 (2000 sq ft): $ŷ_3 = 1 \times 50 + 2000 \times 0.1 = 50 + 200 = 250$
- For house 4 (2500 sq ft): $ŷ_4 = 1 \times 50 + 2500 \times 0.1 = 50 + 250 = 300$
- For house 5 (3000 sq ft): $ŷ_5 = 1 \times 50 + 3000 \times 0.1 = 50 + 300 = 350$

#### Step 4.2: Calculate residuals (e = y - ŷ)

The residual for each observation is the difference between the actual value and the predicted value:
- For house 1: $e_1 = 150 - 150 = 0$
- For house 2: $e_2 = 200 - 200 = 0$
- For house 3: $e_3 = 250 - 250 = 0$
- For house 4: $e_4 = 300 - 300 = 0$
- For house 5: $e_5 = 350 - 350 = 0$

#### Step 4.3: Calculate squared residuals (e²)

The squared residuals are:
- For house 1: $e_1^2 = 0^2 = 0$
- For house 2: $e_2^2 = 0^2 = 0$
- For house 3: $e_3^2 = 0^2 = 0$
- For house 4: $e_4^2 = 0^2 = 0$
- For house 5: $e_5^2 = 0^2 = 0$

#### Step 4.4: Calculate Residual Sum of Squares (RSS)

The Residual Sum of Squares (RSS) is the sum of the squared residuals:
$\text{RSS} = e_1^2 + e_2^2 + e_3^2 + e_4^2 + e_5^2 = 0 + 0 + 0 + 0 + 0 = 0$

We can also calculate the RSS using matrix notation:
$\text{RSS} = \boldsymbol{e}^T \boldsymbol{e}$, where $\boldsymbol{e}$ is the vector of residuals.

In matrix form, this gives us the same result: $\text{RSS} = 0$.

In this particular example, the RSS is 0, which is a special case indicating that our linear regression model perfectly fits the data. All the data points lie exactly on the regression line. This is not typical in real-world data, where there's usually some noise and variation that can't be explained by the model.

#### Summary table

| House Size (x) | Price (y) | Predicted Price (ŷ) | Residual (y - ŷ) | Squared Residual (y - ŷ)² |
|----------------|-----------|---------------------|------------------|----------------------------|
| 1000           | 150       | 150                 | 0                | 0                          |
| 1500           | 200       | 200                 | 0                | 0                          |
| 2000           | 250       | 250                 | 0                | 0                          |
| 2500           | 300       | 300                 | 0                | 0                          |
| 3000           | 350       | 350                 | 0                | 0                          |

## Visual Explanations

### Regression Line and Data Points
![Linear Regression: House Price vs Size](../Images/L3_2_Quiz_1_Matrix/matrix_regression_line.png)

This visualization shows how house prices relate to house sizes, with the regression line. The green point shows our prediction for a house of 1800 sq ft.

### Residuals Plot
![Residuals Plot](../Images/L3_2_Quiz_1_Matrix/matrix_residuals.png)

The residuals plot shows the difference between actual and predicted values. In this case, all residuals are zero, so they all lie on the horizontal line.

### Squared Residuals Visualization
![Visualization of Squared Residuals](../Images/L3_2_Quiz_1_Matrix/matrix_squared_residuals.png)

A visualization of the squared residuals (though in this perfect-fit case, they're all zero).

### Actual vs. Predicted Values
![Actual vs. Predicted Prices](../Images/L3_2_Quiz_1_Matrix/matrix_actual_vs_predicted.png)

Actual vs. predicted prices, with all points falling exactly on the 45-degree line, indicating perfect prediction.

### Residuals Distribution
![Histogram of Residuals](../Images/L3_2_Quiz_1_Matrix/matrix_residuals_histogram.png)

A histogram of residuals, which in this case is just a single bar at zero.

### Matrix Operations Visualization
![Matrix Operations](../Images/L3_2_Quiz_1_Matrix/matrix_operations.png)

A visualization of the matrix operations involved in the linear regression calculation.

## Key Insights

### Mathematical Foundations
- Linear regression finds the line that minimizes the sum of squared residuals (RSS).
- The matrix formula $\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T y$ gives us the least squares estimates directly.
- This matrix approach is more general and can be extended to multiple regression with more than one predictor.
- In this special case, we have a perfect fit with RSS = 0, but in real data, this is extremely rare.

### Practical Applications
- The slope coefficient has a practical interpretation: each additional square foot adds $100 to the house price.
- With the linear model, we can make predictions for houses of any size within a reasonable range.
- The linear relationship between house size and price is perfect in this dataset, but in real estate, other factors often introduce non-linearities.

### Limitations and Extensions
- This model assumes a strictly linear relationship between house size and price.
- In real estate, many other factors affect house prices beyond just size (location, age, features, etc.).
- The perfect fit in this example is unusual and suggests this might be synthetic data or that the relationship was constructed to be exactly linear.

## Conclusion
- We successfully built a linear regression model: Price = 50 + 0.1 × Size, where price is in thousands of dollars and size is in square feet.
- We used the matrix formula $\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T y$ to directly calculate the regression coefficients, with detailed step-by-step calculations.
- The slope coefficient (0.1) indicates that each additional square foot adds $100 to the house price.
- The model predicts that a house with 1800 square feet would cost $230,000.
- The model fits the data perfectly, with zero residuals and zero RSS, which is uncommon in real-world data.

This problem demonstrates both the computational approach to simple linear regression using matrix algebra and the fundamental concepts of parameter estimation, interpretation, prediction, and evaluation of model fit. 