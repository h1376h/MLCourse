# Question 17: Geometric Interpretation of Linear Regression

## Problem Statement
The geometric interpretation of linear regression provides insights into how the least squares method minimizes prediction errors and relates to vector projections in $n$-dimensional space.

### Task
Which of the following statements correctly describes the geometric interpretation of the least squares method?

A) It minimizes the sum of vertical distances between points and the regression line
B) It minimizes the sum of horizontal distances between points and the regression line
C) It minimizes the sum of perpendicular distances between points and the regression line
D) It maximizes the sum of squared distances between points and the regression line

## Understanding the Problem
This problem asks us to identify the correct geometric interpretation of the least squares method in linear regression. Linear regression aims to find a linear relationship between predictor variables $X$ and a response variable $Y$ that best fits the observed data. The least squares method is a specific approach to finding this "best fit" line by minimizing some measure of error between the observed data points $(x_i, y_i)$ and the predictions from the model $\hat{y}_i$.

The geometric interpretation helps us visualize what the least squares method actually does in terms of distances between data points and the fitted line in a geometric space. Understanding this geometric perspective is crucial for comprehending the fundamental principles of regression analysis and the properties of the resulting estimators.

## Solution

### Step 1: Understanding the Least Squares Method Geometrically
In linear regression, we aim to find the line that best fits our data points. For a simple linear regression with one predictor variable, this line has the form:

$$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$$

Where $\hat{\beta}_0$ is the intercept and $\hat{\beta}_1$ is the slope of the line.

The least squares method finds the values of $\hat{\beta}_0$ and $\hat{\beta}_1$ that minimize the sum of squared residuals (SSR):

$$\text{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2$$

The term $(y_i - \hat{y}_i)$ represents the vertical distance between each observed data point and the predicted value on the regression line. These vertical distances are also called residuals $\varepsilon_i$.

For multiple regression with $p$ predictors, we can express this in matrix form:

$$\text{SSR} = (Y - X\beta)^T(Y - X\beta)$$

Where $Y$ is the $n \times 1$ vector of response values, $X$ is the $n \times (p+1)$ design matrix, and $\beta$ is the $(p+1) \times 1$ vector of coefficients.

### Step 2: Comparing Different Distance Metrics
To determine which interpretation is correct, we need to consider the different ways we could measure distances between points and the regression line:

1. **Vertical distances**: The distance measured parallel to the $y$-axis, computed as $|y_i - \hat{y}_i|$
2. **Horizontal distances**: The distance measured parallel to the $x$-axis, computed as $|x_i - \frac{y_i - \hat{\beta}_0}{\hat{\beta}_1}|$ 
3. **Perpendicular distances**: The shortest distance from the point to the line, computed as $\frac{|y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i|}{\sqrt{1 + \hat{\beta}_1^2}}$

Using our example data, we calculated the sum of squared distances for each of these metrics:

- Sum of squared vertical distances: $\sum (y_i - \hat{y}_i)^2 = 43.68$
- Sum of squared horizontal distances: $\sum (x_i - \frac{y_i - \hat{\beta}_0}{\hat{\beta}_1})^2 = 34.72$
- Sum of squared perpendicular distances: $\sum \frac{(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2}{1 + \hat{\beta}_1^2} = 19.34$

While the perpendicular distance (option C) results in the smallest overall sum of squared distances, this is not what standard least squares regression minimizes. The least squares method specifically minimizes the sum of squared vertical distances (option A).

### Step 3: Vector Space Interpretation
From a linear algebra perspective, we can view the least squares solution as the projection of the response vector $Y$ onto the column space of the design matrix $X$. This projection creates a fitted value vector $\hat{Y}$ such that the residual vector $(Y - \hat{Y})$ is orthogonal to the column space of $X$.

This orthogonality condition leads to the normal equations:

$$X^T(Y - X\beta) = 0$$

Which can be solved to find the least squares estimates:

$$\hat{\beta} = (X^TX)^{-1}X^TY$$

This orthogonality property is equivalent to minimizing the sum of squared vertical distances between the observed data points and the predicted values on the regression line. Mathematically, this means:

$$X^T\varepsilon = 0$$

Where $\varepsilon = Y - X\hat{\beta}$ is the vector of residuals.

### Step 4: Error Surface Analysis
The least squares method can also be visualized as finding the minimum point on an error surface. This surface represents the sum of squared errors $\text{SSE}(\beta_0, \beta_1)$ as a function of the regression coefficients (intercept and slope). 

For simple linear regression, the SSE function is:

$$\text{SSE}(\beta_0, \beta_1) = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2$$

The partial derivatives with respect to $\beta_0$ and $\beta_1$ are:

$$\frac{\partial \text{SSE}}{\partial \beta_0} = -2\sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)$$

$$\frac{\partial \text{SSE}}{\partial \beta_1} = -2\sum_{i=1}^{n} x_i(y_i - \beta_0 - \beta_1 x_i)$$

Setting these equal to zero and solving yields the least squares estimates.

Our visualization shows that this error surface is a convex function with a single global minimum, which occurs at our calculated values of $\hat{\beta}_0 = 3.55$ and $\hat{\beta}_1 = 1.12$.

## Practical Implementation
Our script implements several visualizations to demonstrate the geometric interpretation of least squares regression:

1. We plotted the data with the regression line and showed different types of distances (vertical, horizontal, and perpendicular).
2. We visualized the error surface as a function of the regression coefficients $\beta_0$ and $\beta_1$.
3. We illustrated the vector projection interpretation of least squares.
4. We demonstrated the orthogonality property of the residuals, showing that $\sum \varepsilon_i = 0$ and $\sum x_i\varepsilon_i = 0$.

By comparing the sum of squared distances for each approach, we confirmed that ordinary least squares regression specifically minimizes vertical distances.

## Visual Explanations

### Different Types of Distances Between Points and the Regression Line
![Different Distance Metrics](../Images/L3_2_Quiz_17/geometric_interpretation.png)

This figure shows three different ways to measure distances between data points $(x_i, y_i)$ and the regression line $\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$:
- Panel A: Vertical distances (in red) - These are the residuals $\varepsilon_i = y_i - \hat{y}_i$ that least squares minimizes
- Panel B: Horizontal distances (in green) - Measured parallel to the $x$-axis
- Panel C: Perpendicular distances (in purple) - The shortest distance from each point to the line

The bottom panels show the vector projection interpretation and the error surface, illustrating how the least squares estimates correspond to the minimum point on this surface.

### Optimization Process
![Optimization Process](../Images/L3_2_Quiz_17/optimization_process.png)

This figure demonstrates how the sum of squared errors $\text{SSE}(\beta_1)$ changes as we vary the slope of the regression line. The top panel shows the error curve with a clear minimum at the least squares estimate $\hat{\beta}_1 = 1.12$. The bottom panel shows the data with the optimal regression line (in red) and several alternative lines, along with their respective sum of squared errors.

### Matrix Algebra Interpretation
![Matrix Interpretation](../Images/L3_2_Quiz_17/matrix_interpretation.png)

This figure illustrates the vector space interpretation of least squares. The top panel shows how the response vector $Y$ is projected onto the column space of the design matrix $X$, with the residual vector $\varepsilon = Y - \hat{Y}$ being orthogonal to this space. The bottom panel demonstrates the orthogonality condition, showing that the residuals are uncorrelated with the predictors.

## Key Insights

### Theoretical Foundations
- The least squares method minimizes the sum of squared vertical distances between data points and the regression line: $\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$.
- This is equivalent to minimizing the sum of squared residuals: $\min_{\beta} \sum_{i=1}^{n} \varepsilon_i^2$.
- From a vector space perspective, the least squares solution projects the response vector $Y$ onto the column space of the design matrix $X$.
- The resulting residual vector $\varepsilon = Y - X\hat{\beta}$ is orthogonal to the column space of $X$, meaning $X^T\varepsilon = 0$.

### Alternative Interpretations
- Total least squares (also called orthogonal regression) minimizes perpendicular distances (option C): $\min_{\beta} \sum_{i=1}^{n} \frac{(y_i - \beta_0 - \beta_1 x_i)^2}{1 + \beta_1^2}$.
- Inverse regression minimizes horizontal distances (option B): $\min_{\beta} \sum_{i=1}^{n} (x_i - \frac{y_i - \beta_0}{\beta_1})^2$.
- These alternative approaches may be preferred in certain applications, but standard OLS specifically minimizes vertical distances.

### Practical Implications
- The choice of which distances to minimize depends on the assumptions about error in the data.
- OLS assumes that error occurs only in the response variable $Y$, not in the predictor $X$.
- When there is measurement error in $X$, other approaches like orthogonal regression might be more appropriate.
- The vertical distance interpretation makes OLS computationally simpler and leads to desirable statistical properties.

## Conclusion
- **The correct answer is A) It minimizes the sum of vertical distances between points and the regression line.**
- This vertical distance represents the residuals $(y_i - \hat{y}_i)$ in the model.
- The sum of squared vertical distances is what is being minimized in the objective function of OLS regression: $\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$.
- This interpretation connects to other important concepts in regression, such as the normal equations and the orthogonality of residuals to the predictors.

Understanding the geometric interpretation of linear regression provides valuable insight into how and why the least squares method works, and helps explain the properties of the resulting estimates. It also clarifies the assumptions inherent in the approach and when alternative methods might be more appropriate.