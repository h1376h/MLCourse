# Proof: The Regression Line Always Passes Through ($\bar{x}$, $\bar{y}$)

## Statement to Prove
In simple linear regression with an intercept term, the regression line always passes through the point ($\bar{x}$, $\bar{y}$), where $\bar{x}$ is the mean of the independent variable and $\bar{y}$ is the mean of the dependent variable.

## Background and Notation
In simple linear regression, we fit a model of the form:

$$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$$

Where:
- $\hat{y}_i$ is the predicted value for the $i$-th observation
- $\hat{\beta}_0$ is the estimated intercept
- $\hat{\beta}_1$ is the estimated slope
- $x_i$ is the value of the independent variable for the $i$-th observation

The least squares method finds the values of $\hat{\beta}_0$ and $\hat{\beta}_1$ that minimize the sum of squared residuals:

$$SSE = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2$$

## Derivation of Normal Equations

To find the values of $\hat{\beta}_0$ and $\hat{\beta}_1$ that minimize $SSE$, we take partial derivatives with respect to each parameter and set them equal to zero.

### For $\hat{\beta}_0$:

$$\frac{\partial SSE}{\partial \hat{\beta}_0} = -2 \sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0$$

Simplifying:
$$\sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0$$
$$\sum_{i=1}^n y_i - n\hat{\beta}_0 - \hat{\beta}_1 \sum_{i=1}^n x_i = 0$$

Solving for $\hat{\beta}_0$:
$$n\hat{\beta}_0 = \sum_{i=1}^n y_i - \hat{\beta}_1 \sum_{i=1}^n x_i$$

Dividing by $n$:
$$\hat{\beta}_0 = \frac{\sum_{i=1}^n y_i}{n} - \hat{\beta}_1 \frac{\sum_{i=1}^n x_i}{n}$$

Since $\bar{y} = \frac{\sum_{i=1}^n y_i}{n}$ and $\bar{x} = \frac{\sum_{i=1}^n x_i}{n}$:

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

### For $\hat{\beta}_1$:

Taking the partial derivative of $SSE$ with respect to $\hat{\beta}_1$:

$$\frac{\partial SSE}{\partial \hat{\beta}_1} = -2 \sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)x_i = 0$$

Simplifying:
$$\sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)x_i = 0$$
$$\sum_{i=1}^n y_i x_i - \hat{\beta}_0 \sum_{i=1}^n x_i - \hat{\beta}_1 \sum_{i=1}^n x_i^2 = 0$$

Substituting $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$:

$$\sum_{i=1}^n y_i x_i - (\bar{y} - \hat{\beta}_1 \bar{x}) \sum_{i=1}^n x_i - \hat{\beta}_1 \sum_{i=1}^n x_i^2 = 0$$
$$\sum_{i=1}^n y_i x_i - \bar{y} \sum_{i=1}^n x_i + \hat{\beta}_1 \bar{x} \sum_{i=1}^n x_i - \hat{\beta}_1 \sum_{i=1}^n x_i^2 = 0$$

Since $\sum_{i=1}^n x_i = n\bar{x}$:

$$\sum_{i=1}^n y_i x_i - n\bar{y}\bar{x} + \hat{\beta}_1 n\bar{x}^2 - \hat{\beta}_1 \sum_{i=1}^n x_i^2 = 0$$

Solving for $\hat{\beta}_1$:

$$\hat{\beta}_1 (\sum_{i=1}^n x_i^2 - n\bar{x}^2) = \sum_{i=1}^n y_i x_i - n\bar{y}\bar{x}$$

$$\hat{\beta}_1 = \frac{\sum_{i=1}^n y_i x_i - n\bar{y}\bar{x}}{\sum_{i=1}^n x_i^2 - n\bar{x}^2}$$

This is the formula for the slope in simple linear regression.

### Formula for Sample Covariance and Variance

The sample covariance between $x$ and $y$ is defined as:

$$Cov(x,y) = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$$

Expanding:

$$Cov(x,y) = \frac{1}{n}\sum_{i=1}^n (x_i y_i - x_i \bar{y} - \bar{x} y_i + \bar{x}\bar{y})$$
$$Cov(x,y) = \frac{1}{n}(\sum_{i=1}^n x_i y_i - \bar{y}\sum_{i=1}^n x_i - \bar{x}\sum_{i=1}^n y_i + n\bar{x}\bar{y})$$

Since $\sum_{i=1}^n x_i = n\bar{x}$ and $\sum_{i=1}^n y_i = n\bar{y}$:

$$Cov(x,y) = \frac{1}{n}(\sum_{i=1}^n x_i y_i - n\bar{y}\bar{x} - n\bar{x}\bar{y} + n\bar{x}\bar{y})$$
$$Cov(x,y) = \frac{1}{n}(\sum_{i=1}^n x_i y_i - n\bar{y}\bar{x})$$

Multiplying by $n$:

$$n \cdot Cov(x,y) = \sum_{i=1}^n x_i y_i - n\bar{y}\bar{x}$$

Similarly, the sample variance of $x$ is:

$$Var(x) = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$
$$Var(x) = \frac{1}{n}(\sum_{i=1}^n x_i^2 - 2\bar{x}\sum_{i=1}^n x_i + n\bar{x}^2)$$
$$Var(x) = \frac{1}{n}(\sum_{i=1}^n x_i^2 - 2n\bar{x}^2 + n\bar{x}^2)$$
$$Var(x) = \frac{1}{n}(\sum_{i=1}^n x_i^2 - n\bar{x}^2)$$

Multiplying by $n$:

$$n \cdot Var(x) = \sum_{i=1}^n x_i^2 - n\bar{x}^2$$

### The Slope in Terms of Covariance and Variance

We can rewrite the formula for $\hat{\beta}_1$ using our expressions for covariance and variance:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^n y_i x_i - n\bar{y}\bar{x}}{\sum_{i=1}^n x_i^2 - n\bar{x}^2} = \frac{n \cdot Cov(x,y)}{n \cdot Var(x)} = \frac{Cov(x,y)}{Var(x)}$$

## Proof that the Regression Line Passes Through $(\bar{x}, \bar{y})$

Now we can prove that the regression line passes through the point $(\bar{x}, \bar{y})$.

From our derivation of the intercept:

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

To show that the regression line passes through $(\bar{x}, \bar{y})$, we need to demonstrate that:

$$\hat{y}(\bar{x}) = \hat{\beta}_0 + \hat{\beta}_1 \bar{x} = \bar{y}$$

Substituting the expression for $\hat{\beta}_0$:

$$\hat{y}(\bar{x}) = (\bar{y} - \hat{\beta}_1 \bar{x}) + \hat{\beta}_1 \bar{x} = \bar{y} - \hat{\beta}_1 \bar{x} + \hat{\beta}_1 \bar{x} = \bar{y}$$

This confirms that the predicted value at $\bar{x}$ is indeed $\bar{y}$, which means the regression line passes through the point $(\bar{x}, \bar{y})$.

## Conclusion

We have proven that in simple linear regression with an intercept term, the regression line always passes through the point $(\bar{x}, \bar{y})$. This is a direct consequence of the normal equations derived from the least squares method, specifically from the equation for the intercept: 

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

This property is not a coincidence but rather a mathematical necessity arising from the optimization criteria of the least squares method.

## Visual Interpretation

Geometrically, this property means that the regression line acts as a pivot around the "center of mass" of the data. The slope determines how the line rotates around this central point to best fit the overall pattern of the data.

This property is also helpful for quick calculations and interpretations. For example, if you know the mean of both variables and the slope, you can immediately determine the intercept without additional calculations.

It also explains why the residuals sum to zero when an intercept is included in the model. The line passes through the mean point, ensuring that positive and negative residuals balance out exactly.

## Practical Implications

This property has several important practical implications:

1. **Simplification of calculations**: If you know $\bar{x}$, $\bar{y}$, and $\hat{\beta}_1$, you can directly calculate $\hat{\beta}_0$ without performing a separate regression.

2. **Interpretation of coefficients**: The intercept $\hat{\beta}_0$ represents the predicted value of $y$ when $x = 0$, but only if $x = 0$ is within the range of the data. Otherwise, it's purely a mathematical construct needed to ensure the line passes through $(\bar{x}, \bar{y})$.

3. **Centering predictors**: When predictors are centered (by subtracting their means), the intercept becomes equal to the mean of the response variable, which simplifies interpretation.

4. **Connection to ANOVA**: This property creates a direct link between regression and ANOVA, as it ensures that the sum of squared residuals is minimized around the mean of the response. 