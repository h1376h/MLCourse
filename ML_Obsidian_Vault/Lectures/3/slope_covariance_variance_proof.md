# Proof: Why Slope Equals Covariance Divided by Variance

In simple linear regression, we're fitting a model of the form:

$$\hat{y} = \beta_0 + \beta_1 x$$

Where:
- $\hat{y}$ is the predicted value
- $\beta_0$ is the intercept
- $\beta_1$ is the slope
- $x$ is the independent variable

## The Least Squares Approach

We want to find values of $\beta_0$ and $\beta_1$ that minimize the sum of squared errors (SSE):

$$SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2$$

To minimize this function, we take partial derivatives with respect to $\beta_0$ and $\beta_1$ and set them equal to zero:

### For $\beta_0$:

$$\frac{\partial SSE}{\partial \beta_0} = -2\sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i) = 0$$

Simplifying:
$$\sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i) = 0$$
$$\sum_{i=1}^{n} y_i - n\beta_0 - \beta_1 \sum_{i=1}^{n} x_i = 0$$
$$n\beta_0 = \sum_{i=1}^{n} y_i - \beta_1 \sum_{i=1}^{n} x_i$$
$$\beta_0 = \frac{\sum_{i=1}^{n} y_i}{n} - \beta_1 \frac{\sum_{i=1}^{n} x_i}{n} = \bar{y} - \beta_1 \bar{x}$$

### For $\beta_1$:

$$\frac{\partial SSE}{\partial \beta_1} = -2\sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)x_i = 0$$

Simplifying:
$$\sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)x_i = 0$$
$$\sum_{i=1}^{n} y_i x_i - \beta_0 \sum_{i=1}^{n} x_i - \beta_1 \sum_{i=1}^{n} x_i^2 = 0$$

Substituting $\beta_0 = \bar{y} - \beta_1 \bar{x}$ from above:

$$\sum_{i=1}^{n} y_i x_i - (\bar{y} - \beta_1 \bar{x}) \sum_{i=1}^{n} x_i - \beta_1 \sum_{i=1}^{n} x_i^2 = 0$$
$$\sum_{i=1}^{n} y_i x_i - \bar{y} \sum_{i=1}^{n} x_i + \beta_1 \bar{x} \sum_{i=1}^{n} x_i - \beta_1 \sum_{i=1}^{n} x_i^2 = 0$$

Note that $\sum_{i=1}^{n} x_i = n\bar{x}$, so:

$$\sum_{i=1}^{n} y_i x_i - \bar{y} \cdot n\bar{x} + \beta_1 \bar{x} \cdot n\bar{x} - \beta_1 \sum_{i=1}^{n} x_i^2 = 0$$
$$\sum_{i=1}^{n} y_i x_i - n\bar{y}\bar{x} + \beta_1 n\bar{x}^2 - \beta_1 \sum_{i=1}^{n} x_i^2 = 0$$

Solving for $\beta_1$:

$$\beta_1 \left(\sum_{i=1}^{n} x_i^2 - n\bar{x}^2\right) = \sum_{i=1}^{n} y_i x_i - n\bar{y}\bar{x}$$

## Connecting to Covariance and Variance

The sample covariance between $x$ and $y$ is defined as:

$$Cov(x,y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$

Expanding:
$$Cov(x,y) = \frac{1}{n}\sum_{i=1}^{n}(x_i y_i - x_i\bar{y} - \bar{x}y_i + \bar{x}\bar{y})$$
$$Cov(x,y) = \frac{1}{n}\left(\sum_{i=1}^{n}x_i y_i - \bar{y}\sum_{i=1}^{n}x_i - \bar{x}\sum_{i=1}^{n}y_i + n\bar{x}\bar{y}\right)$$

Since $\sum_{i=1}^{n}x_i = n\bar{x}$ and $\sum_{i=1}^{n}y_i = n\bar{y}$:

$$Cov(x,y) = \frac{1}{n}\left(\sum_{i=1}^{n}x_i y_i - n\bar{y}\bar{x} - n\bar{x}\bar{y} + n\bar{x}\bar{y}\right)$$
$$Cov(x,y) = \frac{1}{n}\left(\sum_{i=1}^{n}x_i y_i - n\bar{y}\bar{x}\right)$$
$$n \cdot Cov(x,y) = \sum_{i=1}^{n}x_i y_i - n\bar{y}\bar{x}$$

Similarly, the sample variance of $x$ is:

$$Var(x) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$
$$Var(x) = \frac{1}{n}\sum_{i=1}^{n}(x_i^2 - 2x_i\bar{x} + \bar{x}^2)$$
$$Var(x) = \frac{1}{n}\left(\sum_{i=1}^{n}x_i^2 - 2\bar{x}\sum_{i=1}^{n}x_i + n\bar{x}^2\right)$$
$$Var(x) = \frac{1}{n}\left(\sum_{i=1}^{n}x_i^2 - 2n\bar{x}^2 + n\bar{x}^2\right)$$
$$Var(x) = \frac{1}{n}\left(\sum_{i=1}^{n}x_i^2 - n\bar{x}^2\right)$$
$$n \cdot Var(x) = \sum_{i=1}^{n}x_i^2 - n\bar{x}^2$$

## Final Proof

From our earlier derivation of $\beta_1$:

$$\beta_1 = \frac{\sum_{i=1}^{n} y_i x_i - n\bar{y}\bar{x}}{\sum_{i=1}^{n} x_i^2 - n\bar{x}^2}$$

Substituting our expressions for covariance and variance:

$$\beta_1 = \frac{n \cdot Cov(x,y)}{n \cdot Var(x)} = \frac{Cov(x,y)}{Var(x)}$$

This proves that the slope coefficient in simple linear regression equals the covariance between $x$ and $y$ divided by the variance of $x$.

## Intuitive Interpretation

This relationship makes intuitive sense:
- The covariance measures how $x$ and $y$ vary together
- The variance of $x$ measures how much $x$ varies
- Their ratio tells us how much $y$ changes for a unit change in $x$, which is exactly what the slope represents

Additionally, this explains why the units of the slope are units of $y$ per unit of $x$ - the covariance has units of $x \times y$, while the variance has units of $x^2$, so their ratio has units of $y/x$. 