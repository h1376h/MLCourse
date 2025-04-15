# Simple Linear Regression

## Introduction
Simple Linear Regression is the most basic form of linear regression, involving only one independent variable (feature) to predict a dependent variable (target). It establishes a linear relationship between two variables and serves as a foundation for understanding more complex regression models.

## Mathematical Formulation
The simple linear regression model is expressed as:

$$y = \beta_0 + \beta_1 x + \epsilon$$

Where:
- $y$ is the dependent variable (target)
- $x$ is the independent variable (feature)
- $\beta_0$ is the y-intercept (bias term)
- $\beta_1$ is the slope coefficient
- $\epsilon$ is the error term (residual)

## Geometric Interpretation
- $\beta_0$ represents the expected value of $y$ when $x = 0$
- $\beta_1$ represents the change in $y$ for a one-unit change in $x$
- The model attempts to find the best-fit line through the data points

## Parameter Estimation

### Method of Least Squares
The parameters $\beta_0$ and $\beta_1$ are typically estimated by minimizing the sum of squared residuals (SSR):

$$\text{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2$$

### Analytical Solution
The closed-form solutions for the parameters are:

$$\beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}$$

$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$

Where:
- $\bar{x}$ is the mean of the independent variable
- $\bar{y}$ is the mean of the dependent variable
- $\text{Cov}(x, y)$ is the covariance between $x$ and $y$
- $\text{Var}(x)$ is the variance of $x$

## Assumptions
Simple linear regression relies on several key assumptions:
1. **Linearity**: The relationship between $x$ and $y$ is linear
2. **Independence**: The observations are independent of each other
3. **Homoscedasticity**: The error terms have constant variance
4. **Normality**: The error terms are normally distributed
5. **Fixed X**: The independent variable is measured without error

## Statistical Inference

### Standard Errors of Coefficients
Under the standard assumptions, the estimated coefficients have the following standard errors:

$$\text{SE}(\beta_1) = \sqrt{\frac{\sigma^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}}$$

$$\text{SE}(\beta_0) = \sigma \sqrt{\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}}$$

Where $\sigma^2$ is the error variance, typically estimated as:

$$\hat{\sigma}^2 = \frac{\text{SSR}}{n-2}$$

### Confidence Intervals
95% confidence intervals for the coefficients are:

$$\beta_j \pm t_{n-2, 0.975} \cdot \text{SE}(\beta_j)$$

Where $t_{n-2, 0.975}$ is the 97.5th percentile of the t-distribution with $n-2$ degrees of freedom.

### Hypothesis Testing
To test the significance of the slope coefficient:
- Null hypothesis: $H_0: \beta_1 = 0$ (no linear relationship)
- Alternative hypothesis: $H_1: \beta_1 \neq 0$ (linear relationship exists)

The test statistic is:

$$t = \frac{\beta_1}{\text{SE}(\beta_1)}$$

Which follows a t-distribution with $n-2$ degrees of freedom under the null hypothesis.

## Model Evaluation

### Coefficient of Determination (R-squared)
The R-squared value measures the proportion of variance in the dependent variable that is explained by the independent variable:

$$R^2 = 1 - \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Where SST is the total sum of squares.

### Pearson Correlation Coefficient
In simple linear regression, the square of the Pearson correlation coefficient equals the R-squared value:

$$R^2 = r^2$$

Where $r$ is the Pearson correlation coefficient:

$$r = \frac{\text{Cov}(x, y)}{\sqrt{\text{Var}(x) \cdot \text{Var}(y)}}$$

## Prediction and Confidence Intervals

### Point Prediction
For a new value $x_0$, the predicted value is:

$$\hat{y}_0 = \beta_0 + \beta_1 x_0$$

### Prediction Interval
A $(1-\alpha)$ prediction interval for a new observation at $x_0$ is:

$$\hat{y}_0 \pm t_{n-2, 1-\alpha/2} \cdot \text{SE}_{\text{pred}}$$

Where:

$$\text{SE}_{\text{pred}} = \hat{\sigma} \sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}}$$

### Confidence Interval for Mean Response
A $(1-\alpha)$ confidence interval for the mean response at $x_0$ is:

$$\hat{y}_0 \pm t_{n-2, 1-\alpha/2} \cdot \text{SE}_{\text{mean}}$$

Where:

$$\text{SE}_{\text{mean}} = \hat{\sigma} \sqrt{\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}}$$

## Limitations
- Cannot capture nonlinear relationships
- Only considers one predictor variable
- Sensitive to outliers
- May not be appropriate if assumptions are violated

## Applications
- Establishing relationships between pairs of variables
- Preliminary analysis before multiple regression
- Forecasting trends over time
- Educational studies of achievement
- Simple business and economic modeling
