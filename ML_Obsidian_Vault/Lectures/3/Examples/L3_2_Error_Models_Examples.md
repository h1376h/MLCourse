# Error Models Examples

This document provides practical examples of error models in linear regression, illustrating the concept of how different error distributions affect model assumptions, estimation, and inference.

## Key Concepts and Formulas

Error models in linear regression characterize the statistical properties of the residuals (the differences between observed and predicted values). The most common assumption is that errors follow a normal (Gaussian) distribution, but other distributions may be more appropriate in certain contexts.

### Gaussian Error Model

$$y_i = \beta_0 + \beta_1 x_i + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma^2)$$

Where:
- $\epsilon_i$ = Error term for observation $i$
- $N(0, \sigma^2)$ = Normal distribution with mean 0 and variance $\sigma^2$
- $\sigma^2$ = Error variance (constant for all observations in homoscedastic models)

### Other Error Models

- **Laplace (Double Exponential)**: $\epsilon_i \sim Laplace(0, b)$
- **Student's t**: $\epsilon_i \sim t(\nu)$ (heavier tails than normal)
- **Heteroscedastic Normal**: $\epsilon_i \sim N(0, \sigma_i^2)$ (variance varies by observation)
- **Log-Normal**: Used when errors are multiplicative rather than additive

## Examples

The following examples demonstrate error models in linear regression:

- **Gaussian Error Analysis**: Examining residuals for normality
- **Heteroscedasticity Detection**: Identifying non-constant error variance
- **Robust Regression**: Handling non-Gaussian errors
- **Maximum Likelihood Estimation**: Different error distributions lead to different cost functions

### Example 1: Gaussian Error Analysis

#### Problem Statement
A researcher has fit a linear regression model to predict crop yield (Y) based on rainfall amount (X). After fitting the model, the researcher wants to verify if the Gaussian error assumption is valid.

The fitted model is: $\hat{Y} = 30 + 2.5X$

The data and residuals are as follows:

| Rainfall (X) | Yield (Y) | Predicted (Ŷ) | Residual (e) |
|--------------|-----------|---------------|--------------|
| 10           | 55        | 55            | 0            |
| 15           | 70        | 67.5          | 2.5          |
| 20           | 80        | 80            | 0            |
| 25           | 85        | 92.5          | -7.5         |
| 30           | 100       | 105           | -5           |
| 35           | 120       | 117.5         | 2.5          |
| 40           | 125       | 130           | -5           |
| 45           | 140       | 142.5         | -2.5         |
| 50           | 150       | 155           | -5           |
| 55           | 175       | 167.5         | 7.5          |

Analyze these residuals to check if they follow a Gaussian distribution.

#### Solution

We'll examine the residuals using various techniques to check for normality.

##### Step 1: Calculate basic statistics of the residuals
First, let's compute some basic statistics:

- Mean of residuals: $\bar{e} = \frac{0 + 2.5 + 0 + (-7.5) + (-5) + 2.5 + (-5) + (-2.5) + (-5) + 7.5}{10} = \frac{-12.5}{10} = -1.25$

The mean should be close to zero. While -1.25 isn't exactly zero, it's relatively small compared to the scale of the data.

- Variance of residuals: 
$$\sigma^2 = \frac{\sum_{i=1}^{n}(e_i - \bar{e})^2}{n-1}$$

$$\sigma^2 = \frac{(0-(-1.25))^2 + (2.5-(-1.25))^2 + ... + (7.5-(-1.25))^2}{10-1}$$

$$\sigma^2 = \frac{(1.25)^2 + (3.75)^2 + (1.25)^2 + (-6.25)^2 + (-3.75)^2 + (3.75)^2 + (-3.75)^2 + (-1.25)^2 + (-3.75)^2 + (8.75)^2}{9}$$

$$\sigma^2 = \frac{1.56 + 14.06 + 1.56 + 39.06 + 14.06 + 14.06 + 14.06 + 1.56 + 14.06 + 76.56}{9} = \frac{190.6}{9} \approx 21.18$$

- Standard deviation: $\sigma = \sqrt{21.18} \approx 4.6$

##### Step 2: Create a histogram of residuals
A histogram of the residuals would look approximately like:

```
Frequency
    |     
  3 |     x       x
    |   
  2 |     x   x   
    |   
  1 | x       x   x   x
    |___________________
      -7.5 -5 -2.5 0 2.5 5 7.5
```

This distribution appears roughly symmetric around the mean, which is consistent with a normal distribution.

##### Step 3: Perform statistical tests
We can use the Shapiro-Wilk test for normality. The null hypothesis is that the data comes from a normal distribution.

For this example, let's assume the test gives a p-value of 0.87, which is greater than 0.05, so we fail to reject the null hypothesis, suggesting the residuals are consistent with a normal distribution.

##### Step 4: Create a Q-Q plot
A Q-Q plot compares the quantiles of the residuals with those expected from a normal distribution. If the points approximately follow a straight line, this suggests normality.

For our residuals, the Q-Q plot would show points close to a straight line, further supporting normality.

##### Step 5: Check for patterns
We should also check if there's any pattern in the residuals when plotted against the predictor or fitted values:

```
Residuals
    |                   
  5 |       o           o
    |           
  0 | o   o                   o
    |           
 -5 |           o   o   o   o
    |___________________________________
       10  20  30  40  50  60  Rainfall
```

There doesn't appear to be a clear pattern in the residuals, which is good. However, there might be a slight tendency for negative residuals in the middle range of rainfall, which could be worth investigating further.

##### Step 6: Conclusion
Based on our analysis:
- The mean of residuals is close to zero
- The distribution appears roughly symmetric
- Statistical tests do not reject normality
- The Q-Q plot is approximately linear
- No obvious patterns in the residuals

Therefore, the Gaussian error assumption seems reasonable for this model.

### Example 2: Detecting and Handling Heteroscedasticity

#### Problem Statement
An economist is studying the relationship between family income (X, in $1000s) and annual expenditure on luxury goods (Y, in $1000s). After fitting a linear regression model, they notice that the variability of residuals seems to increase with income levels.

The fitted model is: $\hat{Y} = 1.2 + 0.08X$

Data and residuals:

| Income (X) | Luxury Exp (Y) | Predicted (Ŷ) | Residual (e) |
|------------|----------------|---------------|--------------|
| 50         | 5.1            | 5.2           | -0.1         |
| 60         | 6.3            | 6.0           | 0.3          |
| 70         | 6.2            | 6.8           | -0.6         |
| 80         | 7.5            | 7.6           | -0.1         |
| 90         | 6.8            | 8.4           | -1.6         |
| 100        | 10.5           | 9.2           | 1.3          |
| 110        | 12.3           | 10.0          | 2.3          |
| 120        | 8.4            | 10.8          | -2.4         |
| 130        | 15.6           | 11.6          | 4.0          |
| 140        | 15.8           | 12.4          | 3.4          |

How can we detect and address this heteroscedasticity?

#### Solution

##### Step 1: Visualize the residuals
Let's plot the residuals against the income (X) values:

```
Residual
    |                           o   o  
  3 |                               
    |                       o       
  1 |       o                   
    |                               
 -1 | o           o               
    |           o       o           
 -3 |                       o      
    |___________________________________
       50  60  70  80  90 100 110 120 130 140  Income
```

The plot shows a clear pattern: the spread of residuals increases as income increases, which suggests heteroscedasticity.

##### Step 2: Apply statistical tests
We can use the Breusch-Pagan test, which tests the null hypothesis of homoscedasticity. Let's assume the test gives a p-value of 0.02, which is less than 0.05, so we reject the null hypothesis, confirming heteroscedasticity.

##### Step 3: Apply weighted least squares regression
Since the variance of residuals increases with X, we can use weighted least squares with weights inversely proportional to the variance. Based on the pattern, we might use weights $w_i = \frac{1}{X_i}$ or $w_i = \frac{1}{X_i^2}$.

Let's use $w_i = \frac{1}{X_i}$ for this example:

| Income (X) | Weight (w = 1/X) | Weighted X | Weighted Y |
|------------|------------------|------------|------------|
| 50         | 0.02             | 1          | 0.102      |
| 60         | 0.0167           | 1          | 0.105      |
| 70         | 0.0143           | 1          | 0.0886     |
| 80         | 0.0125           | 1          | 0.0938     |
| 90         | 0.0111           | 1          | 0.0756     |
| 100        | 0.01             | 1          | 0.105      |
| 110        | 0.0091           | 1          | 0.112      |
| 120        | 0.0083           | 1          | 0.07       |
| 130        | 0.0077           | 1          | 0.12       |
| 140        | 0.0071           | 1          | 0.113      |

Fitting the weighted least squares regression would give us a new model, such as: $\hat{Y} = 0.8 + 0.09X$

##### Step 4: Consider a log transformation
Another approach is to transform the data to stabilize variance. Since the variance increases with X, a log transformation might be appropriate.

Let's try modeling $\log(Y)$ as a function of $\log(X)$:

| Income (X) | Luxury Exp (Y) | log(X)      | log(Y)     |
|------------|----------------|-------------|------------|
| 50         | 5.1            | 3.91        | 1.63       |
| 60         | 6.3            | 4.09        | 1.84       |
| 70         | 6.2            | 4.25        | 1.82       |
| 80         | 7.5            | 4.38        | 2.01       |
| 90         | 6.8            | 4.50        | 1.92       |
| 100        | 10.5           | 4.61        | 2.35       |
| 110        | 12.3           | 4.70        | 2.51       |
| 120        | 8.4            | 4.79        | 2.13       |
| 130        | 15.6           | 4.87        | 2.75       |
| 140        | 15.8           | 4.94        | 2.76       |

Fitting a linear regression on the transformed data would give a model like: $\log(\hat{Y}) = -4.2 + 1.4\log(X)$, which can be written as $\hat{Y} = e^{-4.2}X^{1.4} \approx 0.015 X^{1.4}$.

This model can then be evaluated for constant variance in the residuals.

##### Step 5: Use robust standard errors
If we want to keep the original model but get more reliable inference, we can use heteroscedasticity-robust standard errors (White standard errors), which adjust the standard errors to account for heteroscedasticity.

##### Step 6: Conclusion
Based on our analysis:
- There is clear evidence of heteroscedasticity
- Weighted least squares provides a more efficient estimator
- Log transformation might stabilize the variance
- Robust standard errors can be used for reliable inference even with heteroscedasticity

The most appropriate approach depends on the specific needs of the analysis:
- For prediction: weighted least squares or transformed model may be better
- For inference: robust standard errors might be sufficient
- For interpretability: the original model with robust standard errors might be preferred

### Example 3: Robust Regression for Heavy-tailed Errors

#### Problem Statement
A medical researcher is studying the relationship between blood pressure medication dosage (X, in mg) and reduction in systolic blood pressure (Y, in mmHg). The dataset includes some outliers that might be due to measurement errors or unusual patient responses.

The data is as follows:

| Dosage (X) | BP Reduction (Y) |
|------------|------------------|
| 10         | 5                |
| 20         | 8                |
| 30         | 12               |
| 40         | 15               |
| 50         | 18               |
| 60         | 20               |
| 70         | 23               |
| 80         | 40               |
| 90         | 27               |
| 100        | 30               |

Compare ordinary least squares (OLS) regression with robust regression methods that are less sensitive to outliers.

#### Solution

##### Step 1: Fit an OLS regression model
Let's fit an OLS model to the data:

$$\hat{Y}_{OLS} = 0.5 + 0.3X$$

##### Step 2: Examine the residuals
Calculate residuals from the OLS model:

| Dosage (X) | BP Reduction (Y) | Predicted (Ŷ) | Residual (e) |
|------------|------------------|---------------|--------------|
| 10         | 5                | 3.5           | 1.5          |
| 20         | 8                | 6.5           | 1.5          |
| 30         | 12               | 9.5           | 2.5          |
| 40         | 15               | 12.5          | 2.5          |
| 50         | 18               | 15.5          | 2.5          |
| 60         | 20               | 18.5          | 1.5          |
| 70         | 23               | 21.5          | 1.5          |
| 80         | 40               | 24.5          | 15.5         |
| 90         | 27               | 27.5          | -0.5         |
| 100        | 30               | 30.5          | -0.5         |

The residual for X = 80 is much larger than the others, suggesting an outlier.

##### Step 3: Apply robust regression methods
Several robust regression methods can be used to mitigate the influence of outliers:

1. **Huber M-estimation**: This method assigns lower weights to observations with large residuals.
   
2. **Tukey's bisquare**: This method completely ignores observations with very large residuals.

3. **Least absolute deviations (LAD)**: This minimizes the sum of absolute residuals instead of squared residuals.

Let's fit a LAD regression:

$$\hat{Y}_{LAD} = 1.0 + 0.28X$$

##### Step 4: Compare the results
Let's compare the predictions from both models:

| Dosage (X) | BP Reduction (Y) | OLS Prediction | LAD Prediction |
|------------|------------------|----------------|----------------|
| 10         | 5                | 3.5            | 3.8            |
| 20         | 8                | 6.5            | 6.6            |
| 30         | 12               | 9.5            | 9.4            |
| 40         | 15               | 12.5           | 12.2           |
| 50         | 18               | 15.5           | 15.0           |
| 60         | 20               | 18.5           | 17.8           |
| 70         | 23               | 21.5           | 20.6           |
| 80         | 40               | 24.5           | 23.4           |
| 90         | 27               | 27.5           | 26.2           |
| 100        | 30               | 30.5           | 29.0           |

The LAD model is less influenced by the outlier at X = 80.

##### Step 5: Evaluate model performance
We can evaluate both models using median absolute error (MAE), which is less sensitive to outliers than mean squared error:

- OLS MAE: 1.5
- LAD MAE: 1.2

The LAD model has a lower MAE, indicating better overall fit to the majority of the data.

##### Step 6: Conclusion
Based on our analysis:
- The OLS model is heavily influenced by the outlier at X = 80
- The LAD model provides more robust estimates that are less sensitive to the outlier
- For this dataset with a heavy-tailed error distribution, robust regression methods are more appropriate

If the outlier is a valid data point representing a real phenomenon, we might need to reconsider our model (perhaps the relationship is non-linear). If it's a measurement error, the robust regression approach is preferable.

## Key Insights

### Theoretical Insights
- The Gaussian error model leads to the familiar least squares estimation
- Non-Gaussian error models may require different estimation methods
- Heteroscedasticity violates the constant variance assumption of standard OLS
- Heavy-tailed error distributions make OLS inefficient and sensitive to outliers

### Practical Applications
- Test for normality of residuals using visual and statistical methods
- Use weighted least squares or transformations for heteroscedastic data
- Apply robust regression methods when outliers are present
- Consider the error distribution when choosing the appropriate estimation method

### Common Pitfalls
- Assuming normality without checking residuals
- Ignoring heteroscedasticity, leading to inefficient estimates and invalid inference
- Using OLS when the data contains influential outliers
- Applying transformations without considering their effect on interpretability

## Related Topics

- [[L3_2_Simple_Linear_Regression|Simple Linear Regression]]: The basic model with error assumptions
- [[L3_2_Least_Squares|Least Squares Method]]: Optimal under Gaussian errors
- [[L3_2_Cost_Function|Cost Function]]: Different error models lead to different cost functions 