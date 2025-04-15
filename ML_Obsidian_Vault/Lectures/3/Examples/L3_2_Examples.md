# Linear Regression Examples

This document provides examples and key concepts on simple linear regression to help you understand this important concept in machine learning and data analysis.

## Key Concepts and Formulas

Simple linear regression models the relationship between a dependent variable and one independent variable by fitting a linear equation to the observed data.

### Basic Formulations

- **Simple Linear Regression**: $y = \beta_0 + \beta_1x + \epsilon$
- **Ordinary Least Squares**: Minimizes the sum of squared residuals

Where:
- $y$ = Dependent variable (target)
- $x$ = Independent variable (feature)
- $\beta_0$ = Intercept (bias term)
- $\beta_1$ = Coefficient (weight)
- $\epsilon$ = Error term (assumes normal distribution with mean 0)

### Important Formulas

- **Cost Function (MSE)**: $J(\beta) = \frac{1}{2n}\sum_{i=1}^{n}(h_\beta(x^{(i)}) - y^{(i)})^2$
- **Ordinary Least Squares (OLS)**: Minimizes the sum of squared errors
  $$\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \min_{\beta} \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1x_i)^2$$

## Practice Questions

For practice multiple-choice questions on linear regression, see:
- [[L3_2_Quiz|Simple Linear Regression Quiz]]

## Examples

### Simple Linear Regression
- [[L3_2_Linear_Regression_Examples|Linear Regression Formulation Examples]]: Problem setup and notation
- [[L3_2_Simple_Linear_Regression_Examples|Simple Linear Regression Examples]]: One-variable linear models
- [[L3_2_Cost_Function_Examples|Cost Function Examples]]: MSE and optimization objectives
- [[L3_2_Least_Squares_Examples|Least Squares Examples]]: Derivation and geometric interpretation
- [[L3_2_Analytical_Solution_Examples|Analytical Solution Examples]]: Closed-form solution for linear regression
- [[L3_2_Error_Models_Examples|Error Models Examples]]: Gaussian and other error distributions

## Related Topics
- [[L3_2_Linear_Regression_Formulation|Linear Regression Formulation]]
- [[L3_2_Simple_Linear_Regression|Simple Linear Regression]]
- [[L3_2_Cost_Function|Cost Function]]
- [[L3_2_Least_Squares|Least Squares Method]]
- [[L3_2_Analytical_Solution|Analytical Solution]]
- [[L3_2_Error_Models|Error Models]]