# Simple Linear Regression Examples

This document provides practical examples of simple linear regression for various scenarios, illustrating the concept of modeling relationships between two variables and making predictions based on these models.

## Key Concepts and Formulas

Simple linear regression models the relationship between a dependent variable and one independent variable by fitting a linear equation to the observed data.

### The Simple Linear Regression Model

$$y = \beta_0 + \beta_1x + \epsilon$$

$$\hat{y} = \hat{\beta_0} + \hat{\beta_1}x$$

Where:
- $y$ = Dependent variable (target)
- $x$ = Independent variable (feature)
- $\beta_0$ = Population intercept (bias term)
- $\beta_1$ = Population slope (weight)
- $\epsilon$ = Error term (assumes normal distribution with mean 0)
- $\hat{y}$ = Predicted value of $y$
- $\hat{\beta_0}, \hat{\beta_1}$ = Estimated intercept and slope

## Examples

The following examples demonstrate simple linear regression:

- **Estimating Car Prices**: Predicting car prices based on age
- **Salary Prediction**: Modeling the relationship between years of experience and salary
- **Temperature Conversion**: Using linear regression to model Celsius-Fahrenheit conversion

### Example 1: Estimating Car Prices

#### Problem Statement
A car dealership wants to model the relationship between a car's age (in years) and its price (in $1000s). They have collected data from recent sales:

| Car Age (years) | Price ($1000s) |
|-----------------|----------------|
| 1               | 32             |
| 3               | 27             |
| 5               | 22             |
| 7               | 20             |
| 9               | 16             |
| 10              | 15             |

The dealership wants to:
1. Find the linear relationship between car age and price
2. Predict the price of a 4-year-old car
3. Interpret what the coefficients mean in this context

#### Solution

We'll fit a simple linear regression model: $Price = \beta_0 + \beta_1 \cdot Age + \epsilon$

##### Step 1: Calculate the means of x (age) and y (price)
$\bar{x} = \frac{1+3+5+7+9+10}{6} = \frac{35}{6} \approx 5.83$ years

$\bar{y} = \frac{32+27+22+20+16+15}{6} = \frac{132}{6} = 22$ thousand dollars

##### Step 2: Calculate the slope coefficient ($\hat{\beta_1}$)
The formula for the slope is:
$$\hat{\beta_1} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

Let's compute this:

| $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i - \bar{x})(y_i - \bar{y})$ | $(x_i - \bar{x})^2$ |
|-------|-------|-----------------|-----------------|----------------------------------|----------------------|
| 1     | 32    | -4.83           | 10              | -48.3                            | 23.33                |
| 3     | 27    | -2.83           | 5               | -14.15                           | 8.01                 |
| 5     | 22    | -0.83           | 0               | 0                                | 0.69                 |
| 7     | 20    | 1.17            | -2              | -2.34                            | 1.37                 |
| 9     | 16    | 3.17            | -6              | -19.02                           | 10.05                |
| 10    | 15    | 4.17            | -7              | -29.19                           | 17.39                |
| Sum   |       |                 |                 | -113                             | 60.84                |

$$\hat{\beta_1} = \frac{-113}{60.84} \approx -1.857$$

##### Step 3: Calculate the intercept ($\hat{\beta_0}$)
Using the formula: $\hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x}$
$$\hat{\beta_0} = 22 - (-1.857)(5.83) \approx 22 + 10.83 \approx 32.83$$

##### Step 4: Write the regression equation
$$\hat{Price} = 32.83 - 1.857 \cdot Age$$

##### Step 5: Predict the price of a 4-year-old car
$$\hat{Price} = 32.83 - 1.857 \cdot 4 = 32.83 - 7.43 = 25.4$$

Therefore, the predicted price of a 4-year-old car is $25,400.

##### Step 6: Interpret the coefficients
- $\hat{\beta_0} = 32.83$: This represents the estimated price (in $1000s) of a brand new car (age = 0)
- $\hat{\beta_1} = -1.857$: This means that for each additional year of age, a car's price decreases by approximately $1,857

### Example 2: Salary Prediction Based on Experience

#### Problem Statement
A company wants to establish a salary guideline based on years of experience. They collect data on current employees:

| Experience (years) | Salary ($1000s) |
|--------------------|-----------------|
| 1                  | 45              |
| 3                  | 60              |
| 5                  | 75              |
| 7                  | 83              |
| 10                 | 100             |

The goal is to:
1. Build a linear regression model
2. Predict the salary for someone with 6 years of experience
3. Interpret the model's coefficients

#### Solution

##### Step 1: Calculate the means
$\bar{x} = \frac{1+3+5+7+10}{5} = \frac{26}{5} = 5.2$ years

$\bar{y} = \frac{45+60+75+83+100}{5} = \frac{363}{5} = 72.6$ thousand dollars

##### Step 2: Calculate the slope coefficient
Using the same approach as Example 1:

| $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i - \bar{x})(y_i - \bar{y})$ | $(x_i - \bar{x})^2$ |
|-------|-------|-----------------|-----------------|----------------------------------|----------------------|
| 1     | 45    | -4.2            | -27.6           | 115.92                           | 17.64                |
| 3     | 60    | -2.2            | -12.6           | 27.72                            | 4.84                 |
| 5     | 75    | -0.2            | 2.4             | -0.48                            | 0.04                 |
| 7     | 83    | 1.8             | 10.4            | 18.72                            | 3.24                 |
| 10    | 100   | 4.8             | 27.4            | 131.52                           | 23.04                |
| Sum   |       |                 |                 | 293.4                            | 48.8                 |

$$\hat{\beta_1} = \frac{293.4}{48.8} \approx 6.01$$

##### Step 3: Calculate the intercept
$$\hat{\beta_0} = 72.6 - 6.01(5.2) \approx 72.6 - 31.25 \approx 41.35$$

##### Step 4: Write the regression equation
$$\hat{Salary} = 41.35 + 6.01 \cdot Experience$$

##### Step 5: Predict the salary for 6 years of experience
$$\hat{Salary} = 41.35 + 6.01 \cdot 6 = 41.35 + 36.06 = 77.41$$

Therefore, the predicted salary for someone with 6 years of experience is $77,410.

##### Step 6: Interpret the coefficients
- $\hat{\beta_0} = 41.35$: The base salary for someone with no experience is approximately $41,350
- $\hat{\beta_1} = 6.01$: For each additional year of experience, the salary increases by approximately $6,010

## Key Insights

### Theoretical Insights
- Simple linear regression assumes a linear relationship between variables
- The model estimates two parameters: intercept and slope
- The intercept represents the predicted value when the independent variable is zero
- The slope represents the change in the dependent variable for a one-unit change in the independent variable

### Practical Applications
- Simple linear regression is useful for quick predictions based on a single feature
- The model can be used to identify trends and relationships between variables
- Extrapolation (predicting beyond the range of observed data) should be done with caution

### Common Pitfalls
- Assuming linearity when the relationship is non-linear
- Not checking the assumptions of the regression model (normality, constant variance)
- Using the model to predict far outside the range of observed data
- Confusing correlation with causation

## Related Topics

- [[L3_2_Cost_Function|Cost Function]]: How to measure the model's accuracy
- [[L3_2_Least_Squares|Least Squares Method]]: The mathematical foundation for finding the optimal parameters
- [[L3_2_Analytical_Solution|Analytical Solution]]: The closed-form solution for linear regression 