# Cost Function Examples

This document provides practical examples of cost functions for linear regression, illustrating the concept of measuring model performance and guiding optimization in finding the best parameters.

## Key Concepts and Formulas

Cost functions (also called loss functions) quantify how well a model fits the data by measuring the difference between predicted values and actual values. They are used to find optimal model parameters through minimization.

### Mean Squared Error (MSE)

$$J(\beta) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1x_i)^2$$

### Mean Absolute Error (MAE)

$$MAE(\beta) = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| = \frac{1}{n}\sum_{i=1}^{n}|y_i - \beta_0 - \beta_1x_i|$$

### Root Mean Squared Error (RMSE)

$$RMSE(\beta) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1x_i)^2}$$

Where:
- $y_i$ = Actual value of the dependent variable for observation $i$
- $\hat{y}_i$ = Predicted value for observation $i$ (equals $\beta_0 + \beta_1x_i$ for simple linear regression)
- $\beta_0$ = Intercept parameter
- $\beta_1$ = Slope parameter
- $n$ = Number of observations

## Examples

The following examples demonstrate cost functions in linear regression:

- **MSE Calculation**: Computing mean squared error for a simple model
- **Comparing Cost Functions**: Analyzing MSE vs. MAE for the same data
- **Visualizing the Cost Surface**: Understanding the relationship between parameters and cost

### Example 1: MSE Calculation for a Simple Linear Model

#### Problem Statement
Consider a simple linear regression model predicting student exam scores based on hours studied. We have the following data:

| Hours Studied ($x_i$) | Exam Score ($y_i$) |
|-----------------------|-----------------|
| 1                     | 60              |
| 2                     | 70              |
| 3                     | 75              |
| 4                     | 82              |
| 5                     | 85              |

Suppose we propose a model: $\hat{y} = 55 + 6x$

Calculate the MSE for this model to evaluate its performance.

#### Solution

We'll compute the MSE using the formula: $MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

##### Step 1: Calculate predicted values for each observation
For each value of $x_i$, we compute $\hat{y}_i = 55 + 6x_i$:

| $x_i$ | $y_i$ | $\hat{y}_i = 55 + 6x_i$ | $y_i - \hat{y}_i$ | $(y_i - \hat{y}_i)^2$ |
|-------|-------|--------------------------|-------------------|------------------------|
| 1     | 60    | 55 + 6(1) = 61          | 60 - 61 = -1      | 1                      |
| 2     | 70    | 55 + 6(2) = 67          | 70 - 67 = 3       | 9                      |
| 3     | 75    | 55 + 6(3) = 73          | 75 - 73 = 2       | 4                      |
| 4     | 82    | 55 + 6(4) = 79          | 82 - 79 = 3       | 9                      |
| 5     | 85    | 55 + 6(5) = 85          | 85 - 85 = 0       | 0                      |
| Sum   |       |                          |                   | 23                     |

##### Step 2: Compute the MSE
$$MSE = \frac{1}{5}(1 + 9 + 4 + 9 + 0) = \frac{23}{5} = 4.6$$

Therefore, the MSE for this model is 4.6, which means that on average, the squared difference between the actual and predicted scores is 4.6.

### Example 2: Comparing MSE and MAE for Model Selection

#### Problem Statement
Consider two different models for predicting housing prices (in $1000s) based on square footage:

- Model A: $\hat{y}_A = 50 + 0.08x$
- Model B: $\hat{y}_B = 75 + 0.05x$

For the following set of houses:

| Square Footage ($x_i$) | Actual Price ($y_i$) |
|------------------------|----------------------|
| 1000                   | 150                  |
| 1500                   | 165                  |
| 2000                   | 200                  |
| 2500                   | 255                  |
| 3000                   | 280                  |

Calculate both MSE and MAE for each model to decide which one performs better.

#### Solution

##### Step 1: Calculate predictions and errors for Model A
For each house, we compute $\hat{y}_A = 50 + 0.08x_i$:

| $x_i$ | $y_i$ | $\hat{y}_A = 50 + 0.08x_i$ | $y_i - \hat{y}_A$ | $(y_i - \hat{y}_A)^2$ | $\|y_i - \hat{y}_A\|$ |
|-------|-------|----------------------------|-------------------|------------------------|------------------------|
| 1000  | 150   | 50 + 0.08(1000) = 130     | 20                | 400                    | 20                     |
| 1500  | 165   | 50 + 0.08(1500) = 170     | -5                | 25                     | 5                      |
| 2000  | 200   | 50 + 0.08(2000) = 210     | -10               | 100                    | 10                     |
| 2500  | 255   | 50 + 0.08(2500) = 250     | 5                 | 25                     | 5                      |
| 3000  | 280   | 50 + 0.08(3000) = 290     | -10               | 100                    | 10                     |
| Sum   |       |                            |                   | 650                    | 50                     |

MSE for Model A: $MSE_A = \frac{650}{5} = 130$
MAE for Model A: $MAE_A = \frac{50}{5} = 10$

##### Step 2: Calculate predictions and errors for Model B
For each house, we compute $\hat{y}_B = 75 + 0.05x_i$:

| $x_i$ | $y_i$ | $\hat{y}_B = 75 + 0.05x_i$ | $y_i - \hat{y}_B$ | $(y_i - \hat{y}_B)^2$ | $\|y_i - \hat{y}_B\|$ |
|-------|-------|----------------------------|-------------------|------------------------|------------------------|
| 1000  | 150   | 75 + 0.05(1000) = 125     | 25                | 625                    | 25                     |
| 1500  | 165   | 75 + 0.05(1500) = 150     | 15                | 225                    | 15                     |
| 2000  | 200   | 75 + 0.05(2000) = 175     | 25                | 625                    | 25                     |
| 2500  | 255   | 75 + 0.05(2500) = 200     | 55                | 3025                   | 55                     |
| 3000  | 280   | 75 + 0.05(3000) = 225     | 55                | 3025                   | 55                     |
| Sum   |       |                            |                   | 7525                   | 175                    |

MSE for Model B: $MSE_B = \frac{7525}{5} = 1505$
MAE for Model B: $MAE_B = \frac{175}{5} = 35$

##### Step 3: Compare models
- MSE: Model A (130) < Model B (1505) → Model A is better by MSE
- MAE: Model A (10) < Model B (35) → Model A is better by MAE

Both metrics indicate that Model A fits the data better. The MSE particularly penalizes Model B heavily because of the larger errors for the higher-priced houses, as squaring the errors magnifies the large deviations.

### Example 3: Cost Function Visualization

#### Problem Statement
Visualize the cost function landscape for a simple dataset to understand how the cost varies with different values of intercept ($\beta_0$) and slope ($\beta_1$).

Consider a simple dataset with three points: (1, 2), (2, 3), and (3, 5).

#### Solution

##### Step 1: Define the cost function (MSE)
For a simple linear regression model $\hat{y} = \beta_0 + \beta_1x$, the MSE is:
$$MSE(\beta_0, \beta_1) = \frac{1}{3}\sum_{i=1}^{3}(y_i - \beta_0 - \beta_1x_i)^2$$

##### Step 2: Expand the sum
$$MSE(\beta_0, \beta_1) = \frac{1}{3}[(2 - \beta_0 - \beta_1 \cdot 1)^2 + (3 - \beta_0 - \beta_1 \cdot 2)^2 + (5 - \beta_0 - \beta_1 \cdot 3)^2]$$

##### Step 3: Computing MSE for specific values of $\beta_0$ and $\beta_1$
We'll compute MSE for selected values to visualize the cost surface:

For $\beta_0 = 0, \beta_1 = 0$:
$$MSE(0, 0) = \frac{1}{3}[(2 - 0 - 0)^2 + (3 - 0 - 0)^2 + (5 - 0 - 0)^2] = \frac{1}{3}[4 + 9 + 25] = \frac{38}{3} \approx 12.67$$

For $\beta_0 = 0, \beta_1 = 1$:
$$MSE(0, 1) = \frac{1}{3}[(2 - 0 - 1 \cdot 1)^2 + (3 - 0 - 1 \cdot 2)^2 + (5 - 0 - 1 \cdot 3)^2] = \frac{1}{3}[1 + 1 + 4] = 2$$

For $\beta_0 = 0, \beta_1 = 1.5$:
$$MSE(0, 1.5) = \frac{1}{3}[(2 - 0 - 1.5 \cdot 1)^2 + (3 - 0 - 1.5 \cdot 2)^2 + (5 - 0 - 1.5 \cdot 3)^2] = \frac{1}{3}[0.25 + 0 + 0.25] = \frac{0.5}{3} \approx 0.167$$

For $\beta_0 = 1, \beta_1 = 1$:
$$MSE(1, 1) = \frac{1}{3}[(2 - 1 - 1 \cdot 1)^2 + (3 - 1 - 1 \cdot 2)^2 + (5 - 1 - 1 \cdot 3)^2] = \frac{1}{3}[0 + 0 + 1] = \frac{1}{3} \approx 0.333$$

For $\beta_0 = 0.5, \beta_1 = 1.5$:
$$MSE(0.5, 1.5) = \frac{1}{3}[(2 - 0.5 - 1.5 \cdot 1)^2 + (3 - 0.5 - 1.5 \cdot 2)^2 + (5 - 0.5 - 1.5 \cdot 3)^2]$$
$$= \frac{1}{3}[(2 - 2)^2 + (3 - 3.5)^2 + (5 - 5)^2] = \frac{1}{3}[0 + 0.25 + 0] = \frac{0.25}{3} \approx 0.083$$

##### Step 4: Finding the optimal parameters
By computing more values and applying calculus, we can determine that the optimal values are approximately $\beta_0 = 0.5$ and $\beta_1 = 1.5$, which gives us the lowest MSE of about 0.083.

The resulting regression line is $\hat{y} = 0.5 + 1.5x$, which closely fits our three data points.

## Key Insights

### Theoretical Insights
- The MSE is a quadratic function of the parameters, resulting in a convex optimization problem
- The MSE heavily penalizes large errors due to the squaring operation
- The MAE is less sensitive to outliers than the MSE
- A cost function of zero would indicate perfect prediction (rarely achieved in practice)

### Practical Applications
- Cost functions guide optimization algorithms in finding the best model parameters
- Different cost functions may lead to different optimal parameters
- Selecting an appropriate cost function depends on the specific problem and what types of errors are most important to minimize

### Common Pitfalls
- Using MSE when outliers are present can lead to models that are overly influenced by extreme values
- Using MAE can lead to multiple solutions as it's not differentiable at zero
- Focusing solely on minimizing the cost function on training data can lead to overfitting

## Related Topics

- [[L3_2_Simple_Linear_Regression|Simple Linear Regression]]: The basic model whose performance is measured by cost functions
- [[L3_2_Least_Squares|Least Squares Method]]: The optimization method that minimizes the MSE
- [[L3_2_Analytical_Solution|Analytical Solution]]: Closed-form solution that directly computes the parameters minimizing MSE 