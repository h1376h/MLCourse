# Linear Regression Formulation

## Problem Setup
Linear regression models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the observed data.

## Basic Formulation
For a dataset with $n$ observations and $d$ features, the linear regression model can be expressed as:

$$y_i = w_0 + w_1 x_{i1} + w_2 x_{i2} + \ldots + w_d x_{id} + \epsilon_i$$

Where:
- $y_i$ is the target value for the $i$-th observation
- $x_{ij}$ is the value of the $j$-th feature for the $i$-th observation
- $w_0, w_1, \ldots, w_d$ are the model parameters (weights)
- $\epsilon_i$ is the error term (residual) for the $i$-th observation

## Matrix Notation
The linear regression model can be expressed more compactly using matrix notation:

$$\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}$$

Where:
- $\mathbf{y} \in \mathbb{R}^n$ is the vector of target values
- $\mathbf{X} \in \mathbb{R}^{n \times (d+1)}$ is the design matrix (with a column of ones for the intercept)
- $\mathbf{w} \in \mathbb{R}^{d+1}$ is the vector of weights
- $\boldsymbol{\epsilon} \in \mathbb{R}^n$ is the vector of error terms

The design matrix $\mathbf{X}$ is structured as:

$$\mathbf{X} = 
\begin{pmatrix} 
1 & x_{11} & x_{12} & \cdots & x_{1d} \\
1 & x_{21} & x_{22} & \cdots & x_{2d} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{nd}
\end{pmatrix}$$

## Objective Function
The goal of linear regression is to find the weights $\mathbf{w}$ that minimize the sum of squared residuals (SSR), also known as the residual sum of squares (RSS):

$$\text{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (\mathbf{w}^T\mathbf{x}_i))^2$$

In matrix form:

$$\text{SSR} = (\mathbf{y} - \mathbf{X}\mathbf{w})^T(\mathbf{y} - \mathbf{X}\mathbf{w})$$

## Mean Squared Error (MSE)
The MSE is the average of the squared residuals:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n}\text{SSR}$$

## Parameter Estimation

### Analytical Solution
The optimal weights that minimize the SSR can be found by setting the derivative of the SSR with respect to $\mathbf{w}$ to zero:

$$\frac{\partial \text{SSR}}{\partial \mathbf{w}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w}) = \mathbf{0}$$

Solving for $\mathbf{w}$ gives the normal equations:

$$\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}$$

If $\mathbf{X}^T\mathbf{X}$ is invertible, the optimal weights are:

$$\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

## Assumptions of Linear Regression
1. **Linearity**: The relationship between features and target is linear
2. **Independence**: The observations are independent
3. **Homoscedasticity**: The error terms have constant variance
4. **Normality**: The error terms are normally distributed
5. **No perfect multicollinearity**: The features are not perfectly correlated

## Prediction
Once the model is trained (parameters estimated), predictions for new data points can be made:

$$\hat{y}_{\text{new}} = \mathbf{w}^T\mathbf{x}_{\text{new}}$$

## Basic Evaluation Metrics
Common metrics to evaluate linear regression models include:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (coefficient of determination)
- Adjusted R-squared 