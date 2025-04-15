# Cost Function

## Introduction
Cost functions, also known as loss functions, are essential components in linear regression that quantify how well a model fits the training data. They provide a mathematical measure of error between predicted values and actual outcomes, serving as the objective function that the learning algorithm aims to minimize.

## Mean Squared Error (MSE)

### Definition
The most common cost function for linear regression is the Mean Squared Error (MSE), defined as:

$$J(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

Where:
- $n$ is the number of training examples
- $y_i$ is the actual target value
- $\hat{y}_i$ is the predicted value
- $\mathbf{w}$ is the weight vector
- $\mathbf{x}_i$ is the feature vector

### Properties of MSE
- **Non-negative**: MSE is always ≥ 0, with 0 indicating perfect prediction
- **Convex function**: Guarantees a global minimum for linear models
- **Differentiable**: Allows for gradient-based optimization
- **Penalizes larger errors**: Squares the errors, giving greater weight to outliers
- **Same units as target variable squared**: MSE is in squared units of the target variable

## Alternative Cost Functions

### Mean Absolute Error (MAE)
$$J(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- Less sensitive to outliers than MSE
- Not differentiable at zero, complicating optimization
- Results in median predictions rather than mean

### Root Mean Squared Error (RMSE)
$$\text{RMSE}(\mathbf{w}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

- Same units as the target variable
- Often used for reporting model performance
- Differentiable everywhere except at the global minimum if perfect prediction occurs

## Matrix Formulation
For multiple linear regression, the MSE can be expressed in matrix form:

$$J(\mathbf{w}) = \frac{1}{n} (\mathbf{y} - \mathbf{X}\mathbf{w})^T (\mathbf{y} - \mathbf{X}\mathbf{w})$$

Where:
- $\mathbf{y}$ is the vector of target values
- $\mathbf{X}$ is the design matrix with rows representing feature vectors
- $\mathbf{w}$ is the weight vector

## Optimization Objective
The primary goal in linear regression is to find the parameter vector $\mathbf{w}$ that minimizes the cost function:

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} J(\mathbf{w})$$

This optimization problem can be solved:
1. Analytically using the normal equations (covered in [[L3_2_Analytical_Solution]])
2. Using the method of least squares (covered in [[L3_2_Least_Squares]])

## Geometric Interpretation
The MSE cost function can be visualized as a quadratic (paraboloid) surface in the parameter space:
- For simple linear regression, it's a 3D parabolic surface with parameters $w_0$ and $w_1$
- The minimum point corresponds to the optimal parameter values
- The convexity ensures that there is exactly one global minimum

## Connection to Maximum Likelihood
Under the assumption of Gaussian-distributed errors, minimizing the MSE is equivalent to maximizing the likelihood function:

- If we assume the errors follow a normal distribution: $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$
- Then maximizing likelihood is equivalent to minimizing the sum of squared errors

## Practical Considerations
When working with cost functions in linear regression:
- Feature scaling can improve numerical stability
- Careful choice of cost function should reflect the specific application needs
- In practice, MSE is often preferred due to its mathematical properties and ease of optimization

## Applications
In the context of linear regression, cost functions are used for:
- Parameter estimation
- Model selection 
- Evaluating predictive performance
- Comparing different regression approaches 