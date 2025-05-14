# Linear Regression Notes

## Topics

- Linear regression
  - Error (cost) function
  - Optimization
  - Generalization

## Regression problem

The goal is to make (real valued) predictions given features.

Example: predicting house price from 3 attributes

| Size (m²) | Age (year) | Region | Price (10⁶T) |
|-----------|------------|--------|--------------|
| 100       | 2          | 5      | 500          |
| 80        | 25         | 3      | 250          |
| ...       | ...        | ...    | ...          |

## Learning problem

### Selecting a hypothesis space
- Hypothesis space: a set of mappings from feature vector to target

### Learning (estimation): optimization of a cost function
- Based on the training set $D = \{(\boldsymbol{x}^{(i)}, y^{(i)})\}_{i=1}^n$ and a cost function we find (an estimate) $f \in F$ of the target function

### Evaluation: we measure how well $\hat{f}$ generalizes to unseen examples

## Hypothesis space

### Specify the class of functions (e.g., linear)

### We begin by the class of linear functions
- easy to extend to generalized linear and so cover more complex regression functions

## Linear regression: hypothesis space

### Univariate
$$f : \mathbb{R} \rightarrow \mathbb{R} \quad f(x; \boldsymbol{w}) = w_0 + w_1 x$$

### Multivariate
$$f : \mathbb{R}^d \rightarrow \mathbb{R} \quad f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 x_1 + \ldots w_d x_d$$

$$\boldsymbol{w} = [w_0, w_1, \ldots, w_d]^T$$ are parameters we need to set.

![Linear Model Visualization](Codes/plots/linear_model_visualization.png)

## Learning algorithm

The learning algorithm for linear regression follows these key steps:

1. **Input**: Training set $D = \{(\boldsymbol{x}^{(i)}, y^{(i)})\}_{i=1}^n$
   - Each $\boldsymbol{x}^{(i)}$ is a feature vector
   - Each $y^{(i)}$ is the corresponding target value

2. **Processing**: The learning algorithm:
   - Defines a hypothesis space (linear functions)
   - Defines a cost function (typically SSE)
   - Optimizes the cost function to find optimal parameters $\boldsymbol{w}$

3. **Output**: Parameters $w_0, w_1, ..., w_d$ that define our model
   - These parameters minimize the cost function 
   - They represent the best-fitting linear function in our hypothesis space

4. **Prediction**: For a new input $\boldsymbol{x}$, we compute:
   $\hat{y} = f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 x_1 + ... + w_d x_d$

The learning process aims to:
1. Measure how well $f(\boldsymbol{x}; \boldsymbol{w})$ approximates the target function
2. Choose $\boldsymbol{w}$ to minimize the error measure

### Select how to measure the error (i.e. prediction loss)

### Find the minimum of the resulting error or cost function

## How to measure the error

In supervised learning, we need a way to quantify how well our model's predictions match the actual target values. For regression problems, we measure the difference between the predicted value $f(\boldsymbol{x}^{(i)}; \boldsymbol{w})$ and the actual value $y^{(i)}$.

### Squared Error
The most common error measure for linear regression is the squared error:

$$\text{Squared error} = \left(y^{(i)} - f(x^{(i)}; \boldsymbol{w})\right)^2$$

![Squared Error Visualization](Codes/plots/squared_error_visualization.png)

## Learning algorithm

![Learning Algorithm](Codes/plots/learning_algorithm_diagram.png)

Training Set $D$

We need to:
1. Measure how well $f(x; \boldsymbol{w})$ approximates the target
2. Choose $\boldsymbol{w}$ to minimize the error measure

## Linear regression: univariate example

Cost function:

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - f(x; \boldsymbol{w}))^2$$

$$= \sum_{i=1}^{n} (y^{(i)} - w_0 - w_1 x^{(i)})^2$$

The error for each data point is the vertical distance between the predicted value $f(x^{(i)}; \boldsymbol{w})$ and the actual value $y^{(i)}$. We sum the squares of these errors across all training examples.

![Linear Regression Error Visualization](Codes/plots/linear_regression_errors.png)

## Regression: squared loss

In the SSE cost function, we used squared error as the prediction loss:

$$Loss(y, \hat{y}) = (y - \hat{y})^2 \quad \hat{y} = f(\boldsymbol{x}; \boldsymbol{w})$$

Cost function (based on the training set):

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} Loss(y^{(i)}, f(\boldsymbol{x}^{(i)}; \boldsymbol{w}))$$

$$= \sum_{i=1}^{n} (y^{(i)} - f(\boldsymbol{x}^{(i)}; \boldsymbol{w}))^2$$

Minimizing sum (or mean) of squared errors is a common approach in curve fitting, neural network, etc.

## Sum of Squares Error (SSE) cost function

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - f(\boldsymbol{x}^{(i)}; \boldsymbol{w}))^2$$

$J(\boldsymbol{w})$: sum of the squares of the prediction errors on the training set

We want to find the best regression function $f(\boldsymbol{x}^{(i)}; \boldsymbol{w})$
- equivalently, the best $\boldsymbol{w}$

Minimize $J(\boldsymbol{w})$
- Find optimal $\hat{f}(\boldsymbol{x}) = f(\boldsymbol{x}; \hat{\boldsymbol{w}})$ where $\hat{\boldsymbol{w}} = \underset{\boldsymbol{w}}{\operatorname{argmin}}J(\boldsymbol{w})$

## Minimizing the empirical squared loss

The goal is to minimize the empirical squared loss:

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - f(\boldsymbol{x}^{(i)}; \boldsymbol{w}))^2$$

Where our linear model is:

$$f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 x_1 + \ldots w_d x_d$$

And our parameter vector is:

$$\boldsymbol{w} = [w_0, w_1, \ldots, w_d]^T$$

We aim to find the optimal parameters:

$$\hat{\boldsymbol{w}} = \underset{\boldsymbol{w} \in \mathbb{R}^{d+1}}{\operatorname{argmin}}J(\boldsymbol{w})$$

## Cost function: univariate example

$$f(x; w_0, w_1) = w_0 + w_1 x$$

(for fixed $w_0,w_1$, this is a function of $x$)

$$J(w_0, w_1)$$

(function of the parameters $w_0,w_1$)

The left plot shows the hypothesis $f(x; w_0, w_1) = w_0 + w_1 x$ (blue line) and training data points (red x marks). The right plot shows contour lines of the cost function $J(w_0, w_1)$ in the parameter space, with the red x marks showing parameters that have been tried in the gradient descent process.

This example has been adapted from: Prof. Andrew Ng's slides (ML Online Course, Stanford)

### 3D visualization of the cost function:

The 3D surface plot shows how the cost function $J(w_0, w_1)$ varies with different combinations of the intercept $w_0$ and slope $w_1$ parameters. Key features:

- The x-axis represents the intercept parameter $w_0$
- The y-axis represents the slope parameter $w_1$
- The z-axis represents the cost (sum of squared errors)
- The surface has a bowl-like shape, characteristic of convex functions
- The single global minimum corresponds to the optimal parameters that give the best fit
- The surface is steeper in some directions than others, indicating different sensitivities to the parameters

![Cost Function 3D](Codes/plots/cost_function_3d.png)

### Contour plot of the cost function:

The contour plot provides a top-down view of the same cost function, with contour lines connecting points of equal cost value. Key features:

- Each contour line represents combinations of $w_0$ and $w_1$ that produce the same cost value
- Tightly packed contour lines indicate steep gradients
- The center (where contour lines are most dense) represents the global minimum
- The elliptical shape of the contours indicates that the cost function is more sensitive to changes in some directions than others
- The overlaid red path shows how gradient descent iteratively approaches the minimum, starting from an initial guess and following the negative gradient

These visualizations help understand why gradient descent works well for linear regression - the cost function has a clear global minimum and no local minima to get trapped in.

![Cost Function Contour](Codes/plots/cost_function_contour.png)

### Additional visualizations of the cost function

The left plot shows another example of hypothesis $f(x; w_0, w_1) = w_0 + w_1 x$ (blue horizontal line) where the parameters give a constant prediction. The right plot again shows contour lines of $J(w_0, w_1)$ in the parameter space.

![Cost Function Example](Codes/plots/cost_function_horizontal.png)

### Another variant of the linear hypothesis

This version shows a decreasing linear hypothesis (negative slope):

![Cost Function Example Negative Slope](Codes/plots/cost_function_negative_slope.png)

### Increasing linear hypothesis

This version shows an increasing linear hypothesis (positive slope):

![Cost Function Example Positive Slope](Codes/plots/cost_function_positive_slope.png)

## Cost function optimization: univariate

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - w_0 - w_1 x^{(i)})^2$$

Necessary conditions for the "optimal" parameter values:

$$\frac{\partial J(\boldsymbol{w})}{\partial w_0} = 0$$

$$\frac{\partial J(\boldsymbol{w})}{\partial w_1} = 0$$

## Optimality conditions: univariate

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - w_0 - w_1 x^{(i)})^2$$

$$\frac{\partial J(\boldsymbol{w})}{\partial w_1} = \sum_{i=1}^{n} 2(y^{(i)} - w_0 - w_1 x^{(i)})(-x^{(i)}) = 0$$

$$\frac{\partial J(\boldsymbol{w})}{\partial w_0} = \sum_{i=1}^{n} 2(y^{(i)} - w_0 - w_1 x^{(i)})(-1) = 0$$

A systems of 2 linear equations

## Normal equations: univariate

$\frac{\partial J(\boldsymbol{w})}{\partial w_0} = 0 \Rightarrow \sum_{i=1}^{n} (y^{(i)} - w_0 - w_1 x^{(i)}) = 0$

$\frac{\partial J(\boldsymbol{w})}{\partial w_1} = 0 \Rightarrow \sum_{i=1}^{n} (y^{(i)} - w_0 - w_1 x^{(i)})x^{(i)} = 0$

$\sum_{i=1}^{n} y^{(i)} - n w_0 - w_1 \sum_{i=1}^{n} x^{(i)} = 0$

$\sum_{i=1}^{n} y^{(i)}x^{(i)} - w_0 \sum_{i=1}^{n} x^{(i)} - w_1 \sum_{i=1}^{n} (x^{(i)})^2 = 0$

## Normal equations: univariate

$\sum_{i=1}^{n} y^{(i)} - n w_0 - w_1 \sum_{i=1}^{n} x^{(i)} = 0$

$\sum_{i=1}^{n} y^{(i)}x^{(i)} - w_0 \sum_{i=1}^{n} x^{(i)} - w_1 \sum_{i=1}^{n} (x^{(i)})^2 = 0$

$n w_0 + w_1 \sum_{i=1}^{n} x^{(i)} = \sum_{i=1}^{n} y^{(i)}$

$w_0 \sum_{i=1}^{n} x^{(i)} + w_1 \sum_{i=1}^{n} (x^{(i)})^2 = \sum_{i=1}^{n} y^{(i)}x^{(i)}$

Solving for $w_0$ and $w_1$:

$w_1 = \frac{n\sum_{i=1}^{n} x^{(i)}y^{(i)} - \sum_{i=1}^{n} x^{(i)}\sum_{i=1}^{n} y^{(i)}}{n\sum_{i=1}^{n} (x^{(i)})^2 - (\sum_{i=1}^{n} x^{(i)})^2}$

$w_0 = \frac{1}{n}(\sum_{i=1}^{n} y^{(i)} - w_1 \sum_{i=1}^{n} x^{(i)}) = \bar{y} - w_1 \bar{x}$

## Linear regression: multivariate (multiple)

Model: $f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 x_1 + \ldots + w_d x_d$

$\boldsymbol{x} = [x_1, x_2, \ldots, x_d]^T$ and $\boldsymbol{w} = [w_0, w_1, \ldots, w_d]^T$

The training data consists of feature vectors $\boldsymbol{x}^{(i)}$ and target values $y^{(i)}$

Sum of Squares Error:
$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - w_0 - w_1 x_1^{(i)} - \ldots - w_d x_d^{(i)})^2$

Find the $\boldsymbol{w}$ that minimizes $J(\boldsymbol{w})$

## Linear regression: multivariate form

We can rewrite the model in matrix form:

$$f(\boldsymbol{x}; \boldsymbol{w}) = \boldsymbol{w}^T \boldsymbol{x}'$$

where $\boldsymbol{x}' = [1, x_1, x_2, \ldots, x_d]^T$ is the augmented feature vector.

Let's define the input matrix $\boldsymbol{X}$:

$$\boldsymbol{X} = \begin{bmatrix} 
1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_d^{(1)} \\
1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_d^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(n)} & x_2^{(n)} & \cdots & x_d^{(n)}
\end{bmatrix}$$

and the target vector $$\boldsymbol{y} = [y^{(1)}, y^{(2)}, \ldots, y^{(n)}]^T$$

## Linear regression: matrix formulation

SSE cost function in matrix form:

$$J(\boldsymbol{w}) = (\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w})^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w})$$

Gradient w.r.t. $\boldsymbol{w}$:

$$\nabla_{\boldsymbol{w}} J(\boldsymbol{w}) = -2\boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w})$$

Setting the gradient to zero:

$$\boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}) = \boldsymbol{0}$$

$$\boldsymbol{X}^T\boldsymbol{y} - \boldsymbol{X}^T\boldsymbol{X}\boldsymbol{w} = \boldsymbol{0}$$

$$\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{w} = \boldsymbol{X}^T\boldsymbol{y}$$

These are called the normal equations.

## Cost function: matrix notation

The cost function can be expressed in matrix notation:

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - f(\boldsymbol{x}^{(i)}; \boldsymbol{w}))^2 = $$

$$= \sum_{i=1}^{n} (y^{(i)} - \boldsymbol{w}^T\boldsymbol{x}^{(i)})^2$$

We can define the vectors and matrices:

$$\boldsymbol{y} = \begin{bmatrix} y^{(1)} \\ \vdots \\ y^{(n)} \end{bmatrix}, 
\boldsymbol{X} = \begin{bmatrix} 
1 & x_1^{(1)} & \cdots & x_d^{(1)} \\
1 & x_1^{(2)} & \cdots & x_d^{(2)} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(n)} & \cdots & x_d^{(n)}
\end{bmatrix}, 
\boldsymbol{w} = \begin{bmatrix} w_0 \\ w_1 \\ \vdots \\ w_d \end{bmatrix}$$

This allows us to write the cost function as:

$$J(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|^2$$

## Normal equations: solution

Assuming $\boldsymbol{X}^T\boldsymbol{X}$ is invertible, the solution is:

$\boldsymbol{w} = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}$

This is the unique global minimizer of the SSE cost function.

Properties:
- Closed-form solution
- No learning rate or iterations needed
- Need to compute $(\boldsymbol{X}^T\boldsymbol{X})^{-1}$ which is $O(d^3)$

If $\boldsymbol{X}^T\boldsymbol{X}$ is not invertible:
- Features are linearly dependent (multicollinearity)
- More features than training examples ($d > n$)

Solutions: regularization, feature selection, etc.

## Cost function and optimal linear model

![Cost Function and Optimal Linear Model](Codes/plots/cost_function_3d_model.png)

In this visualization, we see:
- A 3D plot showing feature space ($x_1$ and $x_2$) and target values ($y$)
- Training data points (circles)
- The optimal linear model (blue plane) that minimizes the sum of squared errors
- Red lines representing the residuals (prediction errors)

The necessary conditions for the "optimal" parameter values:

$$\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) = \boldsymbol{0}$$

This gives us a system of $d + 1$ linear equations that we can solve to find the optimal parameters.

## Example: Housing price prediction

In this example, we apply linear regression to predict house prices based on house size:

$$f(x; w_0, w_1) = w_0 + w_1 x$$

(for a fixed set of parameters, this is a function of $x$)

$$J(w_0, w_1)$$

(function of the parameters $w_0,w_1$)

where:
- $x$ represents the house size (in feet²)
- $f(x; w_0, w_1)$ is the predicted price (in $1000s$)
- $w_0$ is the intercept
- $w_1$ is the slope (price change per unit of size)

The visualizations show:
- Left plots: Training data points (red x marks) and the linear hypothesis (blue line)
- Right plots: Contour lines of the cost function $J(w_0, w_1)$ with red x marks showing parameters tried during optimization

As we adjust the parameter values, we can observe different fits to the data:
1. First model: A horizontal line hypothesis (with slope near zero) does not capture the trend in the data
2. Second model: A line with negative slope fits the data better than the horizontal line but still doesn't capture the trend correctly
3. Final model: The optimal fit with positive slope captures the general trend of increasing price with increasing size

These examples from Andrew Ng's Stanford ML course illustrate how different parameter choices affect the fit of our linear model, and how finding the minimum of the cost function leads to the best fit.

## Example: simple linear regression

Consider the simple case with one feature:

$f(x; w_0, w_1) = w_0 + w_1 x$

$\boldsymbol{X} = \begin{bmatrix} 
1 & x^{(1)} \\
1 & x^{(2)} \\
\vdots & \vdots \\
1 & x^{(n)}
\end{bmatrix}, \boldsymbol{w} = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}, \boldsymbol{y} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(n)} \end{bmatrix}$

$\boldsymbol{X}^T\boldsymbol{X} = \begin{bmatrix} 
n & \sum_{i=1}^{n} x^{(i)} \\
\sum_{i=1}^{n} x^{(i)} & \sum_{i=1}^{n} (x^{(i)})^2
\end{bmatrix}$

$\boldsymbol{X}^T\boldsymbol{y} = \begin{bmatrix} 
\sum_{i=1}^{n} y^{(i)} \\
\sum_{i=1}^{n} x^{(i)}y^{(i)}
\end{bmatrix}$

## Example: simple linear regression (cont.)

Computing the solution:

$\boldsymbol{w} = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}$

$w_1 = \frac{n\sum_{i=1}^{n} x^{(i)}y^{(i)} - \sum_{i=1}^{n} x^{(i)}\sum_{i=1}^{n} y^{(i)}}{n\sum_{i=1}^{n} (x^{(i)})^2 - (\sum_{i=1}^{n} x^{(i)})^2}$

$w_0 = \frac{1}{n}(\sum_{i=1}^{n} y^{(i)} - w_1 \sum_{i=1}^{n} x^{(i)}) = \bar{y} - w_1 \bar{x}$

Note:
- $w_1$ captures the correlation between $x$ and $y$
- $w_0$ shifts the line to pass through the point $(\bar{x}, \bar{y})$

## Linear regression in higher dimensions

The same approach works for multiple features:

$f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_d x_d$

Solution via normal equations:

$\boldsymbol{w} = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}$

With multiple features:
- The model describes a hyperplane in the feature space
- Each $w_i$ for $i \geq 1$ represents the impact of feature $x_i$ on the prediction
- $w_0$ is the bias or intercept term

## Geometric interpretation: projection

The prediction $\hat{\boldsymbol{y}} = \boldsymbol{X}\boldsymbol{w}$ is the projection of $\boldsymbol{y}$ onto the column space of $\boldsymbol{X}$.

$\hat{\boldsymbol{y}} = \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y} = \boldsymbol{P}\boldsymbol{y}$

where $\boldsymbol{P} = \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T$ is the projection matrix.

Properties:
- $\boldsymbol{P}$ is symmetric: $\boldsymbol{P}^T = \boldsymbol{P}$
- $\boldsymbol{P}$ is idempotent: $\boldsymbol{P}^2 = \boldsymbol{P}$
- The residual $\boldsymbol{y} - \hat{\boldsymbol{y}}$ is orthogonal to the column space of $\boldsymbol{X}$

Geometrically, this means:
- $\hat{\boldsymbol{y}}$ is the point in the column space of $\boldsymbol{X}$ closest to $\boldsymbol{y}$
- The vector $\boldsymbol{y} - \hat{\boldsymbol{y}}$ is perpendicular to the column space
- The sum of squared errors is minimized

In $\mathbb{R}^n$ space (with $n$ observations):
- $\boldsymbol{y}$ is a point in $\mathbb{R}^n$
- The column space of $\boldsymbol{X}$ is a $d+1$ dimensional subspace
- Linear regression finds the projection of $\boldsymbol{y}$ onto this subspace

This interpretation helps visualize why the least squares solution has the properties it does and why it represents the best linear fit to the data.

![Geometric Interpretation](Codes/plots/geometric_interpretation.png)

## Regularization: Ridge regression

When features are correlated or $d$ is large relative to $n$, the normal equations can be ill-conditioned.

Ridge regression adds a penalty on the magnitude of the weights:

$J_{\text{ridge}}(\boldsymbol{w}) = J(\boldsymbol{w}) + \lambda\|\boldsymbol{w}\|_2^2 = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_2^2$

where $\lambda > 0$ is the regularization parameter.

The solution is:

$\boldsymbol{w}_{\text{ridge}} = (\boldsymbol{X}^T\boldsymbol{X} + \lambda\boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{y}$

Benefits:
- Always has a unique solution (since $\boldsymbol{X}^T\boldsymbol{X} + \lambda\boldsymbol{I}$ is invertible)
- Reduces overfitting
- Handles multicollinearity

### How Ridge regression works:
- The L2 penalty term $\lambda\|\boldsymbol{w}\|_2^2$ shrinks all coefficients toward zero
- As $\lambda$ increases, the bias increases and variance decreases
- The regularization effect is stronger for directions with smaller eigenvalues in $\boldsymbol{X}^T\boldsymbol{X}$
- The intercept term $w_0$ is typically not regularized

## Regularization: Lasso regression

Lasso (Least Absolute Shrinkage and Selection Operator) regression uses an L1 penalty:

$J_{\text{lasso}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_1$

where $\|\boldsymbol{w}\|_1 = \sum_{j=0}^{d}|w_j|$ is the L1 norm.

Properties:
- Encourages sparse solutions (many weights exactly zero)
- Performs feature selection
- No closed-form solution (requires optimization algorithms)

### How Lasso regression works:
- The L1 penalty creates a constraint region shaped like a diamond
- This geometry makes it more likely for coefficients to be exactly zero
- Effective when many features have little or no impact on the target
- Computationally more intensive than Ridge regression (typically solved using coordinate descent or LARS)

### Elastic Net regularization:
- Combines both L1 and L2 penalties: $\lambda_1\|\boldsymbol{w}\|_1 + \lambda_2\|\boldsymbol{w}\|_2^2$
- Preserves the feature selection properties of Lasso
- More robust to collinearity than Lasso
- Often outperforms both Ridge and Lasso in practice

Comparison:
- Ridge: shrinks all coefficients toward zero
- Lasso: shrinks some coefficients exactly to zero
- Elastic Net: combines L1 and L2 penalties

### Choosing the regularization parameter:
- Cross-validation is typically used to select $\lambda$
- The regularization path shows how coefficients change with varying $\lambda$
- For high-dimensional data, regularization is often essential

![Regularization Comparison](Codes/plots/regularization_comparison.png)

## Review: Iterative optimization of cost function

- Cost function: $J(\boldsymbol{w})$
- Optimization problem:  $\hat{\boldsymbol{w}} = \underset{\boldsymbol{w}}{\operatorname{argmin}}J(\boldsymbol{w})$

- Steps:
  - Start from $\boldsymbol{w}^0$
  - Repeat
    - Update $\boldsymbol{w}^t$ to $\boldsymbol{w}^{t+1}$ in order to reduce $J$
    - $t \leftarrow t + 1$
  - until we hopefully end up at a minimum

## Review: Gradient descent

-  First-order optimization algorithm to find $\boldsymbol{w}^* = \underset{\boldsymbol{w}}{\operatorname{argmin}}J(\boldsymbol{w})$
   - Also known as "steepest descent"

- In each step, takes steps proportional to the negative of the gradient vector of the function at the current point $\boldsymbol{w}^t$:

$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t - \gamma_t \nabla J(\boldsymbol{w}^t)$$

- $J(\boldsymbol{w})$ decreases fastest if one goes from $\boldsymbol{w}^t$ in the direction of $-\nabla J(\boldsymbol{w}^t)$

- Assumption: $J(\boldsymbol{w})$ is defined and differentiable in a neighborhood of a point $\boldsymbol{w}^t$

#### Gradient ascent
Takes steps proportional to (the positive of) the gradient to find a local maximum of the function

## Review: Problem of gradient descent with non-convex cost functions

When the cost function $J(\boldsymbol{w})$ is not convex, gradient descent may converge to a local minimum rather than the global minimum. This is illustrated in the visualization below, where the algorithm follows the gradient and ends up in one of the local minima, missing the global minimum.

The 3D surface plot shows a non-convex cost function $J(\boldsymbol{w}_0, \boldsymbol{w}_1)$ with multiple local minima. The black path shows how gradient descent might navigate this surface, potentially getting trapped in a local minimum instead of finding the global minimum.

This problem commonly occurs in complex optimization tasks like training deep neural networks, where the loss landscape can be highly non-convex with many local minima and saddle points.

Various techniques have been developed to address this issue:
- Random restarts: Run gradient descent multiple times with different initializations
- Momentum methods: Incorporate information from past gradients to help escape local minima
- Stochastic methods: Add randomness to the optimization process
- Advanced optimizers: Use algorithms specifically designed to handle non-convex optimization

In linear regression with the standard squared error cost function, this is fortunately not an issue because the cost function is convex, which guarantees that gradient descent will converge to the global minimum.

## Gradient Descent for Linear Regression

When $n$ or $d$ is large, inverting matrices becomes computationally expensive.

Gradient descent is an iterative optimization algorithm:

1. Initialize $\boldsymbol{w}$ (often to zeros or small random values)
2. Repeat until convergence:
   $\boldsymbol{w} := \boldsymbol{w} - \alpha \nabla_{\boldsymbol{w}}J(\boldsymbol{w})$
   where $\alpha > 0$ is the learning rate

For linear regression with SSE:
$\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) = -2\boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w})$

Update rule:
$\boldsymbol{w} := \boldsymbol{w} + 2\alpha\boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w})$

### Gradient Descent Variants:

#### Batch gradient descent:
- Uses all examples for each update
- Follows the true gradient direction
- Computationally expensive for large datasets
- Guaranteed to converge to global minimum for convex functions (like linear regression)
- Update rule: $\boldsymbol{w} := \boldsymbol{w} - \alpha \frac{1}{n}\sum_{i=1}^{n}\nabla_{\boldsymbol{w}}(y^{(i)} - \boldsymbol{w}^T\boldsymbol{x}^{(i)})^2$

#### Stochastic gradient descent (SGD):
- Uses one randomly selected example for each update
- Much faster per iteration but noisier updates
- May never converge exactly, but oscillates around the minimum
- Better for very large datasets
- Update rule: $\boldsymbol{w} := \boldsymbol{w} - \alpha \nabla_{\boldsymbol{w}}(y^{(i)} - \boldsymbol{w}^T\boldsymbol{x}^{(i)})^2$

#### Mini-batch gradient descent:
- Uses small batches of examples (e.g., 32, 64, 128)
- Compromise between batch and stochastic variants
- Less noisy than SGD but still efficient
- Most common in practice
- Update rule: $\boldsymbol{w} := \boldsymbol{w} - \alpha \frac{1}{b}\sum_{i \in B}\nabla_{\boldsymbol{w}}(y^{(i)} - \boldsymbol{w}^T\boldsymbol{x}^{(i)})^2$

### Practical considerations:
- Learning rate $\alpha$ is a crucial hyperparameter
  - Too small: slow convergence
  - Too large: may diverge
- Learning rate schedules can improve convergence
  - Start with larger $\alpha$ and decrease over time
  - Common schedules: step decay, exponential decay, 1/t decay
- For high-dimensional problems, adaptive methods like Adam, RMSprop, or Adagrad often work better
- Early stopping can be used to prevent overfitting

### Convergence criteria:
- Fixed number of iterations
- Change in parameters below threshold: $\|\boldsymbol{w}_{k+1} - \boldsymbol{w}_k\| < \epsilon$
- Change in cost function below threshold: $|J(\boldsymbol{w}_{k+1}) - J(\boldsymbol{w}_k)| < \epsilon$
- Gradient magnitude below threshold: $\|\nabla_{\boldsymbol{w}}J(\boldsymbol{w})\| < \epsilon$

![Gradient Descent](Codes/plots/gradient_descent.png)
![Gradient Descent Fitting](Codes/plots/gradient_descent_fitting.png)

## Cost function: 3D visualization

The left plot shows the training data points of housing prices vs. size. The right plot is a 3D surface visualization of the cost function $J(w_0, w_1)$ that shows how the cost varies with different values of the parameters $w_0$ and $w_1$.

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - w_0 - w_1 x^{(i)})^2$$

The 3D plot demonstrates that the cost function has a convex shape, which means it has a single global minimum without local minima where gradient descent might get stuck.

![Cost Function 3D Surface](Codes/plots/cost_function_3d_surface.png)

## Stochastic gradient descent

### Batch techniques process the entire training set in one go
- thus they can be computationally costly for large data sets.

### Stochastic gradient descent: when the cost function can comprise a sum over data points:

$$J(\boldsymbol{w}) = \sum_{i=1}^n J^{(i)}(\boldsymbol{w})$$

### Update after presentation of $(\boldsymbol{x}^{(i)}, y^{(i)})$:

$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t - \eta\nabla_{\boldsymbol{w}}J^{(i)}(\boldsymbol{w})$$

### Example: Linear regression with SSE cost function

For a single training example, the cost function is:

$$J^{(i)}(\boldsymbol{w}) = (y^{(i)} - \boldsymbol{w}^T \boldsymbol{x}^{(i)})^2$$

The weight update using stochastic gradient descent becomes:

$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t - \eta\nabla_{\boldsymbol{w}}J^{(i)}(\boldsymbol{w})$$

When we calculate the gradient and simplify, we get the Least Mean Squares (LMS) update rule:

$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t + \eta(y^{(i)} - \boldsymbol{w}^T\boldsymbol{x}^{(i)})\boldsymbol{x}^{(i)}$$

This update rule is known as the Least Mean Squares (LMS) algorithm.

This form of update is especially suited for sequential or online learning, where parameters are updated after seeing each training example rather than processing the entire dataset at once. It is proper for sequential or online learning scenarios where data arrives in a stream and we want to continuously adapt our model.

## Evaluation and generalization

### Why minimizing the cost function (based on only training data) while we are interested in the performance on new examples?

$$\min_{\theta} \sum_{i=1}^{n} Loss \left(y^{(i)}, f(\boldsymbol{x}^{(i)}; \theta) \right) \longrightarrow \text{Empirical loss}$$

### Evaluation: After training, we need to measure how well the learned prediction function can predicts the target for unseen examples 

## Training and test performance

### Assumption: training and test examples are drawn independently at random from the same but unknown distribution.
- Each training/test example $(\boldsymbol{x}, y)$ is a sample from joint probability distribution $P(\boldsymbol{x}, y)$, i.e., $(\boldsymbol{x}, y) \sim P$

$$\text{Empirical (training) loss} = \frac{1}{n}\sum_{i=1}^{n} Loss \left(y^{(i)}, f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}) \right)$$

$$\text{Expected (test) loss} = E_{\boldsymbol{x},y} \{Loss(y, f(\boldsymbol{x}; \boldsymbol{\theta}))\}$$

### We minimize empirical loss (on the training data) and expect to also find an acceptable expected loss
- Empirical loss as a proxy for the performance over the whole distribution.

## Linear regression: generalization

### By increasing the number of training examples, will solution be better?

### Why the mean squared error does not decrease more after reaching a level?

![MSE vs Number of Training Examples](Codes/plots/mse_vs_training_size.png)

### Structural error: $E_{\boldsymbol{x},y} \left[ \left(y - \boldsymbol{w}^{*T} \boldsymbol{x}\right)^2 \right]$

where $\boldsymbol{w}^* = (w_0^*, \cdots, w_d^*)$ are the optimal linear regression parameters (infinite training data)

## Linear regression: types of errors

### Structural error: the error introduced by the limited function class (infinite training data):

$$\boldsymbol{w}^* = \text{argmin}_{\boldsymbol{w}} E_{\boldsymbol{x},y}[(y - \boldsymbol{w}^T \boldsymbol{x})^2]$$

$$\text{Structural error}: E_{\boldsymbol{x},y} \left[ \left(y - \boldsymbol{w}^{*T} \boldsymbol{x}\right)^2 \right]$$

where $\boldsymbol{w}^* = (w_0^*, \cdots, w_d^*)$ are the optimal linear regression parameters (infinite training data)

## Stochastic gradient descent: online learning

### Sequential learning is also appropriate for real-time applications
- data observations are arriving in a continuous stream
- and predictions must be made before seeing all of the data

### The value of $\eta$ needs to be chosen with care to ensure that the algorithm converges

## Minimizing cost function

Optimal linear weight vector (for SSE cost function):

$$J(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|^2$$

$$\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) = -2\boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w})$$

$$\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) = \boldsymbol{0} \Rightarrow \boldsymbol{X}^T\boldsymbol{X}\boldsymbol{w} = \boldsymbol{X}^T\boldsymbol{y}$$

$$\boldsymbol{w} = (\boldsymbol{X}^T\boldsymbol{X})^{-1} \boldsymbol{X}^T\boldsymbol{y}$$

$$\boldsymbol{w} = \boldsymbol{X}^{\dagger}\boldsymbol{y}$$

$$\boldsymbol{X}^{\dagger} = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T$$

$$\boldsymbol{X}^{\dagger} \text{ is pseudo inverse of } \boldsymbol{X}$$

## Review: Gradient descent

Minimize $J(\boldsymbol{w})$

$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t - \eta\nabla_{\boldsymbol{w}}J(\boldsymbol{w}^t)$$

$$\nabla_{\boldsymbol{w}}J(\boldsymbol{w}) = \left[\frac{\partial J(\boldsymbol{w})}{\partial w_1}, \frac{\partial J(\boldsymbol{w})}{\partial w_2}, ..., \frac{\partial J(\boldsymbol{w})}{\partial w_d}\right]$$

If $\eta$ is small enough, then $J(\boldsymbol{w}^{t+1}) \leq J(\boldsymbol{w}^t)$.

$\eta$ can be allowed to change at every iteration as $\eta_t$.

## Review: Gradient descent disadvantages

- Local minima problem

- However, when $J$ is convex, all local minima are also global minima ⇒ gradient descent can converge to the global solution. 

## Cost function: multivariate

- We have to minimize the empirical squared loss:

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - f(\boldsymbol{x}^{(i)}; \boldsymbol{w}))^2$$

$$f(\boldsymbol{x}, \boldsymbol{w}) = w_0 + w_1 x_1 + ... w_d x_d$$

$$\boldsymbol{w} = [w_0, w_1, ..., w_d]^T$$

$$\hat{\boldsymbol{w}} = \underset{\boldsymbol{w}\in\mathbb{R}^{d+1}}{\operatorname{argmin}}J(\boldsymbol{w})$$

## Another approach for optimizing the sum squared error

### Iterative approach for solving the following optimization problem:

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - f(\boldsymbol{x}^{(i)}; \boldsymbol{w}))^2$$

Where:
$$f(\boldsymbol{x}^{(i)}; \boldsymbol{w}) = \boldsymbol{w}^T\boldsymbol{x}^{(i)}$$

This approach uses gradient descent to iteratively improve the parameter values until convergence to the optimal solution. 

## Gradient descent for SSE cost function

### Minimize $J(\boldsymbol{w})$

$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t - \eta\nabla_{\boldsymbol{w}}J(\boldsymbol{w}^t)$$

### $J(\boldsymbol{w})$: Sum of squares error

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - f(\boldsymbol{x}^{(i)}; \boldsymbol{w}))^2$$

### Weight update rule for $f(\boldsymbol{x}; \boldsymbol{w}) = \boldsymbol{w}^T \boldsymbol{x}$:

$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t + \eta \sum_{i=1}^{n} (y^{(i)} - \boldsymbol{w}^{t T} \boldsymbol{x}^{(i)}) \boldsymbol{x}^{(i)}$$

This is known as batch mode gradient descent, as each step considers all training data. 

### Learning rate considerations:

- $\eta$: too small → gradient descent can be slow.
- $\eta$: too large → gradient descent can overshoot the minimum. It may fail to converge, or even diverge.

The choice of learning rate is crucial for effective optimization:
- If too small, convergence will be unnecessarily slow
- If too large, the algorithm may oscillate around the minimum or diverge completely
- Adaptive learning rate schedules often provide better performance

## Review: First-order optimization algorithm

### First-order optimization algorithm to find $\boldsymbol{w}^* = \underset{\boldsymbol{w}}{\operatorname{argmin}}J(\boldsymbol{w})$

- Also known as "steepest descent"

### In each step, takes steps proportional to the negative of the gradient vector of the function at the current point $\boldsymbol{w}^t$:

$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t - \gamma_t \nabla J(\boldsymbol{w}^t)$$

- $J(\boldsymbol{w})$ decreases fastest if one goes from $\boldsymbol{w}^t$ in the direction of $-\nabla J(\boldsymbol{w}^t)$
- Assumption: $J(\boldsymbol{w})$ is defined and differentiable in a neighborhood of a point $\boldsymbol{w}^t$

#### Gradient ascent
Takes steps proportional to (the positive of) the gradient to find a local maximum of the function