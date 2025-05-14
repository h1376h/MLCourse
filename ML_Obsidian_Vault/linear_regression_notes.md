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

## Cost function: 3D visualization

The left plot shows the training data points of housing prices vs. size. The right plot is a 3D surface visualization of the cost function $J(w_0, w_1)$ that shows how the cost varies with different values of the parameters $w_0$ and $w_1$.

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - w_0 - w_1 x^{(i)})^2$$

The 3D plot demonstrates that the cost function has a convex shape, which means it has a single global minimum without local minima where gradient descent might get stuck.

![Cost Function 3D Surface](Codes/plots/cost_function_3d_surface.png)

## Evaluation and generalization

### Why minimizing the cost function (based on only training data) while we are interested in the performance on new examples?

$$\min_{\theta} \sum_{i=1}^{n} Loss \left(y^{(i)}, f(x^{(i)}; \theta) \right) \longrightarrow \text{Empirical loss}$$

### Evaluation: After training, we need to measure how well the learned prediction function can predicts the target for unseen examples 

## Training and test performance

### Assumption: training and test examples are drawn independently at random from the same but unknown distribution.
- Each training/test example $(x, y)$ is a sample from joint probability distribution $P(x, y)$, i.e., $(x, y) \sim P$

$$\text{Empirical (training) loss} = \frac{1}{n}\sum_{i=1}^{n} Loss \left(y^{(i)}, f(x^{(i)}; \theta) \right)$$

$$\text{Expected (test) loss} = E_{x,y} \{Loss(y, f(x; \theta))\}$$

### We minimize empirical loss (on the training data) and expect to also find an acceptable expected loss
- Empirical loss as a proxy for the performance over the whole distribution. 