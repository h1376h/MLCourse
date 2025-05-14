# Linear Regression

## 1. Introduction to Linear Regression

### Linear Regression Models

Linear regression models map inputs to real-valued outputs:

**Univariate**:
$$f : \mathbb{R} \rightarrow \mathbb{R} \quad f(x; \boldsymbol{w}) = w_0 + w_1 x$$

**Multivariate**:
$$f : \mathbb{R}^d \rightarrow \mathbb{R} \quad f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 x_1 + \ldots + w_d x_d$$

Here, $\boldsymbol{w} = [w_0, w_1, \ldots, w_d]^T$ are the parameters we need to determine.

### Regression Problem

The goal is to make real-valued predictions given features.

Example: predicting house price from attributes

| Size (m²) | Age (year) | Region | Price (10⁶T) |
|-----------|------------|--------|--------------|
| 100       | 2          | 5      | 500          |
| 80        | 25         | 3      | 250          |
| ...       | ...        | ...    | ...          |

### Learning Process

1. **Select hypothesis space**: a set of mappings from feature vectors to targets
2. **Learning (estimation)**: optimize a cost function based on training data
3. **Evaluation**: measure how well the function generalizes to unseen examples

Key insight: The learning algorithm aims to find a function that not only fits the training data well but also generalizes to new, unseen examples.

### The Learning Diagram

Two key perspectives on the learning process:

**Deterministic Target**: 
- An unknown target function h(x) maps inputs to outputs
- Training examples come from this function
- The learning algorithm with hypothesis set H produces a final hypothesis

**Noisy Target**:
- Target includes both a function h(x) and noise
- Joint probability distribution P(x,y) = P(x)P(y|x)
- Learning involves capturing the target distribution and handling noise

This diagram helps visualize how the learning process works with both deterministic and probabilistic perspectives on the target function.

## 2. Hypothesis Space and Bias-Variance Tradeoff

### Hypothesis Space Complexity

- **Simple models** (e.g., constant function $f(x) = b$):
  - Lower variance but higher bias
  - Limited ability to capture complex patterns
  - Straight horizontal line in visualization

- **Complex models** (e.g., higher degree polynomials):
  - Higher variance but lower bias
  - Can fit training data more closely
  - May overfit with limited data

The key principle: **Match model complexity to the data sources, not to the complexity of the target function.**

This principle is crucial because:
- With limited data, simpler models often generalize better
- Even if the true underlying function is complex, without sufficient data to constrain it, a complex model will fit noise rather than signal
- The goal is finding the right balance between underfitting (too simple) and overfitting (too complex)

### Error Decomposition

The expected squared error between a model and the true function can be decomposed:

$$\mathbb{E}_{\mathcal{D}} \left[ \left(f_{\mathcal{D}}(\boldsymbol{x}) - h(\boldsymbol{x})\right)^2 \right] = \underbrace{\mathbb{E}_{\mathcal{D}} \left[ \left(f_{\mathcal{D}}(\boldsymbol{x}) - \bar{f}(\boldsymbol{x})\right)^2 \right]}_{\text{variance}} + \underbrace{\left(\bar{f}(\boldsymbol{x}) - h(\boldsymbol{x})\right)^2}_{\text{bias}^2}$$

Where:
- $f_{\mathcal{D}}(\boldsymbol{x})$ is the model trained on dataset $\mathcal{D}$
- $\bar{f}(\boldsymbol{x})$ is the average prediction across all possible datasets
- $h(\boldsymbol{x})$ is the true function

The total expected error includes a third component - irreducible noise:

Total expected error = Bias² + Variance + Noise

Where noise represents the inherent randomness in the data that cannot be modeled.

### Visual Interpretation

We can visualize the bias-variance tradeoff by considering models trained on different datasets:

- For a constant model ($f(x) = b$):
  - Different datasets result in different horizontal lines
  - These lines have little variation (low variance)
  - But they systematically miss the true pattern (high bias)

- For a linear model ($f(x) = ax + b$):
  - Different datasets result in lines with varying slopes and intercepts
  - There's more variation between these lines (higher variance)
  - But on average they better approximate the true function (lower bias)

### Bias-Variance Tradeoff

- **Bias**: How far the average prediction is from the true function
- **Variance**: How much predictions vary across different datasets
- **Tradeoff**: More complex models have lower bias but higher variance

This tradeoff is fundamental in machine learning:
- Simple models (e.g., constant function): high bias, low variance
- Complex models (e.g., high-degree polynomials): low bias, high variance
- The optimal model complexity minimizes the sum of bias squared and variance

### Example: Fitting a Sine Function with Limited Data

Consider trying to learn a sine function with only two data points:

- Using a constant model $\mathcal{H}_0: f(x) = b$ (horizontal line)
  - bias = 0.50, variance = 0.25, total error = 0.75
  - High bias but very low variance

- Using a linear model $\mathcal{H}_1: f(x) = ax + b$ (straight line)
  - bias = 0.21, variance = 1.69, total error = 1.90
  - Lower bias but much higher variance

Despite having lower bias, the linear model has significantly higher total error due to its high variance. With just two training points, the simpler constant model actually generalizes better to the sine function.

This illustrates why we should match model complexity to available data rather than to the complexity of the target function. With very limited data, simpler models with lower variance often perform better despite having higher bias.

### Expected Training and Test Error Curves

As the number of training samples increases:

For a simple model:
- Test error starts high and gradually decreases
- Training error starts low and gradually increases
- Both eventually converge to a value above zero (limited by model bias)

For a complex model:
- Test error starts very high but can eventually reach lower values
- Training error remains low throughout
- The gap between training and test error is much larger
- Requires more data to achieve good generalization

### Best Unrestricted Regression Function

If we know the joint distribution $P(\boldsymbol{x},y)$ and have no constraints:

$$h^*(\boldsymbol{x}) = \mathbb{E}_{y|\boldsymbol{x}}[y]$$

This minimizes the expected squared error.

Proof:
$$\mathbb{E}_{\boldsymbol{x},y} \left[ (y - h(\boldsymbol{x}))^2 \right] = \iint (y - h(\boldsymbol{x}))^2 p(\boldsymbol{x}, y) d\boldsymbol{x}dy$$

For each $\boldsymbol{x}$ separately minimize loss:
$$\frac{\delta\mathbb{E}_{\boldsymbol{x},y} \left[ (y - h(\boldsymbol{x}))^2 \right]}{\delta h(\boldsymbol{x})} = \int 2(y - h(\boldsymbol{x}))p(\boldsymbol{x}, y)dy = 0$$

$$\Rightarrow h^*(\boldsymbol{x}) = \frac{\int yp(\boldsymbol{x}, y)dy}{\int p(\boldsymbol{x}, y)dy} = \frac{\int yp(\boldsymbol{x}, y)dy}{p(\boldsymbol{x})} = \int yp(y|\boldsymbol{x})dy = \mathbb{E}_{y|\boldsymbol{x}}[y]$$

## 3. Cost Function and Optimization

### Squared Error Loss

For regression, we typically use squared error to measure prediction quality:

$$\text{Squared error} = \left(y^{(i)} - f(x^{(i)}; \boldsymbol{w})\right)^2$$

This measures the squared vertical distance between the prediction and the actual value.

### Sum of Squares Error (SSE) Cost Function

$$J(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - f(\boldsymbol{x}^{(i)}; \boldsymbol{w}))^2$$

This sums the squared errors across all training examples.

### Matrix Formulation

We can express this in matrix form:

$$\boldsymbol{y} = \begin{bmatrix} y^{(1)} \\ \vdots \\ y^{(n)} \end{bmatrix}, 
\boldsymbol{X} = \begin{bmatrix} 
1 & x_1^{(1)} & \cdots & x_d^{(1)} \\
1 & x_1^{(2)} & \cdots & x_d^{(2)} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(n)} & \cdots & x_d^{(n)}
\end{bmatrix}, 
\boldsymbol{w} = \begin{bmatrix} w_0 \\ w_1 \\ \vdots \\ w_d \end{bmatrix}$$

The cost function becomes:
$$J(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|^2 = (\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w})^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w})$$

### Visualization of the Cost Function

The cost function $J(\boldsymbol{w})$ for linear regression forms a convex bowl-shaped surface in parameter space. For a simple univariate case with parameters $w_0$ and $w_1$:

- The 3D surface has a single global minimum (no local minima)
- The contour lines form ellipses when viewed from above
- The contour lines are more densely packed in directions where the cost function changes quickly
- The minimum corresponds to the optimal parameter values that give the best fit

This convex nature guarantees that gradient-based methods will converge to the global optimum regardless of initialization.

### Optimization with Normal Equations

Setting the gradient of the cost function to zero:

$$\nabla_{\boldsymbol{w}} J(\boldsymbol{w}) = -2\boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}) = \boldsymbol{0}$$

$$\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{w} = \boldsymbol{X}^T\boldsymbol{y}$$

The solution (when $\boldsymbol{X}^T\boldsymbol{X}$ is invertible) is:

$$\hat{\boldsymbol{w}} = (\boldsymbol{X}^T \boldsymbol{X})^{-1} \boldsymbol{X}^T \boldsymbol{y}$$

This closed-form solution provides the global minimum of the SSE cost function.

For a simple univariate linear regression, this simplifies to:

$$w_1 = \frac{n\sum_{i=1}^{n} x^{(i)}y^{(i)} - \sum_{i=1}^{n} x^{(i)}\sum_{i=1}^{n} y^{(i)}}{n\sum_{i=1}^{n} (x^{(i)})^2 - (\sum_{i=1}^{n} x^{(i)})^2}$$

$$w_0 = \frac{1}{n}\left(\sum_{i=1}^{n} y^{(i)} - w_1 \sum_{i=1}^{n} x^{(i)}\right) = \bar{y} - w_1 \bar{x}$$

### Geometric Interpretation

The prediction $\hat{\boldsymbol{y}} = \boldsymbol{X}\boldsymbol{w}$ is the projection of $\boldsymbol{y}$ onto the column space of $\boldsymbol{X}$:

$$\hat{\boldsymbol{y}} = \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y} = \boldsymbol{P}\boldsymbol{y}$$

where $\boldsymbol{P} = \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T$ is the projection matrix.

Properties of the projection matrix:
- $\boldsymbol{P}$ is symmetric: $\boldsymbol{P}^T = \boldsymbol{P}$
- $\boldsymbol{P}$ is idempotent: $\boldsymbol{P}^2 = \boldsymbol{P}$
- The residual $\boldsymbol{y} - \hat{\boldsymbol{y}}$ is orthogonal to the column space of $\boldsymbol{X}$

Geometrically, this means:
- $\hat{\boldsymbol{y}}$ is the point in the column space of $\boldsymbol{X}$ closest to $\boldsymbol{y}$
- The residual $\boldsymbol{y} - \hat{\boldsymbol{y}}$ is orthogonal to the column space
- The sum of squared errors is minimized

In $\mathbb{R}^n$ space (with $n$ observations):
- $\boldsymbol{y}$ is a point in $\mathbb{R}^n$
- The column space of $\boldsymbol{X}$ is a $d+1$ dimensional subspace
- Linear regression finds the projection of $\boldsymbol{y}$ onto this subspace

### Gradient Descent for Optimization

When inverting matrices is computationally expensive, we can use gradient descent:

$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t - \alpha \nabla_{\boldsymbol{w}}J(\boldsymbol{w}^t)$$

For linear regression with SSE:
$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t + 2\alpha\boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}^t)$$

**Variants of Gradient Descent**:
- **Batch**: Uses all examples for each update
  - Follows the true gradient direction
  - Computationally expensive for large datasets
  - Update rule: $\boldsymbol{w} := \boldsymbol{w} - \alpha \frac{1}{n}\sum_{i=1}^{n}\nabla_{\boldsymbol{w}}(y^{(i)} - \boldsymbol{w}^T\boldsymbol{x}^{(i)})^2$

- **Stochastic**: Uses one randomly selected example per update
  - Much faster per iteration but noisier updates
  - May never converge exactly, but oscillates around the minimum
  - Better for very large datasets
  - Update rule: $\boldsymbol{w} := \boldsymbol{w} - \alpha \nabla_{\boldsymbol{w}}(y^{(i)} - \boldsymbol{w}^T\boldsymbol{x}^{(i)})^2$
  - Known as the Least Mean Squares (LMS) algorithm when applied to linear regression

- **Mini-batch**: Uses small batches of examples (e.g., 32, 64, 128)
  - Compromise between batch and stochastic variants
  - Less noisy than SGD but still efficient
  - Most common in practice
  - Update rule: $\boldsymbol{w} := \boldsymbol{w} - \alpha \frac{1}{b}\sum_{i \in B}\nabla_{\boldsymbol{w}}(y^{(i)} - \boldsymbol{w}^T\boldsymbol{x}^{(i)})^2$

Learning rate considerations:
- Too small: slow convergence
- Too large: may diverge or oscillate
- Learning rate schedules can improve convergence
  - Start with larger $\alpha$ and decrease over time
  - Common schedules: step decay, exponential decay, 1/t decay

### Sequential or Online Learning

Stochastic gradient descent makes it well-suited for online learning scenarios:
- Data observations arrive in a continuous stream
- Predictions must be made before seeing all data
- Parameters are updated after each new observation
- The model continuously adapts to new patterns

For real-time applications with streaming data, SGD with the LMS update rule provides an effective way to perform continuous learning as new data becomes available.

## 4. Beyond Linear Regression

### Generalized Linear Models

These use linear combinations of fixed non-linear functions:

$$f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 \phi_1(\boldsymbol{x}) + \ldots + w_m \phi_m(\boldsymbol{x})$$

Where $\{\phi_1(\boldsymbol{x}), \ldots, \phi_m(\boldsymbol{x})\}$ are basis functions.

The approach to extend linear regression to non-linear functions:
1. Transform the data using basis functions
2. Learn a linear regression on the new feature vectors (obtained by basis functions)

Optimization follows the same approach as linear regression, using the design matrix:

$$\Phi = \begin{bmatrix} 
1 & \phi_1(\boldsymbol{x}^{(1)}) & \cdots & \phi_m(\boldsymbol{x}^{(1)}) \\
1 & \phi_1(\boldsymbol{x}^{(2)}) & \cdots & \phi_m(\boldsymbol{x}^{(2)}) \\
\vdots & \vdots & \ddots & \vdots \\
1 & \phi_1(\boldsymbol{x}^{(n)}) & \cdots & \phi_m(\boldsymbol{x}^{(n)})
\end{bmatrix}$$

The solution is:
$$\hat{\boldsymbol{w}} = (\Phi^T \Phi)^{-1} \Phi^T \boldsymbol{y}$$

### Polynomial Regression

Using polynomial basis functions:

$$f(x; \boldsymbol{w}) = w_0 + w_1 x + \ldots + w_{m-1} x^{m-1} + w_m x^m$$

Visual comparison of polynomial fits with different degrees:
- $m = 1$ (linear): Simple straight line that may underfit
- $m = 3$ (cubic): More flexible curve that captures moderate complexity
- $m = 5$ (quintic): Highly flexible curve that fits training data closely
- $m = 9$ (9th degree): Extremely flexible curve that may oscillate wildly between data points

As the polynomial degree increases:
- Training error decreases consistently
- Test error initially decreases but then increases as the model begins to overfit
- The coefficient magnitudes typically grow larger, often dramatically

For a 9th degree polynomial without regularization, coefficients can reach extreme values like:
- $w_1 = 232.37$
- $w_5 = 640042.26$
- $w_9 = 125201.43$

These large coefficients are a clear sign of overfitting.

### Radial Basis Functions

Make predictions based on similarity to "prototypes":

$$\phi_j(\boldsymbol{x}) = \exp\left\{-\frac{1}{2\sigma_j^2}\|\boldsymbol{x} - \boldsymbol{c}_j\|^2\right\}$$

Where:
- $\boldsymbol{c}_j$ are prototype/center vectors
- $\sigma_j^2$ controls how quickly the influence vanishes with distance
- Training examples themselves could serve as prototypes

Other basis functions include:
- Sigmoid: $\phi_j(\boldsymbol{x}) = \sigma\left(\frac{\|\boldsymbol{x}-\boldsymbol{c}_j\|}{\sigma_j}\right)$ where $\sigma(a) = \frac{1}{1+\exp(-a)}$
- Gaussian: $\phi_j(\boldsymbol{x}) = \exp\left\{-\frac{(\boldsymbol{x}-\boldsymbol{c}_j)^2}{2\sigma_j^2}\right\}$

## 5. Regularization

### Ridge Regression (L2 Regularization)

Adds a penalty on weight magnitudes:

$$J_{\text{ridge}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_2^2$$

Solution:
$$\boldsymbol{w}_{\text{ridge}} = (\boldsymbol{X}^T\boldsymbol{X} + \lambda\boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{y}$$

Properties:
- Always has a unique solution (since $\boldsymbol{X}^T\boldsymbol{X} + \lambda\boldsymbol{I}$ is invertible)
- As $\lambda$ increases, bias increases and variance decreases
- The regularization effect is stronger for directions with smaller eigenvalues
- The intercept term $w_0$ is typically not regularized

Visual effect of regularization on a 9th degree polynomial:
- With strong regularization ($\ln \lambda = 0$): coefficients are close to zero (e.g., $w_1 = -0.05$, $w_9 = 0.01$)
- With moderate regularization ($\ln \lambda = -18$): coefficients are reasonable (e.g., $w_1 = 4.74$, $w_9 = 72.68$)
- Without regularization: coefficients are extremely large (e.g., $w_1 = 232.37$, $w_9 = 125201.43$)

### Lasso Regression (L1 Regularization)

Uses an L1 penalty for sparse solutions:

$$J_{\text{lasso}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_1$$

Properties:
- Encourages sparse solutions (many weights exactly zero)
- Performs feature selection
- No closed-form solution (requires optimization algorithms)
- The L1 penalty creates a constraint region shaped like a diamond
- This geometry makes it more likely for coefficients to be exactly zero

### Elastic Net

Combines both L1 and L2 penalties:
$$J_{\text{elastic}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda_1\|\boldsymbol{w}\|_1 + \lambda_2\|\boldsymbol{w}\|_2^2$$

- Preserves the feature selection properties of Lasso
- More robust to collinearity than Lasso
- Often outperforms both Ridge and Lasso in practice when features are correlated

### Regularization and Bias-Variance Tradeoff

Comparing models with and without regularization:

For a linear model fitting a sine function:
- Without regularization: bias = 0.21, variance = 1.69
- With regularization: bias = 0.23, variance = 0.33

Regularization slightly increases bias but dramatically reduces variance, leading to better overall performance.

### Regularization Effects

- Prevents overfitting
- Handles collinearity
- Shrinks coefficients toward zero
- Balances bias and variance

As the regularization parameter $\lambda$ changes:
- With large $\lambda$: Models are very simple with high bias, low variance
- With small $\lambda$: Models are flexible with low bias, high variance
- Optimal $\lambda$ balances bias and variance for best predictive performance

Different degrees of regularization (via $\lambda$ values) can be visualized:
- With high regularization (large $\lambda$): All models have similar parameters and predictions, showing low variance but high bias
- With intermediate regularization: Models show a balance between consistency and flexibility
- With low regularization (small $\lambda$): Models vary significantly across different training datasets, showing high variance but lower bias

## 6. Model Evaluation and Selection

### Training vs. Test Performance

- **Training error**: $\frac{1}{n}\sum_{i=1}^{n} \text{Loss}(y^{(i)}, f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}))$
- **Expected (test) error**: $E_{\boldsymbol{x},y} \{\text{Loss}(y, f(\boldsymbol{x}; \boldsymbol{\theta}))\}$

The goal is to minimize expected (test) error, but we can only measure training error directly.

Assumption: Training and test examples are drawn independently at random from the same distribution.
- Each sample $(\boldsymbol{x}, y)$ comes from the same probability distribution $P(\boldsymbol{x}, y)$
- We minimize empirical loss on training data to find an acceptable expected loss

### Error Types in Linear Regression

1. **Structural error**: Error due to the limited function class (even with infinite training data):
   $$\text{Structural error}: E_{\boldsymbol{x},y} \left[ \left(y - \boldsymbol{w}^{*T} \boldsymbol{x}\right)^2 \right]$$

2. **Approximation error**: Error due to limited training data:
   $$\text{Approximation error}: E_{\boldsymbol{x}} \left[ \left(\boldsymbol{w}^{*T} \boldsymbol{x} - \hat{\boldsymbol{w}}^T \boldsymbol{x}\right)^2 \right]$$

The expected error can be decomposed into the sum of structural and approximation errors:

$$E_{\boldsymbol{x},y}[(y - \hat{\boldsymbol{w}}^T \boldsymbol{x})^2] = E_{\boldsymbol{x},y} \left[ \left(y - \boldsymbol{w}^{*T} \boldsymbol{x}\right)^2 \right] + E_{\boldsymbol{x}} \left[ \left(\boldsymbol{w}^{*T} \boldsymbol{x} - \hat{\boldsymbol{w}}^T \boldsymbol{x}\right)^2 \right]$$

### Simple Hold-Out Method

- Divide data into training, validation, and test sets
- Train models on training set
- Select best model based on validation performance
- Evaluate final model on test set

Limitations:
- Wasteful of training data
- Small validation set gives noisy performance estimates
- Performance on validation set may be an optimistic estimate of generalization error

### Cross-Validation (CV)

#### K-Fold CV:
1. Shuffle the dataset and partition into K equal-sized subsets
2. For each subset:
   - Train on K-1 subsets
   - Validate on the remaining subset
3. Average performance across all K runs

Steps for model selection with CV:
- For each model complexity (e.g., polynomial degree)
  - Perform K-fold CV and calculate average error
- Select the model with the best average CV performance

Example results from 5-fold CV (100 runs) for polynomial regression:
- $m = 1$ (linear): CV: $\text{MSE} = 0.30$
- $m = 3$ (cubic): CV: $\text{MSE} = 1.45$
- $m = 5$ (quintic): CV: $\text{MSE} = 45.44$
- $m = 7$ (7th degree): CV: $\text{MSE} = 31759$

We see that as complexity increases, CV error eventually grows dramatically due to overfitting.

#### Leave-One-Out CV (LOOCV):
- Special case where K equals the number of samples
- Each example serves as a validation set once
- Useful for small datasets but computationally expensive
- When data is particularly scarce, this maximizes the training set size

### Choosing Regularization Parameter

- Train models with different $\lambda$ values
- Use validation set or cross-validation to select optimal $\lambda$
- As $\lambda$ varies:
  - Large $\lambda$: Both training and test errors high (underfitting)
  - Optimal $\lambda$: Test error minimized
  - Small $\lambda$: Training error low but test error high (overfitting)

Regularization parameter ($\lambda$) selection process:
1. Train models with different $\lambda$ values
2. Compute validation/CV error for each model
3. Select $\lambda$ with lowest validation/CV error
4. Retrain final model on complete training set with selected $\lambda$

## 7. Overfitting and Generalization

### Causes of Overfitting

- Model complexity exceeding what data supports
- Insufficient training data
- Noise in the data

Signs of overfitting:
- Training error ≈ 0 but test error >> 0
- Large gap between training and test performance
- Coefficient values becoming extremely large

### The Effect of Training Data Size

The overfitting problem becomes less severe as the training data size increases:
- For the same complex model (e.g., 9th degree polynomial):
  - With few samples ($n = 15$): Model fits the noise, producing wild oscillations
  - With more samples ($n = 100$): Model fits the underlying pattern better
  - As data increases, even complex models are forced to capture the true pattern

### Strategies to Avoid Overfitting

1. **Select appropriate model complexity**
   - Cross-validation
   - Validation curves
   - Information criteria (AIC, BIC)

2. **Regularization**
   - Explicit preference for simpler models
   - Penalty terms in cost function
   - Weight decay in gradient descent

3. **More training data**
   - Overfitting becomes less severe with more data
   - Same complex model may generalize well with sufficient data

### Expected Training and True Error Curves

As the number of training samples increases:

For a simple model:
- Test error (true error) starts high and gradually decreases
- Training error starts low and gradually increases
- Both eventually converge to a value above zero (limited by model bias)

For a complex model:
- Test error starts very high but can eventually reach lower values
- Training error remains low throughout
- The gap between training and test error is much larger
- Requires more data to achieve good generalization

This illustrates that:
1. Complex models need more data to generalize well
2. Simple models converge faster but may have higher asymptotic error
3. Training error is typically an optimistic estimate of true error, especially for complex models