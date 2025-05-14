# Linear Regression

## 1. Introduction to Linear Regression

### Linear Regression Models

Linear regression models map inputs to real-valued outputs:

**Univariate**:
$$f : \mathbb{R} \rightarrow \mathbb{R} \quad f(x; \boldsymbol{w}) = w_0 + w_1 x$$

**Multivariate**:
$$f : \mathbb{R}^d \rightarrow \mathbb{R} \quad f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 x_1 + \ldots w_d x_d$$

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

## 2. Learning from Training Sets

Different hypothesis models learn patterns from training data:

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
$$J(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|^2$$

### Optimization with Normal Equations

Setting the gradient of the cost function to zero:

$$\nabla_{\boldsymbol{w}} J(\boldsymbol{w}) = -2\boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}) = \boldsymbol{0}$$

$$\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{w} = \boldsymbol{X}^T\boldsymbol{y}$$

The solution (when $\boldsymbol{X}^T\boldsymbol{X}$ is invertible) is:

$$\hat{\boldsymbol{w}} = (\boldsymbol{X}^T \boldsymbol{X})^{-1} \boldsymbol{X}^T \boldsymbol{y}$$

This closed-form solution provides the global minimum of the SSE cost function.

### Geometric Interpretation

The prediction $\hat{\boldsymbol{y}} = \boldsymbol{X}\boldsymbol{w}$ is the projection of $\boldsymbol{y}$ onto the column space of $\boldsymbol{X}$:

$$\hat{\boldsymbol{y}} = \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y} = \boldsymbol{P}\boldsymbol{y}$$

where $\boldsymbol{P} = \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T$ is the projection matrix.

Geometrically, this means the residual $\boldsymbol{y} - \hat{\boldsymbol{y}}$ is orthogonal to the column space of $\boldsymbol{X}$.

### Gradient Descent for Optimization

When inverting matrices is computationally expensive, we can use gradient descent:

$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t - \alpha \nabla_{\boldsymbol{w}}J(\boldsymbol{w}^t)$$

For linear regression with SSE:
$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t + 2\alpha\boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}^t)$$

**Variants of Gradient Descent**:
- **Batch**: Uses all examples for each update
  - Follows the true gradient direction
  - Computationally expensive for large datasets
- **Stochastic**: Uses one randomly selected example per update
  - Much faster per iteration but noisier updates
  - May never converge exactly, but oscillates around the minimum
- **Mini-batch**: Uses small batches of examples
  - Compromise between batch and stochastic variants
  - Less noisy than SGD but still efficient

Learning rate considerations:
- Too small: slow convergence
- Too large: may diverge or oscillate

## 4. Bias-Variance Tradeoff

### Error Decomposition

The expected squared error between a model and the true function can be decomposed:

$$\mathbb{E}_{\mathcal{D}} \left[ \left(f_{\mathcal{D}}(\boldsymbol{x}) - h(\boldsymbol{x})\right)^2 \right] = \underbrace{\mathbb{E}_{\mathcal{D}} \left[ \left(f_{\mathcal{D}}(\boldsymbol{x}) - \bar{f}(\boldsymbol{x})\right)^2 \right]}_{\text{variance}} + \underbrace{\left(\bar{f}(\boldsymbol{x}) - h(\boldsymbol{x})\right)^2}_{\text{bias}^2}$$

Where:
- $f_{\mathcal{D}}(\boldsymbol{x})$ is the model trained on dataset $\mathcal{D}$
- $\bar{f}(\boldsymbol{x})$ is the average prediction across all possible datasets
- $h(\boldsymbol{x})$ is the true function

### Bias-Variance Tradeoff

- **Bias**: How far the average prediction is from the true function
- **Variance**: How much predictions vary across different datasets
- **Tradeoff**: More complex models have lower bias but higher variance

This tradeoff is fundamental in machine learning:
- Simple models (e.g., constant function): high bias, low variance
- Complex models (e.g., high-degree polynomials): low bias, high variance
- The optimal model complexity minimizes the sum of bias squared and variance

### Best Unrestricted Regression Function

If we know the joint distribution $P(\boldsymbol{x},y)$ and have no constraints:

$$h^*(\boldsymbol{x}) = \mathbb{E}_{y|\boldsymbol{x}}[y]$$

This minimizes the expected squared error.

Proof:
$$\mathbb{E}_{\boldsymbol{x},y} \left[ (y - h(\boldsymbol{x}))^2 \right] = \iint (y - h(\boldsymbol{x}))^2 p(\boldsymbol{x}, y) d\boldsymbol{x}dy$$

For each $\boldsymbol{x}$ separately minimize loss:
$$\frac{\delta\mathbb{E}_{\boldsymbol{x},y} \left[ (y - h(\boldsymbol{x}))^2 \right]}{\delta h(\boldsymbol{x})} = \int 2(y - h(\boldsymbol{x}))p(\boldsymbol{x}, y)dy = 0$$

$$\Rightarrow h^*(\boldsymbol{x}) = \mathbb{E}_{y|\boldsymbol{x}}[y]$$

### Error Types in Linear Regression

1. **Structural error**: Error due to the limited function class (even with infinite training data):
   $$\text{Structural error}: E_{\boldsymbol{x},y} \left[ \left(y - \boldsymbol{w}^{*T} \boldsymbol{x}\right)^2 \right]$$

2. **Approximation error**: Error due to limited training data:
   $$\text{Approximation error}: E_{\boldsymbol{x}} \left[ \left(\boldsymbol{w}^{*T} \boldsymbol{x} - \hat{\boldsymbol{w}}^T \boldsymbol{x}\right)^2 \right]$$

The expected error can be decomposed into the sum of structural and approximation errors.

## 5. Beyond Linear Regression

### Generalized Linear Models

These use linear combinations of fixed non-linear functions:

$$f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 \phi_1(\boldsymbol{x}) + \ldots w_m \phi_m(\boldsymbol{x})$$

Where $\{\phi_1(\boldsymbol{x}), \ldots, \phi_m(\boldsymbol{x})\}$ are basis functions.

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

As the polynomial degree increases:
- Training error decreases consistently
- Test error initially decreases but then increases as the model begins to overfit
- The coefficient magnitudes typically grow larger, often dramatically

### Radial Basis Functions

Make predictions based on similarity to "prototypes":

$$\phi_j(\boldsymbol{x}) = \exp\left\{-\frac{1}{2\sigma_j^2}\|\boldsymbol{x} - \boldsymbol{c}_j\|^2\right\}$$

Where:
- $\boldsymbol{c}_j$ are prototype/center vectors
- $\sigma_j^2$ controls how quickly the influence vanishes with distance
- Training examples themselves could serve as prototypes

Other basis functions include:
- Sigmoid: $\phi_j(\boldsymbol{x}) = \sigma\left(\frac{\|\boldsymbol{x}-\boldsymbol{c}_j\|}{\sigma_j}\right) \quad \sigma(a) = \frac{1}{1+\exp(-a)}$

## 6. Regularization

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

### Lasso Regression (L1 Regularization)

Uses an L1 penalty for sparse solutions:

$$J_{\text{lasso}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_1$$

Properties:
- Encourages sparse solutions (many weights exactly zero)
- Performs feature selection
- No closed-form solution (requires optimization algorithms)

### Elastic Net

Combines both L1 and L2 penalties:
$$J_{\text{elastic}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda_1\|\boldsymbol{w}\|_1 + \lambda_2\|\boldsymbol{w}\|_2^2$$

- Preserves the feature selection properties of Lasso
- More robust to collinearity than Lasso

### Regularization Effects

- Prevents overfitting
- Handles collinearity
- Shrinks coefficients toward zero
- Balances bias and variance

As the regularization parameter λ changes:
- With large λ: Models are very simple with high bias, low variance
- With small λ: Models are flexible with low bias, high variance
- Optimal λ balances bias and variance for best predictive performance

## 7. Model Evaluation and Selection

### Training vs. Test Performance

- **Training error**: $\frac{1}{n}\sum_{i=1}^{n} Loss(y^{(i)}, f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}))$
- **Expected (test) error**: $E_{\boldsymbol{x},y} \{Loss(y, f(\boldsymbol{x}; \boldsymbol{\theta}))\}$

The goal is to minimize expected (test) error, but we can only measure training error directly.

### Simple Hold-Out Method

- Divide data into training, validation, and test sets
- Train models on training set
- Select best model based on validation performance
- Evaluate final model on test set

Limitations:
- Wasteful of training data
- Small validation set gives noisy performance estimates

### Cross-Validation (CV)

#### K-Fold CV:
1. Partition data into K equal-sized subsets
2. For each subset:
   - Train on K-1 subsets
   - Validate on the remaining subset
3. Average performance across all K runs

Steps for model selection with CV:
- For each model complexity (e.g., polynomial degree)
  - Perform K-fold CV and calculate average error
- Select the model with the best average CV performance

#### Leave-One-Out CV (LOOCV):
- Special case where K equals the number of samples
- Each example serves as a validation set once
- Useful for small datasets but computationally expensive

### Choosing Regularization Parameter

- Train models with different λ values
- Use validation set or cross-validation to select optimal λ
- As λ varies:
  - Large λ: Both training and test errors high (underfitting)
  - Optimal λ: Test error minimized
  - Small λ: Training error low but test error high (overfitting)

## 8. Overfitting and Generalization

### Causes of Overfitting

- Model complexity exceeding what data supports
- Insufficient training data
- Noise in the data

Signs of overfitting:
- Training error ≈ 0 but test error >> 0
- Large gap between training and test performance
- Coefficient values becoming extremely large

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

### Error Decomposition

The expected error consists of:

- **Structural error**: The inherent limitation of the model class
- **Approximation error**: Error due to limited training data
- **Noise**: Irreducible error inherent in the data

Total expected error = Bias² + Variance + Noise

This decomposition highlights why the bias-variance tradeoff is central to machine learning: minimizing total error requires balancing bias and variance.

## References

- C. Bishop, "Pattern Recognition and Machine Learning", Chapter 1.1, 1.3, 3.1, 3.2.
- Yaser S. Abu-Mostafa, Malik Maghdon-Ismail, and Hsuan Tien Lin, "Learning from Data", Chapter 2.3, 3.2, 3.4.