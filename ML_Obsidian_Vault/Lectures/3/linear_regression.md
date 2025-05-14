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

## 3. Cost Function and Optimization

### Squared Error Loss

For regression, we typically use squared error to measure prediction quality:

$$\text{Squared error} = \left(y^{(i)} - f(x^{(i)}; \boldsymbol{w})\right)^2$$

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

### Gradient Descent for Optimization

When inverting matrices is computationally expensive, we can use gradient descent:

$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t - \alpha \nabla_{\boldsymbol{w}}J(\boldsymbol{w}^t)$$

For linear regression with SSE:
$$\boldsymbol{w}^{t+1} = \boldsymbol{w}^t + 2\alpha\boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}^t)$$

**Variants of Gradient Descent**:
- **Batch**: Uses all examples for each update
- **Stochastic**: Uses one randomly selected example per update
- **Mini-batch**: Uses small batches of examples

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

### Best Unrestricted Regression Function

If we know the joint distribution $P(\boldsymbol{x},y)$ and have no constraints:

$$h^*(\boldsymbol{x}) = \mathbb{E}_{y|\boldsymbol{x}}[y]$$

This minimizes the expected squared error.

## 5. Beyond Linear Regression

### Generalized Linear Models

These use linear combinations of fixed non-linear functions:

$$f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 \phi_1(\boldsymbol{x}) + \ldots w_m \phi_m(\boldsymbol{x})$$

Where $\{\phi_1(\boldsymbol{x}), \ldots, \phi_m(\boldsymbol{x})\}$ are basis functions.

### Polynomial Regression

Using polynomial basis functions:

$$f(x; \boldsymbol{w}) = w_0 + w_1 x + \ldots + w_{m-1} x^{m-1} + w_m x^m$$

Optimization follows the same approach as linear regression, but with transformed inputs.

### Radial Basis Functions

Make predictions based on similarity to "prototypes":

$$\phi_j(\boldsymbol{x}) = \exp\left\{-\frac{1}{2\sigma_j^2}\|\boldsymbol{x} - \boldsymbol{c}_j\|^2\right\}$$

## 6. Regularization

### Ridge Regression (L2 Regularization)

Adds a penalty on weight magnitudes:

$$J_{\text{ridge}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_2^2$$

Solution:
$$\boldsymbol{w}_{\text{ridge}} = (\boldsymbol{X}^T\boldsymbol{X} + \lambda\boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{y}$$

### Lasso Regression (L1 Regularization)

Uses an L1 penalty for sparse solutions:

$$J_{\text{lasso}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_1$$

### Regularization Effects

- Prevents overfitting
- Handles collinearity
- Shrinks coefficients toward zero
- Balances bias and variance

## 7. Model Evaluation and Selection

### Training vs. Test Performance

- **Training error**: $\frac{1}{n}\sum_{i=1}^{n} Loss(y^{(i)}, f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}))$
- **Expected (test) error**: $E_{\boldsymbol{x},y} \{Loss(y, f(\boldsymbol{x}; \boldsymbol{\theta}))\}$

### Simple Hold-Out Method

- Divide data into training, validation, and test sets
- Train models on training set
- Select best model based on validation performance
- Evaluate final model on test set

### Cross-Validation (CV)

#### K-Fold CV:
1. Partition data into K equal-sized subsets
2. For each subset:
   - Train on K-1 subsets
   - Validate on the remaining subset
3. Average performance across all K runs

#### Leave-One-Out CV (LOOCV):
- Special case where K equals the number of samples
- Useful for small datasets

## 8. Overfitting and Generalization

### Causes of Overfitting

- Model complexity exceeding what data supports
- Insufficient training data
- Noise in the data

### Strategies to Avoid Overfitting

1. **Select appropriate model complexity**
   - Cross-validation
   - Validation curves

2. **Regularization**
   - Explicit preference for simpler models
   - Penalty terms in cost function

3. **More training data**
   - Overfitting becomes less severe with more data

### Error Decomposition

The expected error consists of:

- **Structural error**: The inherent limitation of the model class
- **Approximation error**: Error due to limited training data