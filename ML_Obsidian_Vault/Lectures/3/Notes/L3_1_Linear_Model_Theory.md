# Linear Model Theory

## Introduction
Linear models form the foundation of many machine learning algorithms and statistical methods. They provide a straightforward way to model the relationship between input features and output targets through linear combinations of parameters. Their simplicity, interpretability, and theoretical properties make them essential to understand before advancing to more complex models.

## Historical Context
Linear models have a rich history dating back to the early 19th century with the work of Gauss and Legendre on the method of least squares. They became fundamental tools in statistics and later formed the basis for many machine learning approaches. Understanding linear models provides crucial insights into the mathematical foundations that underpin more complex algorithms.

## Mathematical Formulation
A linear model describes the relationship between a dependent variable $y$ and one or more independent variables $\mathbf{x} = (x_1, x_2, \ldots, x_d)$ through a linear function:

$$y = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_d x_d + \epsilon$$

Where:
- $y$ is the target variable (dependent variable)
- $x_1, x_2, \ldots, x_d$ are the features (independent variables)
- $w_0, w_1, \ldots, w_d$ are the model parameters (weights), where $w_0$ is often called the intercept or bias term
- $\epsilon$ is the error term (representing noise or unexplained variation)

## Vector Form
The linear model can be expressed more concisely in vector form:

$$y = \mathbf{w}^T\mathbf{x} + \epsilon$$

Where:
- $\mathbf{w} = (w_0, w_1, \ldots, w_d)^T$ is the weight vector
- $\mathbf{x} = (1, x_1, \ldots, x_d)^T$ is the feature vector with a prepended 1 for the bias term

For multiple observations, we can represent this in matrix form:
$$\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}$$

Where $\mathbf{X}$ is the design matrix with each row representing an observation and each column a feature (with the first column being all 1s for the bias term).

In this notation:
- $\mathbf{y}$ is an $n \times 1$ vector of target values
- $\mathbf{X}$ is an $n \times (d+1)$ design matrix
- $\mathbf{w}$ is a $(d+1) \times 1$ vector of parameters
- $\boldsymbol{\epsilon}$ is an $n \times 1$ vector of error terms

## Linear Algebra Foundations

### Vector Spaces and Subspaces
The linear model can be understood from the perspective of vector spaces:

- The response vector $\mathbf{y}$ exists in an $n$-dimensional space $\mathbb{R}^n$
- The columns of the design matrix $\mathbf{X}$ span a subspace of $\mathbb{R}^n$ known as the column space $C(\mathbf{X})$
- The linear model attempts to approximate $\mathbf{y}$ with a vector that lies in $C(\mathbf{X})$

### Orthogonal Projection
The least squares solution can be viewed as an orthogonal projection of $\mathbf{y}$ onto the column space of $\mathbf{X}$:

- $\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}}$ is the projection of $\mathbf{y}$ onto $C(\mathbf{X})$
- The residual vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to $C(\mathbf{X})$
- This orthogonality is expressed by $\mathbf{X}^T\mathbf{e} = \mathbf{0}$

### The Normal Equations
From the orthogonality principle, we derive the normal equations:

$$\mathbf{X}^T\mathbf{X}\hat{\mathbf{w}} = \mathbf{X}^T\mathbf{y}$$

When $\mathbf{X}^T\mathbf{X}$ is invertible (full rank), the solution is:

$$\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

### Rank and Dimensionality
The rank of $\mathbf{X}$ is crucial for understanding the behavior of linear models:

- If $\text{rank}(\mathbf{X}) = d+1$ (full column rank), then $\mathbf{X}^T\mathbf{X}$ is invertible and a unique solution exists
- If $\text{rank}(\mathbf{X}) < d+1$ (rank deficiency), then $\mathbf{X}^T\mathbf{X}$ is not invertible and infinitely many solutions exist

## The Hat Matrix

The hat matrix (or projection matrix) $\mathbf{H}$ is a fundamental concept in linear model theory:

$$\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$$

It transforms the observed response vector $\mathbf{y}$ into the fitted values $\hat{\mathbf{y}}$:

$$\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$$

### Properties of the Hat Matrix

The hat matrix has several important properties:

1. **Symmetry**: $\mathbf{H} = \mathbf{H}^T$
2. **Idempotence**: $\mathbf{H}\mathbf{H} = \mathbf{H}$ (applying the projection twice doesn't change the result)
3. **Trace**: $\text{trace}(\mathbf{H}) = d+1$ (the number of parameters)
4. **Eigenvalues**: All eigenvalues are either 0 or 1
5. **Rank**: $\text{rank}(\mathbf{H}) = d+1$ (assuming full column rank of $\mathbf{X}$)

### Leverages and Influence

The diagonal elements of the hat matrix, $h_{ii}$, are called leverages and measure the influence of each observation on its own fitted value:

- $h_{ii} \in [0, 1]$ for all $i$
- $\sum_{i=1}^{n} h_{ii} = d+1$
- The average leverage is $\bar{h} = \frac{d+1}{n}$
- Observations with $h_{ii} > 2\bar{h}$ are considered high-leverage points

## Geometric Interpretation
Linear models can be interpreted geometrically:
- In a 2D space (one feature), the model represents a line
- In a 3D space (two features), the model represents a plane
- In higher dimensions, the model represents a hyperplane

The parameters $\mathbf{w}$ define the orientation and position of this hyperplane in the feature space, determining how the model maps from the input space to the output space.

### Projection Geometry

Geometrically:
- The fitted values $\hat{\mathbf{y}}$ represent the orthogonal projection of $\mathbf{y}$ onto the hyperplane spanned by the columns of $\mathbf{X}$
- The residuals $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ represent the perpendicular distance from the observed points to this hyperplane
- The orthogonality between the residuals and the column space ensures that the sum of squared errors is minimized

## The Gauss-Markov Theorem

The Gauss-Markov theorem is a fundamental result that establishes the optimality of least squares estimation under certain conditions.

### Assumptions

The theorem requires the following assumptions:

1. **Linearity**: The model is linear in parameters
2. **Full rank**: The design matrix $\mathbf{X}$ has full column rank
3. **Exogeneity**: The error terms have zero conditional mean: $E(\boldsymbol{\epsilon}|\mathbf{X}) = \mathbf{0}$
4. **Homoscedasticity**: The error terms have constant variance: $\text{Var}(\boldsymbol{\epsilon}|\mathbf{X}) = \sigma^2\mathbf{I}$
5. **No autocorrelation**: The error terms are uncorrelated: $\text{Cov}(\epsilon_i, \epsilon_j) = 0$ for $i \neq j$

Importantly, the Gauss-Markov theorem does not require the normality of errors.

### BLUE Property

Under these assumptions, the Ordinary Least Squares (OLS) estimator $\hat{\mathbf{w}}_{OLS}$ is BLUE:
- **B**est: has minimum variance among all linear unbiased estimators
- **L**inear: is a linear function of the observed data
- **U**nbiased: expected value equals the true parameter value
- **E**stimator: procedure for estimating parameters

Mathematically, for any other linear unbiased estimator $\tilde{\mathbf{w}}$:

$$\text{Var}(\hat{\mathbf{w}}_{OLS}) \leq \text{Var}(\tilde{\mathbf{w}})$$

where the inequality holds in the matrix sense (the difference is positive semi-definite).

### Implications

The BLUE property has important implications:
- When the Gauss-Markov assumptions hold, OLS provides the most precise parameter estimates among all linear unbiased estimators
- Violations of these assumptions affect either the unbiasedness or the efficiency of OLS
- Even when the normality assumption is added, OLS remains the maximum likelihood estimator

## Core Assumptions
Linear models make several key assumptions:
1. **Linearity**: The relationship between the dependent and independent variables is linear.
2. **Independence**: The observations are independent of each other.
3. **Homoscedasticity**: The error terms have constant variance.
4. **Normality**: The error terms are normally distributed (for inference purposes).
5. **No multicollinearity**: The independent variables are not highly correlated.

### Checking Model Assumptions

These assumptions can be verified through:

1. **Linearity**: Scatter plots of predictors vs. response; residual plots against fitted values or predictors
2. **Independence**: Context of data collection; Durbin-Watson test for time series data
3. **Homoscedasticity**: Residual plots against fitted values; Breusch-Pagan test
4. **Normality**: Q-Q plots of residuals; Shapiro-Wilk test
5. **No multicollinearity**: Correlation matrix; Variance Inflation Factors (VIF)

## Matrix Decompositions in Linear Models

Several matrix decompositions are useful for understanding and computing linear models:

### QR Decomposition

The QR decomposition expresses $\mathbf{X} = \mathbf{Q}\mathbf{R}$ where:
- $\mathbf{Q}$ is an $n \times (d+1)$ matrix with orthogonal columns
- $\mathbf{R}$ is a $(d+1) \times (d+1)$ upper triangular matrix

This decomposition enables efficient computation of $\hat{\mathbf{w}}$:

$$\hat{\mathbf{w}} = \mathbf{R}^{-1}\mathbf{Q}^T\mathbf{y}$$

### Singular Value Decomposition (SVD)

The SVD decomposes $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^T$ where:
- $\mathbf{U}$ is an $n \times n$ orthogonal matrix
- $\mathbf{D}$ is an $n \times (d+1)$ diagonal matrix containing the singular values
- $\mathbf{V}$ is a $(d+1) \times (d+1)$ orthogonal matrix

The SVD provides insights into the conditioning of the linear system and forms the basis for techniques like principal component regression.

## Properties of Linear Models
- **Interpretable**: Parameters directly represent the effect of each feature.
- **Computationally efficient**: Often have closed-form solutions.
- **Stable**: Small changes in data typically result in small changes in the model.
- **Well-understood**: Extensive theoretical analysis available.
- **Foundational**: Serve as building blocks for more complex models.

## Limitations
- Cannot capture nonlinear relationships in their basic form.
- Sensitive to outliers in the training data.
- May oversimplify complex relationships.
- Performance limited when underlying assumptions are violated.

## Applications
- Prediction and forecasting in various domains
- Understanding relationships between variables
- Feature importance analysis
- Baseline model for comparison with more complex approaches
- Educational tool for understanding fundamental ML concepts

## Relationship to Other Models
Many advanced models can be viewed as extensions of linear models:
- Neural networks: Composition of linear models with nonlinear activation functions
- Support vector machines: Linear models with kernel transformations
- Decision trees: Piecewise linear approximations

## Simple Example
Consider predicting house prices based on size. A linear model would take the form:
$$\text{price} = w_0 + w_1 \times \text{size} + \epsilon$$

This represents a line in 2D space, where $w_0$ is the y-intercept (base price) and $w_1$ is the slope (price increase per unit of size). 