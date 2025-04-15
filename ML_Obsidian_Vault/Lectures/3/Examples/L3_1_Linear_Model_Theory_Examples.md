# Linear Model Theory Examples

This document provides examples of linear model theory, illustrating the mathematical foundations and assumptions that underlie linear regression in machine learning.

## Key Concepts and Formulas

Linear models are statistical models that express the relationship between a dependent variable and one or more independent variables as a linear combination.

### Basic Linear Model Formulation

$$y = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

Where:
- $y$ = Dependent variable (target) - an $n \times 1$ vector of observations
- $\mathbf{X}$ = Design matrix of independent variables (features) - an $n \times p$ matrix
- $\boldsymbol{\beta}$ = Vector of model parameters (coefficients) - a $p \times 1$ vector
- $\boldsymbol{\epsilon}$ = Vector of error terms - an $n \times 1$ vector

### Key Assumptions of Linear Models

1. **Linearity**: The relationship between the independent and dependent variables is linear.
2. **Independence**: The error terms are independent of each other.
3. **Homoscedasticity**: The error terms have constant variance.
4. **Normality**: The error terms follow a normal distribution.
5. **Exogeneity**: The independent variables are not correlated with the error term.

## Examples

### Example 1: Understanding the Linear Model Framework

#### Problem Statement
Consider a dataset with 100 observations, 3 features (independent variables), and 1 target variable (dependent variable). Formulate this as a linear model and explain the dimensions of each component.

#### Solution

##### Step 1: Define the Model Components
In this scenario:
- $n = 100$ (number of observations)
- $p = 3$ (number of features)
- $\mathbf{X}$ is a $100 \times 3$ matrix (design matrix)
- $\boldsymbol{\beta}$ is a $3 \times 1$ vector (coefficients)
- $y$ is a $100 \times 1$ vector (target values)
- $\boldsymbol{\epsilon}$ is a $100 \times 1$ vector (error terms)

The linear model is expressed as:

$$y_i = \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3} + \epsilon_i \quad \text{for } i = 1, 2, \ldots, 100$$

Or in matrix form:

$$\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_{100} \end{bmatrix} = 
\begin{bmatrix} 
x_{11} & x_{12} & x_{13} \\ 
x_{21} & x_{22} & x_{23} \\ 
\vdots & \vdots & \vdots \\ 
x_{100,1} & x_{100,2} & x_{100,3}
\end{bmatrix}
\begin{bmatrix} \beta_1 \\ \beta_2 \\ \beta_3 \end{bmatrix} + 
\begin{bmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_{100} \end{bmatrix}$$

##### Step 2: Including the Intercept Term
If we include an intercept term $\beta_0$, we augment the design matrix with a column of ones:

$$\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_{100} \end{bmatrix} = 
\begin{bmatrix} 
1 & x_{11} & x_{12} & x_{13} \\ 
1 & x_{21} & x_{22} & x_{23} \\ 
\vdots & \vdots & \vdots & \vdots \\ 
1 & x_{100,1} & x_{100,2} & x_{100,3}
\end{bmatrix}
\begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \beta_3 \end{bmatrix} + 
\begin{bmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_{100} \end{bmatrix}$$

Now $\mathbf{X}$ is a $100 \times 4$ matrix and $\boldsymbol{\beta}$ is a $4 \times 1$ vector.

### Example 2: Basic Linear Model Assumptions

#### Problem Statement
You have a dataset and want to verify whether it satisfies the key assumptions of linear modeling. Describe basic methods to check each assumption and their importance.

#### Solution

##### Step 1: Understanding the Importance of Assumptions
The assumptions of linear regression are important because they ensure that:
- Our parameter estimates are unbiased and have minimum variance (Best Linear Unbiased Estimators or BLUE)
- The statistical tests (t-tests, F-tests) and confidence intervals are valid
- Our predictions are reliable

##### Step 2: Checking Linearity
The linearity assumption can be verified by:
- Creating scatter plots of each independent variable against the dependent variable
- Plotting residuals versus fitted values

If the relationship is linear, the residual plot should show a random pattern around the zero line with no systematic curvature.

$$\text{residuals} = y - \hat{y} = y - \mathbf{X}\hat{\boldsymbol{\beta}}$$

![Residual Plot for Linearity](../Images/Linear_Model_Theory/residual_linearity.png)

##### Step 3: Checking Independence
For data that has no natural ordering (like time), independence is often assumed based on the sampling design.

For time series or spatial data, we can check for patterns in residuals that might indicate dependence.

##### Step 4: Checking Homoscedasticity
Homoscedasticity (constant variance of errors) can be checked by:
- Plotting residuals versus fitted values

A fan-shaped pattern indicates heteroscedasticity (non-constant variance).

![Homoscedasticity Plot](../Images/Linear_Model_Theory/homoscedasticity.png)

##### Step 5: Checking Normality of Errors
Normality of error terms can be assessed by:
- Creating a histogram of residuals
- Creating a Q-Q plot of residuals

For the Q-Q plot, the residuals are plotted against the theoretical quantiles of a normal distribution. If the points fall approximately along a straight line, the normality assumption is satisfied.

![Q-Q Plot for Normality](../Images/Linear_Model_Theory/qq_plot.png)

### Example 3: Geometric Interpretation of Linear Models

#### Problem Statement
Explain the geometric interpretation of a linear model with two predictors ($x_1$ and $x_2$) and an intercept term. Visualize the model in 3D space.

#### Solution

##### Step 1: Geometric Representation
In a linear model with two predictors and an intercept:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \epsilon$$

The model represents a plane in 3D space:
- The $x_1$ and $x_2$ axes represent the feature space
- The $y$ axis represents the target variable
- The coefficients $\beta_1$ and $\beta_2$ represent the slopes of the plane in the $x_1$ and $x_2$ directions
- The intercept $\beta_0$ represents the height of the plane at the origin

![3D Linear Model](../Images/Linear_Model_Theory/linear_model_3d.png)

##### Step 2: Projecting the Data onto the Model
The fitted values $\hat{y}$ are the projections of the actual $y$ values onto the plane:

$$\hat{y} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^Ty = \mathbf{H}y$$

Where $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is called the "hat matrix" or "projection matrix".

The residuals $e = y - \hat{y}$ represent the vertical distances from the actual data points to the plane.

![Projection in Linear Model](../Images/Linear_Model_Theory/projection.png)

### Example 4: Linear Model Matrix Properties

#### Problem Statement
Explain the properties of the key matrices in linear models, particularly focusing on the hat matrix (projection matrix) and its significance in understanding linear transformations.

#### Solution

##### Step 1: Understanding the Hat Matrix
The hat matrix $\mathbf{H}$ is a fundamental concept in linear model theory that maps the observed responses $\mathbf{y}$ to the fitted values $\hat{\mathbf{y}}$:

$$\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$$

Where:
$$\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$$

The name "hat matrix" comes from the fact that it puts the "hat" on $\mathbf{y}$ to produce $\hat{\mathbf{y}}$.

##### Step 2: Key Properties of the Hat Matrix
The hat matrix has several important properties:

1. **Symmetric**: $\mathbf{H} = \mathbf{H}^T$
2. **Idempotent**: $\mathbf{H}\mathbf{H} = \mathbf{H}$ (applying the projection twice doesn't change the result)
3. **Trace**: $\text{trace}(\mathbf{H}) = p$ (equals the number of parameters, which is the rank of $\mathbf{X}$)
4. **Eigenvalues**: All eigenvalues are either 0 or 1
5. **Range**: The column space of $\mathbf{X}$
6. **Null space**: Orthogonal to the column space of $\mathbf{X}$

##### Step 3: Geometric Interpretation of the Hat Matrix
The hat matrix represents a projection onto the column space of $\mathbf{X}$:

- $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ is the projection of $\mathbf{y}$ onto the space spanned by the columns of $\mathbf{X}$
- $(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y} - \hat{\mathbf{y}} = \mathbf{e}$ (the residuals) is the projection onto the orthogonal complement

This geometric view helps understand why:
- The fitted values are the closest points in the model space to the observed values
- The residuals are orthogonal to the model space: $\mathbf{X}^T\mathbf{e} = \mathbf{0}$

![Hat Matrix Projection](../Images/Linear_Model_Theory/hat_matrix_projection.png)

##### Step 4: Leverages and Influential Points
The diagonal elements $h_{ii}$ of the hat matrix are called leverages and measure the influence of the $i$-th observation on its own fitted value:

$$\hat{y}_i = \sum_{j=1}^{n} h_{ij}y_j$$

A high leverage ($h_{ii}$ close to 1) indicates an observation that has a strong influence on the model fit. These points are typically far from the center of the feature space.

For a balanced dataset with no extreme points:
- Average leverage: $\bar{h}_{ii} = \frac{p}{n}$
- Points with $h_{ii} > 2\frac{p}{n}$ are considered high leverage points

### Example 5: Linear Algebra Foundations of Linear Models

#### Problem Statement
Explain how linear algebra concepts underpin linear models, particularly focusing on vector spaces, orthogonality, and the role of these concepts in understanding model fitting.

#### Solution

##### Step 1: Vector Spaces in Linear Models
Linear modeling can be understood through the lens of vector spaces:

- The response vector $\mathbf{y} \in \mathbb{R}^n$ lies in an $n$-dimensional space
- The column space of $\mathbf{X}$ (denoted as $C(\mathbf{X})$) is a $p$-dimensional subspace of $\mathbb{R}^n$
- The linear model assumes that $\mathbf{y}$ can be approximated by a vector in $C(\mathbf{X})$

$$\mathbf{y} \approx \mathbf{X}\boldsymbol{\beta} \in C(\mathbf{X})$$

##### Step 2: Orthogonal Projection and Least Squares
The method of least squares finds the point in $C(\mathbf{X})$ that is closest to $\mathbf{y}$ in terms of Euclidean distance.

This is achieved by an orthogonal projection:
- $\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}$ is the orthogonal projection of $\mathbf{y}$ onto $C(\mathbf{X})$
- The residual vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to every vector in $C(\mathbf{X})$

This orthogonality principle is expressed mathematically as:

$$\mathbf{X}^T(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \mathbf{0}$$

From which we derive the normal equations:

$$\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$$

![Orthogonal Projection](../Images/Linear_Model_Theory/orthogonal_projection.png)

##### Step 3: Rank and Dimensionality
The rank of the design matrix $\mathbf{X}$ is crucial in linear models:

- If $\text{rank}(\mathbf{X}) = p$ (full rank), then $\mathbf{X}^T\mathbf{X}$ is invertible and there's a unique solution for $\hat{\boldsymbol{\beta}}$
- If $\text{rank}(\mathbf{X}) < p$ (rank deficient), then $\mathbf{X}^T\mathbf{X}$ is not invertible and there are infinitely many solutions for $\hat{\boldsymbol{\beta}}$

Rank deficiency typically occurs due to:
- More parameters than observations ($p > n$)
- Perfect multicollinearity (linear dependencies among predictors)

In rank-deficient cases, additional constraints (like regularization) are needed to find a unique solution.

##### Step 4: Decompositions in Linear Models
Matrix decompositions provide powerful insights into linear models:

1. **Eigendecomposition** of $\mathbf{X}^T\mathbf{X}$:
   - Reveals the directions of maximum variance in the feature space
   - Small eigenvalues indicate near multicollinearity

2. **QR Decomposition** of $\mathbf{X} = \mathbf{Q}\mathbf{R}$:
   - Used for numerically stable computation of $\hat{\boldsymbol{\beta}}$
   - $\hat{\boldsymbol{\beta}} = \mathbf{R}^{-1}\mathbf{Q}^T\mathbf{y}$

3. **Singular Value Decomposition (SVD)** of $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^T$:
   - Provides insights into the subspaces and conditioning of the linear system
   - Basis for many advanced techniques like principal component regression

### Example 6: The Gauss-Markov Theorem and BLUE Properties

#### Problem Statement
Explain the Gauss-Markov theorem, its assumptions, and why it's important for understanding the properties of linear model estimators.

#### Solution

##### Step 1: The Gauss-Markov Assumptions
The Gauss-Markov theorem requires the following assumptions:

1. **Linearity**: The relationship between predictors and outcome is linear
2. **Random Sampling**: Observations are drawn randomly from the population
3. **No Perfect Multicollinearity**: No exact linear relationships among predictors
4. **Exogeneity**: The expected value of errors is zero given any values of predictors: $E(\epsilon|X) = 0$
5. **Homoscedasticity**: The variance of errors is constant: $Var(\epsilon|X) = \sigma^2$

Note that normality of errors is NOT required for the Gauss-Markov theorem.

##### Step 2: The BLUE Property
Under the Gauss-Markov assumptions, the Ordinary Least Squares (OLS) estimator is BLUE:
- **B**: Best - has minimum variance among all linear unbiased estimators
- **L**: Linear - is a linear function of the observed data
- **U**: Unbiased - expected value equals the true parameter value
- **E**: Estimator - procedure for estimating parameters

Mathematically, for any other linear unbiased estimator $\tilde{\beta}$ of $\beta$:

$$Var(\hat{\beta}_{OLS}) \leq Var(\tilde{\beta})$$

where the inequality holds in the matrix sense (the difference is positive semi-definite).

##### Step 3: Implications and Importance

1. **Efficiency**: OLS provides the most precise estimates among all linear unbiased estimators
2. **Optimality**: When assumptions are satisfied, OLS is the optimal estimation method
3. **Reliability**: Statistical tests and confidence intervals are trustworthy

When assumptions are violated:
- If homoscedasticity is violated: OLS is still unbiased but no longer efficient
- If exogeneity is violated: OLS becomes biased and inconsistent
- If multicollinearity is present: OLS is unbiased but less efficient (higher variance)

![BLUE Property Visualization](../Images/Linear_Model_Theory/BLUE_property.png)

## Key Insights

### Theoretical Insights
- Linear models are characterized by the linear combination of parameters and features
- The design matrix $\mathbf{X}$ represents the feature space
- The parameter vector $\boldsymbol{\beta}$ represents the weights of each feature
- The error term $\boldsymbol{\epsilon}$ captures the unexplained variation in the data
- The model assumptions are crucial for the validity of inference
- The hat matrix provides a projection interpretation of linear model fitting
- Linear models are fundamentally connected to concepts of vector spaces and orthogonality
- The Gauss-Markov theorem provides theoretical justification for using OLS
- The geometric view of linear models helps visualize complex higher-dimensional relationships

### Practical Implications
- Violation of linearity can be addressed by transforming variables
- Non-constant variance affects the efficiency of estimators
- Non-normality of errors affects inference but not parameter estimation
- The projection interpretation provides a geometric understanding of model fitting
- The Gauss-Markov theorem provides theoretical justification for using OLS
- Categorical variables require proper encoding techniques
- Multicollinearity affects parameter stability but not overall model fit
- Understanding matrix properties helps diagnose issues in model fitting
- The vector space interpretation explains why OLS minimizes the sum of squared errors
- Leverages can identify influential observations that require special attention
- The BLUE properties explain why OLS is preferred when assumptions are met

### Common Pitfalls
- Forgetting to include an intercept term
- Ignoring assumption violations when making inferences
- Extrapolating the model beyond the range of observed data
- Confusing correlation with causation
- Falling into the dummy variable trap (including all dummy categories)
- Overlooking multicollinearity when interpreting individual coefficients
- Assuming OLS is still optimal when Gauss-Markov assumptions are violated
- Ignoring rank deficiency when fitting linear models
- Overlooking the geometric interpretation of residuals and fitted values
- Failing to check for high-leverage points that can distort parameter estimates

## Running the Examples

You can run the code that generates these examples and visualizations using:

```bash
python3 ML_Obsidian_Vault/Lectures/3/Codes/linear_model_theory.py
```

## Related Topics

- [[L3_2_Simple_Linear_Regression|Simple Linear Regression]]: Application of linear model theory to one predictor
- [[L3_2_Cost_Function|Cost Function]]: Optimization objectives for linear models
- [[L3_2_Least_Squares|Least Squares Method]]: Method for estimating model parameters
- [[L3_2_Error_Models|Error Models]]: Different assumptions about the error distribution
- [[L3_2_Analytical_Solution|Analytical Solution]]: Closed-form solution for parameter estimation 