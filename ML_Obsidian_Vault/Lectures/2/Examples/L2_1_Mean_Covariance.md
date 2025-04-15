# Mean Vector and Covariance Matrix Examples

This document provides practical examples of computing and interpreting mean vectors and covariance matrices, which are fundamental statistical tools for analyzing multivariate data.

## Key Concepts and Formulas

The mean vector and covariance matrix provide a statistical summary of multivariate data, capturing both individual variable distributions and their relationships.

### Mean Vector Formula

For a dataset with $n$ observations and $p$ variables, the mean vector $\boldsymbol{\mu}$ is:

$$\boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_p \end{bmatrix} = \begin{bmatrix} \frac{1}{n}\sum_{i=1}^{n}x_{i1} \\ \frac{1}{n}\sum_{i=1}^{n}x_{i2} \\ \vdots \\ \frac{1}{n}\sum_{i=1}^{n}x_{ip} \end{bmatrix}$$

Where:
- $\mu_i$ = Mean of variable $i$
- $x_{ij}$ = Value of variable $i$ for observation $j$
- $n$ = Number of observations

### Covariance Matrix Formula

The covariance matrix $\boldsymbol{\Sigma}$ for a $p$-dimensional dataset is:

$$\boldsymbol{\Sigma} = \begin{bmatrix} 
\sigma_{11} & \sigma_{12} & \cdots & \sigma_{1p} \\
\sigma_{21} & \sigma_{22} & \cdots & \sigma_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{p1} & \sigma_{p2} & \cdots & \sigma_{pp}
\end{bmatrix}$$

For a sample, we calculate:

$$\sigma_{ij} = \frac{1}{n-1}\sum_{k=1}^{n}(x_{ki} - \mu_i)(x_{kj} - \mu_j)$$

Where:
- $\sigma_{ij}$ = Covariance between variables $i$ and $j$
- $\sigma_{ii}$ = Variance of variable $i$
- $\mu_i$ = Mean of variable $i$
- $x_{ki}$ = Value of variable $i$ for observation $k$
- $n$ = Number of observations

## Examples

The following examples demonstrate computing and interpreting mean vectors and covariance matrices:

- **Car Features Analysis**: Analyzing numerical features of cars
- **Patient Physiological Measures**: Examining relationships between health metrics
- **Financial Data Analysis**: Understanding the structure of financial variables

### Example 1: Car Features Analysis

#### Problem Statement
A researcher is analyzing a dataset of cars with three numerical features: horsepower ($X_1$), weight in tons ($X_2$), and fuel efficiency in miles per gallon ($X_3$). A sample of 5 cars yields the following measurements:

| Car | Horsepower ($X_1$) | Weight ($X_2$) | MPG ($X_3$) |
|-----|--------------|--------|-------|
| 1   | 130         | 1.9     | 27    |
| 2   | 165         | 2.2     | 24    |
| 3   | 200         | 2.5     | 20    |
| 4   | 110         | 1.8     | 32    |
| 5   | 220         | 2.8     | 18    |

Calculate the mean vector and covariance matrix for this dataset.

In this example:
- We have 5 observations (cars) and 3 variables
- Each car is represented as a data point in 3-dimensional space
- We need to compute summary statistics that capture both the central tendency and relationships between variables

#### Solution

The mean vector and covariance matrix provide a statistical summary of multivariate data, capturing both individual variable distributions and their relationships.

##### Step 1: Calculate the Mean Vector
We compute the mean of each variable by averaging the corresponding values:

$$\mu_{X_1} = \frac{130 + 165 + 200 + 110 + 220}{5} = \frac{825}{5} = 165$$

$$\mu_{X_2} = \frac{1.9 + 2.2 + 2.5 + 1.8 + 2.8}{5} = \frac{11.2}{5} = 2.24$$

$$\mu_{X_3} = \frac{27 + 24 + 20 + 32 + 18}{5} = \frac{121}{5} = 24.2$$

The mean vector is:

$$\boldsymbol{\mu} = \begin{bmatrix} \mu_{X_1} \\ \mu_{X_2} \\ \mu_{X_3} \end{bmatrix} = \begin{bmatrix} 165 \\ 2.24 \\ 24.2 \end{bmatrix}$$

##### Step 2: Calculate Deviations from the Mean
We compute the deviation of each observation from the corresponding mean:

| Car | $X_1 - \mu_{X_1}$ | $X_2 - \mu_{X_2}$ | $X_3 - \mu_{X_3}$ |
|-----|---------|---------|---------|
| 1   | -35     | -0.34    | 2.8     |
| 2   | 0       | -0.04    | -0.2    |
| 3   | 35      | 0.26     | -4.2    |
| 4   | -55     | -0.44    | 7.8     |
| 5   | 55      | 0.56     | -6.2    |

##### Step 3: Compute the Covariance Matrix Elements
The covariance matrix is computed using the formula:

$$\sigma_{ij} = \frac{1}{n-1}\sum_{k=1}^{n}(x_{ki} - \mu_i)(x_{kj} - \mu_j)$$

For the variances (diagonal elements):

$$\sigma_{11} = \text{Var}(X_1) = \frac{(-35)^2 + (0)^2 + (35)^2 + (-55)^2 + (55)^2}{4} = \frac{7050}{4} = 1762.5$$

$$\sigma_{22} = \text{Var}(X_2) = \frac{(-0.34)^2 + (-0.04)^2 + (0.26)^2 + (-0.44)^2 + (0.56)^2}{4} = \frac{0.6772}{4} = 0.1693$$

$$\sigma_{33} = \text{Var}(X_3) = \frac{(2.8)^2 + (-0.2)^2 + (-4.2)^2 + (7.8)^2 + (-6.2)^2}{4} = \frac{145.24}{4} = 36.31$$

For the covariances (off-diagonal elements):

$$\sigma_{12} = \text{Cov}(X_1, X_2) = \frac{(-35)(-0.34) + (0)(-0.04) + (35)(0.26) + (-55)(-0.44) + (55)(0.56)}{4} = \frac{76}{4} = 19$$

$$\sigma_{13} = \text{Cov}(X_1, X_3) = \frac{(-35)(2.8) + (0)(-0.2) + (35)(-4.2) + (-55)(7.8) + (55)(-6.2)}{4} = \frac{-1015}{4} = -253.75$$

$$\sigma_{23} = \text{Cov}(X_2, X_3) = \frac{(-0.34)(2.8) + (-0.04)(-0.2) + (0.26)(-4.2) + (-0.44)(7.8) + (0.56)(-6.2)}{4} = \frac{-8.94}{4} = -2.235$$

Since the covariance matrix is symmetric, $\sigma_{21} = \sigma_{12}$, $\sigma_{31} = \sigma_{13}$, and $\sigma_{32} = \sigma_{23}$.

##### Step 4: Assemble the Covariance Matrix
The complete covariance matrix is:

$$\boldsymbol{\Sigma} = \begin{bmatrix} 
1762.5 & 19.0 & -253.75 \\
19.0 & 0.1693 & -2.235 \\
-253.75 & -2.235 & 36.31
\end{bmatrix}$$

Therefore, we have successfully calculated the mean vector and covariance matrix for the dataset. These statistics provide valuable insights about the data distribution and relationships between variables.

##### Interpretation
The covariance matrix reveals important relationships between variables:

- The positive covariance (19.0) between horsepower and weight indicates that as horsepower increases, weight tends to increase
- The negative covariance (-253.75) between horsepower and MPG indicates that as horsepower increases, fuel efficiency tends to decrease
- The negative covariance (-2.235) between weight and MPG indicates that as weight increases, fuel efficiency tends to decrease

These findings align with mechanical engineering principles: more powerful engines tend to be heavier and consume more fuel.

## Key Insights

### Theoretical Insights
- The mean vector provides a central reference point in multidimensional space
- The covariance matrix captures both the spread of individual variables and their interactions
- The diagonal elements of the covariance matrix represent variances of individual variables
- The off-diagonal elements represent the degree to which variables are linearly related

### Practical Applications
- Data summarization without losing information about variable relationships
- Input for various statistical methods like principal component analysis, factor analysis, and discriminant analysis
- Foundation for multivariate normal distribution modeling
- Outlier detection when combined with Mahalanobis distance

### Common Pitfalls
- Sensitivity to outliers, as extreme values can dramatically affect covariance estimates
- Scale dependency, where variables with larger scales dominate the covariance structure
- Difficulty in interpretation when dealing with many variables
- Only capturing linear relationships between variables

## Related Topics
- [[L2_1_Linear_Transformation|Linear Transformation]]: How transformations affect mean vectors and covariance matrices
- [[L2_1_Mahalanobis_Distance|Mahalanobis Distance]]: Using covariance matrices for distance calculations
- [[L2_1_PCA|Principal Component Analysis]]: Utilizing covariance structure for dimensionality reduction
- [[L2_1_Correlation_Examples|Correlation]]: Standardized version of covariance 