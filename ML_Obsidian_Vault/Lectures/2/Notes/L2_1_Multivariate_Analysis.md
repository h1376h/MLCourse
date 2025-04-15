# Multivariate Analysis

## Overview
Multivariate analysis deals with the statistical analysis of data involving multiple random variables simultaneously. This field is particularly important in machine learning, where models often work with high-dimensional data and complex interactions between variables.

## Random Vectors

### Definition
A random vector $\mathbf{X} = [X_1, X_2, \ldots, X_n]^T$ is an ordered collection of random variables. In machine learning, these often represent feature vectors or model parameters.

### Joint Distribution
The complete probabilistic description of a random vector is given by its joint probability distribution:
- For discrete random vectors: joint probability mass function (PMF)
  $$p_{\mathbf{X}}(\mathbf{x}) = P(X_1 = x_1, \ldots, X_n = x_n)$$
- For continuous random vectors: joint probability density function (PDF)
  $$f_{\mathbf{X}}(\mathbf{x}) = \frac{\partial^n F_{\mathbf{X}}(\mathbf{x})}{\partial x_1 \cdots \partial x_n}$$
  where $F_{\mathbf{X}}(\mathbf{x})$ is the joint CDF

## Moments of Random Vectors

### Mean Vector
The mean (expected value) of a random vector $\mathbf{X}$ is defined as:
$$\mathbf{\mu} = E[\mathbf{X}] = [E[X_1], E[X_2], \ldots, E[X_n]]^T$$

### Covariance Matrix
The covariance matrix $\mathbf{\Sigma}$ of a random vector $\mathbf{X}$ is:
$$\mathbf{\Sigma} = E[(\mathbf{X} - \mathbf{\mu})(\mathbf{X} - \mathbf{\mu})^T]$$

Properties:
- Symmetric: $\mathbf{\Sigma} = \mathbf{\Sigma}^T$
- Positive semi-definite: $\mathbf{a}^T\mathbf{\Sigma}\mathbf{a} \geq 0$ for all $\mathbf{a}$
- Diagonal entries: $\Sigma_{ii} = \text{Var}(X_i)$
- Off-diagonal entries: $\Sigma_{ij} = \text{Cov}(X_i, X_j)$

### Correlation Matrix
The correlation matrix $\mathbf{R}$ is defined as:
$$\mathbf{R} = \mathbf{D}^{-1/2}\mathbf{\Sigma}\mathbf{D}^{-1/2}$$
where $\mathbf{D}$ is a diagonal matrix with $\text{Var}(X_i)$ on the diagonal.

## Linear Transformations

### Transformation Properties
For a random vector $\mathbf{X}$ and a linear transformation $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$:
- $E[\mathbf{Y}] = \mathbf{A}E[\mathbf{X}] + \mathbf{b}$
- $\text{Cov}(\mathbf{Y}) = \mathbf{A}\text{Cov}(\mathbf{X})\mathbf{A}^T$

### Mahalanobis Distance
The Mahalanobis distance is a measure of the distance between a point and a distribution:
$$d_M(\mathbf{x}, \mathbf{\mu}) = \sqrt{(\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})}$$

Properties:
- Scale-invariant
- Accounts for correlations between variables
- Reduces to Euclidean distance when $\mathbf{\Sigma} = \mathbf{I}$

## Key Multivariate Distributions

### Multivariate Gaussian Distribution
The multivariate Gaussian (normal) distribution is defined by a mean vector $\mathbf{\mu}$ and covariance matrix $\mathbf{\Sigma}$:
$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\mathbf{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\mathbf{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{\mu})\right)$$

Properties:
- Closed under linear transformations
- Maximum entropy distribution for a given covariance structure
- Marginal and conditional distributions are also Gaussian
- Characterized by first and second moments

### Wishart Distribution
The Wishart distribution is a generalization of the chi-squared distribution to multiple dimensions:
$$f(\mathbf{X}) = \frac{|\mathbf{X}|^{(n-p-1)/2} \exp(-\frac{1}{2}\text{tr}(\mathbf{\Sigma}^{-1}\mathbf{X}))}{2^{np/2}|\mathbf{\Sigma}|^{n/2}\Gamma_p(n/2)}$$
where $\Gamma_p$ is the multivariate gamma function.

### Dirichlet Distribution
The Dirichlet distribution is a multivariate generalization of the beta distribution:
$$f(\mathbf{x}) = \frac{\Gamma(\sum_{i=1}^k \alpha_i)}{\prod_{i=1}^k \Gamma(\alpha_i)} \prod_{i=1}^k x_i^{\alpha_i - 1}$$
where $\mathbf{x}$ is in the $(k-1)$-dimensional simplex.

## Applications in Machine Learning

### Dimensionality Reduction
- **Principal Component Analysis (PCA)**:
  $$\mathbf{Y} = \mathbf{U}^T\mathbf{X}$$
  where $\mathbf{U}$ contains eigenvectors of $\mathbf{\Sigma}$ ordered by eigenvalues
- **Linear Discriminant Analysis (LDA)**:
  $$\mathbf{S}_B\mathbf{w} = \lambda\mathbf{S}_W\mathbf{w}$$
  where $\mathbf{S}_B$ and $\mathbf{S}_W$ are between-class and within-class scatter matrices

### Multivariate Regression
$$\mathbf{Y} = \mathbf{X}\mathbf{B} + \mathbf{E}$$
where $\mathbf{B}$ is the coefficient matrix and $\mathbf{E}$ is the error matrix.

### Graphical Models
- **Bayesian Networks**: Directed acyclic graphs representing conditional dependencies
- **Markov Random Fields**: Undirected graphs representing conditional independence

### Mixture Models
$$f(\mathbf{x}) = \sum_{k=1}^K \pi_k f_k(\mathbf{x}|\mathbf{\theta}_k)$$
where $\pi_k$ are mixing coefficients and $f_k$ are component distributions.

## Challenges in Multivariate Analysis

### Curse of Dimensionality
- Data sparsity: $n$ points in $d$ dimensions require $O(n^d)$ samples
- Distance concentration: All distances become similar as $d \rightarrow \infty$
- Computational complexity: Many algorithms scale poorly with dimension

### High-Dimensional Covariance Estimation
- **Shrinkage Estimators**:
  $$\hat{\mathbf{\Sigma}} = (1-\alpha)\mathbf{S} + \alpha\mathbf{T}$$
  where $\mathbf{S}$ is sample covariance and $\mathbf{T}$ is target matrix
- **Sparse Methods**: Graphical lasso, thresholding
- **Factor Models**: $\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^T + \mathbf{\Psi}$

## Related Concepts
- [[L2_1_Joint_Distributions|Joint Distributions]]: Foundation for multivariate analysis
- [[L2_1_Covariance_Correlation|Covariance and Correlation]]: Pairwise relationships between variables
- [[L2_1_Normal_Distribution|Normal Distribution]]: Single-variable case of the multivariate Gaussian
- [[L2_2_Information_Theory|Information Theory]]: Multivariate mutual information and entropy

## Examples
For practical examples and applications of multivariate analysis, see:
- [[L2_1_Multivariate_Analysis_Examples|Multivariate Analysis Examples]]: Mean vectors, covariance matrices, linear transformations, Mahalanobis distance, PCA, and more 