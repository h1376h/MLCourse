# Multivariate Exam Problems

This document provides a structured index of practice problems and solutions covering key topics in multivariate analysis, essential for exam preparation in machine learning and data science.

## Key Concepts and Formulas

Multivariate analysis deals with observations on multiple variables and their relationships. Below are key concepts and formulas important for exam problems:

### Multivariate Normal Distribution

The probability density function (PDF) of a $p$-dimensional multivariate normal distribution is:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

Where:
- $\boldsymbol{\mu}$ = Mean vector (dimension $p \times 1$)
- $\boldsymbol{\Sigma}$ = Covariance matrix (dimension $p \times p$, symmetric and positive-definite)

### Linear Transformations

If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$, then:
$\mathbf{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$

### Conditional Distributions

If $\mathbf{X} = [\mathbf{X}_1, \mathbf{X}_2]^T \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then:
$\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})$

Where:
- $\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$
- $\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$

### Mahalanobis Distance

The squared Mahalanobis distance from $\mathbf{x}$ to $\boldsymbol{\mu}$ is:
$d^2(\mathbf{x}, \boldsymbol{\mu}) = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$

## Examples

For detailed examples and solutions organized by topic:

1. [[L2_1_Multivariate_Exam_Density_Function_Examples|Multivariate Density Function Examples]]: Problems involving multivariate normal densities, probabilities, and marginal distributions
2. [[L2_1_Multivariate_Exam_Linear_Transformation_Examples|Linear Transformation Examples]]: Examples of applying linear transformations to multivariate distributions
3. [[L2_1_Multivariate_Exam_Conditional_Distribution_Examples|Conditional Distribution Examples]]: Problems on conditional distributions, inference, and predictions
4. [[L2_1_Multivariate_Exam_Mahalanobis_Classification_Examples|Mahalanobis Distance and Classification Examples]]: Applications of Mahalanobis distance in classification and outlier detection
5. [[L2_1_Multivariate_Exam_Eigendecomposition_Examples|Eigendecomposition Examples]]: Examples on eigenvalue decomposition, principal components, and dimensionality reduction
6. [[L2_1_Multivariate_Exam_Gaussian_Independence_Properties_Examples|Multivariate Gaussian Independence Properties Examples]]: Problems focusing on independence relationships in multivariate normal distributions and creating independent variables through transformations

### Related Topics

- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: Detailed examples of multivariate normal distributions
- [[L2_1_Joint_Distributions_Examples|Joint Distributions Examples]]: Working with joint probability distributions
- [[L2_1_Mahalanobis_Distance|Mahalanobis Distance]]: In-depth coverage of Mahalanobis distance and its applications
- [[L2_1_Linear_Transformation|Linear Transformations]]: More on transformations of random vectors
- [[L2_1_Conditional_Probability_Examples|Conditional Probability Examples]]: Additional examples of conditional distributions
- [[L2_1_Covariance_Examples|Covariance Examples]]: Understanding covariance structures
- [[L2_1_Correlation_Examples|Correlation Examples]]: Exploring correlation in multivariate data
- [[L2_1_PCA|Principal Component Analysis]]: Applications of eigendecomposition for dimension reduction
