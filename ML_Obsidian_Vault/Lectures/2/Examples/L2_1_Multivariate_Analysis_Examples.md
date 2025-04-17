# Multivariate Analysis Examples

This document provides examples and key concepts on multivariate analysis to help you understand this important concept in machine learning and data analysis.

## Key Concepts and Formulas

Multivariate analysis deals with the statistical analysis of multiple variables simultaneously. It encompasses techniques for analyzing relationships, patterns, and structures within multidimensional data.

### Key Multivariate Analysis Formulas

#### Mean Vector
For a dataset with $n$ observations and $p$ variables, the mean vector $\boldsymbol{\mu}$ is:

$$\boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_p \end{bmatrix} = \begin{bmatrix} \frac{1}{n}\sum_{i=1}^{n}x_{i1} \\ \frac{1}{n}\sum_{i=1}^{n}x_{i2} \\ \vdots \\ \frac{1}{n}\sum_{i=1}^{n}x_{ip} \end{bmatrix}$$

#### Covariance Matrix
The covariance matrix $\boldsymbol{\Sigma}$ for a $p$-dimensional random vector $\mathbf{X}$ is defined as:

$$\boldsymbol{\Sigma} = \begin{bmatrix} 
\sigma_{11} & \sigma_{12} & \cdots & \sigma_{1p} \\
\sigma_{21} & \sigma_{22} & \cdots & \sigma_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{p1} & \sigma_{p2} & \cdots & \sigma_{pp}
\end{bmatrix}$$

Where:
- $\sigma_{ij} = Cov(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]$ = Covariance between variables $i$ and $j$
- $\sigma_{ii} = Var(X_i)$ = Variance of variable $i$

#### Linear Transformation
For a linear transformation $\mathbf{Y} = \mathbf{AX} + \mathbf{b}$:

$$E[\mathbf{Y}] = \mathbf{A}E[\mathbf{X}] + \mathbf{b}$$
$$Cov(\mathbf{Y}) = \mathbf{A}Cov(\mathbf{X})\mathbf{A}^T$$

#### Mahalanobis Distance
The Mahalanobis distance between a point $\mathbf{x}$ and a distribution with mean $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$ is:

$$d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

#### Hotelling's T² Statistic
For testing the hypothesis $H_0: \boldsymbol{\mu} = \boldsymbol{\mu}_0$ with a sample of size $n$, mean vector $\bar{\mathbf{x}}$, and sample covariance matrix $\mathbf{S}$:

$$T^2 = n \cdot (\bar{\mathbf{x}} - \boldsymbol{\mu}_0)^T \cdot \mathbf{S}^{-1} \cdot (\bar{\mathbf{x}} - \boldsymbol{\mu}_0)$$

The related F-statistic for inference is:

$$F = \frac{n - p}{p(n - 1)} \cdot T^2$$

which follows an F-distribution with $p$ and $(n - p)$ degrees of freedom.

## Practice Questions

For practice multiple-choice questions on multivariate analysis, see:
- [[L2_1_Multivariate_Analysis_MCQ|Multivariate Analysis Multiple Choice Questions]]

## Examples

1. [[L2_1_Mean_Covariance|Mean Vector and Covariance Matrix]]: Examples of computing and interpreting basic multivariate statistics
2. [[L2_1_Linear_Transformation|Linear Transformation]]: Examples of applying and analyzing linear transformations of multivariate data
3. [[L2_1_Mahalanobis_Distance|Mahalanobis Distance]]: Examples of computing Mahalanobis distance for outlier detection
4. [[L2_1_PCA|Principal Component Analysis]]: Examples of dimensionality reduction using PCA
5. [[L3_3_Multivariate_Regression|Multivariate Regression]]: Examples of regression with multiple predictor variables
6. [[L2_1_Hotellings_T2|Hotelling's T² Test]]: Examples of multivariate hypothesis testing for comparing mean vectors 