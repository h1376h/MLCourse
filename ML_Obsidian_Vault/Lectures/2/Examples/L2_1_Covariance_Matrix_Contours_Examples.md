# Covariance Matrix Contours Examples

This document provides examples and key concepts on covariance matrices and their effects on multivariate normal distributions, illustrating the concept of covariance and correlation in machine learning and data analysis.

## Key Concepts and Formulas

The covariance matrix captures how variables in a multivariate distribution vary with respect to each other. For a bivariate normal distribution, the shape and orientation of its probability density contours are determined by the covariance matrix.

### The Multivariate Gaussian Formula

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

Where:
- $\mathbf{x}$ = Vector of variables
- $\boldsymbol{\mu}$ = Mean vector
- $\boldsymbol{\Sigma}$ = Covariance matrix
- $|\Sigma|$ = Determinant of the covariance matrix
- $\Sigma^{-1}$ = Inverse of the covariance matrix

The contour plots of this distribution form ellipses described by:

$$(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu}) = c$$

Where $c$ is a constant value. These ellipses represent points of equal probability density.

## Examples

1. [[L2_1_Covariance_Matrix_Contours_Example_1|Example 1: Basic Normal Distributions]]: Visualizing contours for standard bivariate normals
2. [[L2_1_Covariance_Matrix_Contours_Example_2|Example 2: Covariance Matrix Types and Their Effects]]: Comparing diagonal, non-diagonal, and identity matrices
3. [[L2_1_Covariance_Matrix_Contours_Example_3|Example 3: 3D Visualization of Probability Density Functions]]: Surface and contour plots
4. [[L2_1_Covariance_Matrix_Contours_Example_4|Example 4: Eigenvalues, Eigenvectors, and Covariance]]: Principal axes and ellipse orientation
5. [[L2_1_Covariance_Matrix_Contours_Example_5|Example 5: Height-Weight Relationship - Real-World Covariance]]: Real data example
6. [[L2_1_Covariance_Matrix_Contours_Example_6|Example 6: Effects of Rotation on Covariance Structure]]: Rotated ellipses
7. [[L2_1_Covariance_Matrix_Contours_Example_7|Example 7: Mahalanobis Distance vs Euclidean Distance]]: Comparing distance metrics
8. [[L2_1_Covariance_Matrix_Contours_Example_8|Example 8: Intuitive Emoji Visualization of Correlation]]: Fun visual intuition
9. [[L2_1_Covariance_Matrix_Contours_Example_9|Example 9: Sketching Contours of a Bivariate Normal Distribution]]: Manual sketching
10. [[L2_1_Covariance_Matrix_Contours_Example_10|Example 10: Robust Covariance Estimation]]: Outlier-resistant estimation
11. [[L2_1_Covariance_Matrix_Contours_Example_11|Example 11: Geometric Area Interpretation of Covariance]]: Area and spread

## Related Topics

- [[L2_1_Contour_Plots|Contour Plots]]: Core principles and interpretations of level curves
- [[L2_1_Contour_Plot_Examples|Contour Plot Examples]]: Worked examples of contour plots for various functions
- [[L2_1_Contour_Plot_Visual_Examples|Visual Examples]]: Additional visual examples of covariance matrix effects on contours
- [[L2_1_Multivariate_Distributions|Multivariate Distributions]]: Theoretical foundation for multivariate normal distributions
- [[L2_1_Covariance_Correlation|Covariance and Correlation]]: Deeper exploration of relationship measures
- [[L2_1_Eigendecomposition|Eigendecomposition]]: Mathematical tools for analyzing covariance matrices
- [[L2_1_PCA|Principal Component Analysis]]: Dimensionality reduction technique based on eigendecomposition
- [[L2_1_Mahalanobis_Distance|Mahalanobis Distance]]: Advanced distance metric for correlated variables
- [[L2_1_Linear_Transformation|Linear Transformations]]: How transformations affect covariance structure
- [[L2_1_Mean_Covariance|Mean and Covariance Estimation]]: Statistical estimation of distribution parameters
