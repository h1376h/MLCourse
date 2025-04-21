# Covariance Matrix Contours Examples

This document provides practical examples of covariance matrices and their effects on multivariate normal distributions, illustrating the concept of covariance and correlation in machine learning and data analysis contexts.

## Key Concepts and Formulas

The covariance matrix is a square matrix that captures how variables in a multivariate distribution vary with respect to each other. For a bivariate normal distribution, the shape and orientation of its probability density contours are directly determined by the covariance matrix.

### The Multivariate Gaussian Formula

$$f(x,y) = \frac{1}{2\pi\sqrt{|\Sigma|}} \exp\left(-\frac{1}{2}(X-\mu)^T \Sigma^{-1} (X-\mu)\right)$$

Where:
- $X$ = Vector of variables (x, y)
- $\mu$ = Mean vector (μ₁, μ₂)
- $\Sigma$ = Covariance matrix
- $|\Sigma|$ = Determinant of the covariance matrix

The contour plots of this distribution form ellipses described by:

$$(X-\mu)^T \Sigma^{-1} (X-\mu) = \text{constant}$$

## Examples

The following examples demonstrate how different covariance matrices affect the shape and orientation of probability density contours:

- **Basic Normal Distributions**: Visualizing 1D and 2D normal distributions with different variances
- **Diagonal Covariance Matrices**: Exploring axis-aligned elliptical contours
- **Non-Diagonal Covariance Matrices**: Understanding rotated elliptical contours with correlation
- **3D Visualization**: Examining the probability density surface in three dimensions
- **Eigenvalue Effect**: Analyzing how eigenvalues and eigenvectors relate to contour shapes

### Example 1: Basic Normal Distributions

#### Problem Statement
How do variance changes affect 1D normal distributions, and what happens when we extend to 2D with independent variables?

In this example:
- We visualize 1D normal distributions with different variances
- We show how these distributions extend to 2D space
- We examine the standard circular case and the axis-aligned elliptical case

#### Solution

We'll start with 1D normal distributions and extend to 2D with diagonal covariance matrices.

##### Step 1: 1D Normal Distributions with Different Variances
The standard normal distribution has a PDF given by:

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right)$$

Changing σ² alters the width and height of the bell curve, as shown in the left panel of the figure below.

##### Step 2: 2D Standard Normal Distribution
For a 2D standard normal with identity covariance matrix, the PDF is:

$$f(x,y) = \frac{1}{2\pi} \exp\left(-\frac{x^2 + y^2}{2}\right)$$

This creates circular contours as both variables have the same variance and are uncorrelated, as shown in the middle panel.

##### Step 3: 2D Normal with Different Variances
For a 2D normal with different variances but no correlation:

$$f(x,y) = \frac{1}{2\pi\sqrt{\sigma_1^2\sigma_2^2}} \exp\left(-\frac{1}{2}\left(\frac{x^2}{\sigma_1^2} + \frac{y^2}{\sigma_2^2}\right)\right)$$

This creates axis-aligned elliptical contours, as shown in the right panel.

![Basic 2D Normal Examples](../Images/Contour_Plots/basic_2d_normal_examples.png)

### Example 2: Covariance Matrix Types and Their Effects

#### Problem Statement
How do different types of covariance matrices affect the shape, size, and orientation of probability density contours?

#### Solution

We'll explore four cases with different covariance matrices.

##### Step 1: Diagonal Covariance with Equal Variances
When the covariance matrix is a scaled identity matrix:

$$\Sigma = \begin{bmatrix} \sigma^2 & 0 \\ 0 & \sigma^2 \end{bmatrix} = \sigma^2 I$$

The contours form perfect circles, as shown in the top-left panel of the figure below.

##### Step 2: Diagonal Covariance with Different Variances
When the covariance matrix has different variances but no correlation:

$$\Sigma = \begin{bmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{bmatrix}$$

The contours form axis-aligned ellipses, as shown in the top-right panel.

##### Step 3: Non-Diagonal Covariance with Positive Correlation
When the covariance matrix has non-zero off-diagonal elements with positive correlation:

$$\Sigma = \begin{bmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{bmatrix}, \rho > 0$$

The contours form ellipses rotated along the y = x direction, as shown in the bottom-left panel.

##### Step 4: Non-Diagonal Covariance with Negative Correlation
When the covariance matrix has non-zero off-diagonal elements with negative correlation:

$$\Sigma = \begin{bmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{bmatrix}, \rho < 0$$

The contours form ellipses rotated along the y = -x direction, as shown in the bottom-right panel.

![Covariance Matrix Contours](../Images/Contour_Plots/covariance_matrix_contours.png)

### Example 3: 3D Visualization of Probability Density Functions

#### Problem Statement
How does the probability density function of a bivariate normal distribution look in 3D space, and how does the covariance matrix affect this surface?

#### Solution

We'll visualize the probability density surface in 3D for different covariance matrices.

##### Step 1: Standard Bivariate Normal
For a standard bivariate normal (identity covariance), the PDF creates a symmetric bell-shaped surface in 3D, as shown in the left panel of the figure below.

##### Step 2: Diagonal Covariance with Different Variances
When the variances differ but variables remain uncorrelated, the PDF surface becomes stretched along one axis and compressed along the other, as shown in the middle panel.

##### Step 3: Non-Diagonal Covariance with Correlation
When the variables are correlated, the PDF surface becomes tilted, with the peak still at the mean but the spread occurring along a rotated axis, as shown in the right panel.

![Gaussian 3D Visualization](../Images/Contour_Plots/gaussian_3d_visualization.png)

### Example 4: Eigenvalues, Eigenvectors, and Covariance

#### Problem Statement
How do the eigenvalues and eigenvectors of a covariance matrix relate to the shape and orientation of probability density contours?

#### Solution

We'll examine how increasing correlation affects the eigenvalues and eigenvectors of covariance matrices.

##### Step 1: No Correlation (ρ = 0)
With no correlation, the eigenvalues are equal to the variances, and the eigenvectors align with the coordinate axes, as shown in the top-left panel of the figure below.

##### Step 2: Weak Correlation (ρ = 0.3)
With weak correlation, the eigenvectors begin to rotate, and the eigenvalues start to separate, as shown in the top-right panel.

##### Step 3: Moderate Correlation (ρ = 0.6)
With moderate correlation, the rotation becomes more pronounced, and the difference between eigenvalues increases, as shown in the bottom-left panel.

##### Step 4: Strong Correlation (ρ = 0.9)
With strong correlation, the eigenvectors approach the y = x and y = -x directions, and the eigenvalues become significantly different, as shown in the bottom-right panel.

The principal axes of the elliptical contours align with the eigenvectors, and the lengths of the semi-axes are proportional to the square roots of the eigenvalues.

![Covariance Eigenvalue Visualization](../Images/Contour_Plots/covariance_eigenvalue_visualization.png)

## Key Insights

### Theoretical Insights
- The covariance matrix determines the shape, size, and orientation of probability density contours
- Diagonal elements (variances) control the spread along the principal axes
- Off-diagonal elements (covariances) control the rotation of the principal axes
- Eigenvalues and eigenvectors provide direct insight into the shape and orientation of the distribution

### Practical Applications
- Understanding data correlation structure through visualization
- Designing multivariate confidence regions for statistical inference
- Implementing anomaly detection algorithms using Mahalanobis distance
- Performing dimensionality reduction through principal component analysis

### Common Pitfalls
- Mistaking correlation for causation in data analysis
- Failing to recognize that correlation changes the effective area of confidence regions
- Overlooking the importance of variance normalization when comparing variables
- Assuming independence when significant correlation exists

## Running the Examples

You can run the code that generates these examples and visualizations using:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Codes/covariance_matrix_contours.py
```

## Related Topics

- [[L2_1_Contour_Plots|Contour Plots]]: Core principles and interpretations
- [[L2_1_Contour_Plot_Examples|Contour Plot Examples]]: Worked examples of contour plots for various functions
- [[L2_1_Contour_Plot_Visual_Examples|Visual Examples]]: Additional visual examples of covariance matrix effects on contours
- [[L2_1_Multivariate_Distributions|Multivariate Distributions]]: Theoretical foundation for multivariate normal distributions
- [[L2_1_Covariance_Correlation|Covariance and Correlation]]: Related concepts for understanding distribution shapes
- [[L2_1_Eigendecomposition|Eigendecomposition]]: Mathematical tools for analyzing covariance matrices 