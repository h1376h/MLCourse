# Covariance Matrix Contour Plots

This document explains how the covariance matrix of a multivariate normal distribution affects the shape and orientation of its probability density contours.

## Introduction

The contour plots of a multivariate normal distribution provide valuable insights into the relationship between variables and their correlation structure. The shape, orientation, and size of these contours are directly determined by the covariance matrix.

## Mathematical Foundation

The probability density function (PDF) of a bivariate normal distribution with mean μ = (μ₁, μ₂) and covariance matrix Σ is given by:

$$f(x,y) = \frac{1}{2\pi\sqrt{|\Sigma|}} \exp\left(-\frac{1}{2}(X-\mu)^T \Sigma^{-1} (X-\mu)\right)$$

where X = (x, y), μ = (μ₁, μ₂), and |Σ| is the determinant of Σ.

The contours of this function are sets of points where the density is constant. These form ellipses described by:

$$(X-\mu)^T \Sigma^{-1} (X-\mu) = \text{constant}$$

## Types of Covariance Matrices

### Diagonal Covariance Matrices

A diagonal covariance matrix has zeros for all off-diagonal elements:

$$\Sigma = \begin{bmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{bmatrix}$$

There are two important subcases:

#### Case 1: Equal Variances (Scaled Identity Matrix)

When $\sigma_1^2 = \sigma_2^2 = \sigma^2$:

$$\Sigma = \begin{bmatrix} \sigma^2 & 0 \\ 0 & \sigma^2 \end{bmatrix} = \sigma^2 I$$

This produces circular contours. The radius of the circles is proportional to σ.

![Equal Variances](../Images/Contour_Plots/covariance_matrix_contours.png)

#### Case 2: Different Variances

When $\sigma_1^2 \neq \sigma_2^2$:

$$\Sigma = \begin{bmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{bmatrix}$$

This produces axis-aligned elliptical contours. The lengths of the semi-axes are proportional to $\sigma_1$ and $\sigma_2$.

### Non-Diagonal Covariance Matrices

A non-diagonal covariance matrix has non-zero off-diagonal elements:

$$\Sigma = \begin{bmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{bmatrix}$$

where ρ is the correlation coefficient between the variables.

#### Case 3: Positive Correlation

When ρ > 0, the elliptical contours are rotated in a way that they are tilted along the y = x direction. The stronger the correlation, the more elongated and tilted the ellipses.

#### Case 4: Negative Correlation

When ρ < 0, the elliptical contours are rotated in a way that they are tilted along the y = -x direction.

## Step-by-Step Analysis

### Step 1: Understand the multivariate Gaussian PDF

The probability density function of a bivariate Gaussian with mean μ = (0,0) and covariance matrix Σ defines a surface whose contours we want to analyze.

### Step 2: Analyze the quadratic form in the exponent

The key term that determines the shape of the contours is the quadratic form (x,y)ᵀ Σ⁻¹ (x,y), which creates elliptical level curves.

### Step 3: Consider different types of covariance matrices

Different covariance matrices lead to different shaped contours:
- Diagonal with equal variances: Circular contours
- Diagonal with different variances: Axis-aligned ellipses
- Non-diagonal (with correlation): Rotated ellipses

### Step 4: Determine the principal axes

The principal axes of the elliptical contours align with the eigenvectors of Σ.
- The lengths of the semi-axes are proportional to the square roots of the eigenvalues.
- The correlation coefficient ρ determines the rotation angle of the ellipses.

## Key Insights

1. **Shape information**: The shape of the contour ellipses reveals the relative variances of the variables.
2. **Orientation information**: The orientation of the contours reveals the correlation between variables.
3. **Size information**: The size of the contours reveals the overall scale of the distribution.

## Visualization Examples

![Covariance Matrix Effects](../Images/Contour_Plots/covariance_matrix_contours.png)

## Summary of Covariance Effects

### Diagonal Covariance Matrices
- Equal variances (σ₁² = σ₂²): Circular contours
- Different variances (σ₁² ≠ σ₂²): Axis-aligned elliptical contours
- Major/minor axes proportional to the square roots of the variances

### Non-diagonal Covariance Matrices
- Produce rotated elliptical contours not aligned with coordinate axes
- Positive correlation (ρ > 0): Ellipses tilted along y = x direction
- Negative correlation (ρ < 0): Ellipses tilted along y = -x direction
- Principal axes align with the eigenvectors of the covariance matrix
- The lengths of these axes are proportional to the square roots of the eigenvalues

## Practical Applications

- Visualizing multivariate probability distributions
- Understanding correlation structure in data
- Analyzing principal components and directions of maximum variance
- Designing confidence regions for statistical inference
- Implementing anomaly detection based on Mahalanobis distance

## Related Topics

- [[L2_1_Contour_Plots|Contour Plots]]: Core principles and interpretations
- [[L2_1_Multivariate_Distributions|Multivariate Distributions]]: Theoretical foundation
- [[L2_1_Covariance_Correlation|Covariance and Correlation]]: Related concepts for understanding distribution shapes 
- [[L2_1_Contour_Plot_Examples|Contour Plot Examples]]: Worked examples of contour plots for various functions
- [[L2_1_Visual_Examples|Visual Examples]]: Additional visual examples of covariance matrix effects on contours 