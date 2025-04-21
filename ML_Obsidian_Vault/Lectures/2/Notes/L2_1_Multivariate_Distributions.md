# Multivariate Distributions

## Introduction
Multivariate distributions describe the joint behavior of multiple random variables. Understanding these distributions is essential for many machine learning algorithms that deal with high-dimensional data.

## Key Concepts

### Joint Distributions
- Mathematical formulation of relationships between multiple random variables
- Extension of univariate distributions to higher dimensions
- Characterized by joint PMF (discrete) or joint PDF (continuous)

### Basic Bivariate Normal Distribution
![Basic Bivariate Normal](../Images/basic_bivariate_normal.png)

The bivariate normal distribution is the simplest and most widely used multivariate distribution. It extends the familiar bell curve to two dimensions and serves as a foundation for understanding more complex multivariate relationships.

### Correlation Visualization
![Correlation Visualization](../Images/correlation_visualization.png)

This visualization demonstrates how correlation affects the shape of bivariate normal distributions, ranging from strong negative correlation (-0.8) to strong positive correlation (0.8).

### Marginal Distributions
- Derived from joint distributions by integrating or summing out variables
- Relationship between joint and marginal distributions
- Calculation methods and examples

![Joint and Marginal Distributions](../Images/joint_and_marginals.png)

The figure above shows a bivariate normal distribution (center) with its marginal distributions along the x and y axes. The marginals are obtained by integrating the joint distribution over the other variable.

### Joint, Marginal, and Conditional Distributions
![Joint, Marginal, and Conditional Distributions](../Images/joint_marginal_conditional.png)

This comprehensive visualization shows:
- The joint distribution in the center (contour plot)
- Marginal distributions along the axes (red curves)
- A conditional distribution (blue) showing the distribution of Y given X=1

### Transformations
- Change of variables in multivariate settings
- Jacobian determinant and its role in transformations
- Common transformations and their applications

## Visualization Approaches

### Scatter Plots vs. Contour Plots
![Scatter vs Contour](../Images/scatter_vs_contour.png)

Two common approaches to visualizing multivariate data:
- **Scatter plot** (left): Shows actual data points, good for seeing patterns and outliers
- **Contour plot** (right): Shows probability density, good for understanding the underlying distribution

### Conditional Distributions
![Conditional Distributions](../Images/conditional_distributions.png)

This visualization shows several conditional distributions of Y given X=x for different values of x. The red curves represent the probability density of Y for each fixed value of X.

### 3D Visualization of Distributions
![3D Multivariate Distributions](../Images/3d_multivariate_distributions.png)

Different types of bivariate distributions visualized as 3D surfaces:
- **Standard Bivariate Normal**: The classic bell-shaped surface
- **Correlated Bivariate Normal**: Shows the effect of correlation
- **Bivariate Uniform**: Flat probability density within a square region
- **Bivariate Exponential-like**: Shows the characteristic decay in all directions

### Kernel Density Estimation
![Multivariate KDE](../Images/multivariate_kde.png)

Kernel Density Estimation (KDE) is a non-parametric way to estimate the probability density function of multivariate data:
- **Left**: Scatter plot of raw data points from a bimodal distribution
- **Right**: Smooth KDE surface reconstructed from the data points

## Common Multivariate Distributions

### Multivariate Normal (Gaussian) Distribution
- Definition and properties
- Parameters: mean vector and covariance matrix
- Geometric interpretation
- Relationship with univariate normal distribution
- Applications in machine learning

![Bivariate Normal Distributions](../Images/bivariate_normal_contours.png)

The figure above shows four different configurations of the bivariate normal distribution:
1. **Uncorrelated**: Independent variables with equal variances
2. **Positively Correlated**: Variables that tend to increase together
3. **Negatively Correlated**: Variables where one increases as the other decreases
4. **Different Variances**: Uncorrelated variables with different variances

### 3D Visualization
![3D vs 2D Visualization](../Images/surface_vs_contour.png)

This comparison shows a bivariate normal distribution as both a 3D surface (left) and a contour plot (right), demonstrating two ways to visualize the same multivariate distribution.

### Different Types of Multivariate Distributions
![Different Multivariate Distributions](../Images/different_multivariate_distributions.png)

Four different types of bivariate distributions:
- **Standard Bivariate Normal**: Circular contours for uncorrelated variables
- **Bivariate Student's t**: Heavier tails than normal distribution
- **Bivariate Uniform**: Equal probability within a bounded region
- **Bivariate Exponential-like**: Higher probability near the origin with rapid decay

### Mixture Models
- Combinations of multiple distributions
- Often used for modeling complex, multimodal data
- Common application: clustering and density estimation

![Gaussian Mixture Model](../Images/gaussian_mixture_contours.png)

This figure shows a Gaussian mixture model with three components, resulting in a multimodal distribution with three distinct peaks.

### Non-Gaussian Distributions
- Distributions with asymmetric shapes or bounded support
- Examples: multivariate gamma, Dirichlet, Wishart
- Applications in specialized domains

![Non-Gaussian Distribution](../Images/non_gaussian_contours.png)

The figure shows a non-Gaussian bivariate distribution with asymmetric shape and bounded support (positive values only).

### Multivariate Student's t-Distribution
- Heavier tails than the multivariate normal
- Degrees of freedom parameter
- Robustness to outliers
- Applications in robust statistics

### Dirichlet Distribution
- Multivariate generalization of the beta distribution
- Application in Bayesian inference for categorical data
- Relationship with multinomial distribution
- Role in topic models (e.g., Latent Dirichlet Allocation)

### Wishart Distribution
- Distribution over positive-definite matrices
- Role in Bayesian inference for covariance matrices
- Connection to chi-squared distribution

## Statistical Properties and Applications

### Confidence Regions
- Extensions of confidence intervals to multiple dimensions
- Geometric interpretation: ellipsoids for normal distributions
- Applications in parameter estimation and hypothesis testing

![Confidence Regions](../Images/confidence_regions_contour.png)

This visualization shows confidence regions at different levels (50%, 75%, 90%, and 95%) for a bivariate normal distribution.

### Optimization Landscapes
- Multivariate functions often define loss landscapes in ML
- Understanding these landscapes helps in algorithm selection
- Contour plots provide insight into function behavior

![Optimization Landscape](../Images/optimization_landscape_contour.png)

This figure shows an optimization landscape with multiple local minima (marked with dots), illustrating the complexity of multivariate optimization problems in machine learning.

## Applications in Machine Learning

### Density Estimation
- Kernel density estimation in multiple dimensions
- Parametric and non-parametric approaches
- Curse of dimensionality

### Generative Models
- Sampling from multivariate distributions
- Creating synthetic data with realistic dependencies
- Modeling complex real-world phenomena

### Copulas
- Modeling dependency structures separately from marginals
- Flexible approach to multivariate modeling
- Applications in risk modeling

## Mathematical Properties

### Moment Generating Functions
- Extension to multiple dimensions
- Deriving joint moments

### Characteristic Functions
- Fourier transforms of multivariate distributions
- Uniqueness and inversion properties

### Information-Theoretic Measures
- Multivariate entropy
- Kullback-Leibler divergence
- Mutual information

## Code Example

You can generate visualizations for multivariate distributions with the following code:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Codes/1_multivariate_distributions.py
```

## Related Topics
- [[L2_1_Contour_Plots|Contour Plots]]: Techniques for visualizing multivariate distributions
- [[L2_1_Beta_Distribution|Beta Distribution]]: A univariate distribution related to the Dirichlet distribution
- [[L2_5_Bayesian_Inference|Bayesian Inference]]: Applications of multivariate distributions in Bayesian statistics
