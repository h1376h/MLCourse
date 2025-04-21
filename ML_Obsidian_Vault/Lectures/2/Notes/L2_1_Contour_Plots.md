# Contour Plots for Multivariate Distributions

## Introduction
Contour plots provide a powerful visualization technique for multivariate probability distributions, particularly bivariate distributions. They represent regions of equal probability density, allowing for intuitive understanding of the relationship between random variables.

## Key Concepts

### Basic Principles
- Contour lines connect points of equal probability density
- The spacing between contours indicates the steepness of the probability density function
- Closely spaced contours indicate regions where the density changes rapidly
- Widely spaced contours indicate regions where the density changes slowly

### Mathematical Representation
For a bivariate probability density function $f(x,y)$, contour lines represent the sets:
$$\{(x,y) : f(x,y) = c\}$$
where $c$ is a constant probability density value.

## Visualization Techniques

### Simple Contour Plot
![Simple Contour Plot](../Images/simple_contour.png)

The most basic form of contour plot uses lines to connect points of equal value. This visualization shows a simple bivariate normal distribution with clear concentric contours.

### Filled Contour Plot
![Filled Contour Plot](../Images/filled_contour.png)

Filled contour plots use color to indicate regions between contour levels, making it easier to visualize the distribution of values across the domain.

### Labeled Contours
![Labeled Contour Plot](../Images/labeled_contour.png)

Adding labels to contour lines allows for quantitative assessment of the function values at different regions of the plot.

### Different Level Spacings
![Contour Level Spacing](../Images/contour_level_spacing.png)

Contour levels can be spaced in different ways to highlight specific features of a distribution:
- **Linear levels**: Equally spaced values, good for general visualization
- **Logarithmic levels**: Emphasizes small changes near zero and compresses large values
- **Custom levels**: Focused on specific thresholds of interest

### Color Maps
![Contour Colormaps](../Images/contour_colormaps.png)

Different color maps can be used to emphasize different aspects of the data:
- **Sequential** (Blues, Viridis): Good for quantities that progress from low to high
- **Diverging** (Coolwarm): Good for data with meaningful center point
- **Perceptual** (Viridis, Cividis): Designed to be perceived uniformly

### 3D Surface with Projected Contours
![3D Surface with Projected Contours](../Images/contour_3d_projection.png)

Contours can be projected onto different planes to provide additional context about the 3D surface they represent. This can help in understanding how the contours relate to the actual surface.

### Decision Boundaries
![Decision Boundary](../Images/decision_boundary_contour.png)

Contour plots are extremely useful for visualizing decision boundaries in classification problems. Here, the contour at 0.5 represents the decision boundary between two classes.

### Saddle Function
![Saddle Function](../Images/saddle_function_contour.png)

Contour plots can reveal complex topological features such as saddle points, where the function increases in one direction and decreases in another.

## Advanced Visualizations

### Bivariate Normal Distribution
![Bivariate Normal Distributions](../Images/bivariate_normal_contours.png)

Contour plots are excellent for showing how correlation affects bivariate normal distributions:
- **Uncorrelated**: Variables are independent (circular contours)
- **Positively Correlated**: Variables tend to increase together (ellipses tilted upward)
- **Negatively Correlated**: One variable tends to decrease as the other increases (ellipses tilted downward)
- **Different Variances**: One variable has greater variance than the other (elongated ellipses)

### 3D Surface vs Contour Plot
![Surface vs Contour Plot](../Images/surface_vs_contour.png)

This comparison shows the relationship between a 3D surface plot of a probability density function and its corresponding contour plot. The contour plot can be thought of as the "top-down" view of the 3D surface.

### Gaussian Mixture Model
![Gaussian Mixture](../Images/gaussian_mixture_contours.png)

A multimodal distribution created by mixing three Gaussian distributions. Notice how the contour plot clearly shows the three peaks in the distribution.

### Confidence Regions
![Confidence Regions](../Images/confidence_regions_contour.png)

This visualization demonstrates confidence regions (50%, 75%, 90%, and 95%) for a bivariate normal distribution. Each colored contour represents a boundary containing the specified percentage of the probability mass.

### Non-Gaussian Distribution
![Non-Gaussian Distribution](../Images/non_gaussian_contours.png)

Contour plots are useful for non-Gaussian distributions as well. This example shows a bivariate gamma-like distribution with asymmetric characteristics.

### Joint and Marginal Distributions
![Joint and Marginals](../Images/joint_and_marginals.png)

This figure shows a bivariate distribution (center) along with its marginal distributions along each axis, demonstrating how marginal distributions can be visualized alongside contour plots.

### Optimization Landscapes
![Optimization Landscape](../Images/optimization_landscape_contour.png)

Contour plots are widely used to visualize optimization landscapes in machine learning. This example shows a complex function with multiple local minima (marked with dots).

## Applications in Machine Learning

### Visualizing Joint Distributions
- Identifying correlation patterns between variables
- Detecting multimodality in distributions
- Understanding density concentration regions

### Likelihood and Posterior Visualization
- Exploring parameter space in statistical inference
- Visualizing likelihood functions for MLE
- Mapping posterior distributions in Bayesian inference

### Optimization Landscapes
- Visualizing objective functions in parameter space
- Identifying local and global optima
- Understanding the geometry of learning problems

## Best Practices

### Choosing Contour Levels
- Use regularly spaced probability values
- Include contours at significant quantiles (e.g., 50%, 90%, 95%)
- For log-likelihood surfaces, use appropriate level spacing

### Visualization Enhancements
- Color gradients between contours for better visibility
- Adding marginal distributions on axes
- 3D visualization with surface plots when appropriate

## Code Example

You can generate contour plot visualizations with the following code:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Codes/1_contour_plots.py
```

## Related Concepts
- [[L2_1_Multivariate_Distributions|Multivariate Distributions]]
- Level sets in optimization
- Confidence regions in parameter estimation
- Decision boundaries in classification