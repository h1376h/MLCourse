# Contour Plots: Visual Examples

This document provides visual examples of contour plots for multivariate probability distributions, explaining their interpretation and applications in data analysis and machine learning.

## Key Concepts and Formulas

Contour plots represent multivariate probability density functions (PDFs) by showing lines or regions of equal probability density. For a bivariate distribution with PDF $f(x,y)$, a contour line connects all points $(x,y)$ where $f(x,y) = c$ for some constant $c$.

### Bivariate Normal Distribution

The bivariate normal distribution is defined by the PDF:

$$f(x,y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}}\exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_X)^2}{\sigma_X^2}+\frac{(y-\mu_Y)^2}{\sigma_Y^2}-\frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y}\right]\right)$$

Where:
- $\mu_X, \mu_Y$ = Means of $X$ and $Y$
- $\sigma_X, \sigma_Y$ = Standard deviations of $X$ and $Y$
- $\rho$ = Correlation coefficient between $X$ and $Y$

## Visual Examples

The following examples demonstrate contour plots for various probability distributions:

- **Different Covariance Structures**: Visualizing correlation and variance in bivariate normals
- **Non-Normal Distributions**: Contour plots of multimodal and non-Gaussian distributions
- **Probability Regions**: Understanding confidence regions in multivariate space
- **Conditional Distributions**: Visualizing how one variable depends on another
- **Marginal Distributions**: Relationship between joint and marginal distributions
- **3D vs 2D Representation**: Comparing contour plots with 3D surface plots

### Example 1: Bivariate Normal with Different Covariance Structures

This example shows how different correlation values and variances affect the shape of bivariate normal distributions.

![Bivariate Normal Contours](../Images/Contour_Plot_Visual_Question/bivariate_normal_contours.png)

**Key observations:**
- The standard bivariate normal (top left) shows circular contours, indicating independence between variables
- Positive correlation (top right) produces ellipses tilted upward, showing that as X increases, Y tends to increase
- Negative correlation (bottom left) produces ellipses tilted downward, showing that as X increases, Y tends to decrease
- Different variances (bottom right) produce ellipses stretched along the axis with higher variance

#### How Correlation Changes Contour Shapes

As correlation increases, the bivariate normal contours become more elongated and tilted:

![Correlation Comparison](../Images/Contour_Plot_Visual_Answer/correlation_comparison.png)

The pink ellipses show the 95% probability regions. Notice how they become more elliptical as correlation increases from 0 to 0.9, with the major axis aligning with the direction of positive correlation.

### Example 2: Non-Gaussian Probability Distributions

Contour plots can represent any bivariate distribution, not just normal distributions. Here we show contours for several non-Gaussian distributions.

![Non-Gaussian Distributions](../Images/Contour_Plot_Visual_Question/different_distributions_contours.png)

**Key observations:**
- The mixture of two normals (top left) shows two separate peaks, indicating a bimodal distribution
- The ring-shaped distribution (top right) has probability concentrated in a circular band
- The multimodal distribution (bottom left) shows three separate peaks at different locations
- The valley-shaped distribution (bottom right) has higher probability density near the origin

### Example 3: Probability Regions in Multivariate Space

Contour plots can effectively show probability regions, which are particularly useful for hypothesis testing and confidence intervals in multivariate settings.

![Probability Regions](../Images/Contour_Plot_Visual_Question/bivariate_normal_probability_regions.png)

**Key insights:**
- The innermost contour (red) represents the 39.4% probability region (equivalent to 1σ in univariate normal)
- The middle contour (green) represents the 86.5% probability region (equivalent to 2σ)
- The outermost contour (blue) represents the 98.9% probability region (equivalent to 3σ)
- The elliptical shape is due to correlation between variables (ρ = 0.5)

#### The Connection to Mahalanobis Distance

Probability regions in multivariate normal distributions are directly related to the Mahalanobis distance, which accounts for correlation between variables.

![Mahalanobis Distance](../Images/Contour_Plot_Visual_Answer/mahalanobis_distance.png)

**Left**: Probability density contours of a bivariate normal distribution.
**Right**: The same distribution represented as Mahalanobis distance contours, which form perfect circles in the normalized space.

### Example 4: Conditional Distributions

Conditional distributions show the probability distribution of one variable given a specific value of another variable. Contour plots help visualize these relationships.

![Conditional Distributions](../Images/Contour_Plot_Visual_Question/conditional_distributions.png)

**Key insights:**
- The contour plot shows the joint distribution of X and Y with correlation ρ = 0.7
- The colored curves show the conditional distributions of Y given X = -1.5, 0, and 1.5
- Each conditional distribution is a normal distribution with mean μ<sub>Y|X</sub> = ρ·X
- The width of each conditional distribution is the same, reflecting constant conditional variance σ²<sub>Y|X</sub> = σ²<sub>Y</sub>(1-ρ²)

Here's an animation frame showing a conditional distribution for X = 0:

![Conditional Distribution](../Images/Contour_Plot_Visual_Answer/conditional_demo_frame_5.png)

The red curve shows the normal distribution of Y conditional on X = 0. The mean of this conditional distribution is at the point where the red vertical line intersects the joint distribution's contours.

### Example 5: Joint and Marginal Distributions

This example illustrates the relationship between joint and marginal distributions.

![Marginal Distributions](../Images/Contour_Plot_Visual_Question/marginal_distributions.png)

**Key insights:**
- The main panel shows the joint distribution as a contour plot
- The top panel shows the marginal distribution of X (integrating out Y)
- The right panel shows the marginal distribution of Y (integrating out X)
- Both marginals are normal distributions even with correlation in the joint distribution
- This illustrates that marginal distributions lose information about relationships between variables

### Example 6: Contour Plots vs. 3D Surface Plots

Contour plots are 2D representations of 3D surfaces. This example compares the two visualization methods.

![Contour vs 3D](../Images/Contour_Plot_Visual_Question/contour_3d_comparison.png)

**Key insights:**
- Contour plots (left) show lines of equal probability density, like a topographic map
- 3D surface plots (right) show the actual probability density function in three dimensions
- Contour plots are often easier to read precise values from, while 3D plots give a better intuitive feel for the shape
- Both representations show the same underlying bivariate normal distribution with correlation ρ = 0.5

#### Geometric Interpretation

This visualization helps understand the relationship between the 3D probability density surface and its 2D contour representation:

![Geometric Interpretation](../Images/Contour_Plot_Visual_Answer/geometric_interpretation.png)

The colored horizontal planes intersect the probability density surface, creating the contour lines that are projected onto the XY plane below.

## Applications of Contour Plots

Contour plots for multivariate distributions have numerous applications in machine learning and statistics:

![Applications](../Images/Contour_Plot_Visual_Answer/applications.png)

**Clockwise from top left:**
1. **Bayesian Inference**: Visualizing posterior distributions over multiple parameters, with the MAP (maximum a posteriori) estimate and 95% credible region.
2. **Clustering Analysis**: Density-based clustering with contours showing the estimated distribution for each cluster.
3. **Error Ellipses**: Representing uncertainty in correlated measurements with nested confidence regions.
4. **Optimization Landscapes**: Visualizing objective functions to understand algorithm convergence, with optimization path shown in red.

## Questions and Solutions

### Question 1
**How would the contour plot of a bivariate normal distribution change if the correlation increased from 0.5 to 0.9?**

**Solution:** As the correlation increases from 0.5 to 0.9, the contour ellipses become more elongated and narrower, aligning more closely with the diagonal line y = x. The stronger correlation means that the probability concentrates more tightly around this diagonal, reflecting the stronger linear relationship between the variables. The ellipses' eccentricity increases significantly, with the major axis getting longer relative to the minor axis.

### Question 2
**Why are contour plots useful for visualizing multivariate probability distributions compared to other visualization methods?**

**Solution:** Contour plots are particularly effective for multivariate distributions because they:
1. Represent a 3D surface in an easily interpretable 2D format
2. Clearly show the shape, orientation, and spread of the distribution
3. Make it easy to identify regions of high probability density
4. Allow visualization of complex relationships between variables
5. Enable direct comparison of multiple distributions
6. Support quantitative analysis through labeled contour levels
7. Remain interpretable even for non-Gaussian distributions with multiple modes or unusual shapes

### Question 3
**Given the contour plot of a bivariate normal distribution, how can you determine if the variables are: a) Independent, b) Positively correlated, c) Negatively correlated?**

**Solution:**
- **Independent variables**: The contours form perfect circles centered at the mean, with axes aligned with the coordinate axes.
- **Positively correlated variables**: The contours form ellipses tilted upward (from lower left to upper right), indicating that both variables tend to increase or decrease together.
- **Negatively correlated variables**: The contours form ellipses tilted downward (from upper left to lower right), indicating that when one variable increases, the other tends to decrease.

### Question 4
**What does it mean when contour lines are close together in some regions and far apart in others?**

**Solution:** The spacing between contour lines indicates how rapidly the probability density is changing. When contour lines are close together, the probability density is changing rapidly over a short distance, indicating a steep "slope" in the distribution. When contour lines are far apart, the probability density is changing more gradually. This pattern often appears in non-Gaussian distributions, particularly multimodal ones, where contours are close together at the "shoulders" of peaks and far apart at the plateaus or valleys.

### Question 5
**How does the concept of probability regions in multivariate space extend what you know about confidence intervals for a single variable?**

**Solution:** Probability regions in multivariate space extend the concept of univariate confidence intervals by:
1. Accounting for relationships (correlations) between variables
2. Forming regions in multiple dimensions rather than intervals on a line
3. Using Mahalanobis distance instead of standard deviation to determine region boundaries
4. Requiring matrix algebra (covariance matrices) rather than simple variances
5. Producing ellipses (or ellipsoids) rather than line segments as confidence regions
6. Maintaining the same probabilistic interpretation (e.g., 95% confidence)
7. Allowing for testing multivariate hypotheses and constructing multivariate prediction regions

### Question 6
**What information is preserved in a contour plot that might be lost in separate univariate visualizations of each variable?**

**Solution:** Contour plots preserve critical information that would be lost in separate univariate visualizations:
1. Correlation and dependence structure between variables
2. The joint probability distribution's shape and orientation
3. Conditional relationships (how one variable behaves given a value of another)
4. Multimodality patterns that may not be apparent in marginal distributions
5. Regions of high joint probability that might not correspond to high marginal probabilities
6. Interaction effects between variables
7. The complete covariance structure of the data

## Related Topics

- [[L2_1_Multivariate_Distributions|Multivariate Distributions]]: The mathematical foundations of distributions in multiple dimensions
- [[L2_1_Joint_Probability|Joint Probability]]: How to work with probabilities involving multiple random variables
- [[L2_1_Covariance_Correlation|Covariance and Correlation]]: Quantifying relationships between random variables
- [[L2_1_Continuous_Distributions|Continuous Distributions]]: Background on univariate continuous distributions 