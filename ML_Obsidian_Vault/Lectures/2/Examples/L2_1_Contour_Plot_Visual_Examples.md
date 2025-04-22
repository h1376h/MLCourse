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

## Examples

The following examples demonstrate contour plots for various probability distributions:

- **Different Covariance Structures**: Visualizing correlation and variance in bivariate normals
- **Non-Normal Distributions**: Contour plots of multimodal and non-Gaussian distributions
- **Probability Regions**: Understanding confidence regions in multivariate space
- **Conditional Distributions**: Visualizing how one variable depends on another
- **Marginal Distributions**: Relationship between joint and marginal distributions
- **3D vs 2D Representation**: Comparing contour plots with 3D surface plots

### Example 1: Bivariate Normal with Different Covariance Structures

#### Problem Statement
Visualize and analyze how different correlation values and variances affect the shape of bivariate normal distributions in contour plots. Specifically, explore:

1. How does the shape of contour plots change when correlation is zero, positive, or negative?
2. What happens to the contour shapes when the variances of the two variables are different?
3. How can we interpret the resulting patterns in terms of the relationship between the variables?

![Bivariate Normal Contours](../Images/Contour_Plot_Visual_Question/bivariate_normal_contours.png)

#### Solution

**Key observations:**
- The standard bivariate normal (top left) shows circular contours, indicating independence between variables
- Positive correlation (top right) produces ellipses tilted upward, showing that as $X$ increases, $Y$ tends to increase
- Negative correlation (bottom left) produces ellipses tilted downward, showing that as $X$ increases, $Y$ tends to decrease
- Different variances (bottom right) produce ellipses stretched along the axis with higher variance

The shape of contour plots in a bivariate normal distribution changes in distinct ways when we modify correlation and variance:

1. **Zero correlation (ρ = 0)**: Contours form perfect circles when variances are equal, indicating independent variables.

2. **Positive correlation (ρ > 0)**: Contours stretch into ellipses tilted upward (from lower left to upper right), becoming more elongated as correlation increases.

3. **Negative correlation (ρ < 0)**: Contours form ellipses tilted downward (from upper left to lower right), with stronger elongation as correlation becomes more negative.

4. **Different variances**: Contours stretch along the axis with greater variance, forming ellipses aligned with the coordinate axes when correlation is zero.

##### How Correlation Changes Contour Shapes

As correlation increases, the bivariate normal contours become more elongated and tilted:

![Correlation Comparison](../Images/Contour_Plot_Visual_Answer/correlation_comparison.png)

The pink ellipses show the 95% probability regions. Notice how they become more elliptical as correlation increases from $\rho = 0$ to $\rho = 0.9$, with the major axis aligning with the direction of positive correlation.

![Step-by-step Correlation](../Images/Contour_Plot_Visual_Answer/step_by_step_correlation.png)

As correlation increases from 0 to 0.9, we observe:
- The contours transform from circles to increasingly elongated ellipses
- The major axis of the ellipse aligns with the direction of correlation
- The 95% probability region (pink) becomes narrower perpendicular to the correlation direction
- The uncertainty is increasingly concentrated along the correlation direction

This visualization demonstrates how correlation affects our uncertainty about one variable given knowledge of the other, with stronger correlation leading to more predictable relationships.

### Example 2: Non-Gaussian Probability Distributions

#### Problem Statement
Explore how contour plots can effectively represent various types of non-Gaussian bivariate distributions. For each distribution type shown below:

1. Identify the key features visible in the contour plot (number of modes, shape, orientation)
2. Interpret what these features tell us about the underlying probability distribution
3. Compare how these non-Gaussian distributions differ from the bivariate normal in terms of their contour patterns

![Non-Gaussian Distributions](../Images/Contour_Plot_Visual_Question/different_distributions_contours.png)

#### Solution

**Key observations:**
- The mixture of two normals (top left) shows two separate peaks, indicating a bimodal distribution
- The ring-shaped distribution (top right) has probability concentrated in a circular band
- The multimodal distribution (bottom left) shows three separate peaks at different locations
- The valley-shaped distribution (bottom right) has higher probability density near the origin

Contour plots effectively represent complex, non-Gaussian probability distributions through their level curves:

1. **Mixture of Normals**: Multiple peaks appear as separate "islands" of contours, with each peak representing a component of the mixture. The contours reveal both the location and relative height of each mode.

2. **Ring-Shaped Distribution**: Contours form concentric closed curves, with the highest density along a circular band rather than at a central point, indicating probability mass concentrated in a ring.

3. **Multimodal Distribution**: Multiple distinct peaks appear, with contours clustering around each mode. The spacing between contours shows how sharply probability density changes near each peak.

4. **Valley-Shaped Distribution**: Contours form a diamond-like pattern, with density decreasing linearly from the center in all directions, creating a tent-like probability surface.

These visualizations provide valuable insights:
- Location and number of modes in the distribution
- Areas of high probability density
- Shape and orientation of the distribution
- Unusual features like rings, ridges, or valleys
- Regions of rapid vs. gradual change in probability density

Contour plots make these complex structures interpretable in ways that would be difficult to visualize with marginal distributions or summary statistics alone.

### Example 3: Probability Regions in Multivariate Space

#### Problem Statement
Probability regions are essential for hypothesis testing and confidence intervals in multivariate settings. How can contour plots effectively represent these regions?

#### Solution

![Probability Regions](../Images/Contour_Plot_Visual_Question/bivariate_normal_probability_regions.png)

**Key insights:**
- The innermost contour (red) represents the 39.4% probability region (equivalent to $1\sigma$ in univariate normal)
- The middle contour (green) represents the 86.5% probability region (equivalent to $2\sigma$)
- The outermost contour (blue) represents the 98.9% probability region (equivalent to $3\sigma$)
- The elliptical shape is due to correlation between variables ($\rho = 0.5$)

Contour lines in a multivariate normal distribution represent regions of equal probability density. These contours have a direct relationship with probability regions and the Mahalanobis distance:

1. **Elliptical Regions**: For a bivariate normal distribution, contours form ellipses whose shape depends on the covariance structure of the variables.

2. **Probability Content**: Each contour encloses a specific probability mass. For bivariate normal distributions:
   - The innermost contour (red) contains 39.4% of the probability
   - The middle contour (green) contains 86.5% of the probability
   - The outermost contour (blue) contains 98.9% of the probability

3. **Mahalanobis Distance**: Each contour corresponds to a specific Mahalanobis distance from the mean, which accounts for the correlation between variables.

4. **Extension of Confidence Intervals**: These probability regions extend the concept of univariate confidence intervals to multiple dimensions by:
   - Accounting for correlation between variables
   - Forming elliptical regions rather than simple intervals
   - Using the covariance matrix to determine the shape and orientation
   - Providing the same probabilistic interpretation as univariate intervals

This framework allows us to make statistically valid statements about the joint probability of multiple variables, which is essential for hypothesis testing, outlier detection, and uncertainty quantification in multivariate settings.

##### The Connection to Mahalanobis Distance

Probability regions in multivariate normal distributions are directly related to the Mahalanobis distance, which accounts for correlation between variables.

![Mahalanobis Distance](../Images/Contour_Plot_Visual_Answer/mahalanobis_distance.png)

**Left**: Probability density contours of a bivariate normal distribution.
**Right**: The same distribution represented as Mahalanobis distance contours, which form perfect circles in the normalized space.

The step-by-step visualization of Mahalanobis distance illustrates this concept:

![Mahalanobis Step 1: Probability Density Contours](../Images/Contour_Plot_Visual_Answer/mahalanobis_step1.png)

![Mahalanobis Step 2: Covariance Matrix and Its Inverse](../Images/Contour_Plot_Visual_Answer/mahalanobis_step2.png)

![Mahalanobis Step 3: Mahalanobis Distance Contours](../Images/Contour_Plot_Visual_Answer/mahalanobis_step3.png)

![Mahalanobis Step 4: Comparison of Contours](../Images/Contour_Plot_Visual_Answer/mahalanobis_step4.png)

The step-by-step visualization shows:
1. The original bivariate normal distribution with elliptical contours (step 1)
2. The covariance matrix and its inverse used in Mahalanobis distance calculation (step 2)
3. How Mahalanobis distance contours form circles in standardized space (step 3)
4. Side-by-side comparison showing the relationship between probability density and Mahalanobis distance (step 4)

### Example 4: Conditional Distributions

#### Problem Statement
How do conditional distributions change as we fix one variable at different values? Contour plots help visualize these relationships.

#### Solution

![Conditional Distributions](../Images/Contour_Plot_Visual_Question/conditional_distributions.png)

**Key insights:**
- The contour plot shows the joint distribution of $X$ and $Y$ with correlation $\rho = 0.7$
- The colored curves show the conditional distributions of $Y$ given $X = -1.5$, $0$, and $1.5$
- Each conditional distribution is a normal distribution with mean $\mu_{Y|X} = \rho \cdot X$
- The width of each conditional distribution is the same, reflecting constant conditional variance $\sigma^2_{Y|X} = \sigma^2_Y(1-\rho^2)$

Conditioning on specific values of one variable reveals important properties of the relationship between variables in a bivariate normal distribution:

1. **Shape of Conditional Distributions**: For bivariate normal distributions, all conditional distributions are normal, regardless of the conditioning value.

2. **Changing Mean**: As we vary the conditioning value (X), the mean of the conditional distribution (Y|X) shifts along the regression line. For a standard bivariate normal with correlation ρ:
   - The conditional mean follows E[Y|X=x] = ρx
   - This linear relationship is the foundation of regression analysis

3. **Constant Variance**: The variance of Y|X remains constant regardless of the value of X:
   - Var(Y|X) = σ²ᵧ(1-ρ²)
   - Stronger correlation leads to smaller conditional variance

The step-by-step visualization of conditional distributions:

![Conditional Step 1: Joint Distribution](../Images/Contour_Plot_Visual_Answer/conditional_step1.png)

![Conditional Step 2: Conditioning on X=0](../Images/Contour_Plot_Visual_Answer/conditional_step2.png)

![Conditional Step 3: Resulting Conditional Distribution](../Images/Contour_Plot_Visual_Answer/conditional_step3.png)

![Conditional Step 4: Changing the Conditioning Value](../Images/Contour_Plot_Visual_Answer/conditional_step4.png)

The step-by-step visualization shows:
1. The joint bivariate normal distribution (step 1)
2. Conditioning on X = 0 (vertical slice through the joint distribution) (step 2)
3. The resulting conditional distribution of Y given X = 0 (step 3)
4. How conditional distributions change as we vary the conditioning value of X (step 4)

Here's an animation sequence showing how conditional distributions change as we vary the value of $X$:

![Conditional Distribution (X = -2.0)](../Images/Contour_Plot_Visual_Answer/conditional_demo_frame_1.png)

![Conditional Distribution (X = -1.5)](../Images/Contour_Plot_Visual_Answer/conditional_demo_frame_2.png)

![Conditional Distribution (X = -1.0)](../Images/Contour_Plot_Visual_Answer/conditional_demo_frame_3.png)

![Conditional Distribution (X = -0.5)](../Images/Contour_Plot_Visual_Answer/conditional_demo_frame_4.png)

![Conditional Distribution (X = 0.0)](../Images/Contour_Plot_Visual_Answer/conditional_demo_frame_5.png)

![Conditional Distribution (X = 0.5)](../Images/Contour_Plot_Visual_Answer/conditional_demo_frame_6.png)

![Conditional Distribution (X = 1.0)](../Images/Contour_Plot_Visual_Answer/conditional_demo_frame_7.png)

![Conditional Distribution (X = 1.5)](../Images/Contour_Plot_Visual_Answer/conditional_demo_frame_8.png)

![Conditional Distribution (X = 2.0)](../Images/Contour_Plot_Visual_Answer/conditional_demo_frame_9.png)

As $X$ increases from $-2.0$ to $2.0$, notice how the conditional distribution (red curve) shifts along the regression line, with its mean at $\rho \cdot X$. This visualizes how the expected value of $Y$ changes with $X$ while maintaining the same variance.

This visualization demonstrates key principles:
- Conditioning creates a "slice" through the joint distribution
- The mean of Y|X follows the regression line
- The uncertainty in Y given X (conditional variance) is reduced compared to the marginal variance
- The strength of correlation determines how much information X provides about Y

These principles form the foundation of prediction, regression, and conditional inference in statistics and machine learning.

### Example 5: Joint and Marginal Distributions

#### Problem Statement
What is the relationship between joint and marginal distributions, and how does correlation affect this relationship?

#### Solution

![Marginal Distributions](../Images/Contour_Plot_Visual_Question/marginal_distributions.png)

**Key insights:**
- The main panel shows the joint distribution as a contour plot
- The top panel shows the marginal distribution of $X$ (integrating out $Y$)
- The right panel shows the marginal distribution of $Y$ (integrating out $X$)
- Both marginals are normal distributions even with correlation in the joint distribution
- This illustrates that marginal distributions lose information about relationships between variables

Joint and marginal distributions have a complex relationship, and examining only marginals can lead to loss of important information:

1. **Mathematical Relationship**: The marginal distribution of X is obtained by integrating the joint distribution over all values of Y: p(x) = ∫p(x,y)dy. Similarly for the marginal of Y.

2. **Information Preserved in Marginals**:
   - The central tendency (mean) of each variable
   - The dispersion (variance) of each variable
   - The shape of each variable's distribution (skewness, kurtosis, etc.)

3. **Information Lost in Marginals**:
   - The correlation structure between variables
   - Conditional relationships
   - Any interaction effects
   - Multimodal patterns that exist only in the joint space

4. **Normal Distribution Case**: For bivariate normal distributions, the marginals are also normal, but they don't contain information about the correlation between variables. Two joint distributions with completely different correlation structures can have identical marginal distributions.

5. **Simpson's Paradox**: In extreme cases, the relationships apparent in marginal distributions can be completely reversed when examining the joint distribution, leading to Simpson's paradox.

The joint distribution provides a complete picture of the relationship between variables, while marginals provide only a partial view. This highlights the importance of multivariate analysis and visualizations like contour plots that can reveal the full structure of the data.

### Example 6: Contour Plots vs. 3D Surface Plots

#### Problem Statement
How do contour plots compare to 3D surface plots for visualizing bivariate distributions?

#### Solution

![Contour vs 3D](../Images/Contour_Plot_Visual_Question/contour_3d_comparison.png)

**Key insights:**
- Contour plots (left) show lines of equal probability density, like a topographic map
- 3D surface plots (right) show the actual probability density function in three dimensions
- Contour plots are often easier to read precise values from, while 3D plots give a better intuitive feel for the shape
- Both representations show the same underlying bivariate normal distribution with correlation $\rho = 0.5$

Contour plots and 3D surface plots provide complementary representations of the same underlying probability distribution:

1. **Geometric Relationship**: Contour plots are the top-down view of a 3D probability surface, where lines connect points of equal height (probability density).

![Geometric Interpretation](../Images/Contour_Plot_Visual_Answer/geometric_interpretation.png)

2. **Step-by-Step Formation of Contours**:
   - The 3D surface represents the probability density at each point (x,y)
   - Horizontal planes at different heights intersect the surface
   - These intersections form curves of equal probability density
   - Projecting these curves to the xy-plane creates the contour lines

The following sequence illustrates the geometric relationship between 3D surfaces and contour plots:

![Geometric Step 1: 3D Probability Density Surface](../Images/Contour_Plot_Visual_Answer/geometric_step1.png)

![Geometric Step 2: Horizontal Slice at Density = 0.1](../Images/Contour_Plot_Visual_Answer/geometric_step2.png)

![Geometric Step 3: Intersection Curve](../Images/Contour_Plot_Visual_Answer/geometric_step3.png)

![Geometric Step 4: Projecting to 2D](../Images/Contour_Plot_Visual_Answer/geometric_step4.png)

![Geometric Step 5: Multiple Contour Lines](../Images/Contour_Plot_Visual_Answer/geometric_step5.png)

The step-by-step visualization shows:
1. The 3D probability density surface (step 1)
2. Adding a horizontal slice through the surface (step 2)
3. The intersection curve formed at the slice (step 3)
4. Projecting this curve to the XY plane creates a contour (step 4)
5. Multiple slices create multiple contours, forming a complete contour plot (step 5)

3. **Advantages of Contour Plots**:
   - Easier to read precise values and locations
   - Better for comparing multiple distributions
   - Clearer representation of the location of modes and regions
   - Less visual distortion due to perspective
   - More effective for quantitative analysis

4. **Advantages of 3D Surface Plots**:
   - More intuitive visualization of the probability "landscape"
   - Better for understanding the overall shape and relative heights
   - Provides a memorable visual impression of the distribution
   - Helpful for explaining the concept of probability density

The relationship between these visualizations is analogous to topographic maps versus 3D terrain models. Contour plots provide precision and clarity for analysis, while 3D plots give intuition and overall shape understanding. Together, they provide a complete picture of multivariate probability distributions.

## Key Insights

### Theoretical Insights
- Contour plots of bivariate normal distributions form ellipses whose orientation and shape are determined by correlation and variances
- Conditional distributions in bivariate normals follow the regression line with constant variance
- Probability regions in multivariate space generalize univariate confidence intervals
- Mahalanobis distance provides a correlation-adjusted measure of distance in multivariate space

### Practical Applications
- Contour plots reveal correlation structures that might be missed in univariate or marginal analyses
- They allow visualization of complex, non-Gaussian distributions with multiple modes
- Understanding conditional distributions is fundamental for regression and prediction
- Probability regions provide a framework for multivariate hypothesis testing and outlier detection

### Common Pitfalls
- Examining only marginal distributions can hide important correlation structure
- Misinterpreting contour spacing (steep vs. gradual changes in probability density)
- Forgetting that probability content of contours depends on dimensionality
- Overlooking Simpson's paradox effects when analyzing grouped vs. ungrouped data

## Running the Examples

You can run the code that generates these examples and visualizations using:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Codes/L2_1_Contour_Plot_Visual_question.py
python3 ML_Obsidian_Vault/Lectures/2/Codes/L2_1_Contour_Plot_Visual_answer.py
```

## Related Topics

- [[L2_1_Multivariate_Distributions|Multivariate Distributions]]: The mathematical foundations of distributions in multiple dimensions
- [[L2_1_Joint_Probability|Joint Probability]]: How to work with probabilities involving multiple random variables
- [[L2_1_Covariance_Correlation|Covariance and Correlation]]: Quantifying relationships between random variables
- [[L2_1_Continuous_Distributions|Continuous Distributions]]: Background on univariate continuous distributions 