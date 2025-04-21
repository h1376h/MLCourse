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
- **Real-World Covariance**: Exploring height-weight relationships as a natural example of covariance
- **Rotation Effects**: Understanding how rotation affects covariance structure
- **Mahalanobis Distance**: Comparing Euclidean and Mahalanobis distances for correlated data
- **Emoji Visualization**: Using intuitive visual metaphors to understand correlation types
- **Sketching Contours**: Interactive visualization for sketching contours of a bivariate normal distribution

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

### Example 5: Height-Weight Relationship - Real-World Covariance

#### Problem Statement
How does natural covariance appear in the real world, and how can it be visualized using height and weight data?

In this example:
- We use simulated height and weight data that exhibits natural positive correlation
- We calculate and visualize the covariance matrix and correlation
- We demonstrate principal components as directions of maximum variance in the data

#### Solution

We'll analyze how height and weight naturally covary in human measurements.

##### Step 1: Generate and Plot Correlated Data
We create simulated height and weight data with a positive correlation, representing the natural relationship between these variables.

##### Step 2: Calculate the Covariance Matrix
The covariance matrix quantifies the relationship between height and weight, showing how they vary together.

##### Step 3: Visualize with Confidence Ellipses
We draw ellipses to represent the regions containing approximately 68% and 95% of the data (1σ and 2σ), with the orientation determined by the principal components.

##### Step 4: Interpret Principal Components
The principal components (eigenvectors of the covariance matrix) show the main directions of variance:
- The first principal component represents the "growth direction" where both height and weight increase
- The second principal component represents variations in body type (weight relative to height)

![Simple Covariance Real-World Example](../Images/Contour_Plots/simple_covariance_real_world.png)

### Example 6: Effects of Rotation on Covariance Structure

#### Problem Statement
What happens to the covariance matrix when we rotate a dataset, and why is this important?

In this example:
- We start with uncorrelated data (independent variables with diagonal covariance)
- We apply rotation transformations of different angles
- We observe how rotation introduces correlation between variables

#### Solution

We'll explore how geometric rotations affect the covariance structure of data.

##### Step 1: Generate Independent Data
We create a dataset with independent variables (diagonal covariance matrix) centered at the origin.

##### Step 2: Apply Rotation Transformations
We apply rotation matrices with angles of 0°, 30°, and 60° to the original data.

##### Step 3: Calculate New Covariance Matrices
For each rotation, we calculate the resulting covariance matrix and correlation coefficient.

##### Step 4: Visualize and Compare Results
We plot each rotated dataset with its covariance ellipse, demonstrating how rotation systematically introduces correlation:
- At 0° rotation: No correlation (Cov(x,y) = 0)
- At 30° rotation: Moderate positive correlation
- At 60° rotation: Stronger positive correlation (approaching maximum at 45°)

The total variance (trace of covariance matrix) remains constant throughout all rotations.

![Toy Data Covariance Change with Rotation](../Images/Contour_Plots/toy_data_covariance_change.png)

### Example 7: Mahalanobis Distance vs Euclidean Distance

#### Problem Statement
Why is Euclidean distance inadequate for correlated data, and how does Mahalanobis distance address this limitation?

In this example:
- We generate correlated data from a multivariate normal distribution
- We calculate both Euclidean and Mahalanobis distances for selected test points
- We visualize the difference between these distance metrics

#### Solution

We'll compare two distance metrics and understand why Mahalanobis distance is more appropriate for correlated data.

##### Step 1: Generate Correlated Data
We create data from a multivariate normal distribution with positive correlation.

##### Step 2: Calculate Distances for Test Points
We select specific test points and calculate their Mahalanobis distances from the mean, which accounts for the covariance structure.

##### Step 3: Visualize Distance Contours
We plot contours of equal Mahalanobis distance (ellipses) and equal Euclidean distance (circles) to show the fundamental difference:
- Euclidean distance treats all directions equally (circles)
- Mahalanobis distance accounts for correlation (ellipses aligned with the data)

##### Step 4: Interpret the Results
We find that points at the same Euclidean distance can have very different Mahalanobis distances:
- Points along the principal axis of correlation have smaller Mahalanobis distances
- Points perpendicular to the correlation direction have larger Mahalanobis distances

This makes Mahalanobis distance much more suitable for anomaly detection and classification in correlated data.

![Mahalanobis Distance Visualization](../Images/Contour_Plots/simple_mahalanobis_distance.png)

### Example 8: The Emoji Guide to Correlation

#### Problem Statement
How can we intuitively understand positive and negative correlation using everyday visual metaphors?

In this example:
- We create a fun visualization using emoji-like faces to represent correlation concepts
- We contrast positive and negative correlation structures
- We provide an intuitive memory aid for understanding correlation direction

#### Solution

We'll use visual metaphors to make correlation concepts more intuitive and memorable.

##### Step 1: Positive Correlation Visualization
We create a "happy face" alongside data showing positive correlation, where variables tend to increase or decrease together:
- Data points form a pattern from bottom-left to top-right
- The covariance ellipse is tilted along the y = x direction
- This pattern is seen in naturally related quantities (height-weight, study time-grades)

##### Step 2: Negative Correlation Visualization
We create a "sad face" alongside data showing negative correlation, where as one variable increases, the other tends to decrease:
- Data points form a pattern from top-left to bottom-right
- The covariance ellipse is tilted along the y = -x direction
- This pattern is common in trade-off relationships (price-demand, speed-accuracy)

##### Step 3: Visual Mnemonic
We establish a visual mnemonic that connects the emotional expressions to the mathematical concept:
- Smile curves upward ⌣ like positive correlation
- Frown curves downward ⌢ like negative correlation

This visual approach helps anchor abstract statistical concepts in intuitive, memorable imagery.

![Emoji Covariance Visualization](../Images/Contour_Plots/emoji_covariance_example.png)

### Example 9: Sketching Contours of a Bivariate Normal Distribution

#### Problem Statement
Sketch the contour lines for the probability density function of a bivariate normal distribution with mean $\mu = (0,0)$ and covariance matrix $\Sigma = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$.

In this example:
- We need to understand how to translate the mathematical expression into a visual representation
- We must identify the geometric shape formed by the contours
- We must recognize the effect of the identity covariance matrix on these shapes

#### Solution

We'll analyze the mathematical formula and derive the shape of the contour lines.

##### Step 1: Understand the Mathematical Formula
The probability density function of a bivariate normal distribution is:

$$f(x,y) = \frac{1}{2\pi\sqrt{|\Sigma|}} \exp\left(-\frac{1}{2}(X-\mu)^T \Sigma^{-1} (X-\mu)\right)$$

For our specific case with $\mu = (0,0)$ and $\Sigma = I$ (the identity matrix), this simplifies to:

$$f(x,y) = \frac{1}{2\pi} \exp\left(-\frac{x^2 + y^2}{2}\right)$$

##### Step 2: Identify the Equation for Contour Lines
For a specific contour value $c$, the points $(x,y)$ on the contour satisfy:

$$\frac{1}{2\pi} \exp\left(-\frac{x^2 + y^2}{2}\right) = c$$

Taking the natural logarithm of both sides and rearranging:

$$x^2 + y^2 = -2\ln(2\pi c)$$

This is the equation of a circle with radius $r = \sqrt{-2\ln(2\pi c)}$

##### Step 3: Sketch the Contours
The contours form concentric circles centered at the origin $(0,0)$:
- Higher probability density (larger $c$) corresponds to circles with smaller radii
- Lower probability density (smaller $c$) corresponds to circles with larger radii
- The 1σ, 2σ, and 3σ circles have radii of 1, 2, and 3 respectively

##### Step 4: Interpret the Results
The circular contours indicate:
- Equal spread of probability in all directions (isotropic)
- No correlation between variables (axes are aligned with coordinate axes)
- The standard deviations in both dimensions are equal (both 1)

This special case of the bivariate normal is called the standard bivariate normal distribution, and it forms the foundation for understanding more complex covariance structures.

![Sketch Contour Problem Visualization](../Images/Contour_Plots/sketch_contour_problem.png)

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