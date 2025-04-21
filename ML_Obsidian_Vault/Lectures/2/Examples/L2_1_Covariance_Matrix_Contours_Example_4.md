# Example 4: Eigenvalues, Eigenvectors, and Covariance

## Problem Statement
How do the eigenvalues and eigenvectors of a covariance matrix relate to the shape and orientation of probability density contours? How does correlation strength affect the geometric properties of multivariate distributions?

We'll examine how increasing correlation affects the eigenvalues and eigenvectors using four covariance matrices with increasing correlation:
1. No correlation (ρ = 0): $\Sigma = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$
2. Weak correlation (ρ = 0.3): $\Sigma = \begin{bmatrix} 1.0 & 0.3 \\ 0.3 & 1.0 \end{bmatrix}$
3. Moderate correlation (ρ = 0.6): $\Sigma = \begin{bmatrix} 1.0 & 0.6 \\ 0.6 & 1.0 \end{bmatrix}$
4. Strong correlation (ρ = 0.9): $\Sigma = \begin{bmatrix} 1.0 & 0.9 \\ 0.9 & 1.0 \end{bmatrix}$

## Understanding the Problem
The eigendecomposition of a covariance matrix provides fundamental insights into the shape, size, and orientation of the probability density contours. In multivariate statistics, eigenvectors represent the principal directions of variation, while eigenvalues represent the amount of variation along those directions. This relationship is essential for understanding multivariate data distributions and forms the basis for techniques like Principal Component Analysis (PCA).

## Solution

### Step 1: Mathematical Background
The covariance matrix $\Sigma$ can be decomposed as:

$$\Sigma = V \Lambda V^T$$

Where:
- $V$ contains eigenvectors (principal directions) as columns
- $\Lambda$ is a diagonal matrix of eigenvalues (variances along principal directions)
- This decomposition helps us understand the shape and orientation of the contours

The eigenvectors determine the orientation of the principal axes of the elliptical contours, while the eigenvalues determine the length of these axes (proportional to the square root of the eigenvalues).

### Step 2: No Correlation (ρ = 0)
Covariance matrix:
$$\Sigma = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$$

Properties:
- Equal variances in both dimensions
- Zero correlation
- Eigenvalues: $\lambda_1 = \lambda_2 = 1$
- Eigenvectors: $v_1 = [1, 0]$, $v_2 = [0, 1]$ (aligned with coordinate axes)
- Circular contours (equal variance in all directions)
- No preferred direction of variability in the data

When there's no correlation, the standard deviations along all directions are equal, resulting in a perfect circle. The probability density function has the same spread in all directions.

### Step 3: Weak Correlation (ρ = 0.3)
Covariance matrix:
$$\Sigma = \begin{bmatrix} 1.0 & 0.3 \\ 0.3 & 1.0 \end{bmatrix}$$

Properties:
- Equal variances in both dimensions
- Weak positive correlation ($\rho = 0.3$)
- Eigenvalues: $\lambda_1 \approx 1.3$, $\lambda_2 \approx 0.7$
- Eigenvectors begin to rotate away from the coordinate axes
- Slightly elliptical contours with mild rotation
- Beginning to show a preferred direction of variability

The larger eigenvalue ($\lambda_1 \approx 1.3$) corresponds to the direction of maximum variance, which is now rotated slightly toward the $y = x$ line. We can see that the variance along this direction is starting to increase, while the variance in the perpendicular direction decreases.

### Step 4: Moderate Correlation (ρ = 0.6)
Covariance matrix:
$$\Sigma = \begin{bmatrix} 1.0 & 0.6 \\ 0.6 & 1.0 \end{bmatrix}$$

Properties:
- Equal variances in both dimensions
- Moderate positive correlation ($\rho = 0.6$)
- Eigenvalues: $\lambda_1 \approx 1.6$, $\lambda_2 \approx 0.4$
- Eigenvectors rotate further from the coordinate axes
- More eccentric elliptical contours with significant rotation
- Clear preferred direction of variability emerges

The disparity between eigenvalues increases, leading to more elongated ellipses. The first eigenvector (associated with $\lambda_1$) moves closer to the direction $[1, 1]$ (the $y = x$ line). The ellipses are now approximately 2:1 in terms of their axis lengths.

### Step 5: Strong Correlation (ρ = 0.9)
Covariance matrix:
$$\Sigma = \begin{bmatrix} 1.0 & 0.9 \\ 0.9 & 1.0 \end{bmatrix}$$

Properties:
- Equal variances in both dimensions
- Strong positive correlation ($\rho = 0.9$)
- Eigenvalues: $\lambda_1 \approx 1.9$, $\lambda_2 \approx 0.1$
- Eigenvectors nearly align with the $y = x$ and $y = -x$ directions
- Highly eccentric elliptical contours with strong rotation
- Dominant direction of variability along the first eigenvector
- Very little variability along the second eigenvector

The first eigenvector is now very close to $[1, 1]/\sqrt{2}$, and the second eigenvector approaches $[-1, 1]/\sqrt{2}$. The ratio of eigenvalues ($\lambda_1/\lambda_2 \approx 19$) indicates that the ellipses are 19 times longer in one direction than the other. This extreme elongation shows that the variables are almost perfectly correlated.

### Step 6: Mathematical Relationship
For a covariance matrix with equal variances and correlation $\rho$:
$$\Sigma = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}$$

The eigenvalues are:
$$\lambda_1 = 1 + \rho$$
$$\lambda_2 = 1 - \rho$$

The eigenvectors are:
$$v_1 = [1, 1]/\sqrt{2}$$
$$v_2 = [-1, 1]/\sqrt{2}$$

As $\rho$ approaches 1, $\lambda_1$ approaches 2 and $\lambda_2$ approaches 0, making the ellipses increasingly elongated along the $y = x$ direction. This mathematical relationship perfectly explains the geometric progression we observe in the visualizations.

### Step 7: 3D Visualization of the Effect
The correlation also dramatically affects the 3D probability density surface:

- With no correlation (blue surface), the PDF forms a symmetric bell curve
- With strong correlation (red surface), the PDF stretches along the $y = x$ direction and becomes more concentrated
- The total volume under both surfaces remains constant (= 1), as required by probability theory
- The peak height increases with correlation due to the decrease in the determinant of the covariance matrix

### Step 8: Key Insights
- As correlation increases, eigenvalues become more disparate
- The largest eigenvalue increases, the smallest decreases
- The orientation of eigenvectors approaches $y = x$ (for positive correlation)
- The ellipses become increasingly elongated (higher eccentricity)
- This illustrates why PCA works: it identifies the directions of maximum variance
- When variables are strongly correlated, most of the information can be captured by a single principal component
- The determinant of the covariance matrix (which equals the product of eigenvalues) decreases with increasing correlation, affecting the peak height of the PDF

## Visual Explanations

### Basic Concept Visualization
![Concept Visualization](../Images/Contour_Plots/ex4_concept_visualization.png)
*Basic concept visualization showing how eigenvalues and eigenvectors determine the shape and orientation of probability distributions. Left: Circular distribution with equal eigenvalues. Middle: Axis-aligned elliptical distribution with unequal eigenvalues. Right: Rotated elliptical distribution with correlated variables.*

### Correlation and Eigenvalue Relationship
![Eigenvalue Trend](../Images/Contour_Plots/ex4_eigenvalue_trend.png)
*Relationship between correlation coefficient (ρ) and eigenvalues for a 2×2 covariance matrix with equal variances. As correlation increases, the first eigenvalue increases while the second decreases, creating more elongated ellipses.*

### Comprehensive Visualization
![Covariance Eigenvalue Visualization](../Images/Contour_Plots/ex4_covariance_eigenvalue_visualization.png)
*Comprehensive visualization of how correlation affects eigenvalues, eigenvectors, and probability density contours. Top: 3D probability surfaces for uncorrelated (blue) vs strongly correlated (red) distributions. Bottom: Contour plots showing increasing correlation from left to right with eigenvectors (blue arrows) and probability density heatmaps.*

## Key Insights

### Eigendecomposition Fundamentals
- The covariance matrix can be decomposed into eigenvectors and eigenvalues
- Eigenvectors define the principal axes of the elliptical contours
- Eigenvalues determine the spread along each principal axis
- The length of each axis is proportional to the square root of the corresponding eigenvalue

### Correlation Effects on Eigenstructure
- As correlation increases, eigenvalues diverge: one increases, one decreases
- With zero correlation, eigenvalues are equal, resulting in circular contours
- With perfect correlation ($\rho = 1$), one eigenvalue becomes zero, resulting in a line
- The product of eigenvalues equals the determinant of the covariance matrix
- The sum of eigenvalues equals the trace of the covariance matrix (total variance)

### Geometric Interpretation
- Eigenvectors rotate toward the directions of correlation as correlation increases
- For positive correlation, the principal eigenvector rotates toward the $y = x$ line
- For negative correlation, the principal eigenvector rotates toward the $y = -x$ line
- The eccentricity of the ellipses increases with the strength of correlation
- The shape of the distribution directly reflects the underlying statistical relationship

### Applications in Data Analysis
- Principal Component Analysis (PCA) exploits this eigenstructure to reduce dimensionality
- In highly correlated data, most variation can be captured with fewer principal components
- The eigenvalue spectrum reveals how many components are necessary to represent the data
- Correlation structure gives insights into the underlying data generation process
- Eigendecomposition provides a coordinate system optimized for the data's natural variation

## Running the Examples

You can run the code that generates these examples and visualizations using:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Codes/L2_1_CMC_example_4_covariance_eigenvalue_visualization.py
```