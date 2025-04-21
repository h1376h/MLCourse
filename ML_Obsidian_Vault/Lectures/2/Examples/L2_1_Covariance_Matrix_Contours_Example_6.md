### Example 6: Effects of Rotation on Covariance Structure

#### Problem Statement
What happens to the covariance matrix when we rotate a dataset, and why is this important? How does a change in coordinate system affect the correlation structure of data?

For this example, we'll start with uncorrelated 2D data having equal variances and observe how rotation by various angles (0°, 30°, 60°) changes the correlation structure.

![Rotation Concept Visualization](../Images/Contour_Plots/ex6_concept_visualization.png)
*Basic visualization showing how rotation affects correlation structure for initially uncorrelated data. Left: Original data. Middle: After 30° rotation. Right: After 60° rotation. Red dashed ellipses show the covariance structure.*

#### Solution

We'll explore how geometric rotations affect the covariance structure of data, providing insight into coordinate transformations and feature engineering.

##### Step 1: Mathematical Foundation
When we rotate a dataset using a rotation matrix $R$, the covariance matrix transforms according to:

$$\Sigma' = R \Sigma R^T$$

Where:
- $\Sigma$ is the original covariance matrix
- $\Sigma'$ is the transformed covariance matrix
- $R$ is the rotation matrix

For a 2D rotation by angle $\theta$, the rotation matrix is:

$$R = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

This mathematical relationship is crucial for understanding how correlation can be introduced or removed through coordinate transformations.

##### Step 2: Relationship Between Rotation Angle and Correlation

For initially uncorrelated data with equal variances (covariance matrix = identity matrix), rotation by angle $\theta$ introduces correlation according to the theoretical formula:

$$\rho = \frac{\sin(2\theta)}{2}$$

Where $\rho$ is the correlation coefficient. However, real data often differs from this theoretical prediction due to sampling variability.

![Theoretical vs Actual Correlation](../Images/Contour_Plots/ex6_theoretical_vs_actual_correlation.png)
*Comparison of theoretical correlation formula (red line) with actual correlation values observed in rotated data (blue dashed line). Key points are marked at 0°, 45°, 90°, 135°, and 180°, showing how rotation affects correlation in a periodic manner.*

This visualization demonstrates:
- The sinusoidal relationship between rotation angle and correlation
- Maximum positive correlation occurring at 45° rotation (theoretically)
- Maximum negative correlation occurring at 135° rotation (theoretically)
- Actual data correlation may deviate from theoretical values due to initial correlation in the data and sampling variability
- The periodicity of the pattern, with correlation returning to its initial value after 180° rotation

![Correlation vs Angle Curve](../Images/Contour_Plots/ex6_correlation_angle_curve.png)
*How correlation coefficient changes with rotation angle according to the theoretical formula ρ = sin(2θ)/2, reaching maximum correlation of 0.5 at 45° and minimum of -0.5 at 135°.*

##### Step 3: Rotation as a Vector Field Transformation

Rotation is a linear transformation that preserves distances from the origin and angles between vectors. When we rotate the coordinate system, points move along circular paths centered at the origin.

![Rotation Vector Field](../Images/Contour_Plots/ex6_rotation_vector_field.png)
*Vector field visualization of rotation transformation. Blue arrows show how points move under rotation. Concentric circles remain circles after rotation (illustrated by dotted circles), demonstrating that rotation preserves distances from the origin. The coordinate axes also rotate, shown by the red lines.*

Key observations about rotation as a transformation:
- Points farther from the origin move greater distances (longer arrows)
- All points rotate by the same angle around the origin
- The transformation preserves the shape of probability distributions but changes their orientation
- The coordinate system itself rotates, changing our frame of reference

##### Step 4: Step-by-Step Visualization of Rotation Effects

To further understand how rotation affects correlation structure, we can examine the effect of various rotation angles on the same dataset:

![Rotation Steps Visualization](../Images/Contour_Plots/ex6_rotation_steps.png)
*Comprehensive visualization showing the effect of rotation at multiple angles (0°, 30°, 60°, 90°, 135°, 180°). For each angle, the correlation coefficient changes with the covariance ellipse (red dashed line) rotating accordingly.*

This visualization demonstrates several key insights:
1. The correlation oscillates as the rotation angle increases
2. When the dataset is rotated by 180°, it returns to its original correlation structure
3. The shape of the dataset remains constant, only its orientation changes
4. The covariance ellipse rotates with the data, maintaining its shape

##### Step 5: Properties Preserved Under Rotation

Despite the changes in correlation, certain properties remain invariant under rotation:
- Total variance (trace of covariance matrix): $\text{tr}(\Sigma') = \text{tr}(\Sigma)$
- Determinant of covariance matrix: $|\Sigma'| = |\Sigma|$
- Eigenvalues of the covariance matrix (though eigenvectors rotate)

These invariants reflect the fact that rotation merely changes our perspective on the data, not the fundamental structure of the data itself. Our empirical verification confirms this, with traces and determinants remaining constant across all rotation angles.

The theoretical relationship for the transformed covariance matrix of initially uncorrelated data with equal variances ($\Sigma = \sigma^2 I$) under rotation by angle $\theta$ is:

$$\Sigma' = \sigma^2 \begin{pmatrix} 1 & \frac{\sin(2\theta)}{2} \\ \frac{\sin(2\theta)}{2} & 1 \end{pmatrix}$$

##### Step 6: Practical Significance

Understanding rotation effects on covariance has important applications:
1. **Coordinate system choice**: The observed correlation structure depends on how we choose to measure our variables
2. **Feature engineering**: Rotation can introduce or remove correlations, which can be useful for creating independent features
3. **Principal Component Analysis (PCA)**: PCA exploits this property by finding a rotation that diagonalizes the covariance matrix
4. **Statistical independence**: Independence is coordinate-dependent; what looks uncorrelated in one coordinate system may be strongly correlated in another
5. **Sensor orientation**: In practical applications like sensor data analysis, the physical orientation of sensors can affect the observed correlation patterns
6. **Feature transformations in ML**: Transformations in machine learning pipelines can significantly alter correlation structures

This example demonstrates the fundamental importance of coordinate systems in multivariate statistics. What appears as correlation in one coordinate system may disappear in another, highlighting that correlation is not an intrinsic property of the data but rather depends on our chosen reference frame.

## Running the Examples

You can run the code that generates these examples and visualizations using:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Codes/L2_1_CMC_example_6_rotation_covariance_change.py
```