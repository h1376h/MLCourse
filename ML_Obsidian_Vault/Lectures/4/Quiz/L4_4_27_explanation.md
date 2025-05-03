# Question 27: Linear Discriminant Analysis with Singular Scatter Matrix

## Problem Statement
Linear discriminant analysis has many applications, such as dimensionality reduction and feature extraction. In this problem, we consider a special case with two classes expressed as follows:

- Class A: $\mathbf{x}_1^{(A)} = \begin{bmatrix} 1 \\ 3 \end{bmatrix}$, $\mathbf{x}_2^{(A)} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$

- Class B: $\mathbf{x}_1^{(B)} = \begin{bmatrix} 6 \\ 4 \end{bmatrix}$, $\mathbf{x}_2^{(B)} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}$

Note that in this problem we use column vectors for the data points to simplify the calculation.

### Task
1. Compute the mean vector for each class, $\mu_A$ and $\mu_B$.
2. Compute the covariance matrix for each class, $\Sigma_A$ and $\Sigma_B$.

Fisher's linear discriminant analysis aims to maximize the criterion function:

$$S(\mathbf{w}) = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{within}}} = \frac{(\mathbf{w}^{\top} \mu_A - \mathbf{w}^{\top} \mu_B)^2}{\mathbf{w}^{\top} (\Sigma_A + \Sigma_B)\mathbf{w}}$$

An optimal solution $\mathbf{w}^*$ is:

$$\mathbf{w}^* = (\Sigma_A + \Sigma_B)^{-1}(\mu_A - \mu_B)$$

3. Find the optimal $\mathbf{w}^*$ with unit length.

## Understanding the Problem

Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique that projects high-dimensional data onto a lower-dimensional space while preserving class separability. Unlike Principal Component Analysis (PCA), which ignores class labels, LDA specifically aims to find the direction that best distinguishes between classes.

The key idea behind LDA is to maximize the between-class variance while minimizing the within-class variance. Fisher's criterion function quantifies this objective:

$$S(\mathbf{w}) = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{within}}} = \frac{(\mathbf{w}^{\top} \mu_A - \mathbf{w}^{\top} \mu_B)^2}{\mathbf{w}^{\top} (\Sigma_A + \Sigma_B)\mathbf{w}}$$

What makes this problem special is that we'll encounter a singular within-class scatter matrix, which creates computational challenges when finding the optimal projection direction.

## Solution

### Step 1: Calculate the mean vectors for each class

The mean vector for a class is calculated by averaging all data points in that class:

For Class A:
$$\mu_A = \frac{1}{n_A} \sum_{i=1}^{n_A} \mathbf{x}_i^{(A)} = \frac{1}{2} \left( \begin{bmatrix} 1 \\ 3 \end{bmatrix} + \begin{bmatrix} 3 \\ 1 \end{bmatrix} \right) = \frac{1}{2} \begin{bmatrix} 4 \\ 4 \end{bmatrix} = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$

For Class B:
$$\mu_B = \frac{1}{n_B} \sum_{i=1}^{n_B} \mathbf{x}_i^{(B)} = \frac{1}{2} \left( \begin{bmatrix} 6 \\ 4 \end{bmatrix} + \begin{bmatrix} 4 \\ 6 \end{bmatrix} \right) = \frac{1}{2} \begin{bmatrix} 10 \\ 10 \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \end{bmatrix}$$

### Step 2: Calculate the covariance matrices for each class

The covariance matrix for a class is calculated by:

$$\Sigma_k = \frac{1}{n_k} \sum_{i=1}^{n_k} (\mathbf{x}_i^{(k)} - \mu_k)(\mathbf{x}_i^{(k)} - \mu_k)^T$$

For Class A, first we calculate the centered data points:
$$\mathbf{x}_1^{(A)} - \mu_A = \begin{bmatrix} 1 \\ 3 \end{bmatrix} - \begin{bmatrix} 2 \\ 2 \end{bmatrix} = \begin{bmatrix} -1 \\ 1 \end{bmatrix}$$
$$\mathbf{x}_2^{(A)} - \mu_A = \begin{bmatrix} 3 \\ 1 \end{bmatrix} - \begin{bmatrix} 2 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

Then we compute the outer products:
$$\begin{bmatrix} -1 \\ 1 \end{bmatrix} \begin{bmatrix} -1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$
$$\begin{bmatrix} 1 \\ -1 \end{bmatrix} \begin{bmatrix} 1 & -1 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$

Computing the covariance matrix for Class A:
$$\Sigma_A = \frac{1}{2} \left( \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} + \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} \right) = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$

Similarly for Class B:
$$\mathbf{x}_1^{(B)} - \mu_B = \begin{bmatrix} 6 \\ 4 \end{bmatrix} - \begin{bmatrix} 5 \\ 5 \end{bmatrix} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$
$$\mathbf{x}_2^{(B)} - \mu_B = \begin{bmatrix} 4 \\ 6 \end{bmatrix} - \begin{bmatrix} 5 \\ 5 \end{bmatrix} = \begin{bmatrix} -1 \\ 1 \end{bmatrix}$$

Computing the outer products:
$$\begin{bmatrix} 1 \\ -1 \end{bmatrix} \begin{bmatrix} 1 & -1 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$
$$\begin{bmatrix} -1 \\ 1 \end{bmatrix} \begin{bmatrix} -1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$

Computing the covariance matrix for Class B:
$$\Sigma_B = \frac{1}{2} \left( \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} + \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} \right) = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$

Interestingly, both classes have identical covariance matrices with perfect negative correlation between the features.

### Step 3: Calculate the pooled within-class scatter matrix

The pooled within-class scatter matrix is the sum of the individual covariance matrices:

$$S_w = \Sigma_A + \Sigma_B = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} + \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 2 & -2 \\ -2 & 2 \end{bmatrix}$$

### Step 4: Determine if the scatter matrix is singular

We compute the determinant of the scatter matrix:
$$\det(S_w) = 2 \cdot 2 - (-2) \cdot (-2) = 4 - 4 = 0$$

The determinant of $S_w$ is zero, indicating that the matrix is singular. This means that $S_w$ does not have an inverse, and we cannot directly apply the standard formula for finding the optimal projection direction.

### Step 5: Handle the singular scatter matrix

Since the scatter matrix is singular, we have two main approaches to find $\mathbf{w}^*$:

1. **Regularization approach**: Add a small positive value $\lambda$ to the diagonal elements of $S_w$.
$$S_{w,\text{reg}} = S_w + \lambda I = \begin{bmatrix} 2 & -2 \\ -2 & 2 \end{bmatrix} + \lambda \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2+\lambda & -2 \\ -2 & 2+\lambda \end{bmatrix}$$

   Using $\lambda = 10^{-5}$, we get:
   $$S_{w,\text{reg}} = \begin{bmatrix} 2.00001 & -2 \\ -2 & 2.00001 \end{bmatrix}$$

   The inverse is approximately:
   $$S_{w,\text{reg}}^{-1} \approx \begin{bmatrix} 50000.125 & 49999.875 \\ 49999.875 & 50000.125 \end{bmatrix}$$

   The optimal projection direction is:
   $$\mathbf{w}^*_{\text{reg}} = S_{w,\text{reg}}^{-1}(\mu_A - \mu_B) \approx \begin{bmatrix} 50000.125 & 49999.875 \\ 49999.875 & 50000.125 \end{bmatrix} \begin{bmatrix} -3 \\ -3 \end{bmatrix} \approx \begin{bmatrix} -300000 \\ -300000 \end{bmatrix}$$

   After normalization:
   $$\mathbf{w}^*_{\text{norm}} = \frac{\mathbf{w}^*_{\text{reg}}}{||\mathbf{w}^*_{\text{reg}}||} = \frac{1}{\sqrt{(-300000)^2 + (-300000)^2}} \begin{bmatrix} -300000 \\ -300000 \end{bmatrix} \approx \begin{bmatrix} -0.7071 \\ -0.7071 \end{bmatrix}$$

2. **Direct approach**: Use the mean difference vector directly as the projection direction.
   $$\mathbf{w}^* = \mu_A - \mu_B = \begin{bmatrix} 2 \\ 2 \end{bmatrix} - \begin{bmatrix} 5 \\ 5 \end{bmatrix} = \begin{bmatrix} -3 \\ -3 \end{bmatrix}$$

   After normalization:
   $$\mathbf{w}^*_{\text{norm}} = \frac{\mathbf{w}^*}{||\mathbf{w}^*||} = \frac{1}{\sqrt{(-3)^2 + (-3)^2}} \begin{bmatrix} -3 \\ -3 \end{bmatrix} = \frac{1}{\sqrt{18}} \begin{bmatrix} -3 \\ -3 \end{bmatrix} \approx \begin{bmatrix} -0.7071 \\ -0.7071 \end{bmatrix}$$

Both approaches yield the same normalized direction vector. This makes intuitive sense because the singularity of $S_w$ indicates that the within-class variance is concentrated along a single direction.

## Visual Explanations

### Data Points and LDA Direction
![LDA Projection](../Images/L4_4_Quiz_27/lda_projection.png)

This visualization shows the original data points for both classes (Class A in blue, Class B in red). The class means are marked with stars. The green arrow represents the LDA projection direction $\mathbf{w}^*$ (scaled for visibility), and the purple arrow shows the normalized direction $\mathbf{w}^*_{\text{norm}}$.

The key observation is that the optimal projection direction is proportional to the difference between class means. In this special case where the scatter matrix is singular, the direction coincides exactly with the line connecting the class means.

### Projection onto the LDA Direction
![LDA Projection in 1D](../Images/L4_4_Quiz_27/lda_projection_1d.png)

This visualization demonstrates how the data points are projected onto the LDA direction. The horizontal line represents the 1D subspace defined by the LDA direction. The projected points show clear separation between the two classes, with a decision threshold (green dashed line) at the midpoint between the projected class means.

After projection, we can see that all points from Class A are on one side of the threshold and all points from Class B are on the other side, demonstrating perfect class separation.

### Between-class and Within-class Scatter Analysis
![LDA Scatter Analysis](../Images/L4_4_Quiz_27/lda_scatter_analysis.png)

This comprehensive visualization illustrates several aspects of the LDA analysis:

1. **Top-left**: The original data with class means and the line connecting them
2. **Top-right**: Within-class scatter for Class A, showing the distances from each point to its class mean
3. **Bottom-left**: Within-class scatter for Class B, showing the distances from each point to its class mean
4. **Bottom-right**: The LDA direction and the perpendicular decision boundary

## Key Insights

### Theoretical Foundations
- Fisher's Linear Discriminant Analysis seeks to maximize between-class separation while minimizing within-class scatter
- The criterion function $S(\mathbf{w})$ quantifies this trade-off
- When the scatter matrix is singular, the standard approach needs modification

### Computational Considerations
- A singular scatter matrix indicates that the within-class variance is concentrated in specific directions
- Regularization is a practical approach to handle matrix singularity
- The direction of mean difference can be a good approximation of the optimal projection direction when the scatter matrix is singular

### Analysis of the Special Case
- Both classes have identical covariance matrices with perfect negative correlation
- The points in each class lie on a line with slope -1
- The class means lie on a line with slope 1, which is perpendicular to the within-class direction of variance
- This perfect symmetry leads to a singular scatter matrix
- The optimal projection direction is aligned with the mean difference vector

## Conclusion

- The mean vectors for the classes are $\mu_A = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$ and $\mu_B = \begin{bmatrix} 5 \\ 5 \end{bmatrix}$
- The covariance matrices are identical: $\Sigma_A = \Sigma_B = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$
- The pooled within-class scatter matrix is $S_w = \begin{bmatrix} 2 & -2 \\ -2 & 2 \end{bmatrix}$, which is singular
- The optimal projection direction with unit length is $\mathbf{w}^*_{\text{norm}} = \begin{bmatrix} -0.7071 \\ -0.7071 \end{bmatrix}$
- This direction successfully separates the two classes with maximum between-class distance and minimum within-class scatter

This problem illustrates how Linear Discriminant Analysis handles special cases with singular scatter matrices, and demonstrates that in such cases, the mean difference vector often provides the optimal projection direction. 