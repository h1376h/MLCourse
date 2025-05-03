# Question 26: Linear Discriminant Analysis 

## Problem Statement
Linear discriminant analysis has many applications, such as dimensionality reduction and feature extraction. In this problem, we consider a simple task with two classes expressed as follows:

- Class 0: $\mathbf{x}_1^{(0)} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$, $\mathbf{x}_2^{(0)} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

- Class 1: $\mathbf{x}_1^{(1)} = \begin{bmatrix} 4 \\ 3 \end{bmatrix}$, $\mathbf{x}_2^{(1)} = \begin{bmatrix} 5 \\ 4 \end{bmatrix}$

Note that in this problem we use column vectors for the data points to simplify the calculation.

### Task
1. Compute the mean vector for each class, $\mu_0$ and $\mu_1$.
2. Compute the covariance matrix for each class, $\Sigma_0$ and $\Sigma_1$.

The Fisher's linear discriminant analysis is defined to maximize criterion function:

$$S(\mathbf{w}) = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{within}}} = \frac{(\mathbf{w}^{\top} \mu_0 - \mathbf{w}^{\top} \mu_1)^2}{\mathbf{w}^{\top} (\Sigma_0 + \Sigma_1)\mathbf{w}}$$

An optimal solution $\mathbf{w}^*$ is:

$$\mathbf{w}^* = (\Sigma_0 + \Sigma_1)^{-1}(\mu_0 - \mu_1)$$

3. Find the optimal $\mathbf{w}^*$ with unit length.

## Understanding the Problem

Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique that projects high-dimensional data onto a lower-dimensional space while preserving class separability. Unlike Principal Component Analysis (PCA), which ignores class labels, LDA specifically aims to find the direction that best distinguishes between classes.

The key idea behind LDA is to maximize the between-class variance while minimizing the within-class variance. Fisher's criterion function quantifies this objective:

$$S(\mathbf{w}) = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{within}}} = \frac{(\mathbf{w}^{\top} \mu_0 - \mathbf{w}^{\top} \mu_1)^2}{\mathbf{w}^{\top} (\Sigma_0 + \Sigma_1)\mathbf{w}}$$

The optimal projection direction $\mathbf{w}^*$ that maximizes this criterion can be found analytically:

$$\mathbf{w}^* = (\Sigma_0 + \Sigma_1)^{-1}(\mu_0 - \mu_1)$$

In this problem, we will use this formula to find the optimal projection direction for our two classes in a 2D space.

## Solution

### Step 1: Calculate the mean vectors for each class

The mean vector for a class is calculated by averaging all data points in that class:

For Class 0:
$$\mu_0 = \frac{1}{n_0} \sum_{i=1}^{n_0} \mathbf{x}_i^{(0)} = \frac{1}{2} \left( \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 2 \\ 1 \end{bmatrix} \right) = \frac{1}{2} \begin{bmatrix} 3 \\ 3 \end{bmatrix} = \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix}$$

For Class 1:
$$\mu_1 = \frac{1}{n_1} \sum_{i=1}^{n_1} \mathbf{x}_i^{(1)} = \frac{1}{2} \left( \begin{bmatrix} 4 \\ 3 \end{bmatrix} + \begin{bmatrix} 5 \\ 4 \end{bmatrix} \right) = \frac{1}{2} \begin{bmatrix} 9 \\ 7 \end{bmatrix} = \begin{bmatrix} 4.5 \\ 3.5 \end{bmatrix}$$

The mean vectors represent the "centers" of each class in our feature space. These will be crucial for calculating the between-class scatter and determining the optimal projection direction.

### Step 2: Calculate the covariance matrices for each class

The covariance matrix for a class captures the spread of data points within that class. It is calculated by:

$$\Sigma_k = \frac{1}{n_k} \sum_{i=1}^{n_k} (\mathbf{x}_i^{(k)} - \mu_k)(\mathbf{x}_i^{(k)} - \mu_k)^T$$

For Class 0, we first calculate the centered data points:
$$\mathbf{x}_1^{(0)} - \mu_0 = \begin{bmatrix} 1 \\ 2 \end{bmatrix} - \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix} = \begin{bmatrix} -0.5 \\ 0.5 \end{bmatrix}$$
$$\mathbf{x}_2^{(0)} - \mu_0 = \begin{bmatrix} 2 \\ 1 \end{bmatrix} - \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$$

The outer products for each centered data point:
$$\begin{bmatrix} -0.5 \\ 0.5 \end{bmatrix} \begin{bmatrix} -0.5 & 0.5 \end{bmatrix} = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix}$$
$$\begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} \begin{bmatrix} 0.5 & -0.5 \end{bmatrix} = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix}$$

Computing the covariance matrix for Class 0:
$$\Sigma_0 = \frac{1}{2} \left( \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix} + \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix} \right) = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix}$$

Similarly for Class 1, the centered data points:
$$\mathbf{x}_1^{(1)} - \mu_1 = \begin{bmatrix} 4 \\ 3 \end{bmatrix} - \begin{bmatrix} 4.5 \\ 3.5 \end{bmatrix} = \begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix}$$
$$\mathbf{x}_2^{(1)} - \mu_1 = \begin{bmatrix} 5 \\ 4 \end{bmatrix} - \begin{bmatrix} 4.5 \\ 3.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}$$

The outer products:
$$\begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix} \begin{bmatrix} -0.5 & -0.5 \end{bmatrix} = \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix}$$
$$\begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix} \begin{bmatrix} 0.5 & 0.5 \end{bmatrix} = \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix}$$

The covariance matrix for Class 1:
$$\Sigma_1 = \frac{1}{2} \left( \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix} + \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix} \right) = \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix}$$

Notice that each class has a distinct covariance structure. For Class 0, there is a negative correlation between the features, while for Class 1, there is a positive correlation.

### Step 3: Calculate the pooled within-class scatter matrix

The pooled within-class scatter matrix is the sum of the covariance matrices:

$$S_w = \Sigma_0 + \Sigma_1 = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix} + \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix} = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix}$$

Interestingly, the off-diagonal elements cancel out, resulting in a diagonal matrix.

### Step 4: Calculate the optimal projection direction

The between-class mean difference is:

$$\mu_0 - \mu_1 = \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix} - \begin{bmatrix} 4.5 \\ 3.5 \end{bmatrix} = \begin{bmatrix} -3 \\ -2 \end{bmatrix}$$

Now we can compute the optimal projection direction using the formula:

$$\mathbf{w}^* = S_w^{-1}(\mu_0 - \mu_1)$$

Since $S_w$ is a diagonal matrix with equal diagonal elements, its inverse is also a diagonal matrix:

$$S_w^{-1} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$

Therefore:

$$\mathbf{w}^* = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} -3 \\ -2 \end{bmatrix} = \begin{bmatrix} -6 \\ -4 \end{bmatrix}$$

### Step 5: Normalize the optimal projection direction

To normalize $\mathbf{w}^*$ to unit length, we compute:

$$\|\mathbf{w}^*\| = \sqrt{(-6)^2 + (-4)^2} = \sqrt{36 + 16} = \sqrt{52} \approx 7.211$$

$$\mathbf{w}_{\text{normalized}}^* = \frac{\mathbf{w}^*}{\|\mathbf{w}^*\|} = \frac{1}{7.211} \begin{bmatrix} -6 \\ -4 \end{bmatrix} = \begin{bmatrix} -0.832 \\ -0.555 \end{bmatrix}$$

This normalized vector is our optimal projection direction.

## Visual Explanations

### Data Points and LDA Direction
![LDA Projection](../Images/L4_4_Quiz_26/lda_projection.png)

This visualization shows the original data points for both classes (Class 0 in blue, Class 1 in red). The class means are marked with stars. The green arrow represents the LDA projection direction $\mathbf{w}^*$ (scaled for visibility), and the purple arrow shows the normalized direction $\mathbf{w}_{\text{normalized}}^*$.

The key observation is that the direction is proportional to the difference between class means but adjusted by the pooled covariance structure.

### Projection onto the LDA Direction
![LDA Projection in 1D](../Images/L4_4_Quiz_26/lda_projection_1d.png)

This visualization demonstrates how the data points are projected onto the LDA direction. The horizontal line represents the 1D subspace defined by the LDA direction. The projected points show clear separation between the two classes, with a decision threshold (green dashed line) at the midpoint between the projected class means.

After projection, we can see that all points from Class 0 are on one side of the threshold and all points from Class 1 are on the other side, demonstrating perfect class separation.

### Between-class and Within-class Scatter Analysis
![LDA Scatter Analysis](../Images/L4_4_Quiz_26/lda_scatter_analysis.png)

This comprehensive visualization illustrates several aspects of the LDA analysis:

1. **Top-left**: The original data with class means and the line connecting them
2. **Top-right**: Within-class scatter for Class 0, showing the distances from each point to its class mean
3. **Bottom-left**: Within-class scatter for Class 1, showing the distances from each point to its class mean
4. **Bottom-right**: The LDA direction and the perpendicular decision boundary

## Key Insights

### Theoretical Foundations
- Fisher's Linear Discriminant Analysis seeks to maximize between-class separation while minimizing within-class scatter
- The criterion function $S(\mathbf{w})$ quantifies this trade-off
- The optimal projection is analytically given by $\mathbf{w}^* = S_w^{-1}(\mu_0 - \mu_1)$

### Computational Considerations
- The covariance matrices capture the correlation structure within each class
- The pooled within-class scatter matrix combines these covariance patterns
- The inverse of the pooled scatter matrix scales the mean difference in the optimal direction

### Practical Applications
- LDA can be used for dimensionality reduction before classification
- In binary classification problems, LDA provides an optimal linear decision boundary
- The projection can be used for visualizing high-dimensional data

## Conclusion

- The mean vectors for the classes are $\mu_0 = \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix}$ and $\mu_1 = \begin{bmatrix} 4.5 \\ 3.5 \end{bmatrix}$
- The covariance matrices are $\Sigma_0 = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix}$ and $\Sigma_1 = \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix}$
- The pooled within-class scatter matrix is $S_w = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix}$
- The optimal projection direction is $\mathbf{w}^* = \begin{bmatrix} -6 \\ -4 \end{bmatrix}$, and with unit length is $\mathbf{w}_{\text{normalized}}^* = \begin{bmatrix} -0.832 \\ -0.555 \end{bmatrix}$
- This direction successfully separates the two classes with maximum between-class distance and minimum within-class scatter

This problem illustrates how Linear Discriminant Analysis provides an optimal projection direction for separating classes by taking into account both the between-class separation and the within-class variance structure. 