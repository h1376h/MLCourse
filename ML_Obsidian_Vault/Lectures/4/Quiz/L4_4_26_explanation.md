# Question 26: Linear Discriminant Analysis with Simple Data

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
Linear Discriminant Analysis (LDA) is a dimensionality reduction technique that finds a linear combination of features that characterizes or separates two or more classes. Unlike PCA, which ignores class labels, LDA specifically attempts to model differences between classes. Fisher's LDA seeks to find a projection that maximizes the separation between classes while minimizing the variance within classes.

In this problem, we have two classes with two data points each in a 2D feature space. We need to calculate the class means, covariance matrices, and ultimately find the optimal projection direction that best separates these classes.

It's important to note that the specific data points in this problem create a special case where the covariance matrices will be zero matrices. This is because the points in each class are perfectly symmetric around their means. To fully demonstrate the calculation procedure, we'll show both approaches: first using the original points (which leads to a degenerate case) and then using slightly perturbed points to see how LDA would work in a more general case.

## Solution for the Original Data Points

### Step 1: Calculate the mean vectors for each class
First, we need to calculate the mean vectors for both classes by taking the average of the data points in each class.

For Class 0:
$$\mu_0 = \frac{1}{2} \left( \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 2 \\ 1 \end{bmatrix} \right) = \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix}$$

For Class 1:
$$\mu_1 = \frac{1}{2} \left( \begin{bmatrix} 4 \\ 3 \end{bmatrix} + \begin{bmatrix} 5 \\ 4 \end{bmatrix} \right) = \begin{bmatrix} 4.5 \\ 3.5 \end{bmatrix}$$

### Step 2: Calculate the covariance matrices for each class
Next, we compute the covariance matrices. For each class, we center the data points by subtracting the class mean, and then compute the scatter matrix.

For Class 0:
The centered data points are:
$$\mathbf{x}_1^{(0)} - \mu_0 = \begin{bmatrix} 1 \\ 2 \end{bmatrix} - \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix} = \begin{bmatrix} -0.5 \\ 0.5 \end{bmatrix}$$
$$\mathbf{x}_2^{(0)} - \mu_0 = \begin{bmatrix} 2 \\ 1 \end{bmatrix} - \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$$

Note that these two vectors are negatives of each other. This is a special case that will lead to a zero covariance matrix.

The covariance matrix $\Sigma_0$ is calculated as:
$$\Sigma_0 = \frac{1}{n} \sum_{i=1}^{n} (\mathbf{x}_i^{(0)} - \mu_0)(\mathbf{x}_i^{(0)} - \mu_0)^T$$

$$\Sigma_0 = \frac{1}{2} \left( \begin{bmatrix} -0.5 \\ 0.5 \end{bmatrix} \begin{bmatrix} -0.5 & 0.5 \end{bmatrix} + \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} \begin{bmatrix} 0.5 & -0.5 \end{bmatrix} \right)$$

Computing the first outer product:
$$\begin{bmatrix} -0.5 \\ 0.5 \end{bmatrix} \begin{bmatrix} -0.5 & 0.5 \end{bmatrix} = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix}$$

Computing the second outer product:
$$\begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} \begin{bmatrix} 0.5 & -0.5 \end{bmatrix} = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix}$$

Adding these matrices and dividing by the number of samples:
$$\Sigma_0 = \frac{1}{2} \left( \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix} + \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix} \right) = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix}$$

Interestingly, this is not zero as initially expected. Let's double-check our calculation.

The centered data for Class 0:
$$\mathbf{x}_1^{(0)} - \mu_0 = \begin{bmatrix} 1 - 1.5 \\ 2 - 1.5 \end{bmatrix} = \begin{bmatrix} -0.5 \\ 0.5 \end{bmatrix}$$
$$\mathbf{x}_2^{(0)} - \mu_0 = \begin{bmatrix} 2 - 1.5 \\ 1 - 1.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$$

The outer products:
$$\begin{bmatrix} -0.5 \\ 0.5 \end{bmatrix} \begin{bmatrix} -0.5 & 0.5 \end{bmatrix} = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix}$$
$$\begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} \begin{bmatrix} 0.5 & -0.5 \end{bmatrix} = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix}$$

The covariance matrix for Class 0:
$$\Sigma_0 = \frac{1}{2} \left( \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix} + \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix} \right) = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix}$$

For Class 1:
The centered data points are:
$$\mathbf{x}_1^{(1)} - \mu_1 = \begin{bmatrix} 4 \\ 3 \end{bmatrix} - \begin{bmatrix} 4.5 \\ 3.5 \end{bmatrix} = \begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix}$$
$$\mathbf{x}_2^{(1)} - \mu_1 = \begin{bmatrix} 5 \\ 4 \end{bmatrix} - \begin{bmatrix} 4.5 \\ 3.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}$$

Again, these two vectors are negatives of each other, creating a special case.

The outer products:
$$\begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix} \begin{bmatrix} -0.5 & -0.5 \end{bmatrix} = \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix}$$
$$\begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix} \begin{bmatrix} 0.5 & 0.5 \end{bmatrix} = \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix}$$

The covariance matrix for Class 1:
$$\Sigma_1 = \frac{1}{2} \left( \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix} + \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix} \right) = \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix}$$

### Step 3: Calculate the pooled within-class scatter matrix
Next, we compute the pooled within-class scatter matrix:

$$S_w = \Sigma_0 + \Sigma_1 = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix} + \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix} = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix}$$

### Step 4: Calculate the between-class mean difference
The difference between the class means is:

$$\mu_0 - \mu_1 = \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix} - \begin{bmatrix} 4.5 \\ 3.5 \end{bmatrix} = \begin{bmatrix} -3 \\ -2 \end{bmatrix}$$

### Step 5: Find the optimal projection direction w*
According to Fisher's LDA, the optimal projection direction is:

$$\mathbf{w}^* = (\Sigma_0 + \Sigma_1)^{-1}(\mu_0 - \mu_1)$$

First, we compute the inverse of the pooled within-class scatter matrix:

$$S_w^{-1} = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix}^{-1} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$

Now we can compute $\mathbf{w}^*$:

$$\mathbf{w}^* = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} -3 \\ -2 \end{bmatrix} = \begin{bmatrix} -6 \\ -4 \end{bmatrix}$$

To normalize $\mathbf{w}^*$ to unit length, we divide by its norm:

$$\|\mathbf{w}^*\| = \sqrt{(-6)^2 + (-4)^2} = \sqrt{36 + 16} = \sqrt{52} \approx 7.211$$

$$\mathbf{w}_{\text{normalized}}^* = \frac{\mathbf{w}^*}{\|\mathbf{w}^*\|} = \frac{1}{7.211} \begin{bmatrix} -6 \\ -4 \end{bmatrix} = \begin{bmatrix} -0.832 \\ -0.555 \end{bmatrix}$$

This is our optimal LDA projection direction (normalized). It points from Class 1 towards Class 0, which is what we would expect.

## Solution with Perturbed Data Points

Since the original data creates a special case with covariance matrices that can be problematic for numerical calculations, we can slightly perturb the data points to demonstrate how LDA would work in a more general case.

Using the perturbed data:
- Class 0: (1.1, 1.9), (1.9, 1.1)
- Class 1: (3.9, 3.1), (5.1, 3.9)

### Step 1: Calculate the mean vectors
For the perturbed data, the mean vectors remain the same:
$$\mu_0 = \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix}$$
$$\mu_1 = \begin{bmatrix} 4.5 \\ 3.5 \end{bmatrix}$$

### Step 2: Calculate the covariance matrices
With the perturbed data, we get non-zero covariance matrices:

$$\Sigma_0 = \begin{bmatrix} 0.16 & -0.16 \\ -0.16 & 0.16 \end{bmatrix}$$
$$\Sigma_1 = \begin{bmatrix} 0.36 & 0.24 \\ 0.24 & 0.16 \end{bmatrix}$$

### Step 3: Calculate the pooled within-class scatter matrix
$$S_w = \Sigma_0 + \Sigma_1 = \begin{bmatrix} 0.52 & 0.08 \\ 0.08 & 0.32 \end{bmatrix}$$

### Step 4: Calculate the inverse of Sw
$$S_w^{-1} = \begin{bmatrix} 2.00 & -0.50 \\ -0.50 & 3.25 \end{bmatrix}$$

### Step 5: Find the optimal projection direction
$$\mathbf{w}^* = S_w^{-1}(\mu_0 - \mu_1) = \begin{bmatrix} 2.00 & -0.50 \\ -0.50 & 3.25 \end{bmatrix} \begin{bmatrix} -3 \\ -2 \end{bmatrix} = \begin{bmatrix} -5.00 \\ -5.00 \end{bmatrix}$$

Normalizing to unit length:
$$\|\mathbf{w}^*\| = \sqrt{(-5)^2 + (-5)^2} = \sqrt{50} \approx 7.071$$
$$\mathbf{w}_{\text{normalized}}^* = \frac{\mathbf{w}^*}{\|\mathbf{w}^*\|} = \frac{1}{7.071} \begin{bmatrix} -5 \\ -5 \end{bmatrix} = \begin{bmatrix} -0.707 \\ -0.707 \end{bmatrix}$$

This result indicates that with the perturbed data, the optimal LDA projection direction is along the negative diagonal, corresponding to the direction that best separates the classes.

## Visual Explanations

### Data Points and LDA Projection Direction
![LDA Projection](../Images/L4_4_Quiz_26/lda_projection.png)

This visualization shows the original data points (faded) and the perturbed data points used for the LDA calculation. The class means are marked with stars. The green arrow represents the optimal LDA projection direction w* (scaled), and the purple arrow shows the normalized direction.

## Key Insights

### Theoretical Foundations
- Fisher's LDA finds the direction that maximizes between-class separation while minimizing within-class scatter
- The criterion function $S(\mathbf{w})$ quantifies this trade-off: higher values indicate better class separation
- The solution involves the inverse of the pooled covariance matrix and the difference between class means
- LDA is a supervised dimensionality reduction technique that takes class labels into account

### Computational Considerations
- Special cases like perfectly symmetrical data points can lead to singular or nearly singular covariance matrices
- For degenerate cases, regularization techniques or small perturbations can help stabilize the solution
- The direction connecting the class means provides a reasonable projection direction when covariance information is not reliable
- The analytical solution for LDA can be sensitive to numerical precision issues in certain cases

### Practical Applications
- LDA can be used for dimensionality reduction before classification
- It's particularly useful when the discriminatory information is in the means rather than the covariances
- In binary classification, the projection can directly serve as a classifier using a threshold
- For datasets with special structure (like the original problem), simpler solutions might be equally effective

## Conclusion
- We successfully computed the mean vectors: $\mu_0 = \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix}$ and $\mu_1 = \begin{bmatrix} 4.5 \\ 3.5 \end{bmatrix}$
- For the original dataset, the covariance matrices are $\Sigma_0 = \begin{bmatrix} 0.25 & -0.25 \\ -0.25 & 0.25 \end{bmatrix}$ and $\Sigma_1 = \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix}$
- The optimal projection direction with unit length is $\mathbf{w}_{\text{normalized}}^* = \begin{bmatrix} -0.832 \\ -0.555 \end{bmatrix}$ for the original data
- For the perturbed data, we found $\mathbf{w}_{\text{normalized}}^* = \begin{bmatrix} -0.707 \\ -0.707 \end{bmatrix}$

This analysis demonstrates the application of Fisher's Linear Discriminant Analysis in finding the optimal projection for class separation. The comparison between the original and perturbed data highlights the importance of proper covariance estimation in LDA. 