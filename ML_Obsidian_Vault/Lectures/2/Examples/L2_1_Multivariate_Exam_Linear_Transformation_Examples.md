# Linear Transformation Examples

This document provides examples and key concepts on linear transformations of multivariate distributions, a fundamental concept in machine learning, statistics, and multivariate analysis.

## Key Concepts and Formulas

Linear transformations are fundamental operations in multivariate analysis that preserve the Gaussian nature of multivariate normal distributions. They are essential in dimension reduction, feature extraction, and data preprocessing.

### Linear Transformation Formula

If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ is a multivariate normal random vector and $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$ is a linear transformation of $\mathbf{X}$, then:

$$\mathbf{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$$

Where:
- $\mathbf{A}$ = Transformation matrix
- $\mathbf{b}$ = Shift vector
- $\mathbf{A}\boldsymbol{\mu} + \mathbf{b}$ = Mean vector of the transformed distribution
- $\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T$ = Covariance matrix of the transformed distribution

## Example 1: Linear Transformations of Multivariate Normal Distributions

### Problem Statement
Let $\mathbf{X} = \begin{bmatrix} X_1 \\ X_2 \\ X_3 \end{bmatrix}$ follow a multivariate normal distribution with mean vector $\boldsymbol{\mu} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 4 & 1 & 0 \\ 1 & 9 & 2 \\ 0 & 2 & 16 \end{bmatrix}$.

Define $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$ where $\mathbf{A} = \begin{bmatrix} 2 & 1 & 0 \\ 0 & 3 & 1 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 5 \\ -2 \end{bmatrix}$.

a) Find the distribution of $\mathbf{Y}$.
b) Calculate $Cov(Y_1, Y_2)$.
c) Are $Y_1$ and $Y_2$ independent? Why or why not?

### Solution

#### Part a: Finding the distribution of $\mathbf{Y}$

When a random vector $\mathbf{X}$ follows a multivariate normal distribution, any linear transformation $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$ also follows a multivariate normal distribution with:

$$\mathbf{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$$

Let's compute the mean vector first:

$$\boldsymbol{\mu}_Y = \mathbf{A}\boldsymbol{\mu} + \mathbf{b} = \begin{bmatrix} 2 & 1 & 0 \\ 0 & 3 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 5 \\ -2 \end{bmatrix}$$

$$\boldsymbol{\mu}_Y = \begin{bmatrix} 2(1) + 1(2) + 0(3) \\ 0(1) + 3(2) + 1(3) \end{bmatrix} + \begin{bmatrix} 5 \\ -2 \end{bmatrix} = \begin{bmatrix} 4 \\ 9 \end{bmatrix} + \begin{bmatrix} 5 \\ -2 \end{bmatrix} = \begin{bmatrix} 9 \\ 7 \end{bmatrix}$$

Now, let's calculate the covariance matrix:

$$\boldsymbol{\Sigma}_Y = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T = \begin{bmatrix} 2 & 1 & 0 \\ 0 & 3 & 1 \end{bmatrix} \begin{bmatrix} 4 & 1 & 0 \\ 1 & 9 & 2 \\ 0 & 2 & 16 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 1 & 3 \\ 0 & 1 \end{bmatrix}$$

First, computing $\mathbf{A}\boldsymbol{\Sigma}$:

$$\mathbf{A}\boldsymbol{\Sigma} = \begin{bmatrix} 2(4) + 1(1) + 0(0) & 2(1) + 1(9) + 0(2) & 2(0) + 1(2) + 0(16) \\ 0(4) + 3(1) + 1(0) & 0(1) + 3(9) + 1(2) & 0(0) + 3(2) + 1(16) \end{bmatrix}$$

$$\mathbf{A}\boldsymbol{\Sigma} = \begin{bmatrix} 9 & 11 & 2 \\ 3 & 29 & 22 \end{bmatrix}$$

Then, computing $\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T$:

$$\boldsymbol{\Sigma}_Y = \begin{bmatrix} 9 & 11 & 2 \\ 3 & 29 & 22 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 1 & 3 \\ 0 & 1 \end{bmatrix}$$

$$\boldsymbol{\Sigma}_Y = \begin{bmatrix} 9(2) + 11(1) + 2(0) & 9(0) + 11(3) + 2(1) \\ 3(2) + 29(1) + 22(0) & 3(0) + 29(3) + 22(1) \end{bmatrix}$$

$$\boldsymbol{\Sigma}_Y = \begin{bmatrix} 29 & 35 \\ 35 & 109 \end{bmatrix}$$

Therefore, $\mathbf{Y} \sim \mathcal{N}\left(\begin{bmatrix} 9 \\ 7 \end{bmatrix}, \begin{bmatrix} 29 & 35 \\ 35 & 109 \end{bmatrix}\right)$

#### Part b: Calculating $Cov(Y_1, Y_2)$

From the covariance matrix, we can directly read that:
$$Cov(Y_1, Y_2) = \boldsymbol{\Sigma}_Y[1,2] = 35$$

#### Part c: Determining independence

For multivariate normal distributions, zero covariance means independence. Since $Cov(Y_1, Y_2) = 35 \neq 0$, $Y_1$ and $Y_2$ are not independent.

The non-zero covariance indicates that knowledge of one variable provides information about the other. This is also evident from the structure of the linear transformation, where both $Y_1$ and $Y_2$ depend on overlapping components of the original vector $\mathbf{X}$.

## Example 2: Orthogonal Transformations and Preservation of Distances

### Problem Statement
Consider a random vector $\mathbf{X} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_3)$, i.e., a standard multivariate normal in 3 dimensions. Let $\mathbf{Q}$ be an orthogonal matrix:

$$\mathbf{Q} = \begin{bmatrix} 
\frac{1}{\sqrt{3}} & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
\frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
\frac{1}{\sqrt{3}} & 0 & -\frac{2}{\sqrt{6}}
\end{bmatrix}$$

Define $\mathbf{Y} = \mathbf{Q}\mathbf{X}$.

a) Find the distribution of $\mathbf{Y}$.
b) Show that this transformation preserves Euclidean distances between points.
c) Explain the geometric interpretation of this transformation.

### Solution

#### Part a: Finding the distribution of $\mathbf{Y}$

Using the formula for linear transformations of multivariate normals:

$$\mathbf{Y} = \mathbf{Q}\mathbf{X} \sim \mathcal{N}(\mathbf{Q}\boldsymbol{\mu}, \mathbf{Q}\boldsymbol{\Sigma}\mathbf{Q}^T)$$

Since $\mathbf{X} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_3)$, we have $\boldsymbol{\mu} = \mathbf{0}$ and $\boldsymbol{\Sigma} = \mathbf{I}_3$.

Therefore:
$$\mathbf{Y} \sim \mathcal{N}(\mathbf{Q}\mathbf{0}, \mathbf{Q}\mathbf{I}_3\mathbf{Q}^T) = \mathcal{N}(\mathbf{0}, \mathbf{Q}\mathbf{Q}^T)$$

Since $\mathbf{Q}$ is orthogonal, $\mathbf{Q}\mathbf{Q}^T = \mathbf{I}_3$. Thus:
$$\mathbf{Y} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_3)$$

This shows that $\mathbf{Y}$ follows the same distribution as $\mathbf{X}$, namely a standard multivariate normal distribution.

#### Part b: Preservation of Euclidean distances

For any two points $\mathbf{x}_1$ and $\mathbf{x}_2$ in the original space, their Euclidean distance is:
$$d(\mathbf{x}_1, \mathbf{x}_2) = \|\mathbf{x}_1 - \mathbf{x}_2\| = \sqrt{(\mathbf{x}_1 - \mathbf{x}_2)^T(\mathbf{x}_1 - \mathbf{x}_2)}$$

After transformation, these points become $\mathbf{y}_1 = \mathbf{Q}\mathbf{x}_1$ and $\mathbf{y}_2 = \mathbf{Q}\mathbf{x}_2$. Their distance is:
$$d(\mathbf{y}_1, \mathbf{y}_2) = \|\mathbf{y}_1 - \mathbf{y}_2\| = \|\mathbf{Q}\mathbf{x}_1 - \mathbf{Q}\mathbf{x}_2\| = \|\mathbf{Q}(\mathbf{x}_1 - \mathbf{x}_2)\|$$

Using the property of orthogonal matrices:
$$\|\mathbf{Q}(\mathbf{x}_1 - \mathbf{x}_2)\|^2 = (\mathbf{Q}(\mathbf{x}_1 - \mathbf{x}_2))^T(\mathbf{Q}(\mathbf{x}_1 - \mathbf{x}_2)) = (\mathbf{x}_1 - \mathbf{x}_2)^T\mathbf{Q}^T\mathbf{Q}(\mathbf{x}_1 - \mathbf{x}_2)$$

Since $\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$:
$$(\mathbf{x}_1 - \mathbf{x}_2)^T\mathbf{Q}^T\mathbf{Q}(\mathbf{x}_1 - \mathbf{x}_2) = (\mathbf{x}_1 - \mathbf{x}_2)^T(\mathbf{x}_1 - \mathbf{x}_2) = \|\mathbf{x}_1 - \mathbf{x}_2\|^2$$

Therefore, $d(\mathbf{y}_1, \mathbf{y}_2) = d(\mathbf{x}_1, \mathbf{x}_2)$, showing that the orthogonal transformation preserves distances.

#### Part c: Geometric interpretation

The orthogonal transformation $\mathbf{Q}$ represents a rotation or reflection (or a combination of both) in the 3-dimensional space. Since it preserves distances and angles, it is considered a rigid transformation.

In this case, $\mathbf{Q}$ transforms the standard basis vectors to a new orthonormal basis. The columns of $\mathbf{Q}$ represent the directions of this new basis. The geometric interpretation is that we are viewing the same multivariate normal distribution but from a different coordinate system.

Since the covariance matrix of the original distribution is the identity matrix (meaning the variables are uncorrelated and have unit variance), and the transformation preserves this property, the distribution looks the same from any orthogonal perspective. This is why a standard multivariate normal distribution is spherically symmetric.

## Related Topics

- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: More examples of multivariate normal distributions
- [[L2_1_Linear_Transformation|Linear Transformations]]: Detailed theory of linear transformations
- [[L2_1_PCA|Principal Component Analysis]]: Application of eigendecomposition for dimension reduction
- [[L2_1_Covariance_Examples|Covariance Examples]]: Understanding covariance structures
- [[L2_1_Eigendecomposition|Eigendecomposition]]: Mathematical basis for understanding transformations 