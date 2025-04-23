# Eigendecomposition Examples

This document provides examples and key concepts on eigendecomposition of covariance matrices, a fundamental concept in multivariate analysis, dimensionality reduction, and principal component analysis.

## Key Concepts and Formulas

Eigendecomposition is a fundamental matrix factorization that reveals the underlying structure of a covariance matrix. For symmetric matrices like covariance matrices, eigendecomposition provides insight into the directions of maximum variance.

### Eigendecomposition Formula

For a symmetric matrix $\boldsymbol{\Sigma}$, the eigendecomposition is:

$$\boldsymbol{\Sigma} = \mathbf{P} \boldsymbol{\Lambda} \mathbf{P}^T$$

Where:
- $\mathbf{P}$ = Orthogonal matrix whose columns are the eigenvectors of $\boldsymbol{\Sigma}$
- $\boldsymbol{\Lambda}$ = Diagonal matrix containing the eigenvalues of $\boldsymbol{\Sigma}$
- $\mathbf{P}^T\mathbf{P} = \mathbf{I}$ (orthogonality property)

The eigenvalues and eigenvectors are found by solving:
$$\boldsymbol{\Sigma}\mathbf{v} = \lambda\mathbf{v}$$

## Example 1: Eigenvalue Decomposition of Covariance Matrices

### Problem Statement
Consider a bivariate normal distribution with covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 9 & 5 \\ 5 & 4 \end{bmatrix}$.

a) Find the eigenvalues and eigenvectors of the covariance matrix.
b) Interpret these eigenvalues and eigenvectors in terms of the principal components.
c) Determine the directions of maximum and minimum variance, and the corresponding variances.
d) If you generate samples from this distribution with mean $\boldsymbol{\mu} = (0, 0)$, how would you transform these samples to make the variables uncorrelated?

### Solution

#### Part a: Finding eigenvalues and eigenvectors

To find the eigenvalues of $\boldsymbol{\Sigma}$, we solve:
$$|\boldsymbol{\Sigma} - \lambda \mathbf{I}| = 0$$

$$\begin{vmatrix} 9 - \lambda & 5 \\ 5 & 4 - \lambda \end{vmatrix} = 0$$

$$(9 - \lambda)(4 - \lambda) - 5 \times 5 = 0$$

$$36 - 9\lambda - 4\lambda + \lambda^2 - 25 = 0$$

$$\lambda^2 - 13\lambda + 11 = 0$$

Using the quadratic formula:
$$\lambda = \frac{13 \pm \sqrt{13^2 - 4 \times 11}}{2} = \frac{13 \pm \sqrt{169 - 44}}{2} = \frac{13 \pm \sqrt{125}}{2} \approx \frac{13 \pm 11.18}{2}$$

$$\lambda_1 \approx 12.09, \lambda_2 \approx 0.91$$

Now, for each eigenvalue, we find the corresponding eigenvector by solving:
$$(\boldsymbol{\Sigma} - \lambda_i \mathbf{I})\mathbf{v}_i = \mathbf{0}$$

For $\lambda_1 \approx 12.09$:
$$\begin{bmatrix} 9 - 12.09 & 5 \\ 5 & 4 - 12.09 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$\begin{bmatrix} -3.09 & 5 \\ 5 & -8.09 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

This gives us:
$$-3.09v_{11} + 5v_{12} = 0$$
$$5v_{11} - 8.09v_{12} = 0$$

From the first equation:
$$v_{11} = \frac{5v_{12}}{3.09} \approx 1.62v_{12}$$

Now we normalize the eigenvector so that $\|\mathbf{v}_1\| = 1$:
$$\sqrt{(1.62v_{12})^2 + v_{12}^2} = 1$$
$$\sqrt{2.62^2 + 1^2}v_{12} = 1$$
$$\sqrt{7.86}v_{12} = 1$$
$$v_{12} \approx 0.357$$
$$v_{11} \approx 1.62 \times 0.357 \approx 0.578$$

Therefore, $\mathbf{v}_1 \approx (0.85, 0.53)$ after normalizing.

Similarly, for $\lambda_2 \approx 0.91$, we find $\mathbf{v}_2 \approx (-0.53, 0.85)$.

#### Part b: Interpreting in terms of principal components

The eigenvalues of the covariance matrix represent the variances along the principal component directions, and the eigenvectors specify these directions.

In this case:
- The first principal component (corresponding to $\lambda_1 \approx 12.09$) points in the direction $(0.85, 0.53)$ and has a variance of 12.09. This is the direction of maximum variance in the data.
- The second principal component (corresponding to $\lambda_2 \approx 0.91$) points in the direction $(-0.53, 0.85)$ and has a variance of 0.91. This is the direction of minimum variance and is orthogonal to the first principal component.

The ratio of eigenvalues (approximately 13:1) indicates that the distribution is highly elongated along the first principal component.

#### Part c: Directions of maximum and minimum variance

The direction of maximum variance is given by the eigenvector corresponding to the largest eigenvalue, which is $\mathbf{v}_1 \approx (0.85, 0.53)$. The variance in this direction is $\lambda_1 \approx 12.09$.

The direction of minimum variance is given by the eigenvector corresponding to the smallest eigenvalue, which is $\mathbf{v}_2 \approx (-0.53, 0.85)$. The variance in this direction is $\lambda_2 \approx 0.91$.

#### Part d: Transformation to make variables uncorrelated

To make the variables uncorrelated, we need to transform the data using the eigenvectors as a new basis. This is the principal component transformation.

If $\mathbf{X}$ is a random vector from the original distribution, then the transformed vector $\mathbf{Y} = \mathbf{P}^T(\mathbf{X} - \boldsymbol{\mu})$ will have uncorrelated components, where $\mathbf{P}$ is the matrix whose columns are the eigenvectors of $\boldsymbol{\Sigma}$.

In this case:
$$\mathbf{P} = \begin{bmatrix} 0.85 & -0.53 \\ 0.53 & 0.85 \end{bmatrix}$$

The covariance matrix of $\mathbf{Y}$ will be diagonal:
$$\text{Cov}(\mathbf{Y}) = \mathbf{P}^T \boldsymbol{\Sigma} \mathbf{P} = \begin{bmatrix} 12.09 & 0 \\ 0 & 0.91 \end{bmatrix}$$

This transformation is equivalent to rotating the data so that the axes align with the directions of maximum and minimum variance.

## Example 2: Eigendecomposition for Dimension Reduction

### Problem Statement
A data scientist is analyzing a dataset with three variables, whose covariance matrix is:

$$\boldsymbol{\Sigma} = \begin{bmatrix} 
10 & 7 & 3 \\
7 & 8 & 2 \\
3 & 2 & 5
\end{bmatrix}$$

a) Find the eigenvalues and eigenvectors of this covariance matrix.
b) How much of the total variance is captured by the first two principal components?
c) If the data scientist wants to reduce the dimensionality to 2 while preserving at least 90% of the variance, is this possible?
d) What is the projection matrix for reducing the dimensionality to 2?

### Solution

#### Part a: Finding eigenvalues and eigenvectors

The characteristic polynomial of $\boldsymbol{\Sigma}$ is:
$$|\boldsymbol{\Sigma} - \lambda \mathbf{I}| = 0$$

This is a cubic equation, which we can solve to find the eigenvalues:
$$\lambda^3 - 23\lambda^2 + 148\lambda - 240 = 0$$

The eigenvalues are approximately:
$$\lambda_1 \approx 18.19, \lambda_2 \approx 3.83, \lambda_3 \approx 0.98$$

For each eigenvalue, we find the corresponding eigenvector. Let's solve for the eigenvector corresponding to $\lambda_1 \approx 18.19$:

$$(\boldsymbol{\Sigma} - 18.19\mathbf{I})\mathbf{v}_1 = \mathbf{0}$$

$$\begin{bmatrix} 
-8.19 & 7 & 3 \\
7 & -10.19 & 2 \\
3 & 2 & -13.19
\end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \\ v_{13} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$

Solving this system and normalizing, we get:
$$\mathbf{v}_1 \approx (0.71, 0.62, 0.33)$$

Similarly, we find the other eigenvectors:
$$\mathbf{v}_2 \approx (-0.19, 0.32, 0.93)$$
$$\mathbf{v}_3 \approx (0.68, -0.72, 0.15)$$

#### Part b: Variance captured by the first two principal components

The total variance is the sum of all eigenvalues:
$$\text{Total Variance} = \lambda_1 + \lambda_2 + \lambda_3 = 18.19 + 3.83 + 0.98 = 23$$

The variance captured by the first two principal components is:
$$\text{Captured Variance} = \lambda_1 + \lambda_2 = 18.19 + 3.83 = 22.02$$

The percentage of variance captured is:
$$\frac{\text{Captured Variance}}{\text{Total Variance}} \times 100\% = \frac{22.02}{23} \times 100\% \approx 95.7\%$$

#### Part c: Possibility of preserving at least 90% variance

Since the first two principal components capture approximately 95.7% of the variance, which is greater than 90%, it is indeed possible to reduce the dimensionality to 2 while preserving at least 90% of the variance.

#### Part d: Projection matrix for dimensionality reduction

The projection matrix for reducing the dimensionality to 2 is formed by using the first two eigenvectors as columns:

$$\mathbf{P} = \begin{bmatrix} 
0.71 & -0.19 \\
0.62 & 0.32 \\
0.33 & 0.93
\end{bmatrix}$$

To project data from the original 3D space to the 2D principal component space, we compute:
$$\mathbf{Y} = \mathbf{P}^T(\mathbf{X} - \boldsymbol{\mu})$$

where $\mathbf{X}$ is the original data point and $\boldsymbol{\mu}$ is the mean of the data.

## Example 3: Eigendecomposition for Whitening Transformation

### Problem Statement
A machine learning researcher wants to preprocess multivariate data by whitening it (transforming it to have uncorrelated variables with unit variance). The original data has a covariance matrix:

$$\boldsymbol{\Sigma} = \begin{bmatrix} 
6 & 2 \\
2 & 4
\end{bmatrix}$$

a) Find the eigendecomposition of this covariance matrix.
b) Derive the whitening transformation matrix.
c) Verify that applying this transformation to the original data will result in a covariance matrix equal to the identity matrix.
d) What are the geometric implications of this transformation?

### Solution

#### Part a: Finding the eigendecomposition

First, we find the eigenvalues:
$$|\boldsymbol{\Sigma} - \lambda \mathbf{I}| = 0$$

$$\begin{vmatrix} 6 - \lambda & 2 \\ 2 & 4 - \lambda \end{vmatrix} = 0$$

$$(6 - \lambda)(4 - \lambda) - 2 \times 2 = 0$$

$$24 - 6\lambda - 4\lambda + \lambda^2 - 4 = 0$$

$$\lambda^2 - 10\lambda + 20 = 0$$

Using the quadratic formula:
$$\lambda = \frac{10 \pm \sqrt{10^2 - 4 \times 20}}{2} = \frac{10 \pm \sqrt{100 - 80}}{2} = \frac{10 \pm \sqrt{20}}{2} \approx \frac{10 \pm 4.47}{2}$$

$$\lambda_1 \approx 7.24, \lambda_2 \approx 2.76$$

Now, we find the eigenvectors for each eigenvalue.

For $\lambda_1 \approx 7.24$:
$$\begin{bmatrix} 6 - 7.24 & 2 \\ 2 & 4 - 7.24 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$\begin{bmatrix} -1.24 & 2 \\ 2 & -3.24 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

This gives us:
$$-1.24v_{11} + 2v_{12} = 0$$
$$2v_{11} - 3.24v_{12} = 0$$

From the first equation:
$$v_{11} = \frac{2v_{12}}{1.24} \approx 1.61v_{12}$$

Normalizing:
$$\sqrt{(1.61v_{12})^2 + v_{12}^2} = 1$$
$$\sqrt{2.59^2 + 1^2}v_{12} = 1$$
$$\sqrt{3.59}v_{12} = 1$$
$$v_{12} \approx 0.528$$
$$v_{11} \approx 1.61 \times 0.528 \approx 0.850$$

Therefore, $\mathbf{v}_1 \approx (0.85, 0.53)$.

Similarly, for $\lambda_2 \approx 2.76$, we get $\mathbf{v}_2 \approx (-0.53, 0.85)$.

The eigendecomposition is:
$$\boldsymbol{\Sigma} = \mathbf{P} \boldsymbol{\Lambda} \mathbf{P}^T = \begin{bmatrix} 0.85 & -0.53 \\ 0.53 & 0.85 \end{bmatrix} \begin{bmatrix} 7.24 & 0 \\ 0 & 2.76 \end{bmatrix} \begin{bmatrix} 0.85 & 0.53 \\ -0.53 & 0.85 \end{bmatrix}$$

#### Part b: Deriving the whitening transformation

The whitening transformation matrix is:
$$\mathbf{W} = \mathbf{P} \boldsymbol{\Lambda}^{-1/2} \mathbf{P}^T$$

where $\boldsymbol{\Lambda}^{-1/2}$ is a diagonal matrix with elements $1/\sqrt{\lambda_i}$.

$$\boldsymbol{\Lambda}^{-1/2} = \begin{bmatrix} 1/\sqrt{7.24} & 0 \\ 0 & 1/\sqrt{2.76} \end{bmatrix} \approx \begin{bmatrix} 0.37 & 0 \\ 0 & 0.60 \end{bmatrix}$$

Therefore:
$$\mathbf{W} = \begin{bmatrix} 0.85 & -0.53 \\ 0.53 & 0.85 \end{bmatrix} \begin{bmatrix} 0.37 & 0 \\ 0 & 0.60 \end{bmatrix} \begin{bmatrix} 0.85 & 0.53 \\ -0.53 & 0.85 \end{bmatrix}$$

Computing this product:
$$\mathbf{W} \approx \begin{bmatrix} 0.41 & -0.07 \\ -0.07 & 0.54 \end{bmatrix}$$

#### Part c: Verifying the transformation

To verify that this transformation whitens the data, we need to show that:
$$\mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^T = \mathbf{I}$$

Using the eigendecomposition:
$$\mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^T = (\mathbf{P} \boldsymbol{\Lambda}^{-1/2} \mathbf{P}^T)(\mathbf{P} \boldsymbol{\Lambda} \mathbf{P}^T)(\mathbf{P} \boldsymbol{\Lambda}^{-1/2} \mathbf{P}^T)^T$$

$$= \mathbf{P} \boldsymbol{\Lambda}^{-1/2} \mathbf{P}^T \mathbf{P} \boldsymbol{\Lambda} \mathbf{P}^T \mathbf{P} \boldsymbol{\Lambda}^{-1/2} \mathbf{P}^T$$

Since $\mathbf{P}$ is orthogonal, $\mathbf{P}^T \mathbf{P} = \mathbf{I}$, so:
$$= \mathbf{P} \boldsymbol{\Lambda}^{-1/2} \boldsymbol{\Lambda} \boldsymbol{\Lambda}^{-1/2} \mathbf{P}^T$$

$$= \mathbf{P} \boldsymbol{\Lambda}^{-1/2} \boldsymbol{\Lambda}^{1/2} \boldsymbol{\Lambda}^{1/2} \boldsymbol{\Lambda}^{-1/2} \mathbf{P}^T$$

$$= \mathbf{P} \mathbf{I} \mathbf{P}^T = \mathbf{P} \mathbf{P}^T = \mathbf{I}$$

This confirms that the transformation whitens the data.

#### Part d: Geometric implications

Geometrically, the whitening transformation:
1. Rotates the data to align with the principal components (using $\mathbf{P}$)
2. Scales each principal component by the inverse square root of its variance (using $\boldsymbol{\Lambda}^{-1/2}$)
3. Rotates back to the original coordinate system (using $\mathbf{P}^T$)

The result is a transformation that converts the elliptical contours of the original distribution into circular contours. This removes correlations between variables and equalizes the variance in all directions, making the data more suitable for algorithms that assume spherical distributions or equal feature importance.

## Related Topics

- [[L2_1_PCA|Principal Component Analysis]]: Application of eigendecomposition for dimension reduction
- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: Examples of multivariate normal distributions
- [[L2_1_Covariance_Matrix_Contours_Examples|Covariance Matrix Contours]]: Visualization of eigendecomposition effects
- [[L2_1_Linear_Transformation|Linear Transformations]]: Mathematical basis for eigendecomposition transformations
- [[L2_1_Dimension_Reduction|Dimension Reduction]]: Techniques for reducing data dimensionality 