# Question 24: LDA Projection for Two-Dimensional Dataset

## Problem Statement
Compute LDA projection for the following two-dimensional dataset:
- $X_1=(x_1, x_2)=\{(4, 1), (2, 4), (2, 3), (3, 6), (4, 4)\}$
- $X_2=(x_1, x_2)=\{(9, 10), (6, 8), (9, 5), (8, 7), (10, 8)\}$

### Task
1. Calculate the mean vectors $\mu_1$ and $\mu_2$ for each class
2. Compute the within-class scatter matrices $S_1$ and $S_2$ for each class
3. Determine the total within-class scatter matrix $S_W$
4. Calculate the between-class scatter matrix $S_B$
5. Find the optimal projection direction $\mathbf{w}$ by solving the generalized eigenvalue problem
6. For a new data point $(5, 5)$, determine which class it would be assigned to using LDA

## Understanding the Problem
Linear Discriminant Analysis (LDA) is a dimensionality reduction technique that aims to find a linear projection that maximizes the separation between classes while minimizing the variance within each class. Unlike Principal Component Analysis (PCA), which is an unsupervised method that maximizes variance without considering class labels, LDA is a supervised technique that explicitly uses class information to find an optimal projection for classification.

In this problem, we have a two-dimensional dataset with points from two classes. We need to apply LDA to find the optimal projection direction that best separates these classes, and then use this projection to classify a new data point.

## Solution

### Step 1: Calculate the mean vectors for each class
First, we compute the mean vectors for each class by averaging the coordinates of all points in that class:

For Class 1:
$$\text{Sum}_1 = (4, 1) + (2, 4) + (2, 3) + (3, 6) + (4, 4) = (15, 18)$$
$$\mu_1 = \frac{1}{n_1} \text{Sum}_1 = \frac{1}{5}(15, 18) = (3.0, 3.6)$$

For Class 2:
$$\text{Sum}_2 = (9, 10) + (6, 8) + (9, 5) + (8, 7) + (10, 8) = (42, 38)$$
$$\mu_2 = \frac{1}{n_2} \text{Sum}_2 = \frac{1}{5}(42, 38) = (8.4, 7.6)$$

The mean vectors represent the centers of each class in the feature space:
- Class 1 mean: $\mu_1 = (3.0, 3.6)$
- Class 2 mean: $\mu_2 = (8.4, 7.6)$

![Data Points and Means](../Images/L4_4_Quiz_24/data_points_means.png)

### Step 2: Compute the within-class scatter matrices for each class
The within-class scatter matrix measures the dispersion of points within each class. For each class, we compute:

$$S_k = \sum_{x \in X_k} (x - \mu_k)(x - \mu_k)^T$$

For Class 1:
$$(4, 1) - \mu_1 = (1.0, -2.6) \implies \begin{bmatrix} 1.0 \\ -2.6 \end{bmatrix} \begin{bmatrix} 1.0 & -2.6 \end{bmatrix} = \begin{bmatrix} 1.0 & -2.6 \\ -2.6 & 6.76 \end{bmatrix}$$
$$(2, 4) - \mu_1 = (-1.0, 0.4) \implies \begin{bmatrix} -1.0 \\ 0.4 \end{bmatrix} \begin{bmatrix} -1.0 & 0.4 \end{bmatrix} = \begin{bmatrix} 1.0 & -0.4 \\ -0.4 & 0.16 \end{bmatrix}$$
$$(2, 3) - \mu_1 = (-1.0, -0.6) \implies \begin{bmatrix} -1.0 \\ -0.6 \end{bmatrix} \begin{bmatrix} -1.0 & -0.6 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.6 \\ 0.6 & 0.36 \end{bmatrix}$$
$$(3, 6) - \mu_1 = (0.0, 2.4) \implies \begin{bmatrix} 0.0 \\ 2.4 \end{bmatrix} \begin{bmatrix} 0.0 & 2.4 \end{bmatrix} = \begin{bmatrix} 0.0 & 0.0 \\ 0.0 & 5.76 \end{bmatrix}$$
$$(4, 4) - \mu_1 = (1.0, 0.4) \implies \begin{bmatrix} 1.0 \\ 0.4 \end{bmatrix} \begin{bmatrix} 1.0 & 0.4 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.4 \\ 0.4 & 0.16 \end{bmatrix}$$

$$S_1 = \text{Sum of above matrices} = 
\begin{bmatrix}
4.0 & -2.0 \\
-2.0 & 13.2
\end{bmatrix}
$$

For Class 2:
$$(9, 10) - \mu_2 = (0.6, 2.4) \implies \begin{bmatrix} 0.6 \\ 2.4 \end{bmatrix} \begin{bmatrix} 0.6 & 2.4 \end{bmatrix} = \begin{bmatrix} 0.36 & 1.44 \\ 1.44 & 5.76 \end{bmatrix}$$
$$(6, 8) - \mu_2 = (-2.4, 0.4) \implies \begin{bmatrix} -2.4 \\ 0.4 \end{bmatrix} \begin{bmatrix} -2.4 & 0.4 \end{bmatrix} = \begin{bmatrix} 5.76 & -0.96 \\ -0.96 & 0.16 \end{bmatrix}$$
$$(9, 5) - \mu_2 = (0.6, -2.6) \implies \begin{bmatrix} 0.6 \\ -2.6 \end{bmatrix} \begin{bmatrix} 0.6 & -2.6 \end{bmatrix} = \begin{bmatrix} 0.36 & -1.56 \\ -1.56 & 6.76 \end{bmatrix}$$
$$(8, 7) - \mu_2 = (-0.4, -0.6) \implies \begin{bmatrix} -0.4 \\ -0.6 \end{bmatrix} \begin{bmatrix} -0.4 & -0.6 \end{bmatrix} = \begin{bmatrix} 0.16 & 0.24 \\ 0.24 & 0.36 \end{bmatrix}$$
$$(10, 8) - \mu_2 = (1.6, 0.4) \implies \begin{bmatrix} 1.6 \\ 0.4 \end{bmatrix} \begin{bmatrix} 1.6 & 0.4 \end{bmatrix} = \begin{bmatrix} 2.56 & 0.64 \\ 0.64 & 0.16 \end{bmatrix}$$

$$S_2 = \text{Sum of above matrices} = 
\begin{bmatrix}
9.2 & -0.2 \\
-0.2 & 13.2
\end{bmatrix}
$$

These matrices capture how the data points in each class are distributed around their respective means.

### Step 3: Determine the total within-class scatter matrix SW
The total within-class scatter matrix is the sum of the scatter matrices for each class:

$$S_W = S_1 + S_2$$

$$S_W = 
\begin{bmatrix}
4.0 & -2.0 \\
-2.0 & 13.2
\end{bmatrix} +
\begin{bmatrix}
9.2 & -0.2 \\
-0.2 & 13.2
\end{bmatrix} =
\begin{bmatrix}
13.2 & -2.2 \\
-2.2 & 26.4
\end{bmatrix}
$$

This matrix represents the overall within-class variability that we want to minimize in LDA.

### Step 4: Calculate the between-class scatter matrix SB
The between-class scatter matrix measures the dispersion of class means around the global mean. First, we compute the global mean:

$$\mu = \frac{n_1\mu_1 + n_2\mu_2}{n_1 + n_2} = \frac{5(3.0, 3.6) + 5(8.4, 7.6)}{10} = (5.7, 5.6)$$

Then, we calculate the between-class scatter matrix:

$$S_B = n_1(\mu_1 - \mu)(\mu_1 - \mu)^T + n_2(\mu_2 - \mu)(\mu_2 - \mu)^T$$

$$\mu_1 - \mu = (3.0, 3.6) - (5.7, 5.6) = (-2.7, -2.0)$$
$$\mu_2 - \mu = (8.4, 7.6) - (5.7, 5.6) = (2.7, 2.0)$$

$$S_B = 5 \begin{bmatrix} -2.7 \\ -2.0 \end{bmatrix} \begin{bmatrix} -2.7 & -2.0 \end{bmatrix} + 5 \begin{bmatrix} 2.7 \\ 2.0 \end{bmatrix} \begin{bmatrix} 2.7 & 2.0 \end{bmatrix}$$
$$S_B = 5 \begin{bmatrix} 7.29 & 5.4 \\ 5.4 & 4.0 \end{bmatrix} + 5 \begin{bmatrix} 7.29 & 5.4 \\ 5.4 & 4.0 \end{bmatrix} = \begin{bmatrix} 36.45 & 27.0 \\ 27.0 & 20.0 \end{bmatrix} + \begin{bmatrix} 36.45 & 27.0 \\ 27.0 & 20.0 \end{bmatrix}$$

$$S_B = 
\begin{bmatrix}
72.9 & 54.0 \\
54.0 & 40.0
\end{bmatrix}
$$

(Alternatively for 2 classes: $S_B = \frac{n_1 n_2}{n_1+n_2} (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T = \frac{25}{10} \begin{bmatrix} -5.4 \\ -4.0 \end{bmatrix} \begin{bmatrix} -5.4 & -4.0 \end{bmatrix} = 2.5 \begin{bmatrix} 29.16 & 21.6 \\ 21.6 & 16.0 \end{bmatrix} = S_B$)

This matrix represents the between-class variability that we want to maximize in LDA.

![Scatter Matrices Visualization](../Images/L4_4_Quiz_24/scatter_matrices.png)

### Step 5: Find the optimal projection direction w
The optimal projection direction in LDA is given by the eigenvector corresponding to the largest eigenvalue of the matrix $S_W^{-1}S_B$. This can be formulated as a generalized eigenvalue problem:

$$S_B w = \lambda S_W w$$ 
Or equivalently, solve the standard eigenvalue problem:
$$S_W^{-1} S_B w = \lambda w$$

1. Calculate $S_W^{-1}$:
$$S_W^{-1} = \begin{bmatrix} 13.2 & -2.2 \\ -2.2 & 26.4 \end{bmatrix}^{-1} = \frac{1}{13.2 \times 26.4 - (-2.2)^2} \begin{bmatrix} 26.4 & 2.2 \\ 2.2 & 13.2 \end{bmatrix} = \frac{1}{348.48 - 4.84} \begin{bmatrix} 26.4 & 2.2 \\ 2.2 & 13.2 \end{bmatrix}$$
$$S_W^{-1} = \frac{1}{343.64} \begin{bmatrix} 26.4 & 2.2 \\ 2.2 & 13.2 \end{bmatrix} \approx \begin{bmatrix} 0.0768 & 0.0064 \\ 0.0064 & 0.0384 \end{bmatrix}$$

2. Calculate $M = S_W^{-1}S_B$:
$$M = \begin{bmatrix} 0.0768 & 0.0064 \\ 0.0064 & 0.0384 \end{bmatrix} \begin{bmatrix} 72.9 & 54.0 \\ 54.0 & 40.0 \end{bmatrix} = \begin{bmatrix} 5.946 & 4.405 \\ 2.541 & 1.882 \end{bmatrix}$$

3. Find eigenvalues ($\lambda$) and eigenvectors ($w$) of $M$:
   Solving $\det(M - \lambda I) = 0$. The eigenvalues are found to be:
   $$\lambda_1 \approx 7.8284, \quad \lambda_2 \approx 0$$ 
   The eigenvector corresponding to the largest eigenvalue $\lambda_1 \approx 7.8284$ is:
   $$w_{\text{unnormalized}} \approx \begin{bmatrix} 0.9196 \\ 0.3930 \end{bmatrix}$$

4. Normalize the eigenvector:
   $$\|w_{\text{unnormalized}}\| = \sqrt{0.9196^2 + 0.3930^2} \approx 1.000$$
   $$w = \frac{w_{\text{unnormalized}}}{\|w_{\text{unnormalized}}\|} \approx \begin{bmatrix} 0.9196 \\ 0.3930 \end{bmatrix}$$ 

This vector $w = [0.9196, 0.3930]^T$ is the optimal projection direction for LDA that maximizes the between-class variance relative to the within-class variance when the data is projected onto it.

![LDA Projection](../Images/L4_4_Quiz_24/lda_projection.png)

### Step 6: Classify a new data point (5, 5)
To classify a new data point $x_{\text{new}} = (5, 5)$ using LDA, we project it and the class means onto the LDA direction $w$, and then assign the point to the class whose projected mean is closest to the projected point.

1. Project the new point onto the LDA direction:
   $$\text{proj}_{\text{new}} = w^T x_{\text{new}} = \begin{bmatrix} 0.9196 & 0.3930 \end{bmatrix} \begin{bmatrix} 5 \\ 5 \end{bmatrix} = 0.9196 \times 5 + 0.3930 \times 5 \approx 6.5626$$

2. Project the class means onto the LDA direction:
   $$\text{proj}_{\mu_1} = w^T \mu_1 = \begin{bmatrix} 0.9196 & 0.3930 \end{bmatrix} \begin{bmatrix} 3.0 \\ 3.6 \end{bmatrix} = 0.9196 \times 3.0 + 0.3930 \times 3.6 \approx 4.1733$$
   $$\text{proj}_{\mu_2} = w^T \mu_2 = \begin{bmatrix} 0.9196 & 0.3930 \end{bmatrix} \begin{bmatrix} 8.4 \\ 7.6 \end{bmatrix} = 0.9196 \times 8.4 + 0.3930 \times 7.6 \approx 10.7107$$

3. Calculate distances between projected points:
   $$\text{distance to } \mu_1 = |\text{proj}_{\text{new}} - \text{proj}_{\mu_1}| = |6.5626 - 4.1733| \approx 2.3893$$
   $$\text{distance to } \mu_2 = |\text{proj}_{\text{new}} - \text{proj}_{\mu_2}| = |6.5626 - 10.7107| \approx 4.1482$$

4. Assign the point to the class with the minimum distance:
   Since $2.3893 < 4.1482$, the distance to the projected mean of Class 1 is smaller. Therefore, the new point $(5, 5)$ is assigned to **Class 1**.

![Classification Result](../Images/L4_4_Quiz_24/classification_result.png)

## Visual Explanations

### Data Points and Class Means
![Data Points and Means](../Images/L4_4_Quiz_24/data_points_means.png)

This visualization shows the original data points from both classes in a 2D feature space. Blue circles represent Class 1 points, and red crosses represent Class 2 points. The star markers show the mean of each class: $\mu_1 = (3.0, 3.6)$ for Class 1 and $\mu_2 = (8.4, 7.6)$ for Class 2. The classes are visibly separated in the feature space, with Class 1 concentrated in the lower-left region and Class 2 in the upper-right region.

### LDA Projection Direction
![LDA Projection](../Images/L4_4_Quiz_24/lda_projection.png)

This figure illustrates the optimal LDA projection direction (black arrow along $w = [0.9196, 0.3930]^T$) that maximizes the separation between classes while minimizing within-class variance. The dotted lines show the projections of individual data points onto this direction (represented by the black line passing through the global mean). Notice how the projections of points from the same class (blue/red '+' markers) are relatively close together along the LDA direction, while the projections of points from different classes are well-separated.

### Classification of New Point
![Classification Result](../Images/L4_4_Quiz_24/classification_result.png)

This visualization shows the classification of the new point $(5, 5)$, marked as a green diamond. The green line perpendicular to the LDA direction represents the decision boundary, located at the midpoint between the projected means. The light blue and light red regions indicate the classification areas for Class 1 and Class 2, respectively. The new point falls in the blue region and is correctly classified as belonging to Class 1, as its projection onto the LDA direction (green '+') is closer to the projection of the Class 1 mean than the Class 2 mean.

## Key Insights

### Mathematical Understanding
- LDA finds the projection direction $w$ that maximizes the Fisher criterion $J(w) = \frac{w^T S_B w}{w^T S_W w}$
- The optimal projection direction $w$ is given by the eigenvector corresponding to the largest eigenvalue of $S_W^{-1}S_B$
- In a two-class problem with a two-dimensional feature space, LDA reduces the dimensionality to one dimension (a line)
- The decision boundary in the original space is a hyperplane (a line in 2D) perpendicular to the LDA projection direction $w$ and passing through a point determined by the projected means and class priors (often the midpoint of projected means for balanced classes).

### Geometric Interpretation
- LDA seeks a direction that maximizes the distance between projected class means while simultaneously minimizing the scatter (variance) of projections within each class.
- The within-class scatter matrices ($S_1, S_2$) describe the shape and orientation of each class cluster.
- The between-class scatter matrix ($S_B$) captures the separation between the class centroids.
- The optimal projection $w$ is the direction along which the projected class distributions have the least overlap.

### Practical Considerations
- LDA inherently assumes that the classes have equal covariance matrices ($S_1 \approx S_2$). While not strictly true here ($S_1 \neq S_2$), LDA often performs reasonably well even with moderate violations of this assumption.
- It also assumes data within each class is normally distributed, although it's relatively robust to deviations.
- LDA provides both a dimensionality reduction technique (projecting onto $w$) and a classification method (comparing distances in the projected space).
- For two classes, LDA always reduces to a single projection direction, regardless of the original dimensionality.

## Conclusion
We successfully applied Linear Discriminant Analysis to a two-dimensional dataset with two classes:

1. We calculated the mean vectors: $\mu_1 = (3.0, 3.6)$ and $\mu_2 = (8.4, 7.6)$
2. We computed the within-class scatter matrices $S_1 = \begin{bmatrix} 4.0 & -2.0 \\ -2.0 & 13.2 \end{bmatrix}$ and $S_2 = \begin{bmatrix} 9.2 & -0.2 \\ -0.2 & 13.2 \end{bmatrix}$, leading to the total within-class scatter matrix $S_W = \begin{bmatrix} 13.2 & -2.2 \\ -2.2 & 26.4 \end{bmatrix}$
3. We calculated the between-class scatter matrix $S_B = \begin{bmatrix} 72.9 & 54.0 \\ 54.0 & 40.0 \end{bmatrix}$
4. We found the optimal projection direction $w \approx [0.9196, 0.3930]^T$ by solving the generalized eigenvalue problem $S_B w = \lambda S_W w$
5. We classified the new point $(5, 5)$ by projecting it and the class means onto $w$ and assigned it to **Class 1** based on the minimum distance in the projected 1D space ($\|\text{proj}_{\text{new}} - \text{proj}_{\mu_1}\| < \|\text{proj}_{\text{new}} - \text{proj}_{\mu_2}\|$)

LDA effectively identified the direction maximizing class separation and facilitated the classification of the new point, demonstrating its utility as both a dimensionality reduction and classification tool. 