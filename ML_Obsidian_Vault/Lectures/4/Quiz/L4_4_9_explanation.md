# Question 9: LDA Assumptions and Decision Boundary

## Problem Statement
Linear Discriminant Analysis (LDA) makes several assumptions about the underlying data distributions.

### Task
1. List two key assumptions of LDA
2. Given two classes with equal covariance matrices and equal prior probabilities, if the means are $\mu_1 = [2, 3]^T$ and $\mu_2 = [4, 1]^T$, at what point would the posterior probabilities $P(C_1|x) = P(C_2|x) = 0.5$?
3. For a two-class LDA with shared covariance matrix $\Sigma = I$ (identity matrix), write the decision boundary equation in terms of the class means $\mu_1$ and $\mu_2$
4. How does LDA differ from the Perceptron in terms of how it finds the decision boundary? Answer in one sentence

## Understanding the Problem
Linear Discriminant Analysis (LDA) is a statistical approach to classification that assumes specific properties of the underlying data distributions. This problem explores the key assumptions of LDA, how it calculates the decision boundary, and how it compares to other classification methods like the Perceptron. Understanding these concepts is crucial for applying LDA appropriately and interpreting its results.

## Solution

### Step 1: Understand key assumptions of LDA
LDA makes several important assumptions about the data:

1. **Classes follow multivariate Gaussian distributions**: LDA assumes that the data in each class is drawn from a multivariate normal distribution.

2. **Homoscedasticity (equal covariance matrices)**: All classes share the same covariance matrix, meaning the shape and orientation of the data distributions are the same for all classes.

3. **No perfect multicollinearity**: The features should not be perfectly correlated, and the covariance matrix should be invertible.

4. **Sufficient sample size**: The sample size should be larger than the number of predictors.

When these assumptions are met, LDA provides the optimal decision boundary in the Bayes sense, minimizing the classification error.

![LDA Assumptions: Equal Covariance Matrices](../Images/L4_4_Quiz_9/lda_assumptions_equal_covariance.png)

As shown in the figure above, both classes have the same shape and orientation, differing only in their means. This illustrates the equal covariance assumption of LDA.

### Step 2: Calculate the LDA projection direction
For the problem with means $\boldsymbol{\mu}_1 = [1, 2]^T$ and $\boldsymbol{\mu}_2 = [3, 0]^T$ and shared covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$, we need to find the LDA projection direction $\mathbf{w}$.

The formula for the LDA projection direction is:
$$\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$$

Let's calculate this step-by-step:

1. First, calculate the inverse of the covariance matrix $\boldsymbol{\Sigma}$:
   
   For a $2 \times 2$ matrix $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse is:
   $$\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$
   
   Our covariance matrix is $\boldsymbol{\Sigma} = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$
   
   Determinant = $(2 \times 1) - (0 \times 0) = 2$
   
   Inverse = $\frac{1}{2} \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 0.5 & 0 \\ 0 & 1 \end{bmatrix}$
   
   Therefore, $\boldsymbol{\Sigma}^{-1} = \begin{bmatrix} 0.5 & 0 \\ 0 & 1 \end{bmatrix}$

2. Calculate the difference between class means:
   $$\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2 = \begin{bmatrix} 1 \\ 2 \end{bmatrix} - \begin{bmatrix} 3 \\ 0 \end{bmatrix} = \begin{bmatrix} -2 \\ 2 \end{bmatrix}$$

3. Calculate the projection vector:
   $$\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2) = \begin{bmatrix} 0.5 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} -2 \\ 2 \end{bmatrix}$$
   
   Computing this multiplication:
   $$\mathbf{w} = \begin{bmatrix} 0.5 \times (-2) + 0 \times 2 \\ 0 \times (-2) + 1 \times 2 \end{bmatrix} = \begin{bmatrix} -1 \\ 2 \end{bmatrix}$$

![LDA Projection](../Images/L4_4_Quiz_9/lda_projection.png)

The figure above illustrates the LDA projection direction $\mathbf{w}$ and the resulting decision boundary. Note that $\mathbf{w}$ is not perpendicular to the line connecting the class means unless the covariance matrix is a scalar multiple of the identity matrix.

### Step 3: Determine the threshold for classification
For LDA with equal prior probabilities, the threshold is the midpoint of the projected class means.

1. Project the class means onto the direction $\mathbf{w}$:
   $$\mathbf{w}^T\boldsymbol{\mu}_1 = [-1, 2] \cdot [1, 2]^T = (-1) \times 1 + 2 \times 2 = -1 + 4 = 3$$
   $$\mathbf{w}^T\boldsymbol{\mu}_2 = [-1, 2] \cdot [3, 0]^T = (-1) \times 3 + 2 \times 0 = -3 + 0 = -3$$

2. Calculate the threshold as the midpoint:
   $$\text{threshold} = \frac{\mathbf{w}^T\boldsymbol{\mu}_1 + \mathbf{w}^T\boldsymbol{\mu}_2}{2} = \frac{3 + (-3)}{2} = 0$$

3. The decision rule becomes:
   - If $\mathbf{w}^T \mathbf{x} > 0$, classify as Class 1
   - If $\mathbf{w}^T \mathbf{x} < 0$, classify as Class 2
   
The point where the posterior probabilities are equal ($P(C_1|\mathbf{x}) = P(C_2|\mathbf{x}) = 0.5$) is precisely at this decision boundary, where $\mathbf{w}^T \mathbf{x} = 0$.

Expanding the decision boundary equation:
$$\mathbf{w}^T \mathbf{x} = 0$$
$$[-1, 2] \cdot [x_1, x_2]^T = 0$$
$$-x_1 + 2x_2 = 0$$
$$x_1 = 2x_2$$

This is the equation of the decision boundary in the original feature space.

### Step 4: Classify new data points
Let's classify two new points using our LDA model:

**Point 1**: $\mathbf{x}_1 = [2, 1]^T$
- Projection onto $\mathbf{w}$: $\mathbf{w}^T \mathbf{x}_1 = [-1, 2] \cdot [2, 1]^T = (-1) \times 2 + 2 \times 1 = -2 + 2 = 0$
- Since $\mathbf{w}^T \mathbf{x}_1 = 0$ (exactly at the threshold), we classify this as Class 2 (by convention)

**Point 2**: $\mathbf{x}_2 = [0, 3]^T$
- Projection onto $\mathbf{w}$: $\mathbf{w}^T \mathbf{x}_2 = [-1, 2] \cdot [0, 3]^T = (-1) \times 0 + 2 \times 3 = 0 + 6 = 6$
- Since $\mathbf{w}^T \mathbf{x}_2 > 0$, we classify this as Class 1

![LDA Classification](../Images/L4_4_Quiz_9/lda_classification.png)

The figure above shows the classification of the two points. Notice that $\mathbf{x}_1$ lies exactly on the decision boundary, while $\mathbf{x}_2$ is clearly in the Class 1 region.

### Step 5: Visualize posterior probabilities
The LDA model allows us to calculate posterior probabilities for each class. These probabilities can be visualized as a continuous field over the feature space.

![LDA Posterior Probabilities](../Images/L4_4_Quiz_9/lda_posterior.png)

The figure above shows the posterior probability of Class 1 throughout the feature space. The decision boundary corresponds to the 0.5 contour where $P(C_1|\mathbf{x}) = P(C_2|\mathbf{x}) = 0.5$.

We can also visualize these posterior probabilities in 3D, which gives a better understanding of how the probability changes across the feature space:

![LDA Posterior Probabilities in 3D](../Images/L4_4_Quiz_9/lda_posterior_3d.png)

In this 3D visualization, the height represents the posterior probability of Class 1, and the decision boundary is where the surface crosses the 0.5 level.

### Step 6: Compare LDA with Perceptron
Let's visualize the difference between LDA and Perceptron decision boundaries:

![LDA vs Perceptron](../Images/L4_4_Quiz_9/lda_vs_perceptron.png)

LDA differs from the Perceptron in several important ways:

1. **Approach**:
   - LDA takes a probabilistic approach based on modeling class distributions
   - Perceptron uses an iterative, error-driven approach

2. **Direction vector**:
   - LDA uses $\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$ as the projection direction
   - Perceptron often uses $\mathbf{w} = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_2$ (or variations based on misclassified points)

3. **Statistical properties**:
   - LDA is statistically optimal when its assumptions are met
   - Perceptron makes no distributional assumptions

4. **Posterior probabilities**:
   - LDA provides posterior class probabilities
   - Standard Perceptron only provides class assignments

## Key Insights

### Theoretical Framework
- LDA is derived from a generative modeling approach, modeling class-conditional densities and using Bayes' rule
- When its assumptions are met, LDA provides the optimal decision boundary in the Bayes sense
- The projection direction $\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$ maximizes class separation relative to the shared covariance

### Practical Considerations
- LDA performs well when the class distributions are approximately Gaussian with equal covariance matrices
- The direction $\mathbf{w}$ is influenced by both the class means and the covariance structure
- When covariances differ significantly between classes, Quadratic Discriminant Analysis (QDA) may be more appropriate
- LDA can provide good results even with moderate violations of its assumptions

### Computational Aspects
- LDA has a closed-form solution (no iterative optimization required)
- The decision boundary is always linear
- The orientation of the boundary depends on the covariance structure, not just on the class means
- LDA naturally handles multi-class problems

## Conclusion
1. Two key assumptions of LDA are:
   - Classes follow multivariate Gaussian distributions
   - Classes share the same covariance matrix (homoscedasticity)

2. For the given means $\boldsymbol{\mu}_1 = [1, 2]^T$ and $\boldsymbol{\mu}_2 = [3, 0]^T$ with shared covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$, the decision boundary (where posterior probabilities equal 0.5) is given by the equation $-x_1 + 2x_2 = 0$ or equivalently $x_1 = 2x_2$.

3. Classification results:
   - Point $\mathbf{x}_1 = [2, 1]^T$ is classified as Class 2 (lies exactly on the boundary)
   - Point $\mathbf{x}_2 = [0, 3]^T$ is classified as Class 1 (has a positive projection onto $\mathbf{w}$)

4. LDA differs from the Perceptron in that it takes a statistical approach to find the optimal decision boundary based on class distributions and covariance structure, while the Perceptron simply seeks any hyperplane that separates the classes without considering the underlying distributions. 