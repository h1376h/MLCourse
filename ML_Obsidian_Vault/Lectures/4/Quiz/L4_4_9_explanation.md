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

### Task 1: Key Assumptions of LDA
LDA makes several important assumptions about the data:

1. **Classes follow multivariate Gaussian distributions**: LDA assumes that the data in each class is drawn from a multivariate normal distribution.

2. **Homoscedasticity (equal covariance matrices)**: All classes share the same covariance matrix, meaning the shape and orientation of the data distributions are the same for all classes.

Additional assumptions include:
- No perfect multicollinearity (the covariance matrix should be invertible)
- Sufficient sample size (larger than the number of predictors)

When these assumptions are met, LDA provides the optimal decision boundary in the Bayes sense, minimizing the classification error.

![[../Images/L4_4_Quiz_9/lda_assumptions_equal_covariance.png]]

As shown in the figure above, both classes have the same shape and orientation, differing only in their means. This illustrates the equal covariance assumption of LDA.

### Task 2: Finding the Decision Boundary with Equal Posterior Probabilities
For the problem with means $\mu_1 = [2, 3]^T$ and $\mu_2 = [4, 1]^T$ and shared covariance matrix $\Sigma = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$, we need to find the points where $P(C_1|x) = P(C_2|x) = 0.5$.

The approach involves:
1. Finding the LDA projection direction $w$
2. Calculating the classification threshold
3. Deriving the decision boundary equation

#### Step 1: Calculate the LDA projection direction

The formula for the LDA projection direction is:
$$w = \Sigma^{-1}(\mu_1 - \mu_2)$$

First, calculate the inverse of the covariance matrix $\Sigma$:

For a $2 \times 2$ matrix $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse is:
$$\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

Our covariance matrix is $\Sigma = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$

Determinant = $(2 \times 1) - (0 \times 0) = 2$

Inverse = $\frac{1}{2} \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 0.5 & 0 \\ 0 & 1 \end{bmatrix}$

Therefore, $\Sigma^{-1} = \begin{bmatrix} 0.5 & 0 \\ 0 & 1 \end{bmatrix}$

Next, calculate the difference between class means:
$$\mu_1 - \mu_2 = \begin{bmatrix} 2 \\ 3 \end{bmatrix} - \begin{bmatrix} 4 \\ 1 \end{bmatrix} = \begin{bmatrix} -2 \\ 2 \end{bmatrix}$$

Finally, calculate the projection vector:
$$w = \Sigma^{-1}(\mu_1 - \mu_2) = \begin{bmatrix} 0.5 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} -2 \\ 2 \end{bmatrix} = \begin{bmatrix} -1 \\ 2 \end{bmatrix}$$

![[../Images/L4_4_Quiz_9/lda_projection.png]]

#### Step 2: Determine the classification threshold

For LDA with equal prior probabilities, the threshold is the midpoint of the projected class means.

Project the class means onto the direction $w$:
$$w^T\mu_1 = [-1, 2] \cdot [2, 3]^T = (-1) \times 2 + 2 \times 3 = -2 + 6 = 4$$
$$w^T\mu_2 = [-1, 2] \cdot [4, 1]^T = (-1) \times 4 + 2 \times 1 = -4 + 2 = -2$$

Calculate the threshold as the midpoint:
$$\text{threshold} = \frac{w^T\mu_1 + w^T\mu_2}{2} = \frac{4 + (-2)}{2} = 1$$

#### Step 3: Derive the decision boundary equation

The decision boundary occurs where $w^T x = \text{threshold}$, which is precisely where the posterior probabilities are equal ($P(C_1|x) = P(C_2|x) = 0.5$).

Expanding the decision boundary equation:
$$w^T x = 1$$
$$[-1, 2] \cdot [x_1, x_2]^T = 1$$
$$-x_1 + 2x_2 = 1$$
$$x_1 = 2x_2 - 1$$

This is the equation of the decision boundary in the original feature space. All points on this line have equal posterior probabilities of belonging to either class.

#### Example: Classifying Data Points

Let's classify two example points using our LDA model:

1. For point $x_1 = [2, 1]^T$:
   $$w^T x_1 = [-1, 2] \cdot [2, 1]^T = -2 + 2 = 0$$
   Since $0 < 1$ (threshold), $x_1$ is classified as Class 2.

2. For point $x_2 = [0, 3]^T$:
   $$w^T x_2 = [-1, 2] \cdot [0, 3]^T = 0 + 6 = 6$$
   Since $6 > 1$ (threshold), $x_2$ is classified as Class 1.

![[../Images/L4_4_Quiz_9/lda_classification.png]]

The figure above illustrates the classification of these test points. The black dashed line represents the decision boundary, while the colored points show the test samples and their classifications.

#### Feature Importance in LDA

An important aspect of LDA is understanding which features contribute most to the classification decision. We can analyze this by examining the magnitudes of the components of the projection vector $w$.

![[../Images/L4_4_Quiz_9/lda_feature_importance.png]]

In our example, the magnitude of $w_2$ (2.0) is greater than that of $w_1$ (1.0), indicating that the second feature ($x_2$) has a stronger influence on the classification decision. This is expected given our covariance matrix, where the first feature has higher variance (2 vs. 1), thus reducing its relative weight in the decision boundary.

![[../Images/L4_4_Quiz_9/lda_posterior.png]]

The figure above shows the posterior probability of Class 1 throughout the feature space. The decision boundary (black dashed line) corresponds to the 0.5 contour where $P(C_1|x) = P(C_2|x) = 0.5$.

### Task 3: Decision Boundary with Identity Covariance Matrix
For a two-class LDA with shared covariance matrix $\Sigma = I$ (identity matrix), we need to derive the decision boundary equation in terms of the class means.

When $\Sigma = I$, the projection vector simplifies to:
$$w = \Sigma^{-1}(\mu_1 - \mu_2) = I \cdot (\mu_1 - \mu_2) = \mu_1 - \mu_2$$

The threshold calculation remains conceptually the same, but the projection vector is different:
$$\begin{align}
\text{threshold} &= \frac{w^T\mu_1 + w^T\mu_2}{2}\\
&= \frac{(\mu_1 - \mu_2)^T\mu_1 + (\mu_1 - \mu_2)^T\mu_2}{2}\\
&= \frac{\mu_1^T\mu_1 - \mu_2^T\mu_1 + \mu_1^T\mu_2 - \mu_2^T\mu_2}{2}
\end{align}$$

Since $\mu_2^T\mu_1 = \mu_1^T\mu_2$ (both are scalar dot products), this simplifies to:
$$\begin{align}
\text{threshold} &= \frac{\mu_1^T\mu_1 - \mu_2^T\mu_2}{2}\\
&= \frac{\|\mu_1\|^2 - \|\mu_2\|^2}{2}
\end{align}$$

Therefore, the general decision boundary equation for LDA with $\Sigma = I$ is:
$$(\mu_1 - \mu_2)^T x = \frac{\|\mu_1\|^2 - \|\mu_2\|^2}{2}$$

Applying this to our example values $\mu_1 = [2, 3]^T$ and $\mu_2 = [4, 1]^T$:
$$w = \mu_1 - \mu_2 = \begin{bmatrix} -2 \\ 2 \end{bmatrix}$$

$$\|\mu_1\|^2 = 2^2 + 3^2 = 4 + 9 = 13$$
$$\|\mu_2\|^2 = 4^2 + 1^2 = 16 + 1 = 17$$

$$\text{threshold} = \frac{13 - 17}{2} = -2$$

The decision boundary equation becomes:
$$\begin{bmatrix} -2 & 2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = -2$$
$$-2x_1 + 2x_2 = -2$$
$$-x_1 + x_2 = -1$$
$$x_1 = x_2 + 1$$

![[../Images/L4_4_Quiz_9/lda_identity_covariance.png]]

The figure above illustrates the case of LDA with identity covariance matrix. Note that the decision boundary is perpendicular to the line connecting the class means, which is not generally true when using a non-identity covariance matrix.

### Task 4: LDA vs. Perceptron Decision Boundary
LDA differs from the Perceptron in how it finds the decision boundary:

**LDA** takes a statistical approach to find the optimal decision boundary based on class distributions and covariance structure, while the **Perceptron** simply seeks any hyperplane that separates the classes without considering the underlying distributions.

![[../Images/L4_4_Quiz_9/lda_vs_perceptron.png]]

Key differences include:
- LDA uses $w = \Sigma^{-1}(\mu_1 - \mu_2)$ as the projection direction, accounting for covariance
- Perceptron often uses $w = \mu_1 - \mu_2$ or iteratively adjusts based on misclassifications
- LDA is statistically optimal when its assumptions are met
- LDA provides posterior probabilities, while Perceptron only provides class assignments

## Conclusion

1. **Two key assumptions of LDA are:**
   - Classes follow multivariate Gaussian distributions
   - Classes share the same covariance matrix (homoscedasticity)

2. **For the given means $\mu_1 = [2, 3]^T$ and $\mu_2 = [4, 1]^T$ with shared covariance matrix $\Sigma = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$**, the posterior probabilities are equal ($P(C_1|x) = P(C_2|x) = 0.5$) at the decision boundary given by the equation $-x_1 + 2x_2 = 1$ or equivalently $x_1 = 2x_2 - 1$.

3. **For a two-class LDA with shared covariance matrix $\Sigma = I$ (identity matrix)**, the decision boundary equation in terms of the class means $\mu_1$ and $\mu_2$ is:
   $$(\mu_1 - \mu_2)^T x = \frac{\|\mu_1\|^2 - \|\mu_2\|^2}{2}$$

4. **LDA differs from the Perceptron** in that it takes a statistical approach to find the optimal decision boundary based on class distributions and covariance structure, while the Perceptron simply seeks any hyperplane that separates the classes without considering the underlying distributions. 