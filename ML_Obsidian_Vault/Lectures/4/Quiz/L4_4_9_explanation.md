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

### Step 2: Calculate the LDA projection direction
For the problem with means $\mu_1 = [1, 2]^T$ and $\mu_2 = [3, 0]^T$ and shared covariance matrix $\Sigma = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$, we need to find the LDA projection direction $w$.

The formula for the LDA projection direction is:
$$w = \Sigma^{-1}(\mu_1 - \mu_2)$$

Step-by-step calculation:

1. Calculate the inverse of the covariance matrix:
   $$\Sigma^{-1} = \begin{bmatrix} 0.5 & 0 \\ 0 & 1 \end{bmatrix}$$

2. Calculate the difference between class means:
   $$\mu_1 - \mu_2 = [1, 2]^T - [3, 0]^T = [-2, 2]^T$$

3. Calculate the projection vector:
   $$w = \Sigma^{-1}(\mu_1 - \mu_2) = \begin{bmatrix} 0.5 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} -2 \\ 2 \end{bmatrix} = \begin{bmatrix} -1 \\ 2 \end{bmatrix}$$

![LDA Projection](../Images/L4_4_Quiz_9/lda_projection.png)

### Step 3: Determine the threshold for classification
For LDA with equal prior probabilities, the threshold is the midpoint of the projected class means.

1. Project the class means onto the direction w:
   $$w^T\mu_1 = [-1, 2] \cdot [1, 2]^T = -1 + 4 = 3$$
   $$w^T\mu_2 = [-1, 2] \cdot [3, 0]^T = -3 + 0 = -3$$

2. Calculate the threshold as the midpoint:
   $$\text{threshold} = \frac{w^T\mu_1 + w^T\mu_2}{2} = \frac{3 + (-3)}{2} = 0$$

3. The decision rule becomes:
   - If $w^T x > 0$, classify as Class 1
   - If $w^T x < 0$, classify as Class 2
   
The point where the posterior probabilities are equal ($P(C_1|x) = P(C_2|x) = 0.5$) is precisely at this decision boundary, where $w^T x = 0$.

### Step 4: Classify new data points
Let's classify two new points using our LDA model:

**Point 1**: $x_1 = [2, 1]^T$
- Projection onto w: $w^T x_1 = [-1, 2] \cdot [2, 1]^T = -2 + 2 = 0$
- Since $w^T x_1 = 0$ (exactly at the threshold), we classify this as Class 2 (by convention)

**Point 2**: $x_2 = [0, 3]^T$
- Projection onto w: $w^T x_2 = [-1, 2] \cdot [0, 3]^T = 0 + 6 = 6$
- Since $w^T x_2 > 0$, we classify this as Class 1

![LDA Classification](../Images/L4_4_Quiz_9/lda_classification.png)

### Step 5: Derive the generic LDA decision boundary equation
For a two-class LDA with covariance matrix $\Sigma = I$ (identity matrix), the decision boundary equation simplifies significantly:

$$w = \Sigma^{-1}(\mu_1 - \mu_2) = I \cdot (\mu_1 - \mu_2) = \mu_1 - \mu_2$$

The decision boundary occurs where $w^T x = \text{threshold}$. With equal priors, the threshold is:
$$\text{threshold} = \frac{w^T\mu_1 + w^T\mu_2}{2} = \frac{(\mu_1 - \mu_2)^T\mu_1 + (\mu_1 - \mu_2)^T\mu_2}{2}$$

Simplifying:
$$\text{threshold} = \frac{(\mu_1 - \mu_2)^T\mu_1 + (\mu_1 - \mu_2)^T\mu_2}{2} = \frac{\mu_1^T\mu_1 - \mu_2^T\mu_1 + \mu_1^T\mu_2 - \mu_2^T\mu_2}{2}$$

Since $\mu_1^T\mu_2 = \mu_2^T\mu_1$, this reduces to:
$$\text{threshold} = \frac{\mu_1^T\mu_1 - \mu_2^T\mu_2}{2}$$

Therefore, the decision boundary equation is:
$$(\mu_1 - \mu_2)^T x = \frac{\mu_1^T\mu_1 - \mu_2^T\mu_2}{2}$$

This can be rewritten as:
$$x^T(\mu_1 - \mu_2) = \frac{||\mu_1||^2 - ||\mu_2||^2}{2}$$

### Step 6: Compare LDA with Perceptron
Let's visualize the difference between LDA and Perceptron decision boundaries:

![LDA vs Perceptron](../Images/L4_4_Quiz_9/lda_vs_perceptron.png)

LDA differs from the Perceptron in several important ways:

1. **Approach**:
   - LDA takes a probabilistic approach based on modeling class distributions
   - Perceptron uses an iterative, error-driven approach

2. **Objective**:
   - LDA finds the optimal boundary that maximizes the separation between classes in terms of means relative to the shared covariance
   - Perceptron simply tries to find any hyperplane that separates the classes

3. **Statistical properties**:
   - LDA is statistically optimal when its assumptions are met
   - Perceptron makes no distributional assumptions

4. **Posterior probabilities**:
   - LDA provides posterior class probabilities
   - Standard Perceptron only provides class assignments

![LDA Posterior Probabilities](../Images/L4_4_Quiz_9/lda_posterior.png)

## Key Insights

### Theoretical Framework
- LDA is derived from a generative modeling approach, modeling class-conditional densities and using Bayes' rule
- When its assumptions are met, LDA provides the optimal decision boundary in the Bayes sense
- The projection direction $w = \Sigma^{-1}(\mu_1 - \mu_2)$ maximizes class separation relative to the shared covariance

### Practical Considerations
- LDA performs well when the class distributions are approximately Gaussian with equal covariance matrices
- The direction $w$ is influenced by both the class means and the covariance structure
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

2. For the given means $\mu_1 = [1, 2]^T$ and $\mu_2 = [3, 0]^T$ with shared covariance matrix $\Sigma = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$, the decision boundary (where posterior probabilities equal 0.5) is given by the equation $-x_1 + 2x_2 = 0$.

3. For a two-class LDA with $\Sigma = I$, the decision boundary equation is:
   $$(\mu_1 - \mu_2)^T x = \frac{\mu_1^T\mu_1 - \mu_2^T\mu_2}{2}$$

4. LDA differs from the Perceptron in that it takes a statistical approach to find the optimal decision boundary based on class distributions, while the Perceptron simply seeks any hyperplane that separates the classes without considering the underlying distributions. 