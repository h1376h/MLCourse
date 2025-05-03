# Question 12: LDA vs. Logistic Regression

## Problem Statement
Linear Discriminant Analysis (LDA) approaches classification from a generative modeling perspective, unlike the discriminative approach of Logistic Regression.

### Task
1. Given the between-class scatter matrix $S_B = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}$ and within-class scatter matrix $S_W = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$, find the direction that maximizes class separation in LDA
2. For binary classification with LDA, if the prior probabilities are equal, where is the decision boundary located relative to the two class means?
3. Compare and contrast how LDA and Logistic Regression would behave with outliers in the training data in one sentence
4. When would you prefer Logistic Regression over LDA? List one specific scenario

## Understanding the Problem
This problem explores the fundamental differences between Linear Discriminant Analysis (LDA) and Logistic Regression, two popular approaches to linear classification. LDA is a generative model that models the class-conditional densities and uses Bayes' rule, while Logistic Regression is a discriminative model that directly models the posterior probability. Understanding these differences is crucial for selecting the appropriate algorithm for specific classification tasks.

## Solution

### Step 1: Finding the LDA Direction for Maximum Class Separation

In LDA, we aim to find a projection direction that maximizes the ratio of between-class scatter to within-class scatter. This is achieved by finding the eigenvectors of $S_W^{-1}S_B$.

Let's first compute the inverse of the within-class scatter matrix $S_W$:

$$S_W = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$

Since $S_W$ is a diagonal matrix, its inverse is simply:

$$S_W^{-1} = \begin{bmatrix} \frac{1}{2} & 0 \\ 0 & \frac{1}{2} \end{bmatrix}$$

Now we compute the product $S_W^{-1}S_B$:

$$S_W^{-1}S_B = \begin{bmatrix} \frac{1}{2} & 0 \\ 0 & \frac{1}{2} \end{bmatrix} \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}$$

Let's multiply these matrices element by element:
- Position (1,1): $\frac{1}{2} \times 4 = 2$
- Position (1,2): $\frac{1}{2} \times 2 = 1$
- Position (2,1): $\frac{1}{2} \times 2 = 1$
- Position (2,2): $\frac{1}{2} \times 1 = 0.5$

So:
$$S_W^{-1}S_B = \begin{bmatrix} 2 & 1 \\ 1 & 0.5 \end{bmatrix}$$

Next, we need to find the eigenvalues and eigenvectors of this matrix. For a 2×2 matrix 
$\begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the characteristic equation is:

$$\det(A - \lambda I) = 0$$
$$\begin{vmatrix} a-\lambda & b \\ c & d-\lambda \end{vmatrix} = 0$$
$$(a-\lambda)(d-\lambda) - bc = 0$$

Substituting our values:
$$\begin{vmatrix} 2-\lambda & 1 \\ 1 & 0.5-\lambda \end{vmatrix} = 0$$
$$(2-\lambda)(0.5-\lambda) - 1 \times 1 = 0$$
$$(2-\lambda)(0.5-\lambda) - 1 = 0$$
$$1 - 0.5\lambda - 2\lambda + \lambda^2 - 1 = 0$$
$$\lambda^2 - 2.5\lambda = 0$$
$$\lambda(\lambda - 2.5) = 0$$

So the eigenvalues are $\lambda_1 = 2.5$ and $\lambda_2 = 0$.

For $\lambda_1 = 2.5$, we find the corresponding eigenvector by solving:
$$(A - \lambda_1 I)v_1 = 0$$

$$\begin{bmatrix} 2-2.5 & 1 \\ 1 & 0.5-2.5 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$\begin{bmatrix} -0.5 & 1 \\ 1 & -2 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

This gives us:
$$-0.5v_{11} + v_{12} = 0$$
$$v_{11} + (-2)v_{12} = 0$$

From the first equation: $v_{12} = 0.5v_{11}$
Substituting into the second equation: $v_{11} + (-2)(0.5v_{11}) = 0$
$v_{11} - v_{11} = 0$

This confirms our eigenvector calculation. Setting $v_{11} = 2$, we get $v_{12} = 1$, so our unnormalized eigenvector is $[2, 1]^T$.

Normalizing this vector:
$$\|[2, 1]^T\| = \sqrt{2^2 + 1^2} = \sqrt{5} \approx 2.236$$

So the normalized eigenvector is:
$$v_1 = \frac{[2, 1]^T}{\sqrt{5}} = [\frac{2}{\sqrt{5}}, \frac{1}{\sqrt{5}}]^T \approx [0.894, 0.447]^T$$

This matches our computed result:
$$v_1 = [0.89442719, 0.4472136]^T$$

The direction that maximizes class separation is therefore $v_1 = [0.89442719, 0.4472136]^T$, which corresponds to the eigenvector with the largest eigenvalue $\lambda_1 = 2.5$.

![LDA Direction](../Images/L4_4_Quiz_12/lda_direction.png)

The visualization shows:
- The blue ellipse represents the between-class scatter matrix $S_B$
- The red ellipse represents the within-class scatter matrix $S_W$
- The green arrow indicates the LDA projection direction that maximizes class separation

### Step 2: LDA Decision Boundary with Equal Prior Probabilities

For binary classification with LDA, when the prior probabilities are equal ($P(C_1) = P(C_2)$), the decision boundary has specific geometric properties derived from Bayes' theorem.

In LDA, the class-conditional densities are modeled as multivariate Gaussians:
$$p(x|C_k) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu_k)^T\Sigma^{-1}(x-\mu_k)\right)$$

where $\mu_k$ is the mean of class $k$ and $\Sigma$ is the shared covariance matrix.

Using Bayes' rule and taking the natural logarithm, we get the discriminant function:
$$\delta_k(x) = \ln(P(C_k)) - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + x^T\Sigma^{-1}\mu_k$$

For the decision boundary between two classes with equal priors ($P(C_1) = P(C_2)$), we have $\delta_1(x) = \delta_2(x)$, which gives us:

$$-\frac{1}{2}\mu_1^T\Sigma^{-1}\mu_1 + x^T\Sigma^{-1}\mu_1 = -\frac{1}{2}\mu_2^T\Sigma^{-1}\mu_2 + x^T\Sigma^{-1}\mu_2$$

Rearranging:
$$x^T\Sigma^{-1}(\mu_1 - \mu_2) = \frac{1}{2}(\mu_1^T\Sigma^{-1}\mu_1 - \mu_2^T\Sigma^{-1}\mu_2)$$

By completing the square, this can be rewritten as:
$$x^T\Sigma^{-1}(\mu_1 - \mu_2) = \frac{1}{2}(\mu_1 + \mu_2)^T\Sigma^{-1}(\mu_1 - \mu_2)$$

Which simplifies to:
$$(x - \frac{\mu_1 + \mu_2}{2})^T\Sigma^{-1}(\mu_1 - \mu_2) = 0$$

This is the equation of a hyperplane that:
1. Has a normal vector in the direction of $\Sigma^{-1}(\mu_1 - \mu_2)$
2. Passes through the point $\frac{\mu_1 + \mu_2}{2}$ (the midpoint between the two means)

Since the normal vector is in the direction of $\Sigma^{-1}(\mu_1 - \mu_2)$, and when $\Sigma$ is proportional to the identity matrix (as in our case where $\Sigma \propto I$), the normal vector is in the same direction as $\mu_1 - \mu_2$. This means the decision boundary is perpendicular to the line connecting the two means.

![LDA Decision Boundary](../Images/L4_4_Quiz_12/lda_decision_boundary.png)

In our visualization:
- Blue points represent Class 1 with mean $\mu_1 = [1, 2]^T$
- Red points represent Class 2 with mean $\mu_2 = [4, 0]^T$
- The green line connects the two means
- The purple dot marks the midpoint at $[2.5, 1]^T$
- The dashed black line shows the decision boundary, which is perpendicular to the line connecting the means and passes through the midpoint

### Step 3: LDA vs. Logistic Regression with Outliers

Let's examine how LDA and Logistic Regression behave in the presence of outliers through the lens of their mathematical formulations.

LDA models the class-conditional densities $p(x|C_k)$ as multivariate Gaussians, and estimates the parameters (mean vectors and covariance matrices) using maximum likelihood:

$$\hat{\mu}_k = \frac{1}{N_k}\sum_{i:y_i=k}x_i$$

$$\hat{\Sigma} = \frac{1}{N}\sum_{k=1}^K\sum_{i:y_i=k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$$

These parameter estimates are sensitive to outliers because they directly incorporate every data point.

In contrast, Logistic Regression directly models the decision boundary by optimizing:

$$\min_w -\sum_{i=1}^n [y_i \log(\sigma(w^T x_i)) + (1-y_i)\log(1-\sigma(w^T x_i))] + \lambda \|w\|^2$$

where $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function.

This optimization focuses on the decision boundary itself, with the loss function treating all points more uniformly in terms of their contribution to the boundary placement.

![LDA vs Logistic Regression with Outliers](../Images/L4_4_Quiz_12/lda_vs_logreg_outliers.png)

LDA is more sensitive to outliers than Logistic Regression because it explicitly models the class distributions (estimating mean and covariance parameters) which can be substantially skewed by extreme values, whereas Logistic Regression directly models the decision boundary with its logistic loss function providing some inherent robustness to extreme observations.

### Step 4: When to Prefer Logistic Regression over LDA

To systematically compare when to prefer Logistic Regression over LDA, let's analyze their theoretical assumptions and practical implications:

**1. Non-Gaussian data distributions:**

LDA assumes that the class-conditional densities follow multivariate Gaussian distributions:
$$p(x|C_k) \sim \mathcal{N}(\mu_k, \Sigma)$$

Logistic Regression makes no such assumption about the data distribution. It directly models:
$$P(C_k|x) = \frac{e^{w_k^T x}}{\sum_{j=1}^K e^{w_j^T x}}$$

For data that follows a non-Gaussian distribution (e.g., exponential, uniform, bimodal), the LDA assumptions are violated, potentially leading to suboptimal decision boundaries.

**2. Different covariance structures:**

LDA typically assumes that all classes share the same covariance matrix $\Sigma$ (unless using Quadratic Discriminant Analysis). This assumption is captured in the decision boundary equation:
$$(x - \frac{\mu_1 + \mu_2}{2})^T\Sigma^{-1}(\mu_1 - \mu_2) = 0$$

When classes have different covariance matrices $\Sigma_1 \neq \Sigma_2$, this assumption is violated.

**3. Presence of outliers:**

As discussed in Step 3, LDA estimates parameters that directly incorporate every data point, making it sensitive to outliers.

**4. Direct probability estimation vs. generative modeling:**

LDA provides class probabilities through Bayes' rule:
$$P(C_k|x) = \frac{P(x|C_k)P(C_k)}{\sum_j P(x|C_j)P(C_j)}$$

Logistic Regression directly estimates $P(C_k|x)$ without modeling the generative process.

When accurate probability calibration is more important than understanding the data generation process, Logistic Regression often provides better calibrated probabilities.

![LDA vs Logistic Regression with Non-Gaussian Data](../Images/L4_4_Quiz_12/lda_vs_logreg_non_gaussian.png)

In summary, Logistic Regression is preferred over LDA when:
1. Data doesn't follow a Gaussian distribution
2. Classes have different covariance structures
3. The dataset contains outliers
4. Direct probability estimation is more important than generative modeling
5. The training set is large enough to compensate for Logistic Regression's potentially higher variance

## Key Insights

### Theoretical Foundations
- LDA is a generative model that estimates class-conditional densities $p(x|C_k)$ and applies Bayes' rule to obtain $P(C_k|x)$
- Logistic Regression is a discriminative model that directly estimates $P(C_k|x)$ without modeling the data generation process
- LDA maximizes the ratio of between-class variance to within-class variance through eigendecomposition of $S_W^{-1}S_B$
- The eigenvector corresponding to the largest eigenvalue of $S_W^{-1}S_B$ gives the direction of maximum class separation

### Geometric Interpretation
- The LDA direction is the eigenvector corresponding to the largest eigenvalue of $S_W^{-1}S_B$
- With equal priors, the LDA decision boundary is perpendicular to the line connecting class means and passes through their midpoint
- The decision boundary equation can be written as $(x - \frac{\mu_1 + \mu_2}{2})^T\Sigma^{-1}(\mu_1 - \mu_2) = 0$
- When $\Sigma \propto I$, the normal vector to the decision boundary is parallel to $(\mu_1 - \mu_2)$

### Practical Considerations
- LDA is typically more data-efficient when its assumptions are met
- Logistic Regression is more robust to outliers and non-Gaussian distributions
- LDA provides insights into the data structure through its estimated parameters
- Logistic Regression often provides better-calibrated probabilities, making it suitable for applications requiring reliable probability estimates

### Mathematical Formulations
- LDA assumes class-conditional densities: $p(x|C_k) \sim \mathcal{N}(\mu_k, \Sigma)$
- Logistic Regression models posterior probabilities: $P(C_k|x) = \frac{e^{w_k^T x}}{\sum_{j=1}^K e^{w_j^T x}}$
- LDA parameter estimation: $\hat{\mu}_k = \frac{1}{N_k}\sum_{i:y_i=k}x_i$ and $\hat{\Sigma} = \frac{1}{N}\sum_{k=1}^K\sum_{i:y_i=k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$
- Logistic Regression optimization: $\min_w -\sum_{i=1}^n [y_i \log(\sigma(w^T x_i)) + (1-y_i)\log(1-\sigma(w^T x_i))] + \lambda \|w\|^2$

## Conclusion
- The direction that maximizes class separation in LDA is the eigenvector of $S_W^{-1}S_B$ with the largest eigenvalue, which in this case is $[0.89442719, 0.4472136]^T$
- For binary LDA with equal prior probabilities, the decision boundary is perpendicular to the line connecting the two class means and passes through their midpoint
- LDA is more sensitive to outliers than Logistic Regression because it models class distributions and estimates parameters directly influenced by all data points
- Logistic Regression is preferred over LDA when data doesn't follow Gaussian distributions, has different class covariance structures, contains outliers, requires accurate probability calibration, or when the training set is sufficiently large
 