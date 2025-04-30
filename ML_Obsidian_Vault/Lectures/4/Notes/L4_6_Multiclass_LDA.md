# Multiclass Linear Discriminant Analysis

## Overview
Multiclass Linear Discriminant Analysis (LDA) extends the binary classification capabilities of LDA to problems involving more than two classes. It maintains the core principles of maximizing between-class scatter while minimizing within-class scatter, but adapts the framework to handle multiple decision boundaries simultaneously.

## Multiclass LDA Framework
In multiclass problems with $c$ classes, LDA seeks to find up to $c-1$ discriminant functions that optimally separate the classes. These functions form a projection matrix that maps the original feature space to a lower-dimensional space where class separation is maximized.

## Generalized Eigenvalue Problem
For multiclass LDA, we solve the generalized eigenvalue problem:

$$S_B v = \lambda S_W v$$

Where:
- $S_B$ is the between-class scatter matrix
- $S_W$ is the within-class scatter matrix
- $v$ are the eigenvectors representing the discriminant directions
- $\lambda$ are the corresponding eigenvalues

The solution involves finding eigenvectors corresponding to the largest eigenvalues of $S_W^{-1}S_B$.

## Within-Class Scatter Matrix for Multiple Classes
The within-class scatter matrix is computed in the same way as in binary LDA:

$$S_W = \sum_{i=1}^{c} \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T$$

This represents the pooled covariance of all classes, assuming they share the same covariance structure.

## Between-Class Scatter Matrix for Multiple Classes
The between-class scatter matrix measures the dispersion of class means around the global mean:

$$S_B = \sum_{i=1}^{c} N_i (\mu_i - \mu)(\mu_i - \mu)^T$$

Where:
- $N_i$ is the number of samples in class $i$
- $\mu_i$ is the mean of class $i$
- $\mu$ is the global mean of all samples

## Dimensionality Reduction Properties
Multiclass LDA produces at most $c-1$ non-zero eigenvalues, meaning we can project the data into a space with at most $c-1$ dimensions. This is a significant advantage when dealing with high-dimensional data and multiple classes.

## Implementation Steps
1. Compute the mean vector for each class: $\mu_1, \mu_2, ..., \mu_c$
2. Compute the global mean vector: $\mu$
3. Compute the within-class scatter matrix $S_W$
4. Compute the between-class scatter matrix $S_B$
5. Solve the eigenvalue problem for $S_W^{-1}S_B$
6. Select the top $k \leq c-1$ eigenvectors to form the projection matrix
7. Project the data onto the new subspace
8. Perform classification in the reduced space

## Classification in the Reduced Space
After projection, we can classify new samples using:

1. **Minimum Euclidean Distance**: Assign the sample to the class with the nearest centroid in the projected space
2. **Bayesian Approach**: Compute posterior probabilities using Bayes' rule and assign to the class with highest probability
3. **Mahalanobis Distance**: Account for covariance structure when measuring distance to centroids

## Regularized Multiclass LDA
In high-dimensional settings or with small sample sizes, the within-class scatter matrix may be singular or ill-conditioned. Regularization addresses this:

$$S_W^* = (1-\alpha)S_W + \alpha I$$

Where:
- $\alpha$ is the regularization parameter (0 ≤ α ≤ 1)
- $I$ is the identity matrix

## Comparison with One-vs-All and One-vs-One Approaches
Unlike strategies that decompose multiclass problems into multiple binary problems, multiclass LDA:
- Considers all classes simultaneously
- Creates a unified subspace that optimally separates all classes
- Utilizes information about relationships between all class pairs
- Generally requires fewer computations than one-vs-one approaches

## Special Cases and Extensions
- **Heteroscedastic LDA**: Accounts for different covariance matrices between classes
- **Stepwise LDA**: Performs feature selection during discriminant analysis
- **Penalized LDA**: Adds regularization penalties to handle high-dimensional data
- **Kernel Multiclass LDA**: Handles nonlinear class boundaries through kernel methods

## Practical Considerations
- Pre-processing: Standardize features before applying multiclass LDA
- Singular covariance: Use pseudoinverse or regularization when $S_W$ is singular
- Unbalanced classes: Consider class priors in the classification rule
- Dimensionality: Select an appropriate number of discriminant functions

## Advantages in Multiclass Settings
- Provides explicit dimensionality reduction to $c-1$ dimensions
- Can reveal class relationships in lower-dimensional visualizations
- Often outperforms binary decomposition strategies on problems with natural class structure
- Computational efficiency compared to constructing multiple binary classifiers

## Limitations
- Still assumes Gaussian distributions and homoscedasticity
- Maximum of $c-1$ discriminant functions may be insufficient for complex class boundaries
- Performance degrades when assumptions are violated

## Related Readings
- Chapter 4.3 of "Pattern Recognition and Machine Learning" by Bishop
- Rao, C.R. (1948). "The utilization of multiple measurements in problems of biological classification" 