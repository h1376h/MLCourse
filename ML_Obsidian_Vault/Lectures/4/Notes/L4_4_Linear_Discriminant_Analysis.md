# Linear Discriminant Analysis (LDA)

## Overview
Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction and classification technique that finds a linear combination of features to separate two or more classes. Unlike the Perceptron or Logistic Regression which directly model decision boundaries, LDA takes a generative approach by modeling the distribution of each class and using Bayes' rule for classification.

## Fundamental Concept
LDA aims to find a projection that maximizes the separation between classes while minimizing the variance within each class. This is achieved by maximizing the ratio of between-class scatter to within-class scatter.

## Statistical Framework
LDA makes the following assumptions:
- The data for each class is normally distributed
- All classes share the same covariance matrix (homoscedasticity)
- Features are statistically independent
- Each class has a different mean vector

## Within-Class Scatter Matrix
The within-class scatter matrix $S_W$ measures the spread of data points around their respective class means. It represents the covariance of data points within each class and is calculated as:

$$S_W = \sum_{i=1}^{c} \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T$$

Where:
- $c$ is the number of classes
- $C_i$ is the set of data points in class $i$
- $\mu_i$ is the mean vector of class $i$
- $x$ represents a data point

The within-class scatter matrix can also be expressed as the sum of covariance matrices for each class:

$$S_W = \sum_{i=1}^{c} P(C_i) \Sigma_i$$

Where:
- $P(C_i)$ is the prior probability of class $i$
- $\Sigma_i$ is the covariance matrix of class $i$

## Between-Class Scatter Matrix
The between-class scatter matrix $S_B$ measures the distance between the means of different classes, weighted by the number of samples in each class. It is calculated as:

$$S_B = \sum_{i=1}^{c} N_i (\mu_i - \mu)(\mu_i - \mu)^T$$

Where:
- $N_i$ is the number of samples in class $i$
- $\mu_i$ is the mean vector of class $i$
- $\mu$ is the overall mean vector of all data points

Alternatively, it can be expressed as:

$$S_B = \sum_{i=1}^{c} P(C_i) (\mu_i - \mu)(\mu_i - \mu)^T$$

## Objective Function
LDA aims to find a projection matrix $W$ that maximizes the ratio of between-class scatter to within-class scatter in the projected space. This is formulated as:

$$J(W) = \frac{W^T S_B W}{W^T S_W W}$$

The solution involves finding the eigenvectors of $S_W^{-1}S_B$.

## Computing LDA Projection
To find the optimal projection for binary classification:

1. Compute the mean vectors for each class: $\mu_1$ and $\mu_2$
2. Compute the within-class scatter matrix $S_W$
3. Compute the between-class scatter matrix $S_B$
4. Compute $S_W^{-1}S_B$ and find its eigenvectors
5. Select the eigenvectors corresponding to the largest eigenvalues to form the projection matrix $W$

For multi-class problems, we can have at most $c-1$ discriminant functions, where $c$ is the number of classes.

## Classification with LDA
After projecting the data, classification can be performed by:

1. Computing the posterior probability for each class using Bayes' rule
2. Assigning the sample to the class with the highest posterior probability

For Gaussian class-conditional densities with shared covariance, this results in a linear decision boundary.

## Relationship to Fisher's Discriminant
Fisher's Linear Discriminant is closely related to LDA. For the two-class case, the optimal projection direction $w$ is:

$$w = S_W^{-1}(\mu_1 - \mu_2)$$

This direction maximizes class separation while minimizing within-class variance.

## Advantages of LDA
- Works well with small sample sizes
- Handles multicollinearity well
- Provides dimensionality reduction
- Has good interpretability
- Computationally efficient

## Limitations of LDA
- Assumes Gaussian distribution of data
- Assumes homoscedasticity (equal covariance matrices)
- Sensitive to outliers
- Not suitable for non-linear class boundaries

## Comparison with PCA
While both LDA and PCA are dimensionality reduction techniques:
- PCA finds directions of maximum variance in the data (unsupervised)
- LDA finds directions that maximize class separation (supervised)
- LDA uses class labels, while PCA doesn't

## Extensions and Variants
- Quadratic Discriminant Analysis (QDA): Allows different covariance matrices for each class
- Regularized Discriminant Analysis (RDA): Adds regularization to handle ill-conditioned covariance matrices
- Kernel LDA: Extends LDA to non-linear decision boundaries using the kernel trick

## Implementation Considerations
- Feature standardization is often beneficial
- Regularization can improve performance with high-dimensional data
- Singular value decomposition can handle singularity issues in the covariance matrix

## Related Readings
- Chapter 4.3 of "Pattern Recognition and Machine Learning" by Bishop
- Fisher, R.A. (1936). "The use of multiple measurements in taxonomic problems" 