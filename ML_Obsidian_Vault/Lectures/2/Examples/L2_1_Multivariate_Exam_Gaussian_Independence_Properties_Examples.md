# Multivariate Gaussian Independence Properties Examples

This document provides examples and key concepts on independence properties in multivariate normal distributions, a fundamental concept in machine learning, statistics, and data analysis.

## Key Concepts and Formulas

In multivariate normal distributions, independence and correlation have specific relationships that are important for understanding data relationships, creating transformations, and developing statistical models.

### Independence and Correlation in Multivariate Normal Distributions

For multivariate normal distributions, the following key properties hold:

1. **Correlation and Independence**: For multivariate normal random variables, zero correlation implies independence (unlike general random variables where zero correlation only implies lack of linear relationship).

2. **Covariance Matrix Interpretation**: If the covariance matrix $\boldsymbol{\Sigma}$ is diagonal, the variables are independent. Off-diagonal elements represent covariances between variables.

3. **Linear Transformations and Independence**: If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$, then $\mathbf{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$.

4. **Conditional Independence**: Variables $X$ and $Y$ are conditionally independent given $Z$ if $P(X, Y | Z) = P(X | Z)P(Y | Z)$.

## Example 1: Independence in Multivariate Normal Variables

### Problem Statement
Let $\mathbf{X} = \begin{bmatrix} X_1 \\ X_2 \\ X_3 \end{bmatrix}$ follow a multivariate normal distribution with mean vector $\boldsymbol{\mu} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and covariance matrix:

$$\boldsymbol{\Sigma} = \begin{bmatrix} 
4 & 0 & 2 \\
0 & 3 & 0 \\
2 & 0 & 6
\end{bmatrix}$$

a) Which pairs of variables are independent? Explain your reasoning.
b) Define $Z = 3X_1 - 6X_3$. Is $Z$ independent of $X_2$? Prove your answer.
c) Are $X_1$ and $X_3$ conditionally independent given $X_2$? Explain.

### Solution

#### Part a: Identifying independent pairs of variables

For multivariate normal distributions, zero covariance implies independence. Looking at the covariance matrix $\boldsymbol{\Sigma}$, we can identify which pairs have zero covariance:

- $\text{Cov}(X_1, X_2) = \boldsymbol{\Sigma}_{12} = 0$, so $X_1$ and $X_2$ are independent.
- $\text{Cov}(X_2, X_3) = \boldsymbol{\Sigma}_{23} = 0$, so $X_2$ and $X_3$ are independent.
- $\text{Cov}(X_1, X_3) = \boldsymbol{\Sigma}_{13} = 2 \neq 0$, so $X_1$ and $X_3$ are not independent.

Therefore, the pairs $(X_1, X_2)$ and $(X_2, X_3)$ are independent.

#### Part b: Determining if $Z$ is independent of $X_2$

We need to find the covariance between $Z = 3X_1 - 6X_3$ and $X_2$.

$$\text{Cov}(Z, X_2) = \text{Cov}(3X_1 - 6X_3, X_2) = 3\text{Cov}(X_1, X_2) - 6\text{Cov}(X_3, X_2)$$

From the covariance matrix:
- $\text{Cov}(X_1, X_2) = 0$
- $\text{Cov}(X_3, X_2) = 0$

Therefore:
$$\text{Cov}(Z, X_2) = 3 \cdot 0 - 6 \cdot 0 = 0$$

Since $Z$ and $X_2$ are jointly normal (as linear combinations of multivariate normal variables are also normal) and have zero covariance, they are independent.

#### Part c: Conditional independence of $X_1$ and $X_3$ given $X_2$

To determine if $X_1$ and $X_3$ are conditionally independent given $X_2$, we need to calculate the conditional covariance matrix. Let's do this step by step:

1. First, we partition the covariance matrix for $(X_1, X_3)$ and $X_2$:

$$\boldsymbol{\Sigma}_{aa} = \begin{bmatrix} 4 & 2 \\ 2 & 6 \end{bmatrix} \text{ (Covariance of } X_1, X_3)$$

$$\boldsymbol{\Sigma}_{ab} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \text{ (Covariance between } (X_1,X_3) \text{ and } X_2)$$

$$\boldsymbol{\Sigma}_{bb} = \begin{bmatrix} 3 \end{bmatrix} \text{ (Variance of } X_2)$$

2. Calculate $\boldsymbol{\Sigma}_{bb}^{-1}$:
$$\boldsymbol{\Sigma}_{bb}^{-1} = \begin{bmatrix} \frac{1}{3} \end{bmatrix}$$

3. Calculate $\boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}$:
$$\begin{bmatrix} 0 \\ 0 \end{bmatrix} \cdot \frac{1}{3} \cdot \begin{bmatrix} 0 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$$

4. Calculate the conditional covariance matrix:
$$\boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba} = \begin{bmatrix} 4 & 2 \\ 2 & 6 \end{bmatrix} - \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 4 & 2 \\ 2 & 6 \end{bmatrix}$$

Since the off-diagonal element in the conditional covariance matrix is 2 (not 0), $X_1$ and $X_3$ are not conditionally independent given $X_2$.

This makes intuitive sense as conditioning on $X_2$ doesn't add any new information about the relationship between $X_1$ and $X_3$ since $X_2$ is already independent of both $X_1$ and $X_3$.

## Example 2: Creating Independent Variables Through Linear Transformations

### Problem Statement
Let $\mathbf{X} = \begin{bmatrix} X_1 \\ X_2 \\ X_3 \end{bmatrix}$ follow a multivariate normal distribution with mean vector $\boldsymbol{\mu} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$ and covariance matrix:

$$\boldsymbol{\Sigma} = \begin{bmatrix} 
4 & 2 & 0 \\
2 & 5 & 1 \\
0 & 1 & 3
\end{bmatrix}$$

a) Which variables or pairs of variables, if any, are independent?
b) Define $Z_1 = X_1 - \frac{1}{2}X_2$ and $Z_2 = X_2 - \frac{1}{5}X_3$. Show whether $Z_1$ and $Z_3=X_3$ are independent.
c) Find a linear transformation of $X_1$ and $X_2$ that creates two independent variables.

### Solution

#### Part a: Identifying independent variables

For multivariate normal distributions, zero covariance implies independence. Looking at the covariance matrix $\boldsymbol{\Sigma}$:

- $\text{Cov}(X_1, X_2) = \boldsymbol{\Sigma}_{12} = 2 \neq 0$, so $X_1$ and $X_2$ are not independent.
- $\text{Cov}(X_2, X_3) = \boldsymbol{\Sigma}_{23} = 1 \neq 0$, so $X_2$ and $X_3$ are not independent.
- $\text{Cov}(X_1, X_3) = \boldsymbol{\Sigma}_{13} = 0$, so $X_1$ and $X_3$ are independent.

Therefore, only $X_1$ and $X_3$ are independent.

#### Part b: Independence of $Z_1$ and $Z_2$

We need to find the covariance between $Z_1 = X_1 - \frac{1}{2}X_2$ and $Z_2 = X_2 - \frac{1}{5}X_3$.

$$\text{Cov}(Z_1, Z_2) = \text{Cov}(X_1 - \frac{1}{2}X_2, X_2 - \frac{1}{5}X_3)$$
$$= \text{Cov}(X_1,X_2) - \frac{1}{5}\text{Cov}(X_1,X_3) - \frac{1}{2}\text{Cov}(X_2,X_2) + \frac{1}{10}\text{Cov}(X_2,X_3)$$
$$= 2 - \frac{1}{5} \cdot 0 - \frac{1}{2} \cdot 5 + \frac{1}{10} \cdot 1$$
$$= 2 - 0 - 2.5 + 0.1 = -0.4$$

Since the covariance is not zero, $Z_1$ and $Z_2$ are not independent.

#### Part c: Finding a linear transformation for independence

To create independent variables from $X_1$ and $X_2$, we use eigendecomposition of their covariance matrix:

1. Extract the covariance matrix for $(X_1, X_2)$:
$$\boldsymbol{\Sigma}_{12} = \begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix}$$

2. Find eigenvalues and eigenvectors:
$$\text{eigenvalues} = \begin{bmatrix} 2.4384 \\ 6.5616 \end{bmatrix}$$

$$\text{eigenvectors} = \begin{bmatrix} -0.7882 & -0.6154 \\ 0.6154 & -0.7882 \end{bmatrix}$$

3. Define transformation matrix $A$ as the transpose of the eigenvector matrix:
$$A = \begin{bmatrix} -0.7882 & 0.6154 \\ -0.6154 & -0.7882 \end{bmatrix}$$

4. Calculate transformed covariance matrix:
$$\text{Transformed covariance} = A\boldsymbol{\Sigma}_{12}A^T = \begin{bmatrix} 2.4384 & 0 \\ 0 & 6.5616 \end{bmatrix}$$

5. Calculate transformed mean:
$$\text{Transformed mean} = A\boldsymbol{\mu} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

The resulting transformed variables $Y_1$ and $Y_2$ are independent because:
1. The transformed covariance matrix is diagonal (off-diagonal elements are zero up to numerical precision)
2. For multivariate normal distributions, uncorrelated variables are independent

Therefore, we can define new independent variables:
$$Y_1 = -0.7882X_1 + 0.6154X_2$$
$$Y_2 = -0.6154X_1 - 0.7882X_2$$

These variables $Y_1$ and $Y_2$ are independent since they are projections onto orthogonal eigenvectors of the covariance matrix. The eigenvalues (2.4384 and 6.5616) represent the variances of $Y_1$ and $Y_2$ respectively.

## Example 3: Independence Properties in Statistical Inference

### Problem Statement
Let $\mathbf{X} = \begin{bmatrix} X_1 \\ X_2 \\ X_3 \end{bmatrix}$ follow a multivariate normal distribution with mean vector $\boldsymbol{\mu} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and covariance matrix:

$$\boldsymbol{\Sigma} = \begin{bmatrix} 
3 & 1 & 2 \\
1 & 4 & 0 \\
2 & 0 & 5
\end{bmatrix}$$

a) Which pairs of variables are independent? Explain your reasoning.
b) Define $Z = 3X_1 - 6X_3$. Is $Z$ independent of $X_2$? Prove your answer.
c) Are $X_1$ and $X_3$ conditionally independent given $X_2$? Explain.

### Solution

#### Part a: Identifying independent pairs of variables

For multivariate normal distributions, zero covariance implies independence. Looking at the covariance matrix $\boldsymbol{\Sigma}$:

- $\text{Cov}(X_1, X_2) = \boldsymbol{\Sigma}_{12} = 1 \neq 0$, so $X_1$ and $X_2$ are not independent.
- $\text{Cov}(X_2, X_3) = \boldsymbol{\Sigma}_{23} = 0$, so $X_2$ and $X_3$ are independent.
- $\text{Cov}(X_1, X_3) = \boldsymbol{\Sigma}_{13} = 2 \neq 0$, so $X_1$ and $X_3$ are not independent.

Therefore, only $X_2$ and $X_3$ are independent.

#### Part b: Independence of $Z$ and $X_2$

We need to find the covariance between $Z = 3X_1 - 6X_3$ and $X_2$.

$$\text{Cov}(Z, X_2) = \text{Cov}(3X_1 - 6X_3, X_2) = 3\text{Cov}(X_1, X_2) - 6\text{Cov}(X_3, X_2)$$

From the covariance matrix:
- $\text{Cov}(X_1, X_2) = 1$
- $\text{Cov}(X_3, X_2) = 0$

Therefore:
$$\text{Cov}(Z, X_2) = 3 \cdot 1 - 6 \cdot 0 = 3$$

Since the covariance is not zero, $Z$ and $X_2$ are not independent.

#### Part c: Conditional independence of $X_1$ and $X_3$ given $X_2$

To determine conditional independence, we calculate the conditional covariance matrix of $(X_1, X_3)$ given $X_2$.

First, we partition the covariance matrix:
$$\boldsymbol{\Sigma}_{aa} = \begin{bmatrix} 3 & 2 \\ 2 & 5 \end{bmatrix}$$
$$\boldsymbol{\Sigma}_{ab} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$
$$\boldsymbol{\Sigma}_{ba} = \begin{bmatrix} 1 & 0 \end{bmatrix}$$
$$\boldsymbol{\Sigma}_{bb} = \begin{bmatrix} 4 \end{bmatrix}$$

The conditional covariance matrix is:
$$\boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}$$

$$\boldsymbol{\Sigma}_{a|b} = \begin{bmatrix} 3 & 2 \\ 2 & 5 \end{bmatrix} - \begin{bmatrix} 1 \\ 0 \end{bmatrix} \cdot \frac{1}{4} \cdot \begin{bmatrix} 1 & 0 \end{bmatrix}$$

$$\boldsymbol{\Sigma}_{a|b} = \begin{bmatrix} 3 & 2 \\ 2 & 5 \end{bmatrix} - \begin{bmatrix} 0.25 & 0 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 2.75 & 2 \\ 2 & 5 \end{bmatrix}$$

Since the off-diagonal element in the conditional covariance matrix is 2 (not 0), $X_1$ and $X_3$ are not conditionally independent given $X_2$.

This means that even after knowing the value of $X_2$, there is still a statistical dependence between $X_1$ and $X_3$. The conditional covariance of 2 indicates a positive relationship between $X_1$ and $X_3$ that persists even after conditioning on $X_2$.

## Example 4: Multivariate Normal with Partitioned Vectors

### Problem Statement

A random vector $\mathbf{X} = \begin{bmatrix} X_1 \\ X_2 \\ X_3 \\ X_4 \\ X_5 \end{bmatrix}$ has a multivariate normal distribution with mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$ given by:

$$\boldsymbol{\mu} = \begin{bmatrix} 4 \\ 45 \\ 30 \\ 35 \\ 40 \end{bmatrix}$$

$$\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 1 & 0 & 0 & 0 \\
1 & 15 & 1 & 4 & 0 \\
0 & 1 & 5 & 4 & 0 \\
0 & 4 & 4 & 8 & 0 \\
0 & 0 & 0 & 0 & 9
\end{bmatrix}$$

Let $\mathbf{X}_1 = \begin{bmatrix} X_2 \\ X_4 \end{bmatrix}$ and $\mathbf{X}_2 = \begin{bmatrix} X_1 \\ X_3 \end{bmatrix}$.

a) Determine the probability density function of $\mathbf{X}_2$.
b) If $\mathbf{Y} = \begin{bmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{bmatrix}$, write down the covariance matrix of $\mathbf{Y}$.
c) Determine the distribution of $\mathbf{X}_1$ conditioned on $\mathbf{X}_2 = \begin{bmatrix} 6 \\ 24 \end{bmatrix}$

### Solution

#### Part a: PDF of $\mathbf{X}_2 = \begin{bmatrix} X_1 \\ X_3 \end{bmatrix}$

For a multivariate normal distribution, we need to find the marginal mean vector and covariance matrix for $\mathbf{X}_2$:

Mean vector:
$$\boldsymbol{\mu}_2 = \begin{bmatrix} 4 \\ 30 \end{bmatrix}$$

Covariance matrix:
$$\boldsymbol{\Sigma}_2 = \begin{bmatrix} 1 & 0 \\ 0 & 5 \end{bmatrix}$$

To find the PDF, we need the determinant and inverse of $\boldsymbol{\Sigma}_2$:

1. Determinant calculation:
   For 2×2 matrix:
   $$\begin{vmatrix} 1 & 0 \\ 0 & 5 \end{vmatrix} = (1)(5) - (0)(0) = 5$$

2. Inverse calculation:
   For 2×2 matrix:
   - Calculate determinant = 5
   - Form adjugate matrix:
     $$\text{adj}(\boldsymbol{\Sigma}_2) = \begin{bmatrix} 5 & 0 \\ 0 & 1 \end{bmatrix}$$
   - Multiply by 1/determinant:
     $$\boldsymbol{\Sigma}_2^{-1} = \begin{bmatrix} 1 & 0 \\ 0 & 0.2 \end{bmatrix}$$

The probability density function is:
$$f_{\mathbf{X}_2}(\mathbf{x}_2) = \frac{1}{2\pi \sqrt{|\boldsymbol{\Sigma}_2|}} \exp\left(-\frac{1}{2}(\mathbf{x}_2 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}_2^{-1} (\mathbf{x}_2 - \boldsymbol{\mu}_2)\right)$$

#### Part b: Covariance Matrix of $\mathbf{Y}$

For $\mathbf{Y} = \begin{bmatrix} X_2 \\ X_4 \\ X_5 \\ X_1 \\ X_3 \end{bmatrix}$, we need to reorder the mean vector and covariance matrix:

Mean vector:
$$\boldsymbol{\mu}_Y = \begin{bmatrix} 45 \\ 35 \\ 40 \\ 4 \\ 30 \end{bmatrix}$$

Covariance matrix:
$$\boldsymbol{\Sigma}_Y = \begin{bmatrix}
15 & 4 & 0 & 1 & 1 \\
4 & 8 & 0 & 0 & 4 \\
0 & 0 & 9 & 0 & 0 \\
1 & 0 & 0 & 1 & 0 \\
1 & 4 & 0 & 0 & 5
\end{bmatrix}$$

#### Part c: Conditional Distribution of $\mathbf{X}_1$ given $\mathbf{X}_2$

To find the conditional distribution, we need to calculate:

1. Calculate $\boldsymbol{\Sigma}_{22}^{-1}$:
   For 2×2 matrix:
   - Calculate determinant = 5
   - Form adjugate matrix:
     $$\text{adj}(\boldsymbol{\Sigma}_{22}) = \begin{bmatrix} 5 & 0 \\ 0 & 1 \end{bmatrix}$$
   - Multiply by 1/determinant:
     $$\boldsymbol{\Sigma}_{22}^{-1} = \begin{bmatrix} 1 & 0 \\ 0 & 0.2 \end{bmatrix}$$

2. Calculate $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}$:
   $$\begin{bmatrix} 1 & 1 \\ 0 & 4 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 0.2 \end{bmatrix} = \begin{bmatrix} 1 & 0.2 \\ 0 & 0.8 \end{bmatrix}$$

3. Calculate conditional mean $\boldsymbol{\mu}_{1|2}$:
   $$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$
   $$= \begin{bmatrix} 45 \\ 35 \end{bmatrix} + \begin{bmatrix} 1 & 0.2 \\ 0 & 0.8 \end{bmatrix} \begin{bmatrix} 2 \\ -6 \end{bmatrix}$$
   $$= \begin{bmatrix} 45.8 \\ 30.2 \end{bmatrix}$$

4. Calculate conditional covariance $\boldsymbol{\Sigma}_{1|2}$:
   $$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$
   $$= \begin{bmatrix} 15 & 4 \\ 4 & 8 \end{bmatrix} - \begin{bmatrix} 1 & 0.2 \\ 0 & 0.8 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 4 \end{bmatrix}$$
   $$= \begin{bmatrix} 13.8 & 3.2 \\ 3.2 & 4.8 \end{bmatrix}$$

Therefore, the conditional distribution is:
$$\mathbf{X}_1|\mathbf{X}_2 = \begin{bmatrix} 6 \\ 24 \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} 45.8 \\ 30.2 \end{bmatrix}, \begin{bmatrix} 13.8 & 3.2 \\ 3.2 & 4.8 \end{bmatrix}\right)$$

## Example 5: Independent Variables with Inverse of Covariance Matrix

### Problem Statement

Assume we have the following three dimensional normal random variable

$$\mathbf{X} = \begin{bmatrix} X_1 \\ X_2 \\ X_3 \end{bmatrix} \sim \mathcal{N} \left( \begin{bmatrix} 0 \\ 1 \\ -2 \end{bmatrix}, \begin{bmatrix} 4 & 1 & -1 \\ 1 & 1 & 0 \\ -1 & 0 & 1 \end{bmatrix} \right).$$

We need to find the inverse of the covariance matrix $\boldsymbol{\Sigma}^{-1}$ and verify various independence properties.

### Solution

#### Part 1: Finding the Inverse of the Covariance Matrix

To find $\boldsymbol{\Sigma}^{-1}$, we follow these steps:

1. Calculate the cofactor matrix:
   For each element $(i,j)$, we calculate the cofactor $C_{ij} = (-1)^{i+j}M_{ij}$ where $M_{ij}$ is the minor (determinant of the 2×2 matrix after removing row $i$ and column $j$).

   For example, for position (1,1):
   $$M_{11} = \begin{vmatrix} 1 & 0 \\ 0 & 1 \end{vmatrix} = 1$$
   $$C_{11} = 1$$

   Complete cofactor matrix:
   $$\begin{bmatrix} 
   1 & -1 & 1 \\
   -1 & 3 & -1 \\
   1 & -1 & 3
   \end{bmatrix}$$

2. Calculate determinant using first row expansion:
   $$|\boldsymbol{\Sigma}| = 4(1) + 1(-1) + (-1)(1) = 2$$

3. Calculate adjugate matrix (transpose of cofactor matrix):
   $$\text{adj}(\boldsymbol{\Sigma}) = \begin{bmatrix} 
   1 & -1 & 1 \\
   -1 & 3 & -1 \\
   1 & -1 & 3
   \end{bmatrix}$$

4. Divide by determinant to get inverse:
   $$\boldsymbol{\Sigma}^{-1} = \frac{1}{2}\begin{bmatrix} 
   1 & -1 & 1 \\
   -1 & 3 & -1 \\
   1 & -1 & 3
   \end{bmatrix} = \begin{bmatrix} 
   0.5 & -0.5 & 0.5 \\
   -0.5 & 1.5 & -0.5 \\
   0.5 & -0.5 & 1.5
   \end{bmatrix}$$

Verification: $\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{-1} = \mathbf{I}$

#### Part 2: Independence Between Pairs

For multivariate normal distributions, zero covariance implies independence. Looking at the covariance matrix $\boldsymbol{\Sigma}$:

1. For $(X_1,X_2)$:
   $$\text{Cov}(X_1,X_2) = \boldsymbol{\Sigma}_{12} = 1 \neq 0 \implies \text{not independent}$$

2. For $(X_2,X_3)$:
   $$\text{Cov}(X_2,X_3) = \boldsymbol{\Sigma}_{23} = 0 \implies \text{independent}$$

3. For $(X_1,X_3)$:
   $$\text{Cov}(X_1,X_3) = \boldsymbol{\Sigma}_{13} = -1 \neq 0 \implies \text{not independent}$$

Therefore, only $X_2$ and $X_3$ are independent.

#### Part 3: Finding Values of $a$ and $b$ for Independence

For $Z = X_1 - aX_2 - bX_3$ to be independent of $X_1$, we need $\text{Cov}(Z,X_1) = 0$.

Step-by-step calculation:
1. Express covariance:
   $$\text{Cov}(Z,X_1) = \text{Cov}(X_1 - aX_2 - bX_3, X_1)$$
   $$= \text{Var}(X_1) - a\text{Cov}(X_2,X_1) - b\text{Cov}(X_3,X_1)$$

2. Substitute values:
   $$4 - a(1) - b(-1) = 0$$
   $$4 - a + b = 0$$

3. Solve for $a$:
   $$a = 4 + b$$

Any pair $(a,b)$ satisfying this equation will make $Z$ and $X_1$ independent. For example:
- If $b = 0$, then $a = 4$
- If $b = 1$, then $a = 5$
- If $b = -1$, then $a = 3$

#### Part 4: Conditional Independence Given $X_3$

To determine if $Z$ and $X_1$ can be conditionally independent given $X_3$, we:

1. Extract relevant submatrices:
   $$\boldsymbol{\Sigma}_{11} = \begin{bmatrix} 4 & 1 \\ 1 & 1 \end{bmatrix} \text{ (covariance of } X_1,X_2)$$
   
   $$\boldsymbol{\Sigma}_{12} = \begin{bmatrix} -1 \\ 0 \end{bmatrix} \text{ (covariance with } X_3)$$
   
   $$\boldsymbol{\Sigma}_{22} = [1] \text{ (variance of } X_3)$$

2. Calculate conditional covariance matrix:
   $$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$
   
   $$= \begin{bmatrix} 4 & 1 \\ 1 & 1 \end{bmatrix} - \begin{bmatrix} -1 \\ 0 \end{bmatrix} [1] \begin{bmatrix} -1 & 0 \end{bmatrix}$$
   
   $$= \begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix}$$

3. For conditional independence, we need:
   $$\text{Cov}(Z,X_1|X_3) = 0$$
   $$\text{Cov}(X_1 - aX_2,X_1|X_3) = 0$$
   $$3 - a(1) = 0$$
   $$a = 3$$

Therefore, to achieve conditional independence of $Z$ and $X_1$ given $X_3$, we need $a = 3$ and $b$ can be any value (since we're conditioning on $X_3$). For instance, $(a,b) = (3,0)$ would give $Z = X_1 - 3X_2$ which is independent of $X_1$ conditional on $X_3$.

The geometric interpretation is that conditioning on $X_3$ changes the correlation structure between $X_1$ and $X_2$, and with $a = 3$, we find a linear combination that eliminates this conditional correlation.

## Running the Examples

You can run the code that generates these examples and visualizations using:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Codes/1_multivariate_gaussian_independence.py
```

## Related Topics

- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: Detailed examples of multivariate normal distributions
- [[L2_1_Multivariate_Exam_Linear_Transformation_Examples|Linear Transformation Examples]]: Examples of applying linear transformations to multivariate distributions
- [[L2_1_Multivariate_Exam_Conditional_Distribution_Examples|Conditional Distribution Examples]]: Problems on conditional distributions, inference, and predictions
- [[L2_1_Independence_Examples|Independence Examples]]: General examples of independence in probability theory
- [[L2_1_Multivariate_Exam_Eigendecomposition_Examples|Eigendecomposition Examples]]: Examples on eigenvalue decomposition related to independence
- [[L2_1_Mahalanobis_Distance|Mahalanobis Distance]]: Applications of independence concepts in distance metrics 