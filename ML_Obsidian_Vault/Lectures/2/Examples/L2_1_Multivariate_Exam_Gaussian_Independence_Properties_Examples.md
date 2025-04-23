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

From the covariance matrix, we know:
- $\text{Cov}(X_1, X_2) = 0$
- $\text{Cov}(X_3, X_2) = 0$

Therefore:
$$\text{Cov}(Z, X_2) = 3 \cdot 0 - 6 \cdot 0 = 0$$

Since $Z$ and $X_2$ are jointly normal (as linear combinations of multivariate normal variables are also normal) and have zero covariance, they are independent.

#### Part c: Conditional independence of $X_1$ and $X_3$ given $X_2$

To determine if $X_1$ and $X_3$ are conditionally independent given $X_2$, we need to look at the conditional covariance matrix of $(X_1, X_3)$ given $X_2$.

For a multivariate normal distribution partitioned as $\mathbf{X} = (\mathbf{X}_a, \mathbf{X}_b)$ with covariance matrix partitioned accordingly:

$$\boldsymbol{\Sigma} = \begin{bmatrix}
\boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab} \\
\boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb}
\end{bmatrix}$$

The conditional covariance matrix is:
$$\boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}$$

In our case, $\mathbf{X}_a = (X_1, X_3)$ and $\mathbf{X}_b = X_2$, so:

$$\boldsymbol{\Sigma}_{aa} = \begin{bmatrix} 4 & 2 \\ 2 & 6 \end{bmatrix}$$
$$\boldsymbol{\Sigma}_{ab} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
$$\boldsymbol{\Sigma}_{ba} = \begin{bmatrix} 0 & 0 \end{bmatrix}$$
$$\boldsymbol{\Sigma}_{bb} = \begin{bmatrix} 3 \end{bmatrix}$$

Therefore:
$$\boldsymbol{\Sigma}_{a|b} = \begin{bmatrix} 4 & 2 \\ 2 & 6 \end{bmatrix} - \begin{bmatrix} 0 \\ 0 \end{bmatrix} \cdot \frac{1}{3} \cdot \begin{bmatrix} 0 & 0 \end{bmatrix} = \begin{bmatrix} 4 & 2 \\ 2 & 6 \end{bmatrix}$$

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

Examining the covariance matrix $\boldsymbol{\Sigma}$:
- $\text{Cov}(X_1, X_2) = \boldsymbol{\Sigma}_{12} = 2 \neq 0$, so $X_1$ and $X_2$ are not independent.
- $\text{Cov}(X_2, X_3) = \boldsymbol{\Sigma}_{23} = 1 \neq 0$, so $X_2$ and $X_3$ are not independent.
- $\text{Cov}(X_1, X_3) = \boldsymbol{\Sigma}_{13} = 0$, so $X_1$ and $X_3$ are independent.

Therefore, only $X_1$ and $X_3$ are independent.

#### Part b: Independence of $Z_1$ and $Z_3$

We need to find the covariance between $Z_1 = X_1 - \frac{1}{2}X_2$ and $Z_3 = X_3$.

$$\text{Cov}(Z_1, Z_3) = \text{Cov}(X_1 - \frac{1}{2}X_2, X_3) = \text{Cov}(X_1, X_3) - \frac{1}{2}\text{Cov}(X_2, X_3)$$

From the covariance matrix:
- $\text{Cov}(X_1, X_3) = 0$
- $\text{Cov}(X_2, X_3) = 1$

Therefore:
$$\text{Cov}(Z_1, Z_3) = 0 - \frac{1}{2} \cdot 1 = -\frac{1}{2} \neq 0$$

Since the covariance is not zero, $Z_1$ and $Z_3$ are not independent.

#### Part c: Linear transformation for independence

To create independent variables from $X_1$ and $X_2$, we can use eigendecomposition of the covariance matrix of $(X_1, X_2)$:

$$\boldsymbol{\Sigma}_{12} = \begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix}$$

Finding the eigenvalues by solving $|\boldsymbol{\Sigma}_{12} - \lambda \mathbf{I}| = 0$:

$$\begin{vmatrix} 4 - \lambda & 2 \\ 2 & 5 - \lambda \end{vmatrix} = 0$$

$$(4 - \lambda)(5 - \lambda) - 2 \cdot 2 = 0$$

$$\lambda^2 - 9\lambda + 20 - 4 = 0$$

$$\lambda^2 - 9\lambda + 16 = 0$$

Using the quadratic formula:
$$\lambda = \frac{9 \pm \sqrt{81 - 64}}{2} = \frac{9 \pm \sqrt{17}}{2}$$

$$\lambda_1 \approx 6.56, \lambda_2 \approx 2.44$$

The corresponding eigenvectors (after normalization) are:
$$\mathbf{v}_1 \approx \begin{bmatrix} 0.525 \\ 0.851 \end{bmatrix}$$
$$\mathbf{v}_2 \approx \begin{bmatrix} 0.851 \\ -0.525 \end{bmatrix}$$

We can define new variables:
$$Y_1 = 0.525X_1 + 0.851X_2$$
$$Y_2 = 0.851X_1 - 0.525X_2$$

These variables $Y_1$ and $Y_2$ will be independent since they are projections onto orthogonal eigenvectors of the covariance matrix.

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

Therefore, only the pair $(X_2, X_3)$ is independent.

#### Part b: Determining if $Z$ is independent of $X_2$

We need to find the covariance between $Z = 3X_1 - 6X_3$ and $X_2$.

$$\text{Cov}(Z, X_2) = \text{Cov}(3X_1 - 6X_3, X_2) = 3\text{Cov}(X_1, X_2) - 6\text{Cov}(X_3, X_2)$$

From the covariance matrix:
- $\text{Cov}(X_1, X_2) = 1$
- $\text{Cov}(X_3, X_2) = 0$

Therefore:
$$\text{Cov}(Z, X_2) = 3 \cdot 1 - 6 \cdot 0 = 3 \neq 0$$

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

Let $\mathbf{X}_1 = \begin{bmatrix} X_2 \\ X_4 \\ X_5 \end{bmatrix}$ and $\mathbf{X}_2 = \begin{bmatrix} X_1 \\ X_3 \end{bmatrix}$.

a) Determine the probability density function of $\mathbf{X}_2$.
b) If $\mathbf{Y} = \begin{bmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{bmatrix}$, write down the covariance matrix of $\mathbf{Y}$.
c) Determine the distribution of $\mathbf{X}_1$ conditioned on $\mathbf{X}_2 = \begin{bmatrix} 6 \\ 24 \end{bmatrix}$

### Solution

#### Part a: PDF of $\mathbf{X}_2 = \begin{bmatrix} X_1 \\ X_3 \end{bmatrix}$

For a multivariate normal distribution, the marginal distribution of any subset of variables is also multivariate normal, with the corresponding subvector of the mean and the appropriate submatrix of the covariance matrix.

The mean vector of $\mathbf{X}_2$ is:

$$\boldsymbol{\mu}_{\mathbf{X}_2} = \begin{bmatrix} 4 \\ 30 \end{bmatrix}$$

The covariance matrix of $\mathbf{X}_2$ is the submatrix of $\boldsymbol{\Sigma}$ corresponding to variables $X_1$ and $X_3$:

$$\boldsymbol{\Sigma}_{\mathbf{X}_2} = \begin{bmatrix}
1 & 0 \\
0 & 5
\end{bmatrix}$$

Now we can write the probability density function of $\mathbf{X}_2$:

$$f_{\mathbf{X}_2}(\mathbf{x}_2) = \frac{1}{2\pi \sqrt{|\boldsymbol{\Sigma}_{\mathbf{X}_2}|}} \exp\left(-\frac{1}{2}(\mathbf{x}_2 - \boldsymbol{\mu}_{\mathbf{X}_2})^T \boldsymbol{\Sigma}_{\mathbf{X}_2}^{-1} (\mathbf{x}_2 - \boldsymbol{\mu}_{\mathbf{X}_2})\right)$$

Where:
- $|\boldsymbol{\Sigma}_{\mathbf{X}_2}| = 1 \cdot 5 = 5$
- $\boldsymbol{\Sigma}_{\mathbf{X}_2}^{-1} = \begin{bmatrix} 1 & 0 \\ 0 & \frac{1}{5} \end{bmatrix}$

Therefore:

$$f_{\mathbf{X}_2}(\mathbf{x}_2) = \frac{1}{2\pi \sqrt{5}} \exp\left(-\frac{1}{2}\left[(x_1 - 4)^2 + \frac{(x_3 - 30)^2}{5}\right]\right)$$

#### Part b: Covariance matrix of $\mathbf{Y} = \begin{bmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{bmatrix}$

To find the covariance matrix of $\mathbf{Y}$, we need to reorder the covariance matrix $\boldsymbol{\Sigma}$ according to the new arrangement of variables.

$\mathbf{Y} = \begin{bmatrix} X_2 \\ X_4 \\ X_5 \\ X_1 \\ X_3 \end{bmatrix}$, so we need to rearrange the rows and columns of $\boldsymbol{\Sigma}$ to match this order.

$$\text{Cov}(\mathbf{Y}) = \begin{bmatrix}
\text{Cov}(\mathbf{X}_1, \mathbf{X}_1) & \text{Cov}(\mathbf{X}_1, \mathbf{X}_2) \\
\text{Cov}(\mathbf{X}_2, \mathbf{X}_1) & \text{Cov}(\mathbf{X}_2, \mathbf{X}_2)
\end{bmatrix}$$

Where:

$$\text{Cov}(\mathbf{X}_1, \mathbf{X}_1) = \begin{bmatrix}
15 & 4 & 0 \\
4 & 8 & 0 \\
0 & 0 & 9
\end{bmatrix}$$

$$\text{Cov}(\mathbf{X}_1, \mathbf{X}_2) = \begin{bmatrix}
1 & 1 \\
0 & 4 \\
0 & 0
\end{bmatrix}$$

$$\text{Cov}(\mathbf{X}_2, \mathbf{X}_1) = \begin{bmatrix}
1 & 0 & 0 \\
1 & 4 & 0
\end{bmatrix}$$

$$\text{Cov}(\mathbf{X}_2, \mathbf{X}_2) = \begin{bmatrix}
1 & 0 \\
0 & 5
\end{bmatrix}$$

Therefore, the covariance matrix of $\mathbf{Y}$ is:

$$\text{Cov}(\mathbf{Y}) = \begin{bmatrix}
15 & 4 & 0 & 1 & 1 \\
4 & 8 & 0 & 0 & 4 \\
0 & 0 & 9 & 0 & 0 \\
1 & 0 & 0 & 1 & 0 \\
1 & 4 & 0 & 0 & 5
\end{bmatrix}$$

#### Part c: Conditional distribution of $\mathbf{X}_1$ given $\mathbf{X}_2 = \begin{bmatrix} 6 \\ 24 \end{bmatrix}$

For multivariate normal distributions, the conditional distribution of $\mathbf{X}_1$ given $\mathbf{X}_2 = \mathbf{x}_2$ is also multivariate normal with:

- Conditional mean: $\boldsymbol{\mu}_{\mathbf{X}_1|\mathbf{X}_2} = \boldsymbol{\mu}_{\mathbf{X}_1} + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_{\mathbf{X}_2})$
- Conditional covariance: $\boldsymbol{\Sigma}_{\mathbf{X}_1|\mathbf{X}_2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$

Where:
- $\boldsymbol{\mu}_{\mathbf{X}_1} = \begin{bmatrix} 45 \\ 35 \\ 40 \end{bmatrix}$
- $\mathbf{x}_2 = \begin{bmatrix} 6 \\ 24 \end{bmatrix}$
- $\boldsymbol{\mu}_{\mathbf{X}_2} = \begin{bmatrix} 4 \\ 30 \end{bmatrix}$

$$\boldsymbol{\Sigma}_{11} = \text{Cov}(\mathbf{X}_1, \mathbf{X}_1) = \begin{bmatrix}
15 & 4 & 0 \\
4 & 8 & 0 \\
0 & 0 & 9
\end{bmatrix}$$

$$\boldsymbol{\Sigma}_{12} = \text{Cov}(\mathbf{X}_1, \mathbf{X}_2) = \begin{bmatrix}
1 & 1 \\
0 & 4 \\
0 & 0
\end{bmatrix}$$

$$\boldsymbol{\Sigma}_{21} = \text{Cov}(\mathbf{X}_2, \mathbf{X}_1) = \begin{bmatrix}
1 & 0 & 0 \\
1 & 4 & 0
\end{bmatrix}$$

$$\boldsymbol{\Sigma}_{22} = \text{Cov}(\mathbf{X}_2, \mathbf{X}_2) = \begin{bmatrix}
1 & 0 \\
0 & 5
\end{bmatrix}$$

$$\boldsymbol{\Sigma}_{22}^{-1} = \begin{bmatrix}
1 & 0 \\
0 & \frac{1}{5}
\end{bmatrix}$$

First, let's calculate:
$$\mathbf{x}_2 - \boldsymbol{\mu}_{\mathbf{X}_2} = \begin{bmatrix} 6 \\ 24 \end{bmatrix} - \begin{bmatrix} 4 \\ 30 \end{bmatrix} = \begin{bmatrix} 2 \\ -6 \end{bmatrix}$$

Now, calculate $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_{\mathbf{X}_2})$:

$$\begin{bmatrix}
1 & 1 \\
0 & 4 \\
0 & 0
\end{bmatrix} \begin{bmatrix}
1 & 0 \\
0 & \frac{1}{5}
\end{bmatrix} \begin{bmatrix}
2 \\
-6
\end{bmatrix}$$

$$= \begin{bmatrix}
1 & \frac{1}{5} \\
0 & \frac{4}{5} \\
0 & 0
\end{bmatrix} \begin{bmatrix}
2 \\
-6
\end{bmatrix}$$

$$= \begin{bmatrix}
2 - \frac{6}{5} \\
-\frac{24}{5} \\
0
\end{bmatrix} = \begin{bmatrix}
\frac{4}{5} \\
-\frac{24}{5} \\
0
\end{bmatrix}$$

Therefore, the conditional mean is:

$$\boldsymbol{\mu}_{\mathbf{X}_1|\mathbf{X}_2} = \begin{bmatrix}
45 \\
35 \\
40
\end{bmatrix} + \begin{bmatrix}
\frac{4}{5} \\
-\frac{24}{5} \\
0
\end{bmatrix}$$

$$= \begin{bmatrix}
45 + \frac{4}{5} \\
35 - \frac{24}{5} \\
40
\end{bmatrix} = \begin{bmatrix}
\frac{229}{5} \\
\frac{151}{5} \\
40
\end{bmatrix}$$

Next, calculate the conditional covariance matrix:

$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = \begin{bmatrix}
1 & 1 \\
0 & 4 \\
0 & 0
\end{bmatrix} \begin{bmatrix}
1 & 0 \\
0 & \frac{1}{5}
\end{bmatrix} \begin{bmatrix}
1 & 0 & 0 \\
1 & 4 & 0
\end{bmatrix}$$

$$= \begin{bmatrix}
1 + \frac{1}{5} & 0 + \frac{4}{5} & 0 \\
0 + \frac{4}{5} & 0 + \frac{16}{5} & 0 \\
0 & 0 & 0
\end{bmatrix} = \begin{bmatrix}
\frac{6}{5} & \frac{4}{5} & 0 \\
\frac{4}{5} & \frac{16}{5} & 0 \\
0 & 0 & 0
\end{bmatrix}$$

Therefore, the conditional covariance matrix is:

$$\boldsymbol{\Sigma}_{\mathbf{X}_1|\mathbf{X}_2} = \begin{bmatrix}
15 & 4 & 0 \\
4 & 8 & 0 \\
0 & 0 & 9
\end{bmatrix} - \begin{bmatrix}
\frac{6}{5} & \frac{4}{5} & 0 \\
\frac{4}{5} & \frac{16}{5} & 0 \\
0 & 0 & 0
\end{bmatrix}$$

$$= \begin{bmatrix}
15 - \frac{6}{5} & 4 - \frac{4}{5} & 0 \\
4 - \frac{4}{5} & 8 - \frac{16}{5} & 0 \\
0 & 0 & 9
\end{bmatrix} = \begin{bmatrix}
\frac{69}{5} & \frac{16}{5} & 0 \\
\frac{16}{5} & \frac{24}{5} & 0 \\
0 & 0 & 9
\end{bmatrix}$$

The conditional distribution of $\mathbf{X}_1$ given $\mathbf{X}_2 = \begin{bmatrix} 6 \\ 24 \end{bmatrix}$ is therefore:

$$\mathbf{X}_1|\mathbf{X}_2 = \begin{bmatrix} 6 \\ 24 \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \frac{229}{5} \\ \frac{151}{5} \\ 40 \end{bmatrix}, \begin{bmatrix} \frac{69}{5} & \frac{16}{5} & 0 \\ \frac{16}{5} & \frac{24}{5} & 0 \\ 0 & 0 & 9 \end{bmatrix}\right)$$

Which simplifies to:

$$\mathbf{X}_1|\mathbf{X}_2 = \begin{bmatrix} 6 \\ 24 \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} 45.8 \\ 30.2 \\ 40 \end{bmatrix}, \begin{bmatrix} 13.8 & 3.2 & 0 \\ 3.2 & 4.8 & 0 \\ 0 & 0 & 9 \end{bmatrix}\right)$$

## Related Topics

- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: Detailed examples of multivariate normal distributions
- [[L2_1_Multivariate_Exam_Linear_Transformation_Examples|Linear Transformation Examples]]: Examples of applying linear transformations to multivariate distributions
- [[L2_1_Multivariate_Exam_Conditional_Distribution_Examples|Conditional Distribution Examples]]: Problems on conditional distributions, inference, and predictions
- [[L2_1_Independence_Examples|Independence Examples]]: General examples of independence in probability theory
- [[L2_1_Multivariate_Exam_Eigendecomposition_Examples|Eigendecomposition Examples]]: Examples on eigenvalue decomposition related to independence
- [[L2_1_Mahalanobis_Distance|Mahalanobis Distance]]: Applications of independence concepts in distance metrics 