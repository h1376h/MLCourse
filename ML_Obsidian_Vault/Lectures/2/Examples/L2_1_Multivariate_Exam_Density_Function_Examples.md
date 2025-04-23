# Multivariate Density Function Examples

This document provides examples and key concepts on multivariate density functions to help you understand these fundamental concepts in multivariate analysis, machine learning, and data science.

## Key Concepts and Formulas

Multivariate density functions describe the probability distribution of multiple random variables simultaneously. The most important multivariate density is the multivariate normal (Gaussian) distribution.

### The Multivariate Normal Density Function

For a $p$-dimensional random vector $\mathbf{X}$, the multivariate normal density function is:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

Where:
- $\boldsymbol{\mu}$ = Mean vector (dimension $p \times 1$)
- $\boldsymbol{\Sigma}$ = Covariance matrix (dimension $p \times p$, symmetric and positive-definite)
- $|\boldsymbol{\Sigma}|$ = Determinant of the covariance matrix
- $\boldsymbol{\Sigma}^{-1}$ = Inverse of the covariance matrix

## Example 1: Bivariate Normal Density Function

### Problem Statement
Let $\mathbf{X} = (X_1, X_2)$ follow a bivariate normal distribution with mean vector $\boldsymbol{\mu} = (2, 3)$ and covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix}$.

a) Find the probability density function (PDF) of $\mathbf{X}$.
b) Calculate the probability $P(X_1 \leq 3, X_2 \leq 4)$.
c) Find the conditional distribution of $X_1$ given $X_2 = 4$.

### Solution

#### Part a: Finding the PDF

To find the PDF, we need to substitute the given parameters into the multivariate normal density function formula.

First, let's calculate the determinant of the covariance matrix:
$$|\boldsymbol{\Sigma}| = |{\begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix}}| = 4 \times 5 - 2 \times 2 = 20 - 4 = 16$$

Next, we calculate the inverse of the covariance matrix:
$$\boldsymbol{\Sigma}^{-1} = \frac{1}{16}{\begin{bmatrix} 5 & -2 \\ -2 & 4 \end{bmatrix}} = {\begin{bmatrix} 5/16 & -1/8 \\ -1/8 & 1/4 \end{bmatrix}}$$

Now, we substitute into the PDF formula:
$$f(x_1, x_2) = \frac{1}{2\pi \sqrt{16}} \exp\left(-\frac{1}{2}
\begin{bmatrix} x_1 - 2 \\ x_2 - 3 \end{bmatrix}^T
\begin{bmatrix} 5/16 & -1/8 \\ -1/8 & 1/4 \end{bmatrix}
\begin{bmatrix} x_1 - 2 \\ x_2 - 3 \end{bmatrix}
\right)$$

$$f(x_1, x_2) = \frac{1}{8\pi} \exp\left(-\frac{1}{2}\left[\frac{5}{16}(x_1-2)^2 - \frac{1}{4}(x_1-2)(x_2-3) + \frac{1}{4}(x_2-3)^2\right]\right)$$

#### Part b: Calculating $P(X_1 \leq 3, X_2 \leq 4)$

For multivariate normal probabilities, we need to standardize and use numerical integration or statistical software. We can use the following steps:

1. Standardize the variables:
   $(X_1 - 2)/\sqrt{4} = (X_1 - 2)/2$ and $(X_2 - 3)/\sqrt{5} = (X_2 - 3)/\sqrt{5}$
   
2. Account for correlation coefficient $\rho = 2/\sqrt{4 \cdot 5} = 2/\sqrt{20} = 2/2\sqrt{5} = 1/\sqrt{5} \approx 0.447$

3. The standardized bounds are:
   $z_1 = (3 - 2)/2 = 0.5$ and $z_2 = (4 - 3)/\sqrt{5} \approx 0.447$
   
4. Using the bivariate normal CDF with correlation 0.447:
   $P(X_1 \leq 3, X_2 \leq 4) = \Phi_2(0.5, 0.447; 0.447) \approx 0.627$

#### Part c: Finding the conditional distribution

For a bivariate normal, the conditional distribution of $X_1$ given $X_2 = 4$ is also normal with:

$$\mu_{1|2} = \mu_1 + \frac{\sigma_{12}}{\sigma_2^2}(x_2 - \mu_2) = 2 + \frac{2}{5}(4 - 3) = 2 + \frac{2}{5} = 2.4$$

$$\sigma_{1|2}^2 = \sigma_1^2 - \frac{\sigma_{12}^2}{\sigma_2^2} = 4 - \frac{4}{5} = 4 - 0.8 = 3.2$$

Therefore, the conditional distribution is:
$$X_1 | (X_2 = 4) \sim \mathcal{N}(2.4, 3.2)$$

## Example 2: Multivariate Density with Marginal and Conditional Distributions

### Problem Statement
Consider a trivariate normal distribution with density function $f(x, y, z)$ where the mean vector is $\boldsymbol{\mu} = (1, 2, 3)$ and the covariance matrix is:

$$\boldsymbol{\Sigma} = \begin{bmatrix} 3 & 1 & -1 \\ 1 & 2 & 0 \\ -1 & 0 & 4 \end{bmatrix}$$

a) Find the marginal density function $f(x, y)$.
b) Find the conditional density function $f(z | x=2, y=1)$.

### Solution

#### Part a: Finding the marginal density function

For multivariate normal distributions, the marginal distribution of any subset of variables is also multivariate normal. The marginal distribution of $(X, Y)$ is obtained by:
- Taking the corresponding elements of the mean vector: $\boldsymbol{\mu}_{X,Y} = (1, 2)$
- Taking the corresponding submatrix of the covariance matrix: $\boldsymbol{\Sigma}_{X,Y} = \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}$

Therefore, the marginal density function is:

$$f(x, y) = \frac{1}{2\pi\sqrt{|\boldsymbol{\Sigma}_{X,Y}|}} \exp\left(-\frac{1}{2}
\begin{bmatrix} x - 1 \\ y - 2 \end{bmatrix}^T
\boldsymbol{\Sigma}_{X,Y}^{-1}
\begin{bmatrix} x - 1 \\ y - 2 \end{bmatrix}
\right)$$

We have $|\boldsymbol{\Sigma}_{X,Y}| = 3 \times 2 - 1 \times 1 = 5$, and:

$$\boldsymbol{\Sigma}_{X,Y}^{-1} = \frac{1}{5} \begin{bmatrix} 2 & -1 \\ -1 & 3 \end{bmatrix}$$

Substituting, we get:

$$f(x, y) = \frac{1}{2\pi\sqrt{5}} \exp\left(-\frac{1}{2}\left[\frac{2}{5}(x-1)^2 - \frac{2}{5}(x-1)(y-2) + \frac{3}{5}(y-2)^2\right]\right)$$

#### Part b: Finding the conditional density function

For the conditional distribution of $Z$ given $X=2$ and $Y=1$, we need to partition the variables:
- $\mathbf{X}_1 = Z$ (the variable of interest)
- $\mathbf{X}_2 = (X, Y)$ (the conditioning variables)

With this partition:
- $\boldsymbol{\mu}_1 = 3$ (mean of $Z$)
- $\boldsymbol{\mu}_2 = (1, 2)$ (mean of $(X, Y)$)
- $\boldsymbol{\Sigma}_{11} = 4$ (variance of $Z$)
- $\boldsymbol{\Sigma}_{12} = (-1, 0)$ (covariance between $Z$ and $(X, Y)$)
- $\boldsymbol{\Sigma}_{22} = \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}$ (covariance matrix of $(X, Y)$)

The conditional mean is:
$$\mu_{Z|X,Y} = \mu_Z + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

Substituting:
$$\mu_{Z|X,Y} = 3 + (-1, 0) \frac{1}{5} \begin{bmatrix} 2 & -1 \\ -1 & 3 \end{bmatrix} \begin{pmatrix} 2 - 1 \\ 1 - 2 \end{pmatrix}$$

$$\mu_{Z|X,Y} = 3 + (-1, 0) \frac{1}{5} \begin{bmatrix} 2 & -1 \\ -1 & 3 \end{bmatrix} \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

$$\mu_{Z|X,Y} = 3 + (-1, 0) \frac{1}{5} \begin{pmatrix} 3 \\ -4 \end{pmatrix} = 3 - \frac{3}{5} = \frac{12}{5} = 2.4$$

The conditional variance is:
$$\sigma_{Z|X,Y}^2 = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

$$\sigma_{Z|X,Y}^2 = 4 - (-1, 0) \frac{1}{5} \begin{bmatrix} 2 & -1 \\ -1 & 3 \end{bmatrix} \begin{pmatrix} -1 \\ 0 \end{pmatrix}$$

$$\sigma_{Z|X,Y}^2 = 4 - (-1, 0) \frac{1}{5} \begin{pmatrix} -2 \\ 1 \end{pmatrix} = 4 + \frac{2}{5} = \frac{22}{5} = 4.4$$

Therefore, the conditional distribution is:
$$Z | (X=2, Y=1) \sim \mathcal{N}(2.4, 4.4)$$

And the conditional density function is:
$$f(z | x=2, y=1) = \frac{1}{\sqrt{2\pi \cdot 4.4}} \exp\left(-\frac{(z-2.4)^2}{2 \cdot 4.4}\right)$$

## Related Topics

- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: More examples of multivariate normal distributions
- [[L2_1_Joint_Distributions_Examples|Joint Distributions Examples]]: Working with joint probability distributions
- [[L2_1_Conditional_Probability_Examples|Conditional Probability Examples]]: Additional examples of conditional distributions
- [[L2_1_Expectation_Examples|Expectation Examples]]: Calculating expected values for multivariate distributions
- [[L2_1_Covariance_Examples|Covariance Examples]]: Understanding covariance in multivariate settings 