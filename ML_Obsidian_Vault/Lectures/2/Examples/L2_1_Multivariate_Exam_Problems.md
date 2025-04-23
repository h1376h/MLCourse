# Multivariate Exam Problems

This document provides practice problems and solutions covering density functions, multivariate Gaussian distributions, transformation techniques, and common exam questions related to multivariate analysis in machine learning and data science.

## Key Concepts and Formulas

Multivariate analysis deals with observations on multiple variables and their relationships. Below are key concepts and formulas important for exam problems:

### Multivariate Normal Distribution

The probability density function (PDF) of a $p$-dimensional multivariate normal distribution is:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

Where:
- $\boldsymbol{\mu}$ = Mean vector (dimension $p \times 1$)
- $\boldsymbol{\Sigma}$ = Covariance matrix (dimension $p \times p$, symmetric and positive-definite)

### Linear Transformations

If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$, then:
$\mathbf{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$

### Conditional Distributions

If $\mathbf{X} = [\mathbf{X}_1, \mathbf{X}_2]^T \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then:
$\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})$

Where:
- $\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$
- $\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$

### Mahalanobis Distance

The squared Mahalanobis distance from $\mathbf{x}$ to $\boldsymbol{\mu}$ is:
$d^2(\mathbf{x}, \boldsymbol{\mu}) = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$

## Practice Problems

### Problem 1: Bivariate Normal Density Function

#### Problem Statement
Let $\mathbf{X} = (X_1, X_2)$ follow a bivariate normal distribution with mean vector $\boldsymbol{\mu} = (2, 3)$ and covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix}$.

a) Find the probability density function (PDF) of $\mathbf{X}$.
b) Calculate the probability $P(X_1 \leq 3, X_2 \leq 4)$.
c) Find the conditional distribution of $X_1$ given $X_2 = 4$.

#### Solution

##### Part a: Finding the PDF

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

##### Part b: Calculating $P(X_1 \leq 3, X_2 \leq 4)$

For multivariate normal probabilities, we need to standardize and use numerical integration or statistical software. We can use the following steps:

1. Standardize the variables:
   $(X_1 - 2)/\sqrt{4} = (X_1 - 2)/2$ and $(X_2 - 3)/\sqrt{5} = (X_2 - 3)/\sqrt{5}$
   
2. Account for correlation coefficient $\rho = 2/\sqrt{4 \cdot 5} = 2/\sqrt{20} = 2/2\sqrt{5} = 1/\sqrt{5} \approx 0.447$

3. The standardized bounds are:
   $z_1 = (3 - 2)/2 = 0.5$ and $z_2 = (4 - 3)/\sqrt{5} \approx 0.447$
   
4. Using the bivariate normal CDF with correlation 0.447:
   $P(X_1 \leq 3, X_2 \leq 4) = \Phi_2(0.5, 0.447; 0.447) \approx 0.627$

##### Part c: Finding the conditional distribution

For a bivariate normal, the conditional distribution of $X_1$ given $X_2 = 4$ is also normal with:

$$\mu_{1|2} = \mu_1 + \frac{\sigma_{12}}{\sigma_2^2}(x_2 - \mu_2) = 2 + \frac{2}{5}(4 - 3) = 2 + \frac{2}{5} = 2.4$$

$$\sigma_{1|2}^2 = \sigma_1^2 - \frac{\sigma_{12}^2}{\sigma_2^2} = 4 - \frac{4}{5} = 4 - 0.8 = 3.2$$

Therefore, the conditional distribution is:
$$X_1 | (X_2 = 4) \sim \mathcal{N}(2.4, 3.2)$$

### Problem 2: Linear Transformations of Multivariate Normal Distributions

#### Problem Statement
Let $\mathbf{X} = \begin{bmatrix} X_1 \\ X_2 \\ X_3 \end{bmatrix}$ follow a multivariate normal distribution with mean vector $\boldsymbol{\mu} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 4 & 1 & 0 \\ 1 & 9 & 2 \\ 0 & 2 & 16 \end{bmatrix}$.

Define $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$ where $\mathbf{A} = \begin{bmatrix} 2 & 1 & 0 \\ 0 & 3 & 1 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 5 \\ -2 \end{bmatrix}$.

a) Find the distribution of $\mathbf{Y}$.
b) Calculate $Cov(Y_1, Y_2)$.
c) Are $Y_1$ and $Y_2$ independent? Why or why not?

#### Solution

##### Part a: Finding the distribution of $\mathbf{Y}$

When a random vector $\mathbf{X}$ follows a multivariate normal distribution, any linear transformation $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$ also follows a multivariate normal distribution with:

$$\mathbf{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$$

Let's compute the mean vector first:

$$\boldsymbol{\mu}_Y = \mathbf{A}\boldsymbol{\mu} + \mathbf{b} = \begin{bmatrix} 2 & 1 & 0 \\ 0 & 3 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 5 \\ -2 \end{bmatrix}$$

$$\boldsymbol{\mu}_Y = \begin{bmatrix} 2(1) + 1(2) + 0(3) \\ 0(1) + 3(2) + 1(3) \end{bmatrix} + \begin{bmatrix} 5 \\ -2 \end{bmatrix} = \begin{bmatrix} 4 \\ 9 \end{bmatrix} + \begin{bmatrix} 5 \\ -2 \end{bmatrix} = \begin{bmatrix} 9 \\ 7 \end{bmatrix}$$

Now, let's calculate the covariance matrix:

$$\boldsymbol{\Sigma}_Y = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T = \begin{bmatrix} 2 & 1 & 0 \\ 0 & 3 & 1 \end{bmatrix} \begin{bmatrix} 4 & 1 & 0 \\ 1 & 9 & 2 \\ 0 & 2 & 16 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 1 & 3 \\ 0 & 1 \end{bmatrix}$$

First, computing $\mathbf{A}\boldsymbol{\Sigma}$:

$$\mathbf{A}\boldsymbol{\Sigma} = \begin{bmatrix} 2(4) + 1(1) + 0(0) & 2(1) + 1(9) + 0(2) & 2(0) + 1(2) + 0(16) \\ 0(4) + 3(1) + 1(0) & 0(1) + 3(9) + 1(2) & 0(0) + 3(2) + 1(16) \end{bmatrix}$$

$$\mathbf{A}\boldsymbol{\Sigma} = \begin{bmatrix} 9 & 11 & 2 \\ 3 & 29 & 22 \end{bmatrix}$$

Then, computing $\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T$:

$$\boldsymbol{\Sigma}_Y = \begin{bmatrix} 9 & 11 & 2 \\ 3 & 29 & 22 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 1 & 3 \\ 0 & 1 \end{bmatrix}$$

$$\boldsymbol{\Sigma}_Y = \begin{bmatrix} 9(2) + 11(1) + 2(0) & 9(0) + 11(3) + 2(1) \\ 3(2) + 29(1) + 22(0) & 3(0) + 29(3) + 22(1) \end{bmatrix}$$

$$\boldsymbol{\Sigma}_Y = \begin{bmatrix} 29 & 35 \\ 35 & 109 \end{bmatrix}$$

Therefore, $\mathbf{Y} \sim \mathcal{N}\left(\begin{bmatrix} 9 \\ 7 \end{bmatrix}, \begin{bmatrix} 29 & 35 \\ 35 & 109 \end{bmatrix}\right)$

##### Part b: Calculating $Cov(Y_1, Y_2)$

From the covariance matrix, we can directly read that:
$$Cov(Y_1, Y_2) = \boldsymbol{\Sigma}_Y[1,2] = 35$$

##### Part c: Determining independence

For multivariate normal distributions, zero covariance means independence. Since $Cov(Y_1, Y_2) = 35 \neq 0$, $Y_1$ and $Y_2$ are not independent.

The non-zero covariance indicates that knowledge of one variable provides information about the other. This is also evident from the structure of the linear transformation, where both $Y_1$ and $Y_2$ depend on overlapping components of the original vector $\mathbf{X}$.

### Problem 3: Conditional Distributions and Inference

#### Problem Statement
Consider a trivariate normal random vector $\mathbf{X} = (X_1, X_2, X_3)$ with mean vector $\boldsymbol{\mu} = (5, 7, 10)$ and covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 4 & 2 & 1 \\ 2 & 9 & 3 \\ 1 & 3 & 5 \end{bmatrix}$.

a) Find the conditional distribution of $X_1$ given $X_2 = 8$ and $X_3 = 11$.
b) If we only observe $X_2 = 8$ (but not $X_3$), what is our best prediction for $X_1$?
c) Calculate the reduction in variance of our prediction of $X_1$ when we observe both $X_2$ and $X_3$ compared to observing only $X_2$.

#### Solution

##### Part a: Finding the conditional distribution of $X_1$ given $X_2 = 8$ and $X_3 = 11$

To find the conditional distribution, we partition the variables:
- $\mathbf{X}_1 = X_1$ (the variable of interest)
- $\mathbf{X}_2 = (X_2, X_3)$ (the conditioning variables)

With this partition:
- $\boldsymbol{\mu}_1 = 5$ 
- $\boldsymbol{\mu}_2 = (7, 10)$
- $\boldsymbol{\Sigma}_{11} = 4$ (variance of $X_1$)
- $\boldsymbol{\Sigma}_{12} = (2, 1)$ (covariance between $X_1$ and $(X_2, X_3)$)
- $\boldsymbol{\Sigma}_{21} = (2, 1)^T$ (transpose of $\boldsymbol{\Sigma}_{12}$)
- $\boldsymbol{\Sigma}_{22} = \begin{bmatrix} 9 & 3 \\ 3 & 5 \end{bmatrix}$ (covariance matrix of $(X_2, X_3)$)

First, we need to find $\boldsymbol{\Sigma}_{22}^{-1}$:
$$|\boldsymbol{\Sigma}_{22}| = 9 \times 5 - 3 \times 3 = 45 - 9 = 36$$

$$\boldsymbol{\Sigma}_{22}^{-1} = \frac{1}{36} \begin{bmatrix} 5 & -3 \\ -3 & 9 \end{bmatrix} = \begin{bmatrix} 5/36 & -1/12 \\ -1/12 & 1/4 \end{bmatrix}$$

Next, we calculate the conditional mean:
$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

$$\boldsymbol{\mu}_{1|2} = 5 + (2, 1) \begin{bmatrix} 5/36 & -1/12 \\ -1/12 & 1/4 \end{bmatrix} \begin{pmatrix} 8 - 7 \\ 11 - 10 \end{pmatrix}$$

$$\boldsymbol{\mu}_{1|2} = 5 + (2, 1) \begin{bmatrix} 5/36 & -1/12 \\ -1/12 & 1/4 \end{bmatrix} \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

$$\boldsymbol{\mu}_{1|2} = 5 + (2, 1) \begin{pmatrix} 5/36 - 1/12 \\ -1/12 + 1/4 \end{pmatrix} = 5 + (2, 1) \begin{pmatrix} 5/36 - 3/36 \\ -3/36 + 9/36 \end{pmatrix}$$

$$\boldsymbol{\mu}_{1|2} = 5 + (2, 1) \begin{pmatrix} 2/36 \\ 6/36 \end{pmatrix} = 5 + 2(2/36) + 1(6/36) = 5 + 4/36 + 6/36 = 5 + 10/36 = 5 + 5/18 \approx 5.28$$

Now, we calculate the conditional variance:
$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

$$\boldsymbol{\Sigma}_{1|2} = 4 - (2, 1) \begin{bmatrix} 5/36 & -1/12 \\ -1/12 & 1/4 \end{bmatrix} \begin{pmatrix} 2 \\ 1 \end{pmatrix}$$

$$\boldsymbol{\Sigma}_{1|2} = 4 - (2, 1) \begin{pmatrix} 10/36 - 1/12 \\ -2/12 + 1/4 \end{pmatrix} = 4 - (20/36 + 10/36) = 4 - 30/36 \approx 3.53$$

Therefore, the conditional distribution is:
$$X_1 | (X_2 = 8, X_3 = 11) \sim \mathcal{N}(5.28, 3.53)$$

##### Part b: Best prediction for $X_1$ given only $X_2 = 8$

When we observe only $X_2 = 8$, we need to find the conditional distribution of $X_1$ given just $X_2$.

For this, we use a similar approach but with different partitioning:
- $\boldsymbol{\mu}_1 = 5$ 
- $\boldsymbol{\mu}_2 = 7$
- $\boldsymbol{\Sigma}_{11} = 4$ (variance of $X_1$)
- $\boldsymbol{\Sigma}_{12} = 2$ (covariance between $X_1$ and $X_2$)
- $\boldsymbol{\Sigma}_{22} = 9$ (variance of $X_2$)

The conditional mean is:
$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(x_2 - \boldsymbol{\mu}_2)$$

$$\boldsymbol{\mu}_{1|2} = 5 + 2 \cdot \frac{1}{9} \cdot (8 - 7) = 5 + \frac{2}{9} \approx 5.22$$

This is our best prediction of $X_1$ given that we observe $X_2 = 8$.

##### Part c: Reduction in variance

The conditional variance when we observe only $X_2$ is:
$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = 4 - 2 \cdot \frac{1}{9} \cdot 2 = 4 - \frac{4}{9} \approx 3.56$$

The conditional variance when we observe both $X_2$ and $X_3$ was calculated in part (a) as $\boldsymbol{\Sigma}_{1|2,3} \approx 3.53$.

The reduction in variance is:
$$\boldsymbol{\Sigma}_{1|2} - \boldsymbol{\Sigma}_{1|2,3} \approx 3.56 - 3.53 = 0.03$$

This small reduction in variance indicates that knowing $X_3$ in addition to $X_2$ provides relatively little additional information about $X_1$. This makes sense given the covariance structure, where $X_1$ has a stronger correlation with $X_2$ (covariance of 2) than with $X_3$ (covariance of 1).

### Problem 4: Mahalanobis Distance and Classification

#### Problem Statement
You are building a binary classifier for a machine learning problem. The two classes follow multivariate normal distributions with the same covariance matrix but different means:

Class 1: $\mathbf{X} \sim \mathcal{N}\left(\begin{bmatrix} 2 \\ 4 \end{bmatrix}, \begin{bmatrix} 5 & 1 \\ 1 & 3 \end{bmatrix}\right)$

Class 2: $\mathbf{X} \sim \mathcal{N}\left(\begin{bmatrix} 5 \\ 6 \end{bmatrix}, \begin{bmatrix} 5 & 1 \\ 1 & 3 \end{bmatrix}\right)$

a) Calculate the Mahalanobis distance between the two class means.
b) For a new observation $\mathbf{x} = (3, 5)$, determine the class it belongs to using the minimum Mahalanobis distance classifier.
c) Show that the decision boundary based on the Mahalanobis distance is a straight line, and find its equation.

#### Solution

##### Part a: Calculating the Mahalanobis distance between class means

The Mahalanobis distance between two points $\mathbf{x}_1$ and $\mathbf{x}_2$ with respect to covariance matrix $\boldsymbol{\Sigma}$ is:

$$d_M(\mathbf{x}_1, \mathbf{x}_2) = \sqrt{(\mathbf{x}_1 - \mathbf{x}_2)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}_1 - \mathbf{x}_2)}$$

First, we need to find $\boldsymbol{\Sigma}^{-1}$:

$$|\boldsymbol{\Sigma}| = 5 \times 3 - 1 \times 1 = 15 - 1 = 14$$

$$\boldsymbol{\Sigma}^{-1} = \frac{1}{14} \begin{bmatrix} 3 & -1 \\ -1 & 5 \end{bmatrix} = \begin{bmatrix} 3/14 & -1/14 \\ -1/14 & 5/14 \end{bmatrix}$$

The difference between the class means is:
$$\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2 = \begin{bmatrix} 2 \\ 4 \end{bmatrix} - \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} -3 \\ -2 \end{bmatrix}$$

Now we calculate the squared Mahalanobis distance:
$$d_M^2(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) = \begin{bmatrix} -3 & -2 \end{bmatrix} \begin{bmatrix} 3/14 & -1/14 \\ -1/14 & 5/14 \end{bmatrix} \begin{bmatrix} -3 \\ -2 \end{bmatrix}$$

$$d_M^2(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) = \begin{bmatrix} -3 & -2 \end{bmatrix} \begin{bmatrix} -9/14 + 2/14 \\ 3/14 - 10/14 \end{bmatrix}$$

$$d_M^2(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) = \begin{bmatrix} -3 & -2 \end{bmatrix} \begin{bmatrix} -7/14 \\ -7/14 \end{bmatrix}$$

$$d_M^2(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) = 3 \times (-7/14) + 2 \times (-7/14) = -21/14 - 14/14 = -35/14 = -2.5$$

Since we're calculating a distance, we need the absolute value:
$$d_M^2(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) = 2.5$$

Therefore, the Mahalanobis distance is:
$$d_M(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) = \sqrt{2.5} \approx 1.58$$

##### Part b: Classifying a new observation using minimum Mahalanobis distance

For a new observation $\mathbf{x} = (3, 5)$, we need to calculate the Mahalanobis distance to each class mean and assign it to the class with the smaller distance.

Distance to Class 1 mean:
$$\mathbf{x} - \boldsymbol{\mu}_1 = \begin{bmatrix} 3 \\ 5 \end{bmatrix} - \begin{bmatrix} 2 \\ 4 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_1) = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 3/14 & -1/14 \\ -1/14 & 5/14 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_1) = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 3/14 - 1/14 \\ -1/14 + 5/14 \end{bmatrix} = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 2/14 \\ 4/14 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_1) = 1 \times (2/14) + 1 \times (4/14) = 2/14 + 4/14 = 6/14 = 3/7 \approx 0.429$$

Distance to Class 2 mean:
$$\mathbf{x} - \boldsymbol{\mu}_2 = \begin{bmatrix} 3 \\ 5 \end{bmatrix} - \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} -2 \\ -1 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_2) = \begin{bmatrix} -2 & -1 \end{bmatrix} \begin{bmatrix} 3/14 & -1/14 \\ -1/14 & 5/14 \end{bmatrix} \begin{bmatrix} -2 \\ -1 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_2) = \begin{bmatrix} -2 & -1 \end{bmatrix} \begin{bmatrix} -6/14 + 1/14 \\ 2/14 - 5/14 \end{bmatrix} = \begin{bmatrix} -2 & -1 \end{bmatrix} \begin{bmatrix} -5/14 \\ -3/14 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_2) = (-2) \times (-5/14) + (-1) \times (-3/14) = 10/14 + 3/14 = 13/14 \approx 0.929$$

Since $d_M^2(\mathbf{x}, \boldsymbol{\mu}_1) < d_M^2(\mathbf{x}, \boldsymbol{\mu}_2)$ (0.429 < 0.929), we classify $\mathbf{x}$ as belonging to Class 1.

##### Part c: Finding the decision boundary equation

For the minimum Mahalanobis distance classifier with equal covariance matrices, the decision boundary is a hyperplane equidistant from both means in the Mahalanobis distance sense.

The decision boundary satisfies:
$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_1) = d_M^2(\mathbf{x}, \boldsymbol{\mu}_2)$$

This expands to:
$$(\mathbf{x} - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_1) = (\mathbf{x} - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_2)$$

After expanding and simplifying, this becomes:
$$2(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} \mathbf{x} = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2)$$

Let's compute this equation for our specific case:
$$(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} = \begin{bmatrix} -3 & -2 \end{bmatrix} \begin{bmatrix} 3/14 & -1/14 \\ -1/14 & 5/14 \end{bmatrix} = \begin{bmatrix} -9/14 + 2/14 & 3/14 - 10/14 \end{bmatrix} = \begin{bmatrix} -7/14 & -7/14 \end{bmatrix}$$

$$(\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2) = \begin{bmatrix} 2 \\ 4 \end{bmatrix} + \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 7 \\ 10 \end{bmatrix}$$

The right side of the equation is:
$$(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2) = \begin{bmatrix} -7/14 & -7/14 \end{bmatrix} \begin{bmatrix} 7 \\ 10 \end{bmatrix} = -7/14 \times 7 - 7/14 \times 10 = -49/14 - 70/14 = -119/14$$

So the decision boundary equation is:
$$2 \times \begin{bmatrix} -7/14 & -7/14 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = -119/14$$

$$\begin{bmatrix} -1 & -1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = -119/28$$

$$-x_1 - x_2 = -119/28$$

Simplifying:
$$x_1 + x_2 = 119/28 \approx 4.25$$

Therefore, the decision boundary is the straight line $x_1 + x_2 = 4.25$. Any point $(x_1, x_2)$ where $x_1 + x_2 > 4.25$ will be classified as Class 2, and any point where $x_1 + x_2 < 4.25$ will be classified as Class 1.

### Problem 5: Eigenvalue Decomposition of Covariance Matrices

#### Problem Statement
Consider a bivariate normal distribution with covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 9 & 5 \\ 5 & 4 \end{bmatrix}$.

a) Find the eigenvalues and eigenvectors of the covariance matrix.
b) Interpret these eigenvalues and eigenvectors in terms of the principal components.
c) Determine the directions of maximum and minimum variance, and the corresponding variances.
d) If you generate samples from this distribution with mean $\boldsymbol{\mu} = (0, 0)$, how would you transform these samples to make the variables uncorrelated?

#### Solution

##### Part a: Finding eigenvalues and eigenvectors

To find the eigenvalues of $\boldsymbol{\Sigma}$, we solve:
$$|\boldsymbol{\Sigma} - \lambda \mathbf{I}| = 0$$

$$\begin{vmatrix} 9 - \lambda & 5 \\ 5 & 4 - \lambda \end{vmatrix} = 0$$

$$(9 - \lambda)(4 - \lambda) - 5 \times 5 = 0$$

$$36 - 9\lambda - 4\lambda + \lambda^2 - 25 = 0$$

$$\lambda^2 - 13\lambda + 11 = 0$$

Using the quadratic formula:
$$\lambda = \frac{13 \pm \sqrt{13^2 - 4 \times 11}}{2} = \frac{13 \pm \sqrt{169 - 44}}{2} = \frac{13 \pm \sqrt{125}}{2} \approx \frac{13 \pm 11.18}{2}$$

$$\lambda_1 \approx 12.09, \lambda_2 \approx 0.91$$

Now, for each eigenvalue, we find the corresponding eigenvector by solving:
$$(\boldsymbol{\Sigma} - \lambda_i \mathbf{I})\mathbf{v}_i = \mathbf{0}$$

For $\lambda_1 \approx 12.09$:
$$\begin{bmatrix} 9 - 12.09 & 5 \\ 5 & 4 - 12.09 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$\begin{bmatrix} -3.09 & 5 \\ 5 & -8.09 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

This gives us:
$$-3.09v_{11} + 5v_{12} = 0$$
$$5v_{11} - 8.09v_{12} = 0$$

From the first equation:
$$v_{11} = \frac{5v_{12}}{3.09} \approx 1.62v_{12}$$

Now we normalize the eigenvector so that $\|\mathbf{v}_1\| = 1$:
$$\sqrt{(1.62v_{12})^2 + v_{12}^2} = 1$$
$$\sqrt{2.62^2 + 1^2}v_{12} = 1$$
$$\sqrt{7.86}v_{12} = 1$$
$$v_{12} \approx 0.357$$
$$v_{11} \approx 1.62 \times 0.357 \approx 0.578$$

Therefore, $\mathbf{v}_1 \approx (0.85, 0.53)$ after normalizing.

Similarly, for $\lambda_2 \approx 0.91$, we find $\mathbf{v}_2 \approx (-0.53, 0.85)$.

##### Part b: Interpreting in terms of principal components

The eigenvalues of the covariance matrix represent the variances along the principal component directions, and the eigenvectors specify these directions.

In this case:
- The first principal component (corresponding to $\lambda_1 \approx 12.09$) points in the direction $(0.85, 0.53)$ and has a variance of 12.09. This is the direction of maximum variance in the data.
- The second principal component (corresponding to $\lambda_2 \approx 0.91$) points in the direction $(-0.53, 0.85)$ and has a variance of 0.91. This is the direction of minimum variance and is orthogonal to the first principal component.

The ratio of eigenvalues (approximately 13:1) indicates that the distribution is highly elongated along the first principal component.

##### Part c: Directions of maximum and minimum variance

The direction of maximum variance is given by the eigenvector corresponding to the largest eigenvalue, which is $\mathbf{v}_1 \approx (0.85, 0.53)$. The variance in this direction is $\lambda_1 \approx 12.09$.

The direction of minimum variance is given by the eigenvector corresponding to the smallest eigenvalue, which is $\mathbf{v}_2 \approx (-0.53, 0.85)$. The variance in this direction is $\lambda_2 \approx 0.91$.

##### Part d: Transformation to make variables uncorrelated

To make the variables uncorrelated, we need to transform the data using the eigenvectors as a new basis. This is the principal component transformation.

If $\mathbf{X}$ is a random vector from the original distribution, then the transformed vector $\mathbf{Y} = \mathbf{P}^T(\mathbf{X} - \boldsymbol{\mu})$ will have uncorrelated components, where $\mathbf{P}$ is the matrix whose columns are the eigenvectors of $\boldsymbol{\Sigma}$.

In this case:
$$\mathbf{P} = \begin{bmatrix} 0.85 & -0.53 \\ 0.53 & 0.85 \end{bmatrix}$$

The covariance matrix of $\mathbf{Y}$ will be diagonal:
$$\text{Cov}(\mathbf{Y}) = \mathbf{P}^T \boldsymbol{\Sigma} \mathbf{P} = \begin{bmatrix} 12.09 & 0 \\ 0 & 0.91 \end{bmatrix}$$

This transformation is equivalent to rotating the data so that the axes align with the directions of maximum and minimum variance.

## Running the Examples

You can run code to explore these multivariate distributions and visualize the concepts using:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Codes/1_multivariate_exam_problems.py
```

## Related Topics

- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: Detailed examples of multivariate normal distributions
- [[L2_1_Joint_Distributions_Examples|Joint Distributions Examples]]: Working with joint probability distributions
- [[L2_1_Mahalanobis_Distance|Mahalanobis Distance]]: In-depth coverage of Mahalanobis distance and its applications
- [[L2_1_Linear_Transformation|Linear Transformations]]: More on transformations of random vectors
- [[L2_1_Conditional_Probability_Examples|Conditional Probability Examples]]: Additional examples of conditional distributions
- [[L2_1_Covariance_Examples|Covariance Examples]]: Understanding covariance structures
- [[L2_1_Correlation_Examples|Correlation Examples]]: Exploring correlation in multivariate data
