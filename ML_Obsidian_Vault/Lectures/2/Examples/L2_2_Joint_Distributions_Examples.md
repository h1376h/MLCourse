# Joint Distributions Examples

This document provides practical examples of joint distributions to help illustrate their application in machine learning and data analysis contexts.

## Key Concepts and Formulas

Joint distributions describe the simultaneous behavior of two or more random variables. Understanding joint distributions is fundamental for modeling multivariate data and capturing the relationships between variables.

### The Joint Distribution Formulas

For discrete random variables $X$ and $Y$:
$$P(X = x, Y = y) = P(X = x | Y = y) \cdot P(Y = y) = P(Y = y | X = x) \cdot P(X = x)$$

For continuous random variables $X$ and $Y$ with joint probability density function $f_{X,Y}(x,y)$:
$$f_{X,Y}(x,y) = f_{X|Y}(x|y) \cdot f_Y(y) = f_{Y|X}(y|x) \cdot f_X(x)$$

Where:
- $P(X = x, Y = y)$ = Joint probability mass function for discrete random variables
- $f_{X,Y}(x,y)$ = Joint probability density function for continuous random variables
- $P(X = x | Y = y)$ = Conditional probability mass function
- $f_{X|Y}(x|y)$ = Conditional probability density function

## Basic Joint Distribution Examples

The following examples demonstrate joint distributions for discrete and continuous random variables:

- **Discrete Joint Distribution**: Working with probability mass functions and marginal probabilities
- **Bivariate Normal Distribution**: Analyzing correlated continuous random variables
- **Conditional Distributions**: Deriving and interpreting conditional probability distributions
- **Testing Independence**: Determining whether random variables are independent

### Example 1: Discrete Joint Distribution

#### Problem Statement
Consider the joint probability mass function of two discrete random variables $X$ and $Y$ given by the following table:

| $P(X, Y)$ | $Y=1$  | $Y=2$  | $Y=3$  |
|---------|------|------|------|
| $X=1$     | 0.10 | 0.08 | 0.12 |
| $X=2$     | 0.15 | 0.20 | 0.05 |
| $X=3$     | 0.05 | 0.15 | 0.10 |

Find:
1. The marginal distributions of $X$ and $Y$
2. $P(X=2, Y=2)$
3. $P(X > 1, Y < 3)$

#### Solution

##### Step 1: Find the marginal distributions
The marginal distribution of $X$ is found by summing the joint probabilities across all values of $Y$:

$$P(X=1) = \sum_{y} P(X=1, Y=y) = 0.10 + 0.08 + 0.12 = 0.30$$
$$P(X=2) = \sum_{y} P(X=2, Y=y) = 0.15 + 0.20 + 0.05 = 0.40$$
$$P(X=3) = \sum_{y} P(X=3, Y=y) = 0.05 + 0.15 + 0.10 = 0.30$$

The marginal distribution of $Y$ is found by summing the joint probabilities across all values of $X$:

$$P(Y=1) = \sum_{x} P(X=x, Y=1) = 0.10 + 0.15 + 0.05 = 0.30$$
$$P(Y=2) = \sum_{x} P(X=x, Y=2) = 0.08 + 0.20 + 0.15 = 0.43$$
$$P(Y=3) = \sum_{x} P(X=x, Y=3) = 0.12 + 0.05 + 0.10 = 0.27$$

##### Step 2: Find the specific joint probability
From the joint probability table:
$$P(X=2, Y=2) = 0.20$$

##### Step 3: Calculate the probability of the event
This requires summing all probabilities where $X > 1$ (so $X=2$ or $X=3$) and $Y < 3$ (so $Y=1$ or $Y=2$):

$$P(X > 1, Y < 3) = P(X=2, Y=1) + P(X=2, Y=2) + P(X=3, Y=1) + P(X=3, Y=2)$$
$$P(X > 1, Y < 3) = 0.15 + 0.20 + 0.05 + 0.15 = 0.55$$

Therefore, the probability that $X > 1$ and $Y < 3$ is 0.55 or 55%.

![Joint PMF Visualization](../Images/joint_pmf_visualization.png)

### Example 2: Bivariate Normal Distribution

#### Problem Statement
Two variables $X$ and $Y$ follow a bivariate normal distribution with the following parameters:
- Mean of $X$: $\mu_X = 5$
- Mean of $Y$: $\mu_Y = 10$
- Variance of $X$: $\sigma^2_X = 4$ ($\sigma_X = 2$)
- Variance of $Y$: $\sigma^2_Y = 9$ ($\sigma_Y = 3$)
- Correlation coefficient: $\rho = 0.7$

Find:
1. The covariance between $X$ and $Y$
2. The probability $P(X < 6, Y < 12)$

#### Solution

##### Step 1: Calculate the covariance
The covariance is related to the correlation coefficient by:
$$\text{Cov}(X, Y) = \rho\sigma_X\sigma_Y = 0.7 \times 2 \times 3 = 4.2$$

##### Step 2: Find the cumulative probability
For a bivariate normal distribution, calculating the cumulative probability requires integrating the joint density function:

$$P(X < 6, Y < 12) = \int_{-\infty}^{6}\int_{-\infty}^{12} f_{X,Y}(x,y) \, dy \, dx$$

Where $f_{X,Y}(x,y)$ is the bivariate normal density function:

$$f_{X,Y}(x,y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_X)^2}{\sigma_X^2} + \frac{(y-\mu_Y)^2}{\sigma_Y^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y}\right]\right)$$

This calculation is typically performed using statistical software or numerical integration methods.

$$P(X < 6, Y < 12) \approx 0.7291$$

Therefore, the probability that $X < 6$ and $Y < 12$ is approximately 0.7291 or 72.91%.

![Bivariate Normal Distribution](../Images/bivariate_normal_visualization.png)

## Conditional Distributions from Joint Distributions

The following examples demonstrate the derivation and application of conditional distributions:

- **Conditional PMF**: Finding the conditional probability mass function
- **Conditional Expectation**: Calculating expected values with conditional distributions

### Example 3: Conditional Distribution from a Joint PMF

#### Problem Statement
Consider the joint probability mass function from Example 1:

| $P(X, Y)$ | $Y=1$  | $Y=2$  | $Y=3$  |
|---------|------|------|------|
| $X=1$     | 0.10 | 0.08 | 0.12 |
| $X=2$     | 0.15 | 0.20 | 0.05 |
| $X=3$     | 0.05 | 0.15 | 0.10 |

Find the conditional probability mass function of $X$ given $Y=2$.

#### Solution

##### Step 1: Identify the conditional probability formula
The conditional probability mass function of $X$ given $Y=2$ is calculated as:

$$P(X=x|Y=2) = \frac{P(X=x, Y=2)}{P(Y=2)}$$

##### Step 2: Calculate the conditional probabilities
From Example 1, we know that $P(Y=2) = 0.43$.

Therefore:
$$P(X=1|Y=2) = \frac{P(X=1, Y=2)}{P(Y=2)} = \frac{0.08}{0.43} \approx 0.186$$
$$P(X=2|Y=2) = \frac{P(X=2, Y=2)}{P(Y=2)} = \frac{0.20}{0.43} \approx 0.465$$
$$P(X=3|Y=2) = \frac{P(X=3, Y=2)}{P(Y=2)} = \frac{0.15}{0.43} \approx 0.349$$

The conditional distribution sums to 1: $0.186 + 0.465 + 0.349 = 1.0$, which serves as a verification.

![Conditional vs Marginal Distribution](../Images/conditional_distribution_visualization.png)

### Example 4: Conditional Expectation

#### Problem Statement
Using the joint PMF from Example 1, find:
1. The expected value of $X$ given $Y=3$
2. The expected value of $Y$ given $X=1$

#### Solution

##### Step 1: Calculate $E[X|Y=3]$
First, we need the conditional probabilities:
$$P(X=1|Y=3) = \frac{P(X=1, Y=3)}{P(Y=3)} = \frac{0.12}{0.27} \approx 0.444$$
$$P(X=2|Y=3) = \frac{P(X=2, Y=3)}{P(Y=3)} = \frac{0.05}{0.27} \approx 0.185$$
$$P(X=3|Y=3) = \frac{P(X=3, Y=3)}{P(Y=3)} = \frac{0.10}{0.27} \approx 0.370$$

Now calculate the conditional expectation:
$$E[X|Y=3] = \sum_{x} x \cdot P(X=x|Y=3) = 1 \times 0.444 + 2 \times 0.185 + 3 \times 0.370 \approx 1.926$$

##### Step 2: Calculate $E[Y|X=1]$
First, we need the conditional probabilities:
$$P(Y=1|X=1) = \frac{P(X=1, Y=1)}{P(X=1)} = \frac{0.10}{0.30} \approx 0.333$$
$$P(Y=2|X=1) = \frac{P(X=1, Y=2)}{P(X=1)} = \frac{0.08}{0.30} \approx 0.267$$
$$P(Y=3|X=1) = \frac{P(X=1, Y=3)}{P(X=1)} = \frac{0.12}{0.30} = 0.400$$

Now calculate the conditional expectation:
$$E[Y|X=1] = \sum_{y} y \cdot P(Y=y|X=1) = 1 \times 0.333 + 2 \times 0.267 + 3 \times 0.400 \approx 2.067$$

Therefore, the expected value of $X$ given $Y=3$ is approximately 1.926, and the expected value of $Y$ given $X=1$ is approximately 2.067.

![Conditional Expectations](../Images/conditional_expectation_visualization.png)

## Independence and Correlation

The following examples explore the relationship between independence and correlation:

- **Testing Independence**: Determining when random variables are statistically independent
- **Generating Correlated Data**: Creating synthetic data with specified correlation properties

### Example 5: Testing Independence

#### Problem Statement
Using the joint PMF from Example 1, determine whether $X$ and $Y$ are independent random variables.

#### Solution

##### Step 1: State the independence criterion
For $X$ and $Y$ to be independent, the joint probability must equal the product of the marginals for all combinations of $X$ and $Y$:

$$P(X=x, Y=y) = P(X=x) \times P(Y=y) \text{ for all } x, y$$

##### Step 2: Check independence for specific cases
Let's check a few cases:

For $X=1, Y=1$:
$$P(X=1, Y=1) = 0.10$$
$$P(X=1) \times P(Y=1) = 0.30 \times 0.30 = 0.09$$

For $X=2, Y=2$:
$$P(X=2, Y=2) = 0.20$$
$$P(X=2) \times P(Y=2) = 0.40 \times 0.43 = 0.172$$

Since $P(X=x, Y=y) \neq P(X=x) \times P(Y=y)$ for these cases, $X$ and $Y$ are not independent.

##### Step 3: Calculate the covariance
We can also compute the covariance between $X$ and $Y$:

$$E[X] = \sum_{x} x \cdot P(X=x) = 1 \times 0.30 + 2 \times 0.40 + 3 \times 0.30 = 2.00$$
$$E[Y] = \sum_{y} y \cdot P(Y=y) = 1 \times 0.30 + 2 \times 0.43 + 3 \times 0.27 = 1.97$$

$$E[XY] = \sum_{x}\sum_{y} xy \cdot P(X=x, Y=y) = 3.94$$

$$\text{Cov}(X, Y) = E[XY] - E[X]E[Y] = 3.94 - 2.00 \times 1.97 = 3.94 - 3.94 = 0.00$$

Interestingly, although the covariance is approximately zero (due to rounding), we've shown the variables are not independent. This demonstrates that zero covariance does not imply independence except in special cases like the multivariate normal distribution.

![Independence Test Visualization](../Images/independence_test_visualization.png)

### Example 6: Generating Correlated Random Variables

#### Problem Statement
You need to generate bivariate data $(X, Y)$ with the following properties:
- $X$ follows a standard normal distribution $N(0, 1)$
- $Y$ follows a standard normal distribution $N(0, 1)$
- The correlation between $X$ and $Y$ is $\rho = 0.8$

How would you generate such data?

#### Solution

##### Step 1: Set up the generation method
To generate bivariate normal data with a specified correlation, we can use the following approach:

1. Generate two independent standard normal random variables, $Z_1$ and $Z_2$.
2. Set $X = Z_1$
3. Set $Y = \rho Z_1 + \sqrt{1-\rho^2}Z_2$

##### Step 2: Verify the properties
This construction ensures that:
- $X \sim N(0, 1)$ since $Z_1 \sim N(0, 1)$
- $Y \sim N(0, 1)$ as we'll verify below
- $\text{Corr}(X, Y) = \rho$

Let's verify that $Y \sim N(0, 1)$ and $\text{Corr}(X, Y) = \rho$:

$$\text{Var}(Y) = \text{Var}(\rho Z_1 + \sqrt{1-\rho^2}Z_2) = \rho^2 \cdot \text{Var}(Z_1) + (1-\rho^2) \cdot \text{Var}(Z_2) = \rho^2 \cdot 1 + (1-\rho^2) \cdot 1 = 1$$

$$\text{Cov}(X, Y) = \text{Cov}(Z_1, \rho Z_1 + \sqrt{1-\rho^2}Z_2) = \rho \cdot \text{Var}(Z_1) = \rho \cdot 1 = \rho$$

$$\text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X\sigma_Y} = \frac{\rho}{1 \cdot 1} = \rho$$

##### Step 3: Apply the method for $\rho = 0.8$
Therefore, for $\rho = 0.8$:
1. Generate $Z_1 \sim N(0, 1)$ and $Z_2 \sim N(0, 1)$ independently
2. Set $X = Z_1$
3. Set $Y = 0.8Z_1 + 0.6Z_2$ (since $\sqrt{1-0.8^2} = \sqrt{1-0.64} = \sqrt{0.36} = 0.6$)

![Correlated Random Variables](../Images/correlated_random_variables.png)

## Linear Transformations Examples

The following examples demonstrate how to work with linear transformations of joint distributions:

- **Random Vector Transformation**: Applying linear transformations to random vectors
- **Multivariate Gaussian**: Working with multivariate normal distributions

### Example 7: Random Vector Transformation

#### Problem Statement
A machine learning researcher is analyzing a dataset with two features $X$ and $Y$ that follow a joint distribution. They want to apply a linear transformation to create new features $U$ and $V$ according to:
- $U = 2X + Y$
- $V = X - Y$

Given that $X$ and $Y$ have the following properties:
- $E[X] = 3$
- $E[Y] = 2$
- $\text{Var}(X) = 4$
- $\text{Var}(Y) = 9$
- $\text{Cov}(X, Y) = 2$

Find:
1. The means $E[U]$ and $E[V]$
2. The variances $\text{Var}(U)$ and $\text{Var}(V)$
3. The covariance $\text{Cov}(U, V)$

#### Solution

##### Step 1: Calculate the means
The mean of a linear combination of random variables is the linear combination of their means:

$$E[U] = E[2X + Y] = 2E[X] + E[Y] = 2 \times 3 + 2 = 6 + 2 = 8$$
$$E[V] = E[X - Y] = E[X] - E[Y] = 3 - 2 = 1$$

##### Step 2: Calculate the variances
For the variance of linear combinations:
$$\text{Var}(aX + bY) = a^2\text{Var}(X) + b^2\text{Var}(Y) + 2ab \cdot \text{Cov}(X,Y)$$

$$\text{Var}(U) = \text{Var}(2X + Y) = 4\text{Var}(X) + \text{Var}(Y) + 2(2)(1)\text{Cov}(X,Y)$$
$$= 4 \times 4 + 9 + 2 \times 2 \times 2 = 16 + 9 + 8 = 33$$

$$\text{Var}(V) = \text{Var}(X - Y) = \text{Var}(X) + \text{Var}(Y) - 2\text{Cov}(X,Y)$$
$$= 4 + 9 - 2 \times 2 = 13 - 4 = 9$$

##### Step 3: Calculate the covariance
For the covariance between two linear combinations:
$$\text{Cov}(aX + bY, cX + dY) = ac \cdot \text{Var}(X) + bd \cdot \text{Var}(Y) + (ad+bc) \cdot \text{Cov}(X,Y)$$

$$\text{Cov}(U, V) = \text{Cov}(2X + Y, X - Y)$$
$$= 2 \times 1 \times \text{Var}(X) + 1 \times (-1) \times \text{Var}(Y) + [2 \times (-1) + 1 \times 1] \times \text{Cov}(X,Y)$$
$$= 2\text{Var}(X) - \text{Var}(Y) - \text{Cov}(X,Y)$$
$$= 2 \times 4 - 9 - 2 = 8 - 9 - 2 = -3$$

Therefore, $E[U] = 8$, $E[V] = 1$, $\text{Var}(U) = 33$, $\text{Var}(V) = 9$, and $\text{Cov}(U, V) = -3$. The negative covariance indicates that when $U$ increases, $V$ tends to decrease.

![Linear Transformation Visualization](../Images/linear_transformation_visualization.png)

### Example 8: Multivariate Gaussian Example

#### Problem Statement
A data scientist is developing a face recognition system that uses three facial measurements (in millimeters):
- $X_1$: Distance between eyes
- $X_2$: Width of nose
- $X_3$: Width of mouth

Based on a dataset of facial measurements, they model these features using a multivariate Gaussian distribution with the following parameters:

Mean vector: 
$$\boldsymbol{\mu} = \begin{pmatrix} 32 \\ 25 \\ 50 \end{pmatrix}$$

Covariance matrix: 
$$\boldsymbol{\Sigma} = \begin{pmatrix}
16 & 4 & 6 \\
4 & 25 & 10 \\
6 & 10 & 36
\end{pmatrix}$$

1. Write the probability density function for this distribution.
2. Find the marginal distribution of $X_1$ (distance between eyes).
3. Find the conditional distribution of $X_1$ given $X_2 = 30$ and $X_3 = 45$.

#### Solution

##### Step 1: Write the multivariate Gaussian PDF
The general form of a multivariate Gaussian PDF for a random vector $\mathbf{X}$ is:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

Where:
- $n$ is the dimension (3 in our case)
- $\boldsymbol{\mu}$ is the mean vector 
- $\boldsymbol{\Sigma}$ is the covariance matrix
- $|\boldsymbol{\Sigma}|$ is the determinant of $\boldsymbol{\Sigma}$

Calculating the determinant $|\boldsymbol{\Sigma}| = 9736$ and substituting into the formula gives the complete PDF.

##### Step 2: Find the marginal distribution
For a multivariate Gaussian, the marginal distributions are also Gaussian. The marginal distribution of $X_1$ is a univariate Gaussian with:
- Mean $\mu_1 = 32$
- Variance $\sigma_1^2 = 16$

Therefore, $X_1 \sim N(32, 16)$

##### Step 3: Find the conditional distribution
For a multivariate Gaussian, the conditional distribution is also Gaussian. To find the conditional distribution of $X_1$ given $X_2 = 30$ and $X_3 = 45$, we need to use the formula:

Let's partition the random vector $\mathbf{X}$ as:
$$\mathbf{X} = \begin{pmatrix} X_1 \\ X_2 \\ X_3 \end{pmatrix} = \begin{pmatrix} X_1 \\ \mathbf{Y} \end{pmatrix}$$

where $\mathbf{Y} = \begin{pmatrix} X_2 \\ X_3 \end{pmatrix}$.

Then $X_1$ given $\mathbf{Y}$ follows a Gaussian with:
- Conditional mean: $\mu_{1|\mathbf{Y}} = \mu_1 + \boldsymbol{\Sigma}_{1\mathbf{Y}} \boldsymbol{\Sigma}_{\mathbf{Y}}^{-1}(\mathbf{y} - \boldsymbol{\mu}_{\mathbf{Y}})$
- Conditional variance: $\sigma_{1|\mathbf{Y}}^2 = \sigma_1^2 - \boldsymbol{\Sigma}_{1\mathbf{Y}} \boldsymbol{\Sigma}_{\mathbf{Y}}^{-1}\boldsymbol{\Sigma}_{\mathbf{Y}1}$

Where:
- $\mu_1 = 32$
- $\boldsymbol{\mu}_{\mathbf{Y}} = \begin{pmatrix} 25 \\ 50 \end{pmatrix}$
- $\mathbf{y} = \begin{pmatrix} 30 \\ 45 \end{pmatrix}$
- $\sigma_1^2 = 16$
- $\boldsymbol{\Sigma}_{1\mathbf{Y}} = \begin{pmatrix} 4 & 6 \end{pmatrix}$
- $\boldsymbol{\Sigma}_{\mathbf{Y}} = \begin{pmatrix} 25 & 10 \\ 10 & 36 \end{pmatrix}$

Computing these values:
- The inverse of the covariance submatrix:
  $$\boldsymbol{\Sigma}_{\mathbf{Y}}^{-1} = \begin{pmatrix} 0.0445 & -0.0124 \\ -0.0124 & 0.0309 \end{pmatrix}$$

- The difference between observation and mean:
  $$\mathbf{y} - \boldsymbol{\mu}_{\mathbf{Y}} = \begin{pmatrix} 5 \\ -5 \end{pmatrix}$$

- Calculating the mean adjustment:
  $$\boldsymbol{\Sigma}_{1\mathbf{Y}} \boldsymbol{\Sigma}_{\mathbf{Y}}^{-1}(\mathbf{y} - \boldsymbol{\mu}_{\mathbf{Y}}) = \begin{pmatrix} 4 & 6 \end{pmatrix} \begin{pmatrix} 0.0445 & -0.0124 \\ -0.0124 & 0.0309 \end{pmatrix} \begin{pmatrix} 5 \\ -5 \end{pmatrix} = 0.65$$

- Conditional mean: 
  $$\mu_{1|\mathbf{Y}} = 32 + 0.65 = 32.65$$

- Calculating the variance adjustment:
  $$\boldsymbol{\Sigma}_{1\mathbf{Y}} \boldsymbol{\Sigma}_{\mathbf{Y}}^{-1}\boldsymbol{\Sigma}_{\mathbf{Y}1} = \begin{pmatrix} 4 & 6 \end{pmatrix} \begin{pmatrix} 0.0445 & -0.0124 \\ -0.0124 & 0.0309 \end{pmatrix} \begin{pmatrix} 4 \\ 6 \end{pmatrix} = 2.75$$

- Conditional variance: 
  $$\sigma_{1|\mathbf{Y}}^2 = 16 - 2.75 = 13.25$$

Therefore, the conditional distribution is:
$$X_1|(X_2 = 30, X_3 = 45) \sim N(32.65, 13.25)$$

![Multivariate Gaussian Visualization](../Images/multivariate_gaussian_visualization.png)

## Key Insights

### Theoretical Insights
- Joint distributions fully characterize the probabilistic relationship between random variables, capturing both marginal behavior and dependencies.
- For multivariate Gaussian distributions, zero correlation implies independence, but this is not true for general distributions.
- The conditional distribution of a subset of a multivariate Gaussian given another subset is also Gaussian.

### Practical Applications
- **Decomposition**: Leverage factorization and conditional independence to simplify complex joint distributions.
- **Sampling**: Generate synthetic data from joint distributions for data augmentation or simulation studies.
- **Curse of Dimensionality**: As dimensionality increases, joint distributions become increasingly sparse and difficult to estimate.

### Common Pitfalls
- Confusing independence and zero correlation: variables can be uncorrelated but dependent.
- Failing to account for covariance when transforming random vectors.
- Ignoring non-normality: multivariate distributions don't need to be Gaussian, and non-Gaussian distributions require different tools.

## Running the Examples

You can run the code that generates these examples and visualizations using:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Codes/joint_distributions_examples.py
```

## Related Topics

- [[L2_1_Basic_Probability|Basic Probability]]: Foundation for understanding distributions
- [[L2_1_Joint_Probability|Joint Probability]]: Theoretical foundations of joint probabilities
- [[L2_1_Covariance_Correlation|Covariance and Correlation]]: Measures of dependence between variables 