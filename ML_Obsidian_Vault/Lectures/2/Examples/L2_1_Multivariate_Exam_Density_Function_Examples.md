# Multivariate Density Function Examples

This document provides examples and key concepts on multivariate density functions to help you understand these fundamental concepts in multivariate analysis, machine learning, and data science.

This document provides detailed explanations of the step-by-step solutions to the multivariate density function examples, with visualizations to help understand these concepts.

## Example 1: Bivariate Normal Density Function

### Problem Statement
Let $\mathbf{X} = (X_1, X_2)$ follow a bivariate normal distribution with mean vector $\boldsymbol{\mu} = (2, 3)$ and covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix}$.

a) Find the probability density function (PDF) of $\mathbf{X}$.
b) Calculate the probability $P(X_1 \leq 3, X_2 \leq 4)$.
c) Find the conditional distribution of $X_1$ given $X_2 = 4$.

### Solution

#### Part A: Finding the PDF

The multivariate normal density function is given by:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

To find the PDF, we need to determine the components of this formula:

**Step 1: Calculate the determinant of the covariance matrix**

$$|\boldsymbol{\Sigma}| = |{\begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix}}| = 4 \times 5 - 2 \times 2 = 20 - 4 = 16$$

**Step 2: Calculate the inverse of the covariance matrix**

First, we use the formula for the inverse of a 2×2 matrix:
$$\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

Applying this to our covariance matrix:
$$\boldsymbol{\Sigma}^{-1} = \frac{1}{16}{\begin{bmatrix} 5 & -2 \\ -2 & 4 \end{bmatrix}} = {\begin{bmatrix} 5/16 & -2/16 \\ -2/16 & 4/16 \end{bmatrix}} = {\begin{bmatrix} 0.3125 & -0.125 \\ -0.125 & 0.25 \end{bmatrix}}$$

**Step 3: Substitute into the PDF formula**

Now we can write the full PDF:

$$f(x_1, x_2) = \frac{1}{2\pi \sqrt{16}} \exp\left(-\frac{1}{2}\left[\begin{pmatrix} x_1 - 2 \\ x_2 - 3 \end{pmatrix}^T \begin{pmatrix} 0.3125 & -0.125 \\ -0.125 & 0.25 \end{pmatrix} \begin{pmatrix} x_1 - 2 \\ x_2 - 3 \end{pmatrix}\right]\right)$$

$$f(x_1, x_2) = \frac{1}{2\pi \cdot 4} \exp\left(-\frac{1}{2}\left[0.3125(x_1-2)^2 - 0.25(x_1-2)(x_2-3) + 0.25(x_2-3)^2\right]\right)$$

$$f(x_1, x_2) = \frac{1}{8\pi} \exp\left(-\frac{1}{2}\left[0.3125(x_1-2)^2 - 0.25(x_1-2)(x_2-3) + 0.25(x_2-3)^2\right]\right)$$

#### Part B: Calculating the Probability $P(X_1 \leq 3, X_2 \leq 4)$

To calculate the probability $P(X_1 \leq 3, X_2 \leq 4)$, we need to standardize the random variables and use properties of the multivariate normal distribution.

**Step 1: Define the region and standardize the bounds**

We need to find:
$$P(X_1 \leq 3, X_2 \leq 4)$$

Let's standardize the bounds by converting to Z-scores:

For $X_1 \leq 3$:
$$Z_1 = \frac{X_1 - \mu_1}{\sigma_1} = \frac{3 - 2}{\sqrt{4}} = \frac{1}{2} = 0.5$$

For $X_2 \leq 4$:
$$Z_2 = \frac{X_2 - \mu_2}{\sigma_2} = \frac{4 - 3}{\sqrt{5}} = \frac{1}{\sqrt{5}} \approx 0.4472$$

**Step 2: Account for the correlation**

Since $X_1$ and $X_2$ are correlated (with correlation coefficient $\rho$), we need to account for this.

The correlation coefficient is:
$$\rho = \frac{\sigma_{12}}{\sigma_1 \sigma_2} = \frac{2}{\sqrt{4} \cdot \sqrt{5}} = \frac{2}{2 \cdot \sqrt{5}} = \frac{1}{\sqrt{5}} \approx 0.4472$$

**Step 3: Calculate the probability using numeric methods**

For a bivariate normal distribution, this probability requires evaluating the bivariate normal CDF:

$$P(X_1 \leq 3, X_2 \leq 4) = \Phi_2(0.5, 0.4472; 0.4472)$$

Where $\Phi_2(a, b; \rho)$ is the CDF of the standard bivariate normal distribution up to points $a$ and $b$ with correlation $\rho$.

Using numerical computation, we find:
$$P(X_1 \leq 3, X_2 \leq 4) \approx 0.5264 \text{ or } 52.64\%$$

**Alternative Step 3: Use the bivariate normal probability formula**

We can also directly integrate the PDF over the region:

$$P(X_1 \leq 3, X_2 \leq 4) = \int_{-\infty}^{4}\int_{-\infty}^{3} f(x_1, x_2) dx_1 dx_2$$

Where $f(x_1, x_2)$ is the PDF we derived in Part A.

This requires numerical integration, and yields the same result:
$$P(X_1 \leq 3, X_2 \leq 4) \approx 0.5264 \text{ or } 52.64\%$$

#### Part C: Finding the Conditional Distribution

For a bivariate normal distribution, the conditional distribution of $X_1$ given $X_2 = x_2$ is also normal with the following parameters:

**Step 1: Calculate the conditional mean**

The formula for the conditional mean is:
$$\mu_{1|2} = \mu_1 + \frac{\sigma_{12}}{\sigma_2^2}(x_2 - \mu_2)$$

Where:
- $\mu_1 = 2$ (mean of $X_1$)
- $\mu_2 = 3$ (mean of $X_2$)
- $\sigma_{12} = 2$ (covariance between $X_1$ and $X_2$)
- $\sigma_2^2 = 5$ (variance of $X_2$)
- $x_2 = 4$ (the conditioning value)

Substituting the values:
$$\mu_{1|2} = 2 + \frac{2}{5}(4 - 3) = 2 + \frac{2}{5} \times 1 = 2 + 0.4 = 2.4$$

**Step 2: Calculate the conditional variance**

The formula for the conditional variance is:
$$\sigma_{1|2}^2 = \sigma_1^2 - \frac{\sigma_{12}^2}{\sigma_2^2}$$

Where:
- $\sigma_1^2 = 4$ (variance of $X_1$)
- $\sigma_{12} = 2$ (covariance between $X_1$ and $X_2$)
- $\sigma_2^2 = 5$ (variance of $X_2$)

Substituting the values:
$$\sigma_{1|2}^2 = 4 - \frac{2^2}{5} = 4 - \frac{4}{5} = 4 - 0.8 = 3.2$$

**Step 3: Write the conditional distribution**

Therefore, the conditional distribution is:
$$X_1 | (X_2 = 4) \sim \mathcal{N}(2.4, 3.2)$$

### Visualizations

![Bivariate Normal Density](../Images/Multivariate_Density_Examples/example1_bivariate_normal.png)

The contour plot shows the probability density levels, with the mean at $(2, 3)$ marked with a red dot. The contour lines represent equal probability density values. As we move away from the mean, the density decreases in an elliptical pattern defined by the covariance structure.

The 3D plot gives a better perspective of how the density peaks at the mean and falls off in all directions, with the shape determined by the covariance matrix.

![Probability Region for P(X₁≤3, X₂≤4)](../Images/Multivariate_Density_Examples/example1_probability_region.png)

This visualization shows the region where $X_1 \leq 3$ and $X_2 \leq 4$. The shaded area represents the probability we calculated (approximately 52.64%). The point (3,4) is marked, and the shaded region represents all values where both $X_1 \leq 3$ and $X_2 \leq 4$.

![Conditional Distribution](../Images/Multivariate_Density_Examples/example1_conditional_distribution.png)

Notice how the conditional distribution shifts to the right (from mean 2 to 2.4) and becomes narrower (from variance 4 to 3.2) compared to the marginal distribution. This reflects the positive correlation between $X_1$ and $X_2$: when we know $X_2 = 4$ (which is above its mean of 3), the expected value of $X_1$ also increases above its marginal mean.

## Example 2: Trivariate Normal Distribution

### Problem Statement
Consider a trivariate normal distribution with density function $f(x, y, z)$ where the mean vector is $\boldsymbol{\mu} = (1, 2, 3)$ and the covariance matrix is:

$$\boldsymbol{\Sigma} = \begin{bmatrix} 3 & 1 & -1 \\ 1 & 2 & 0 \\ -1 & 0 & 4 \end{bmatrix}$$

a) Find the marginal density function $f(x, y)$.
b) Find the conditional density function $f(z | x=2, y=1)$.

### Solution

#### Part A: Finding the Marginal Density Function

For multivariate normal distributions, the marginal distribution of any subset of variables is also multivariate normal. 

**Step 1: Extract the corresponding elements**

The marginal distribution of $(X, Y)$ is obtained by:
- Taking the corresponding elements of the mean vector: $\boldsymbol{\mu}_{X,Y} = (1, 2)$
- Taking the corresponding submatrix of the covariance matrix: $\boldsymbol{\Sigma}_{X,Y} = \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}$

**Step 2: Calculate the determinant of the submatrix**

$$|\boldsymbol{\Sigma}_{X,Y}| = 3 \times 2 - 1 \times 1 = 6 - 1 = 5$$

**Step 3: Calculate the inverse of the submatrix**

$$\boldsymbol{\Sigma}_{X,Y}^{-1} = \frac{1}{5} \begin{bmatrix} 2 & -1 \\ -1 & 3 \end{bmatrix} = \begin{bmatrix} 0.4 & -0.2 \\ -0.2 & 0.6 \end{bmatrix}$$

**Step 4: Write the marginal PDF**

Therefore, the marginal density function $f(x, y)$ is a bivariate normal distribution with mean $(1, 2)$ and covariance matrix $\begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}$:

$$f(x, y) = \frac{1}{2\pi \sqrt{5}} \exp\left(-\frac{1}{2}\left[\begin{pmatrix} x - 1 \\ y - 2 \end{pmatrix}^T \begin{pmatrix} 0.4 & -0.2 \\ -0.2 & 0.6 \end{pmatrix} \begin{pmatrix} x - 1 \\ y - 2 \end{pmatrix}\right]\right)$$

$$f(x, y) = \frac{1}{2\pi \sqrt{5}} \exp\left(-\frac{1}{2}\left[0.4(x-1)^2 - 0.4(x-1)(y-2) + 0.6(y-2)^2\right]\right)$$

The visualization below shows:
1. A contour plot of the marginal density function (top)
2. A 3D surface plot of the same function (bottom)

![Marginal Distribution](../Images/Multivariate_Density_Examples/example2_marginal_xy.png)

#### Part B: Finding the Conditional Density Function

To find the conditional distribution of $Z$ given $X=2$ and $Y=1$, we use a partitioning approach.

**Step 1: Partition the variables**

We split the variables into two groups:
- $\mathbf{X}_1 = Z$ (the variable of interest)
- $\mathbf{X}_2 = (X, Y)$ (the conditioning variables)

With this partition:
- $\boldsymbol{\mu}_1 = 3$ (mean of $Z$)
- $\boldsymbol{\mu}_2 = (1, 2)$ (mean of $(X, Y)$)
- $\boldsymbol{\Sigma}_{11} = 4$ (variance of $Z$)
- $\boldsymbol{\Sigma}_{12} = (-1, 0)$ (covariance between $Z$ and $(X, Y)$)
- $\boldsymbol{\Sigma}_{21} = \begin{bmatrix} -1 \\ 0 \end{bmatrix}$ (covariance between $(X, Y)$ and $Z$)
- $\boldsymbol{\Sigma}_{22} = \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}$ (covariance matrix of $(X, Y)$)

**Step 2: Calculate the conditional mean**

For conditional multivariate normal distributions, the conditional mean is:
$$\mu_{Z|X,Y} = \mu_Z + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

Given $X=2$ and $Y=1$, we have $\mathbf{x}_2 = (2, 1)$, so:
$$\mathbf{x}_2 - \boldsymbol{\mu}_2 = \begin{pmatrix} 2 - 1 \\ 1 - 2 \end{pmatrix} = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

Using the inverse of $\boldsymbol{\Sigma}_{22}$ which we calculated in Part A:
$$\boldsymbol{\Sigma}_{22}^{-1} = \begin{bmatrix} 0.4 & -0.2 \\ -0.2 & 0.6 \end{bmatrix}$$

We compute the term:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = (-1, 0) \begin{bmatrix} 0.4 & -0.2 \\ -0.2 & 0.6 \end{bmatrix} \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

$$= (-1, 0) \begin{pmatrix} 0.4 \times 1 + (-0.2) \times (-1) \\ (-0.2) \times 1 + 0.6 \times (-1) \end{pmatrix} = (-1, 0) \begin{pmatrix} 0.4 + 0.2 \\ -0.2 - 0.6 \end{pmatrix} = (-1, 0) \begin{pmatrix} 0.6 \\ -0.8 \end{pmatrix}$$

$$= -1 \times 0.6 + 0 \times (-0.8) = -0.6$$

Therefore, the conditional mean is:
$$\mu_{Z|X,Y} = 3 + (-0.6) = 2.4$$

**Step 3: Calculate the conditional variance**

For conditional multivariate normal distributions, the conditional variance is:
$$\sigma_{Z|X,Y}^2 = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

Computing this term:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = (-1, 0) \begin{bmatrix} 0.4 & -0.2 \\ -0.2 & 0.6 \end{bmatrix} \begin{pmatrix} -1 \\ 0 \end{pmatrix}$$

$$= (-1, 0) \begin{pmatrix} 0.4 \times (-1) + (-0.2) \times 0 \\ (-0.2) \times (-1) + 0.6 \times 0 \end{pmatrix} = (-1, 0) \begin{pmatrix} -0.4 \\ 0.2 \end{pmatrix}$$

$$= -1 \times (-0.4) + 0 \times 0.2 = 0.4$$

Therefore, the conditional variance is:
$$\sigma_{Z|X,Y}^2 = 4 - 0.4 = 3.6$$

**Step 4: Write the conditional distribution**

Therefore, the conditional distribution is:
$$Z | (X=2, Y=1) \sim \mathcal{N}(2.4, 3.6)$$

And the conditional density function is:
$$f(z | x=2, y=1) = \frac{1}{\sqrt{2\pi \cdot 3.6}} \exp\left(-\frac{(z-2.4)^2}{2 \cdot 3.6}\right)$$

The visualization below compares the marginal distribution of $Z$ with the conditional distribution:

![Conditional Distribution of Z](../Images/Multivariate_Density_Examples/example2_conditional_z_given_xy.png)

This plot shows how the conditional distribution of $Z$ (solid red line) differs from its marginal distribution (dashed blue line). The conditional mean shifts from 3 to 2.4, reflecting the negative correlation between $Z$ and $X$. The variance decreases from 4 to 3.6, showing how knowledge of $X$ and $Y$ reduces uncertainty about $Z$.

## Key Insights

1. **Marginal Distributions**: For multivariate normal distributions, any marginal distribution is also normal, obtained by extracting the corresponding means and covariance submatrix.

2. **Conditional Distributions**: Conditional distributions in multivariate normal settings are also normal, with means and variances adjusted according to the correlation structure.

3. **Correlation Effects**:
   - Positive correlation causes conditional means to shift in the same direction as the conditioning value
   - Negative correlation causes conditional means to shift in the opposite direction
   - Stronger correlation leads to larger shifts in conditional means
   - Conditioning always reduces variance (unless variables are independent)

4. **Visualization Insights**:
   - Contour plots reveal the orientation and shape of the distribution, determined by the covariance matrix
   - The principal axes of these ellipses correspond to the eigenvectors of the covariance matrix
   - 3D density plots show how probability mass concentrates around the mean

These insights about multivariate distributions are crucial for understanding many machine learning algorithms, especially those involving Gaussian processes, probabilistic graphical models, and Bayesian methods.

## Related Topics

- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: More examples of multivariate normal distributions
- [[L2_1_Joint_Distributions_Examples|Joint Distributions Examples]]: Working with joint probability distributions
- [[L2_1_Conditional_Probability_Examples|Conditional Probability Examples]]: Additional examples of conditional distributions
- [[L2_1_Expectation_Examples|Expectation Examples]]: Calculating expected values for multivariate distributions
- [[L2_1_Covariance_Examples|Covariance Examples]]: Understanding covariance in multivariate settings 