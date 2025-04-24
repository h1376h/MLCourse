# Conditional Distribution Examples

This document provides examples and key concepts on conditional distributions in multivariate settings, an essential concept in machine learning, statistics, and probability theory.

## Key Concepts and Formulas

Conditional distributions allow us to update our belief about some variables given information about other variables. For multivariate normal distributions, conditional distributions remain normal with updated parameters.

### Conditional Distribution Formula for Multivariate Normal

If we have a partitioned multivariate normal random vector:

$$\mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{bmatrix}\right)$$

Then the conditional distribution of $\mathbf{X}_1$ given $\mathbf{X}_2 = \mathbf{x}_2$ follows a normal distribution:

$$\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})$$

Where:
- $\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$ (Conditional mean)
- $\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$ (Conditional covariance)

For the special case of bivariate normal distribution (when $\mathbf{X}_1$ and $\mathbf{X}_2$ are scalar random variables), these formulas simplify to:

$$\mu_{1|2} = \mu_1 + \frac{\sigma_{12}}{\sigma_{22}}(x_2 - \mu_2)$$

$$\sigma_{1|2}^2 = \sigma_{11} - \frac{\sigma_{12}^2}{\sigma_{22}}$$

Where $\frac{\sigma_{12}}{\sigma_{22}}$ represents the regression coefficient of $X_1$ on $X_2$.

### Key Properties of Conditional Distributions

1. **Linearity**: The conditional mean $\boldsymbol{\mu}_{1|2}$ is a linear function of the conditioning variables $\mathbf{x}_2$.

2. **Variance Reduction**: The conditional variance $\boldsymbol{\Sigma}_{1|2}$ is always less than or equal to the unconditional variance $\boldsymbol{\Sigma}_{11}$. This reflects the reduction in uncertainty when we have additional information.

3. **Independence**: If variables are independent (i.e., $\boldsymbol{\Sigma}_{12} = \mathbf{0}$), then conditioning has no effect on the distribution:
   - $\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1$ (the conditional mean equals the unconditional mean)
   - $\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11}$ (the conditional variance equals the unconditional variance)

4. **Normality Preservation**: For multivariate normal distributions, conditional distributions remain normal. This is not true for arbitrary multivariate distributions.

5. **Regression Interpretation**: The conditional mean formula can be interpreted as a regression of $\mathbf{X}_1$ on $\mathbf{X}_2$. The term $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}$ represents the matrix of regression coefficients.

## Example 1: Conditional Distributions in Bivariate Normal

### Problem Statement
Consider a bivariate normal distribution where $\mathbf{X} = (X_1, X_2)$ has mean vector $\boldsymbol{\mu} = (3, 5)$ and covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 9 & 6 \\ 6 & 16 \end{bmatrix}$.

a) Find the conditional distribution of $X_1$ given $X_2 = 7$.
b) What is the best prediction for $X_1$ given $X_2 = 7$?
c) Calculate the reduction in variance when predicting $X_1$ after observing $X_2$.

### Solution

Let's begin by calculating the correlation coefficient to understand the relationship between $X_1$ and $X_2$:

$$\rho = \frac{\sigma_{12}}{\sqrt{\sigma_{11}\sigma_{22}}} = \frac{6}{\sqrt{9 \times 16}}$$

Simplifying step by step:
$$\rho = \frac{6}{\sqrt{144}} = \frac{6}{12} = 0.5$$

This indicates a moderately positive relationship between $X_1$ and $X_2$, with correlation coefficient $\rho = 0.5$.

#### Part a: Finding the conditional distribution

To find the conditional distribution, we need to apply the conditional distribution formulas for multivariate normal distributions.

**Step 1:** Identify the parameters from the given bivariate normal distribution:
- $\mu_1 = 3$ (mean of $X_1$)
- $\mu_2 = 5$ (mean of $X_2$)
- $\sigma_{11} = 9$ (variance of $X_1$)
- $\sigma_{12} = \sigma_{21} = 6$ (covariance between $X_1$ and $X_2$)
- $\sigma_{22} = 16$ (variance of $X_2$)

**Step 2:** Calculate the conditional mean using the formula:
$$\mu_{1|2} = \mu_1 + \frac{\sigma_{12}}{\sigma_{22}}(x_2 - \mu_2)$$

First, compute the regression coefficient $\frac{\sigma_{12}}{\sigma_{22}}$:
$$\frac{\sigma_{12}}{\sigma_{22}} = \frac{6}{16} = \frac{3}{8} = 0.375$$

This coefficient tells us that, on average, $X_1$ increases by 0.375 units for every 1-unit increase in $X_2$.

Now, substituting the values for the conditional mean:
$$\mu_{1|2} = 3 + \frac{6}{16}(7 - 5)$$

Calculate the deviation from the mean of $X_2$:
$$7 - 5 = 2$$

Multiply the deviation by the regression coefficient:
$$\frac{6}{16} \times 2 = 0.375 \times 2 = 0.75$$

Add this adjustment to the mean of $X_1$:
$$\mu_{1|2} = 3 + 0.75 = 3.75$$

**Step 3:** Calculate the conditional variance using the formula:
$$\sigma_{1|2}^2 = \sigma_{11} - \frac{\sigma_{12}^2}{\sigma_{22}}$$

First, compute the squared covariance:
$$\sigma_{12}^2 = 6^2 = 36$$

Next, calculate the reduction in variance $\frac{\sigma_{12}^2}{\sigma_{22}}$:
$$\frac{\sigma_{12}^2}{\sigma_{22}} = \frac{36}{16} = \frac{9}{4} = 2.25$$

Now, subtract this reduction from the unconditional variance:
$$\sigma_{1|2}^2 = 9 - 2.25 = 6.75$$

**Step 4:** Calculate the conditional standard deviation:
$$\sigma_{1|2} = \sqrt{\sigma_{1|2}^2} = \sqrt{6.75} \approx 2.598 \approx 2.60$$

Therefore, the conditional distribution is:
$$X_1 | (X_2 = 7) \sim \mathcal{N}(3.75, 6.75)$$

This means that given $X_2 = 7$, the random variable $X_1$ follows a normal distribution with mean 3.75 and variance 6.75.

#### Part b: Best prediction for $X_1$

The best prediction for $X_1$ given $X_2 = 7$ is the conditional mean, which minimizes the expected squared prediction error:

$$E[X_1 | X_2 = 7] = \mu_{1|2} = 3.75$$

**Verification:** We can verify this is indeed the best predictor by noting that for a random variable $Y$ with mean $\mu$, the value of $c$ that minimizes $E[(Y-c)^2]$ is $c = \mu$. 

To prove this:
$$E[(Y-c)^2] = E[Y^2] - 2cE[Y] + c^2$$

To find the value of $c$ that minimizes this expression, we take the derivative with respect to $c$ and set it to zero:
$$\frac{d}{dc}E[(Y-c)^2] = -2E[Y] + 2c = 0$$

Solving for $c$:
$$c = E[Y]$$

In our case, $Y = X_1|X_2=7$ has mean $\mu_{1|2} = 3.75$, so the best predictor is 3.75.

#### Part c: Reduction in variance

To calculate the reduction in variance, we compare the unconditional (marginal) variance of $X_1$ with the conditional variance after observing $X_2$.

**Step 1:** Identify the unconditional variance of $X_1$:
$$\sigma_{11} = 9$$

**Step 2:** Note the conditional variance after observing $X_2$ (calculated in part a):
$$\sigma_{1|2}^2 = 6.75$$

**Step 3:** Calculate the absolute reduction in variance:
$$\sigma_{11} - \sigma_{1|2}^2 = 9 - 6.75 = 2.25$$

**Step 4:** Calculate the percentage reduction in variance:
$$\frac{\sigma_{11} - \sigma_{1|2}^2}{\sigma_{11}} \times 100\% = \frac{2.25}{9} \times 100\% = 25\%$$

This 25% reduction in variance indicates that knowing $X_2$ reduces our uncertainty about $X_1$ by one-quarter.

**Step 5:** Verify the relationship with correlation coefficient:

We can also express this reduction in terms of the correlation coefficient:
$$\frac{\sigma_{12}^2}{\sigma_{11}\sigma_{22}} = \rho^2 = 0.5^2 = 0.25 = 25\%$$

Let's verify this calculation:
$$\frac{\sigma_{12}^2}{\sigma_{11}\sigma_{22}} = \frac{36}{9 \times 16} = \frac{36}{144} = \frac{1}{4} = 0.25 = 25\%$$

This confirms an important property: the percentage reduction in variance equals the square of the correlation coefficient, which is a general property for bivariate normal distributions.

**Step 6:** Derive the general regression equation for predicting $X_1$ from any value of $X_2$:
$$E[X_1|X_2=x_2] = \mu_1 + \frac{\sigma_{12}}{\sigma_{22}}(x_2 - \mu_2)$$

Substituting our specific values:
$$E[X_1|X_2=x_2] = 3 + 0.375 \times (x_2 - 5)$$

Distributing the coefficient:
$$E[X_1|X_2=x_2] = 3 + 0.375x_2 - 0.375 \times 5$$
$$E[X_1|X_2=x_2] = 1.125 + 0.375 \times x_2$$

This linear equation gives us the expected value of $X_1$ for any observed value of $X_2$.

#### Key Insights

1. The conditional distribution of $X_1$ given $X_2 = 7$ has mean 3.75, which is higher than the unconditional mean of $X_1$ (which is 3). This makes sense because $X_2 = 7$ is higher than its mean of 5, and the variables are positively correlated.

2. The conditional variance (6.75) is reduced by 25% compared to the unconditional variance (9), reflecting the information gained from knowing $X_2$.

3. The regression line shows how the expected value of $X_1$ changes linearly with $X_2$, with slope equal to $\frac{\sigma_{12}}{\sigma_{22}} = 0.375$.

4. The regression coefficient 0.375 tells us that, on average, a one-unit increase in $X_2$ corresponds to a 0.375-unit increase in the expected value of $X_1$.

5. The formula for conditional variance shows that uncertainty decreases as the magnitude of correlation increases, which aligns with our intuition that stronger relationships between variables lead to better predictions.

6. The squared correlation coefficient ($\rho^2 = 0.25$) represents the proportion of variance in $X_1$ that can be explained by $X_2$.

## Example 2: Conditional Distributions and Inference in Trivariate Normal

### Problem Statement
Consider a trivariate normal random vector $\mathbf{X} = (X_1, X_2, X_3)$ with mean vector $\boldsymbol{\mu} = (5, 7, 10)$ and covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 4 & 2 & 1 \\ 2 & 9 & 3 \\ 1 & 3 & 5 \end{bmatrix}$.

a) Find the conditional distribution of $X_1$ given $X_2 = 8$ and $X_3 = 11$.
b) If we only observe $X_2 = 8$ (but not $X_3$), what is our best prediction for $X_1$?
c) Calculate the reduction in variance of our prediction of $X_1$ when we observe both $X_2$ and $X_3$ compared to observing only $X_2$.

### Solution

Let's first compute the correlation matrix to better understand the relationships between variables:

**Step 0:** Compute the correlation matrix from the covariance matrix.

The correlation between variables $i$ and $j$ is:
$$\rho_{ij} = \frac{\sigma_{ij}}{\sqrt{\sigma_{ii}\sigma_{jj}}}$$

Calculating each correlation:

For $X_1$ and $X_2$:
$$\rho_{12} = \frac{\sigma_{12}}{\sqrt{\sigma_{11}\sigma_{22}}} = \frac{2}{\sqrt{4 \times 9}}$$
$$= \frac{2}{\sqrt{36}} = \frac{2}{6} = \frac{1}{3} \approx 0.333$$

For $X_1$ and $X_3$:
$$\rho_{13} = \frac{\sigma_{13}}{\sqrt{\sigma_{11}\sigma_{33}}} = \frac{1}{\sqrt{4 \times 5}}$$
$$= \frac{1}{\sqrt{20}} = \frac{1}{\sqrt{20}} \approx \frac{1}{4.47} \approx 0.224$$

For $X_2$ and $X_3$:
$$\rho_{23} = \frac{\sigma_{23}}{\sqrt{\sigma_{22}\sigma_{33}}} = \frac{3}{\sqrt{9 \times 5}}$$
$$= \frac{3}{\sqrt{45}} = \frac{3}{\sqrt{45}} \approx \frac{3}{6.71} \approx 0.447$$

Therefore, the correlation matrix is:
$$\mathbf{R} = \begin{bmatrix} 
1 & 0.333 & 0.224 \\
0.333 & 1 & 0.447 \\
0.224 & 0.447 & 1
\end{bmatrix}$$

From the correlation matrix, we can see that $X_1$ has a stronger correlation with $X_2$ (0.333) than with $X_3$ (0.224). This suggests that $X_2$ might be more informative for predicting $X_1$ than $X_3$.

#### Part a: Finding the conditional distribution of $X_1$ given $X_2 = 8$ and $X_3 = 11$

When conditioning on multiple variables, we need to use the general matrix form of the conditional distribution formula.

**Step 1:** Partition the parameters for conditional distribution:
- $\mathbf{X}_1 = X_1$ (the variable of interest)
- $\mathbf{X}_2 = (X_2, X_3)$ (the conditioning variables)

With this partition:
- $\boldsymbol{\mu}_1 = 5$ (mean of $X_1$)
- $\boldsymbol{\mu}_2 = \begin{bmatrix} 7 \\ 10 \end{bmatrix}$ (mean of $(X_2, X_3)$)
- $\boldsymbol{\Sigma}_{11} = 4$ (variance of $X_1$)
- $\boldsymbol{\Sigma}_{12} = \begin{bmatrix} 2 & 1 \end{bmatrix}$ (covariance between $X_1$ and $(X_2, X_3)$)
- $\boldsymbol{\Sigma}_{21} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$ (transpose of $\boldsymbol{\Sigma}_{12}$)
- $\boldsymbol{\Sigma}_{22} = \begin{bmatrix} 9 & 3 \\ 3 & 5 \end{bmatrix}$ (covariance matrix of $(X_2, X_3)$)

**Step 2:** Calculate $\boldsymbol{\Sigma}_{22}^{-1}$ using the formula for 2×2 matrix inversion:

For a 2×2 matrix $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse is:
$$A^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

First, find the determinant of $\boldsymbol{\Sigma}_{22}$:
$$|\boldsymbol{\Sigma}_{22}| = 9 \times 5 - 3 \times 3 = 45 - 9 = 36$$

Then, find the adjugate matrix:
$$\text{adj}(\boldsymbol{\Sigma}_{22}) = \begin{bmatrix} 5 & -3 \\ -3 & 9 \end{bmatrix}$$

Finally, compute the inverse by dividing each element of the adjugate matrix by the determinant:
$$\boldsymbol{\Sigma}_{22}^{-1} = \frac{1}{36} \begin{bmatrix} 5 & -3 \\ -3 & 9 \end{bmatrix}$$

Let's calculate each element precisely:
$$\boldsymbol{\Sigma}_{22}^{-1}[1,1] = \frac{5}{36} = 0.1389$$
$$\boldsymbol{\Sigma}_{22}^{-1}[1,2] = \frac{-3}{36} = -0.0833$$
$$\boldsymbol{\Sigma}_{22}^{-1}[2,1] = \frac{-3}{36} = -0.0833$$
$$\boldsymbol{\Sigma}_{22}^{-1}[2,2] = \frac{9}{36} = 0.2500$$

Therefore:
$$\boldsymbol{\Sigma}_{22}^{-1} = \begin{bmatrix} 0.1389 & -0.0833 \\ -0.0833 & 0.2500 \end{bmatrix}$$

**Step 3:** Calculate $(\mathbf{x}_2 - \boldsymbol{\mu}_2)$, the deviation of observed values from their means:
$$\mathbf{x}_2 - \boldsymbol{\mu}_2 = \begin{bmatrix} 8 \\ 11 \end{bmatrix} - \begin{bmatrix} 7 \\ 10 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

**Step 4:** Calculate the conditional mean $\boldsymbol{\mu}_{1|2}$ using the formula:
$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

First, calculate $\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$ - this is a matrix-vector multiplication:
$$\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = \begin{bmatrix} 0.1389 & -0.0833 \\ -0.0833 & 0.2500 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

Let's compute this multiplication in detail:
$$\begin{bmatrix} (0.1389 \times 1) + (-0.0833 \times 1) \\ (-0.0833 \times 1) + (0.2500 \times 1) \end{bmatrix}$$

Calculate the first element:
$$(0.1389 \times 1) + (-0.0833 \times 1) = 0.1389 - 0.0833 = 0.0556$$

Calculate the second element:
$$(-0.0833 \times 1) + (0.2500 \times 1) = -0.0833 + 0.2500 = 0.1667$$

Therefore:
$$\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = \begin{bmatrix} 0.0556 \\ 0.1667 \end{bmatrix}$$

Next, calculate $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$ - this is a vector-vector multiplication:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = \begin{bmatrix} 2 & 1 \end{bmatrix} \begin{bmatrix} 0.0556 \\ 0.1667 \end{bmatrix}$$

Let's compute this multiplication in detail:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = (2 \times 0.0556) + (1 \times 0.1667)$$
$$= 0.1112 + 0.1667 = 0.2779 \approx 0.2778$$

Finally, calculate the complete expression for the conditional mean:
$$\boldsymbol{\mu}_{1|2} = 5 + 0.2778 = 5.2778$$

**Step 5:** Calculate the conditional variance $\boldsymbol{\Sigma}_{1|2}$ using the formula:
$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

First, calculate $\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$ - this is a matrix-vector multiplication:
$$\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = \begin{bmatrix} 0.1389 & -0.0833 \\ -0.0833 & 0.2500 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

Let's compute this multiplication in detail:
$$\begin{bmatrix} (0.1389 \times 2) + (-0.0833 \times 1) \\ (-0.0833 \times 2) + (0.2500 \times 1) \end{bmatrix}$$

Calculate the first element:
$$(0.1389 \times 2) + (-0.0833 \times 1) = 0.2778 - 0.0833 = 0.1945 \approx 0.1944$$

Calculate the second element:
$$(-0.0833 \times 2) + (0.2500 \times 1) = -0.1666 + 0.2500 = 0.0834 \approx 0.0833$$

Therefore:
$$\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = \begin{bmatrix} 0.1944 \\ 0.0833 \end{bmatrix}$$

Next, calculate $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$ - this is a vector-vector multiplication:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = \begin{bmatrix} 2 & 1 \end{bmatrix} \begin{bmatrix} 0.1944 \\ 0.0833 \end{bmatrix}$$

Let's compute this multiplication in detail:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = (2 \times 0.1944) + (1 \times 0.0833)$$
$$= 0.3888 + 0.0833 = 0.4721 \approx 0.4722$$

Finally, calculate the complete expression for the conditional variance:
$$\boldsymbol{\Sigma}_{1|2} = 4 - 0.4722 = 3.5278$$

**Step 6:** Calculate the conditional standard deviation:
$$\sigma_{1|2} = \sqrt{\boldsymbol{\Sigma}_{1|2}} = \sqrt{3.5278} \approx 1.8782$$

Therefore, the conditional distribution is:
$$X_1 | (X_2 = 8, X_3 = 11) \sim \mathcal{N}(5.2778, 3.5278)$$

#### Part b: Best prediction for $X_1$ given only $X_2 = 8$

When we observe only $X_2 = 8$, we can use a simpler bivariate conditioning approach (similar to Example 1).

**Step 1:** Extract the relevant parameters for this bivariate case:
- $\mu_1 = 5$ (mean of $X_1$)
- $\mu_2 = 7$ (mean of $X_2$)
- $\sigma_{11} = 4$ (variance of $X_1$)
- $\sigma_{12} = \sigma_{21} = 2$ (covariance between $X_1$ and $X_2$)
- $\sigma_{22} = 9$ (variance of $X_2$)

**Step 2:** Calculate the regression coefficient of $X_1$ on $X_2$:
$$\beta_{1|2} = \frac{\sigma_{12}}{\sigma_{22}} = \frac{2}{9} \approx 0.2222$$

**Step 3:** Calculate the conditional mean using the bivariate formula:
$$\mu_{1|2} = \mu_1 + \beta_{1|2}(x_2 - \mu_2)$$

Substituting the values:
$$\mu_{1|2} = 5 + \frac{2}{9}(8 - 7)$$

Calculate the deviation from the mean:
$$8 - 7 = 1$$

Multiply by the regression coefficient:
$$\frac{2}{9} \times 1 = 0.2222$$

Add to the mean of $X_1$:
$$\mu_{1|2} = 5 + 0.2222 = 5.2222$$

**Step 4:** Calculate the conditional variance using the bivariate formula:
$$\sigma_{1|2}^2 = \sigma_{11} - \frac{\sigma_{12}^2}{\sigma_{22}}$$

Calculate the squared covariance:
$$\sigma_{12}^2 = 2^2 = 4$$

Calculate the reduction in variance:
$$\frac{\sigma_{12}^2}{\sigma_{22}} = \frac{4}{9} = 0.4444$$

Calculate the conditional variance:
$$\sigma_{1|2}^2 = 4 - 0.4444 = 3.5556$$

**Step 5:** Calculate the conditional standard deviation:
$$\sigma_{1|2} = \sqrt{\sigma_{1|2}^2} = \sqrt{3.5556} \approx 1.8856$$

Therefore, the conditional distribution considering only $X_2$ is:
$$X_1 | (X_2 = 8) \sim \mathcal{N}(5.2222, 3.5556)$$

The best prediction of $X_1$ given only $X_2 = 8$ is:
$$E[X_1 | X_2 = 8] = 5.2222$$

#### Part c: Reduction in variance

Let's analyze the reduction in variance step by step:

**Step 1:** Identify the three variance values:
1. Unconditional variance of $X_1$: $\sigma_{11} = 4$
2. Variance when conditioning on $X_2$ only: $\sigma_{1|2}^2 = 3.5556$
3. Variance when conditioning on both $X_2$ and $X_3$: $\sigma_{1|2,3}^2 = 3.5278$

**Step 2:** Calculate the absolute reduction from unconditional to conditioning on $X_2$:
$$\Delta\sigma^2_1 = \sigma_{11} - \sigma_{1|2}^2 = 4 - 3.5556 = 0.4444$$

**Step 3:** Calculate the percentage reduction:
$$\frac{\Delta\sigma^2_1}{\sigma_{11}} \times 100\% = \frac{0.4444}{4} \times 100\% = 11.11\%$$

**Step 4:** Calculate the additional absolute reduction when also conditioning on $X_3$:
$$\Delta\sigma^2_2 = \sigma_{1|2}^2 - \sigma_{1|2,3}^2 = 3.5556 - 3.5278 = 0.0278$$

**Step 5:** Calculate the percentage additional reduction relative to the variance after conditioning on $X_2$:
$$\frac{\Delta\sigma^2_2}{\sigma_{1|2}^2} \times 100\% = \frac{0.0278}{3.5556} \times 100\% = 0.78\%$$

**Step 6:** Calculate the total absolute reduction from unconditional to conditioning on both $X_2$ and $X_3$:
$$\Delta\sigma^2_{total} = \sigma_{11} - \sigma_{1|2,3}^2 = 4 - 3.5278 = 0.4722$$

**Step 7:** Calculate the percentage total reduction:
$$\frac{\Delta\sigma^2_{total}}{\sigma_{11}} \times 100\% = \frac{0.4722}{4} \times 100\% = 11.81\%$$

This small additional reduction in variance (only 0.78%) indicates that knowing $X_3$ in addition to $X_2$ provides relatively little additional information about $X_1$. This makes sense given the covariance structure, where $X_1$ has a stronger correlation with $X_2$ (covariance of 2) than with $X_3$ (covariance of 1).

**Additional insight:** We can compute the squared multiple correlation coefficient, which represents the proportion of variance explained by the conditioning variables:

$$R^2 = \frac{\sigma_{11} - \sigma_{1|2,3}^2}{\sigma_{11}} = \frac{0.4722}{4} = 0.1181$$

This means that knowing both $X_2$ and $X_3$ explains approximately 11.81% of the variability in $X_1$.

We can also relate this to the partial correlation concept. The small additional variance reduction suggests that the partial correlation between $X_1$ and $X_3$ given $X_2$ is small.

#### Key Insights

1. When conditioning on multiple variables, the calculations become more complex as we need to work with matrices rather than scalar values. The general formula allows us to account for the joint effect of multiple conditioning variables.

2. The conditional mean of $X_1$ given both $X_2 = 8$ and $X_3 = 11$ (5.2778) is very close to the conditional mean of $X_1$ given only $X_2 = 8$ (5.2222). This suggests that $X_3$ provides minimal additional predictive power once we know $X_2$.

3. The variance reduction from additionally conditioning on $X_3$ (after already conditioning on $X_2$) is very small (0.78%), confirming that $X_3$ adds little marginal information beyond what $X_2$ already provides for predicting $X_1$.

4. This example illustrates an important principle in statistical modeling: sometimes adding more predictors may not substantially improve prediction accuracy, especially if the additional variables are not strongly correlated with the target variable or are correlated with predictors already in the model.

5. The concept of partial correlation is relevant here - the small additional variance reduction suggests that the partial correlation between $X_1$ and $X_3$ given $X_2$ is small.

6. In regression analysis, this phenomenon is related to the concept of multicollinearity, where predictor variables are correlated. When predictors are correlated, adding more variables may yield diminishing returns in terms of predictive power.

7. We can see this mathematically by examining the formula for conditional variance reduction:
   $$\boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$
   This represents the variance explained by the conditioning variables. When conditioning variables are highly correlated with each other, the additional explained variance becomes smaller with each added variable.

## Example 3: Prediction and Conditional Inference

### Problem Statement
A professor wants to predict a student's final exam score based on their midterm and homework scores. From historical data, the professor knows that the scores follow a trivariate normal distribution with the following parameters:

- Mean vector: $\boldsymbol{\mu} = (82, 78, 85)$ (Final, Midterm, Homework)
- Covariance matrix: $\boldsymbol{\Sigma} = \begin{bmatrix} 100 & 60 & 40 \\ 60 & 64 & 30 \\ 40 & 30 & 25 \end{bmatrix}$

If a student scores 85 on the midterm and 90 on homework, what is the predicted final exam score? Provide a 95% prediction interval for the student's final exam score.

### Solution

Step 0: Analyze the correlation structure between scores

Let's first calculate the correlation matrix to understand the relationships between the three scores:

- $\rho_{\text{Final,Midterm}} = \frac{\sigma_{\text{FM}}}{\sqrt{\sigma_{\text{FF}}\sigma_{\text{MM}}}} = \frac{60}{\sqrt{100 \times 64}} = \frac{60}{\sqrt{6400}} = \frac{60}{80} = 0.75$

- $\rho_{\text{Final,Homework}} = \frac{\sigma_{\text{FH}}}{\sqrt{\sigma_{\text{FF}}\sigma_{\text{HH}}}} = \frac{40}{\sqrt{100 \times 25}} = \frac{40}{\sqrt{2500}} = \frac{40}{50} = 0.80$

- $\rho_{\text{Midterm,Homework}} = \frac{\sigma_{\text{MH}}}{\sqrt{\sigma_{\text{MM}}\sigma_{\text{HH}}}} = \frac{30}{\sqrt{64 \times 25}} = \frac{30}{\sqrt{1600}} = \frac{30}{40} = 0.75$

From these correlations, we can see that:
- Final exam has a strong positive correlation of 0.75 with midterm scores
- Final exam has an even stronger correlation of 0.80 with homework scores
- Midterm has a strong correlation of 0.75 with homework scores

These strong positive correlations suggest that both midterm and homework scores are good predictors of the final exam score. The slightly higher correlation between final exam and homework suggests that homework scores might be marginally more predictive of final exam performance.

#### Partitioning the Variables for Conditional Distribution

To find the conditional distribution of the final exam score given both the midterm and homework scores, we need to partition our variables:

- $\mathbf{X}_1 = \text{Final}$ (the variable we want to predict)
- $\mathbf{X}_2 = (\text{Midterm}, \text{Homework})$ (the conditioning variables)

With this partition:
- $\boldsymbol{\mu}_1 = 82$ (mean of $X_1$)
- $\boldsymbol{\mu}_2 = (78, 85)$ (mean of midterm and homework scores)
- $\boldsymbol{\Sigma}_{11} = 100$ (variance of $X_1$)
- $\boldsymbol{\Sigma}_{12} = (60, 40)$ (covariance between final and (midterm, homework))
- $\boldsymbol{\Sigma}_{21} = (60, 40)^T$ (transpose of $\boldsymbol{\Sigma}_{12}$)
- $\boldsymbol{\Sigma}_{22} = \begin{bmatrix} 64 & 30 \\ 30 & 25 \end{bmatrix}$ (covariance matrix of midterm and homework)

#### Step 1: Calculate $\boldsymbol{\Sigma}_{22}^{-1}$ (inverse of the covariance matrix of conditioning variables)

First, find the determinant of $\boldsymbol{\Sigma}_{22}$:
$$|\boldsymbol{\Sigma}_{22}| = 64 \times 25 - 30 \times 30 = 1600 - 900 = 700$$

Next, find the adjugate matrix:
$$\text{adj}(\boldsymbol{\Sigma}_{22}) = \begin{bmatrix} 25 & -30 \\ -30 & 64 \end{bmatrix}$$

Now, compute the inverse using the formula $A^{-1} = \frac{1}{|A|}\text{adj}(A)$:
$$\boldsymbol{\Sigma}_{22}^{-1} = \frac{1}{700} \begin{bmatrix} 25 & -30 \\ -30 & 64 \end{bmatrix}$$

Let's calculate each element precisely:
$$\boldsymbol{\Sigma}_{22}^{-1}(1,1) = \frac{25}{700} = 0.0357$$
$$\boldsymbol{\Sigma}_{22}^{-1}(1,2) = \frac{-30}{700} = -0.0429$$
$$\boldsymbol{\Sigma}_{22}^{-1}(2,1) = \frac{-30}{700} = -0.0429$$
$$\boldsymbol{\Sigma}_{22}^{-1}(2,2) = \frac{64}{700} = 0.0914$$

Therefore:
$$\boldsymbol{\Sigma}_{22}^{-1} = \begin{bmatrix} 0.0357 & -0.0429 \\ -0.0429 & 0.0914 \end{bmatrix}$$

#### Step 2: Calculate $(\mathbf{x}_2 - \boldsymbol{\mu}_2)$, the deviation of observed scores from their means

For the student with midterm = 85 and homework = 90:
$$\mathbf{x}_2 - \boldsymbol{\mu}_2 = \begin{bmatrix} 85 \\ 90 \end{bmatrix} - \begin{bmatrix} 78 \\ 85 \end{bmatrix} = \begin{bmatrix} 7 \\ 5 \end{bmatrix}$$

The student scored 7 points above the mean on the midterm and 5 points above the mean on the homework.

#### Step 3: Calculate the conditional mean (predicted final exam score)

We apply the formula for the conditional mean:
$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

First, we calculate $\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$ through matrix-vector multiplication:
$$\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = \begin{bmatrix} 0.0357 & -0.0429 \\ -0.0429 & 0.0914 \end{bmatrix} \begin{bmatrix} 7 \\ 5 \end{bmatrix}$$

Breaking this down step by step:
$$\begin{bmatrix} (0.0357 \times 7) + (-0.0429 \times 5) \\ (-0.0429 \times 7) + (0.0914 \times 5) \end{bmatrix}$$
$$= \begin{bmatrix} 0.2499 - 0.2145 \\ -0.3003 + 0.4570 \end{bmatrix}$$
$$= \begin{bmatrix} 0.0354 \\ 0.1567 \end{bmatrix}$$

Next, we calculate $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$ through vector-vector multiplication:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = \begin{bmatrix} 60 & 40 \end{bmatrix} \begin{bmatrix} 0.0354 \\ 0.1567 \end{bmatrix}$$

Computing this product:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = (60 \times 0.0354) + (40 \times 0.1567)$$
$$= 2.124 + 6.268 = 8.392$$

Now we can calculate the conditional mean:
$$\boldsymbol{\mu}_{1|2} = 82 + 8.392 = 90.392 \approx 90.4$$

Let's verify this calculation more carefully:
$$\boldsymbol{\mu}_{1|2} = 82 + \begin{bmatrix} 60 & 40 \end{bmatrix} \begin{bmatrix} 0.0357 & -0.0429 \\ -0.0429 & 0.0914 \end{bmatrix} \begin{bmatrix} 7 \\ 5 \end{bmatrix}$$

Calculating the matrix product first:
$$\begin{bmatrix} 60 & 40 \end{bmatrix} \begin{bmatrix} 0.0357 & -0.0429 \\ -0.0429 & 0.0914 \end{bmatrix} = \begin{bmatrix} (60 \times 0.0357) + (40 \times -0.0429) & (60 \times -0.0429) + (40 \times 0.0914) \end{bmatrix}$$
$$= \begin{bmatrix} 2.142 - 1.716 & -2.574 + 3.656 \end{bmatrix}$$
$$= \begin{bmatrix} 0.426 & 1.082 \end{bmatrix}$$

Now multiplying by the vector:
$$\begin{bmatrix} 0.426 & 1.082 \end{bmatrix} \begin{bmatrix} 7 \\ 5 \end{bmatrix} = (0.426 \times 7) + (1.082 \times 5) = 2.982 + 5.41 = 8.392$$

So our final result is:
$$\boldsymbol{\mu}_{1|2} = 82 + 8.392 = 90.392 \approx 90.4$$

#### Step 4: Calculate the conditional variance

We apply the formula for the conditional variance:
$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

First, we calculate $\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$ through matrix-vector multiplication:
$$\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = \begin{bmatrix} 0.0357 & -0.0429 \\ -0.0429 & 0.0914 \end{bmatrix} \begin{bmatrix} 60 \\ 40 \end{bmatrix}$$

Breaking this down step by step:
$$\begin{bmatrix} (0.0357 \times 60) + (-0.0429 \times 40) \\ (-0.0429 \times 60) + (0.0914 \times 40) \end{bmatrix}$$
$$= \begin{bmatrix} 2.142 - 1.716 \\ -2.574 + 3.656 \end{bmatrix}$$
$$= \begin{bmatrix} 0.426 & 1.082 \end{bmatrix}$$

Next, we calculate $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$ through vector-vector multiplication:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = \begin{bmatrix} 60 & 40 \end{bmatrix} \begin{bmatrix} 0.426 & 1.082 \end{bmatrix}$$

Computing this product:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = (60 \times 0.426) + (40 \times 1.082) = 25.56 + 43.28 = 68.84$$

Now we can calculate the conditional variance:
$$\boldsymbol{\Sigma}_{1|2} = 100 - 68.84 = 31.16$$

And the conditional standard deviation:
$$\sigma_{1|2} = \sqrt{\boldsymbol{\Sigma}_{1|2}} = \sqrt{31.16} \approx 5.58$$

Therefore, the conditional distribution of the final exam score is:
$$\text{Final} | (\text{Midterm} = 85, \text{Homework} = 90) \sim \mathcal{N}(90.4, 31.16)$$

#### Step 5: Calculate a 95% prediction interval

For a 95% prediction interval, we need to find the values that encompass the middle 95% of the conditional distribution. For a normal distribution, this corresponds to approximately ±1.96 standard deviations from the mean.

The formula for the prediction interval is:
$$[\mu_{1|2} - 1.96\sigma_{1|2}, \mu_{1|2} + 1.96\sigma_{1|2}]$$

Let's substitute our calculated values:
$$[90.4 - 1.96 \times 5.58, 90.4 + 1.96 \times 5.58]$$
$$[90.4 - 10.94, 90.4 + 10.94]$$
$$[79.46, 101.34]$$

Rounded to two decimal places:
$$[79.46, 101.34]$$

Therefore, we are 95% confident that the student's final exam score will be between approximately 79.5 and 101.3.

Note: This is a prediction interval, not a confidence interval. A prediction interval accounts for both the uncertainty in estimating the mean and the inherent variability of individual observations, which is why it's wider than a confidence interval would be for the same level of confidence.

#### Step 6: Analysis of variance explained

The variance of the final exam score is reduced from the marginal variance of 100 to the conditional variance of 31.16. This represents a reduction in variance of:
$$\Delta\sigma^2 = 100 - 31.16 = 68.84$$

The proportion of variance explained (or coefficient of determination, $R^2$) is:
$$R^2 = \frac{\Delta\sigma^2}{\sigma_{11}} = \frac{68.84}{100} \times 100\% = 68.84\%$$

This means that knowing the midterm and homework scores explains approximately 68.84% of the variability in the final exam scores.

We can also compute the multiple correlation coefficient as:
$$R = \sqrt{R^2} = \sqrt{0.6884} \approx 0.8297$$

This high multiple correlation coefficient (0.83) indicates a strong linear relationship between the final exam score and the combination of midterm and homework scores.

#### Regression Equation for Prediction

We can express the prediction as a multiple regression equation:
$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$$

Where:
- $\hat{y}$ is the predicted final exam score
- $x_1$ is the midterm score
- $x_2$ is the homework score

From our calculation of $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}$ above, we found:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1} = \begin{bmatrix} 0.426 & 1.082 \end{bmatrix}$$

These are our regression coefficients $\beta_1 = 0.426$ and $\beta_2 = 1.082$.

The intercept $\beta_0$ can be calculated as:
$$\beta_0 = \mu_1 - \beta_1 \mu_{21} - \beta_2 \mu_{22} = 82 - (0.426 \times 78) - (1.082 \times 85)$$
$$= 82 - 33.228 - 91.97 = -43.198$$

So our regression equation is:
$$\hat{y} = -43.198 + 0.426 \times \text{midterm} + 1.082 \times \text{homework}$$

Plugging in our student's scores:
$$\hat{y} = -43.198 + 0.426 \times 85 + 1.082 \times 90$$
$$= -43.198 + 36.21 + 97.38 = 90.392 \approx 90.4$$

This matches our earlier calculation using the conditional mean formula.

#### Key Insights

1. **Predictive Power**: Both midterm and homework scores are good predictors of the final exam score, with high correlations (0.75 and 0.80, respectively). The regression coefficients suggest that homework scores (β₂ = 1.082) have a slightly larger impact on final exam prediction than midterm scores (β₁ = 0.426).

2. **Precision of Prediction**: The conditional variance (31.16) is significantly lower than the marginal variance (100), indicating that knowing the midterm and homework scores substantially reduces uncertainty in predicting the final exam score.

3. **Variance Explained**: Approximately 68.84% of the variability in final exam scores can be explained by midterm and homework scores. This high percentage indicates that these two predictors are capturing most of the relevant information for predicting final exam performance.

4. **Prediction Interval**: The 95% prediction interval [79.46, 101.34] provides a range within which we can be reasonably confident the student's actual final exam score will fall. This interval accounts for both the uncertainty in our estimate of the mean and the inherent variability in exam performance.

5. **Conditional Expectation**: The predicted final exam score (90.4) is higher than the population mean (82) because the student performed above average on both the midterm and homework. This demonstrates how conditional expectations adjust based on observed information.

6. **Application to Real-World Prediction**: This example illustrates how multivariate normal theory can be applied to practical prediction problems, such as educational assessment. It provides a principled way to make predictions and quantify uncertainty in those predictions.

7. **Relationship to Multiple Regression**: The conditional distribution approach gives the same point predictions as traditional multiple regression, but also provides a complete predictive distribution, allowing for proper quantification of uncertainty through prediction intervals.

## Running the Examples

You can run the code that generates these examples and visualizations using:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Codes/1_conditional_distributions_examples.py
```

## Related Topics

- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: More examples on multivariate normal distributions
- [[L2_1_Joint_Distributions_Examples|Joint Distributions Examples]]: Working with joint probability distributions
- [[L2_1_Conditional_Probability_Examples|Conditional Probability Examples]]: Basic concepts of conditional probability
- [[L2_1_Covariance_Examples|Covariance Examples]]: Understanding covariance structures