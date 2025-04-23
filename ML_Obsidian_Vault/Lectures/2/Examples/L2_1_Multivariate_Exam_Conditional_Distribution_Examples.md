# Conditional Distribution Examples

This document provides examples and key concepts on conditional distributions in multivariate settings, an essential concept in machine learning, statistics, and probability theory.

## Key Concepts and Formulas

Conditional distributions allow us to update our belief about some variables given information about other variables. For multivariate normal distributions, conditional distributions remain normal with updated parameters.

### Conditional Distribution Formula for Multivariate Normal

If $\mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{bmatrix}\right)$, then:

$$\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})$$

Where:
- $\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$
- $\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$

## Example 1: Conditional Distributions in Bivariate Normal

### Problem Statement
Consider a bivariate normal distribution where $\mathbf{X} = (X_1, X_2)$ has mean vector $\boldsymbol{\mu} = (3, 5)$ and covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 9 & 6 \\ 6 & 16 \end{bmatrix}$.

a) Find the conditional distribution of $X_1$ given $X_2 = 7$.
b) What is the best prediction for $X_1$ given $X_2 = 7$?
c) Calculate the reduction in variance when predicting $X_1$ after observing $X_2$.

### Solution

#### Part a: Finding the conditional distribution

For a bivariate normal, the conditional distribution of $X_1$ given $X_2 = x_2$ is also normal with:

$$\mu_{1|2} = \mu_1 + \frac{\sigma_{12}}{\sigma_2^2}(x_2 - \mu_2)$$

$$\sigma_{1|2}^2 = \sigma_1^2 - \frac{\sigma_{12}^2}{\sigma_2^2}$$

With the given parameters:
- $\mu_1 = 3$
- $\mu_2 = 5$
- $\sigma_1^2 = 9$
- $\sigma_2^2 = 16$
- $\sigma_{12} = 6$

Substituting these values:

$$\mu_{1|2} = 3 + \frac{6}{16}(7 - 5) = 3 + \frac{6}{16} \cdot 2 = 3 + 0.75 = 3.75$$

$$\sigma_{1|2}^2 = 9 - \frac{6^2}{16} = 9 - \frac{36}{16} = 9 - 2.25 = 6.75$$

Therefore, the conditional distribution is:
$$X_1 | (X_2 = 7) \sim \mathcal{N}(3.75, 6.75)$$

#### Part b: Best prediction for $X_1$

The best prediction for $X_1$ given $X_2 = 7$ is the conditional mean:
$$E[X_1 | X_2 = 7] = \mu_{1|2} = 3.75$$

#### Part c: Reduction in variance

The unconditional variance of $X_1$ is $\sigma_1^2 = 9$.
The conditional variance after observing $X_2$ is $\sigma_{1|2}^2 = 6.75$.

The reduction in variance is:
$$\sigma_1^2 - \sigma_{1|2}^2 = 9 - 6.75 = 2.25$$

This represents a reduction of $\frac{2.25}{9} \times 100\% = 25\%$ in the variance, indicating that knowledge of $X_2$ reduces our uncertainty about $X_1$ by 25%.

## Example 2: Conditional Distributions and Inference in Trivariate Normal

### Problem Statement
Consider a trivariate normal random vector $\mathbf{X} = (X_1, X_2, X_3)$ with mean vector $\boldsymbol{\mu} = (5, 7, 10)$ and covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 4 & 2 & 1 \\ 2 & 9 & 3 \\ 1 & 3 & 5 \end{bmatrix}$.

a) Find the conditional distribution of $X_1$ given $X_2 = 8$ and $X_3 = 11$.
b) If we only observe $X_2 = 8$ (but not $X_3$), what is our best prediction for $X_1$?
c) Calculate the reduction in variance of our prediction of $X_1$ when we observe both $X_2$ and $X_3$ compared to observing only $X_2$.

### Solution

#### Part a: Finding the conditional distribution of $X_1$ given $X_2 = 8$ and $X_3 = 11$

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

$$\boldsymbol{\Sigma}_{1|2} = 4 - (2, 1) \begin{pmatrix} 10/36 - 1/12 \\ -2/12 + 1/4 \end{pmatrix}$$

$$\boldsymbol{\Sigma}_{1|2} = 4 - (2, 1) \begin{pmatrix} 10/36 - 3/36 \\ -6/36 + 9/36 \end{pmatrix}$$

$$\boldsymbol{\Sigma}_{1|2} = 4 - (2, 1) \begin{pmatrix} 7/36 \\ 3/36 \end{pmatrix}$$

$$\boldsymbol{\Sigma}_{1|2} = 4 - (2 \cdot 7/36 + 1 \cdot 3/36) = 4 - (14/36 + 3/36) = 4 - 17/36 \approx 3.53$$

Therefore, the conditional distribution is:
$$X_1 | (X_2 = 8, X_3 = 11) \sim \mathcal{N}(5.28, 3.53)$$

#### Part b: Best prediction for $X_1$ given only $X_2 = 8$

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

#### Part c: Reduction in variance

The conditional variance when we observe only $X_2$ is:
$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = 4 - 2 \cdot \frac{1}{9} \cdot 2 = 4 - \frac{4}{9} \approx 3.56$$

The conditional variance when we observe both $X_2$ and $X_3$ was calculated in part (a) as $\boldsymbol{\Sigma}_{1|2,3} \approx 3.53$.

The reduction in variance is:
$$\boldsymbol{\Sigma}_{1|2} - \boldsymbol{\Sigma}_{1|2,3} \approx 3.56 - 3.53 = 0.03$$

This small reduction in variance indicates that knowing $X_3$ in addition to $X_2$ provides relatively little additional information about $X_1$. This makes sense given the covariance structure, where $X_1$ has a stronger correlation with $X_2$ (covariance of 2) than with $X_3$ (covariance of 1).

## Example 3: Prediction and Conditional Inference

### Problem Statement
A researcher has developed a model to predict a student's final exam score ($Y$) based on their midterm score ($X_1$) and homework average ($X_2$). Based on historical data, the variables follow a multivariate normal distribution with:
$$\boldsymbol{\mu} = \begin{bmatrix} 75 \\ 80 \\ 78 \end{bmatrix}$$

$$\boldsymbol{\Sigma} = \begin{bmatrix} 
100 & 60 & 70 \\
60 & 64 & 48 \\
70 & 48 & 81
\end{bmatrix}$$

A new student scored 85 on the midterm and has a homework average of 90.

a) What is the predicted final exam score for this student?
b) What is the 95% prediction interval for this student's final exam score?
c) How much of the variance in final exam scores can be explained by knowing both the midterm score and homework average?

### Solution

#### Part a: Predicted final exam score

We need to find the conditional distribution of $Y$ given $X_1 = 85$ and $X_2 = 90$.

Partitioning the parameters:
- $\boldsymbol{\mu}_1 = 78$ (mean of $Y$)
- $\boldsymbol{\mu}_2 = (75, 80)$ (mean of $(X_1, X_2)$)
- $\boldsymbol{\Sigma}_{11} = 81$ (variance of $Y$)
- $\boldsymbol{\Sigma}_{12} = (70, 48)$ (covariance between $Y$ and $(X_1, X_2)$)
- $\boldsymbol{\Sigma}_{21} = (70, 48)^T$ (transpose of $\boldsymbol{\Sigma}_{12}$)
- $\boldsymbol{\Sigma}_{22} = \begin{bmatrix} 100 & 60 \\ 60 & 64 \end{bmatrix}$ (covariance matrix of $(X_1, X_2)$)

First, we need to find $\boldsymbol{\Sigma}_{22}^{-1}$:
$$|\boldsymbol{\Sigma}_{22}| = 100 \times 64 - 60 \times 60 = 6400 - 3600 = 2800$$

$$\boldsymbol{\Sigma}_{22}^{-1} = \frac{1}{2800} \begin{bmatrix} 64 & -60 \\ -60 & 100 \end{bmatrix} = \begin{bmatrix} 0.0229 & -0.0214 \\ -0.0214 & 0.0357 \end{bmatrix}$$

The conditional mean (predicted final exam score) is:
$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

$$\boldsymbol{\mu}_{1|2} = 78 + (70, 48) \begin{bmatrix} 0.0229 & -0.0214 \\ -0.0214 & 0.0357 \end{bmatrix} \begin{pmatrix} 85 - 75 \\ 90 - 80 \end{pmatrix}$$

$$\boldsymbol{\mu}_{1|2} = 78 + (70, 48) \begin{bmatrix} 0.0229 & -0.0214 \\ -0.0214 & 0.0357 \end{bmatrix} \begin{pmatrix} 10 \\ 10 \end{pmatrix}$$

Computing the matrix multiplication:
$$\begin{bmatrix} 0.0229 & -0.0214 \\ -0.0214 & 0.0357 \end{bmatrix} \begin{pmatrix} 10 \\ 10 \end{pmatrix} = \begin{pmatrix} 0.229 - 0.214 \\ -0.214 + 0.357 \end{pmatrix} = \begin{pmatrix} 0.015 \\ 0.143 \end{pmatrix}$$

Continuing:
$$\boldsymbol{\mu}_{1|2} = 78 + (70, 48) \begin{pmatrix} 0.015 \\ 0.143 \end{pmatrix} = 78 + 70(0.015) + 48(0.143) = 78 + 1.05 + 6.864 = 85.91$$

Therefore, the predicted final exam score for this student is approximately 85.91.

#### Part b: 95% prediction interval

To find the prediction interval, we need the conditional variance:
$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

$$\boldsymbol{\Sigma}_{1|2} = 81 - (70, 48) \begin{bmatrix} 0.0229 & -0.0214 \\ -0.0214 & 0.0357 \end{bmatrix} \begin{pmatrix} 70 \\ 48 \end{pmatrix}$$

Computing the inner matrix multiplication:
$$\begin{bmatrix} 0.0229 & -0.0214 \\ -0.0214 & 0.0357 \end{bmatrix} \begin{pmatrix} 70 \\ 48 \end{pmatrix} = \begin{pmatrix} 0.0229(70) - 0.0214(48) \\ -0.0214(70) + 0.0357(48) \end{pmatrix} = \begin{pmatrix} 1.603 - 1.0272 \\ -1.498 + 1.7136 \end{pmatrix} = \begin{pmatrix} 0.5758 \\ 0.2156 \end{pmatrix}$$

Continuing:
$$\boldsymbol{\Sigma}_{1|2} = 81 - (70, 48) \begin{pmatrix} 0.5758 \\ 0.2156 \end{pmatrix} = 81 - (70(0.5758) + 48(0.2156)) = 81 - (40.306 + 10.3488) = 81 - 50.6548 = 30.3452$$

For a 95% prediction interval, we use the conditional standard deviation $\sigma_{1|2} = \sqrt{30.3452} \approx 5.51$ and a normal multiplier of 1.96:

$$\text{Prediction Interval} = \mu_{1|2} \pm 1.96 \times \sigma_{1|2} = 85.91 \pm 1.96 \times 5.51 = 85.91 \pm 10.80 = [75.11, 96.71]$$

Therefore, with 95% confidence, the student's final exam score will be between 75.11 and 96.71.

#### Part c: Variance explained

The total variance in final exam scores is $\boldsymbol{\Sigma}_{11} = 81$.
The residual variance after conditioning on midterm and homework is $\boldsymbol{\Sigma}_{1|2} = 30.3452$.

The proportion of variance explained is:
$$1 - \frac{\boldsymbol{\Sigma}_{1|2}}{\boldsymbol{\Sigma}_{11}} = 1 - \frac{30.3452}{81} = 1 - 0.3746 = 0.6254$$

Therefore, approximately 62.54% of the variance in final exam scores can be explained by knowing both the midterm score and homework average.

## Related Topics

- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: More examples on multivariate normal distributions
- [[L2_1_Joint_Distributions_Examples|Joint Distributions Examples]]: Working with joint probability distributions
- [[L2_1_Conditional_Probability_Examples|Conditional Probability Examples]]: Basic concepts of conditional probability
- [[L2_1_Covariance_Examples|Covariance Examples]]: Understanding covariance structures
- [[L2_1_Regression_Examples|Regression Examples]]: Applications of conditional distributions in regression 