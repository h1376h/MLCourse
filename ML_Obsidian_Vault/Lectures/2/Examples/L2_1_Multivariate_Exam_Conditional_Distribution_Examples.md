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

First, let's identify the parameters from the given bivariate normal distribution:
- $\mu_1 = 3$ (mean of $X_1$)
- $\mu_2 = 5$ (mean of $X_2$)
- $\sigma_{11} = 9$ (variance of $X_1$)
- $\sigma_{12} = \sigma_{21} = 6$ (covariance between $X_1$ and $X_2$)
- $\sigma_{22} = 16$ (variance of $X_2$)
- $\rho = \frac{\sigma_{12}}{\sqrt{\sigma_{11}\sigma_{22}}} = \frac{6}{\sqrt{9 \times 16}} = 0.5$ (correlation coefficient)

For a bivariate normal, the conditional distribution of $X_1$ given $X_2 = x_2$ is also normal with:

$$\mu_{1|2} = \mu_1 + \frac{\sigma_{12}}{\sigma_{22}}(x_2 - \mu_2)$$

Substituting these values for the conditional mean:
$$\mu_{1|2} = 3 + \frac{6}{16}(7 - 5)$$
$$\mu_{1|2} = 3 + \frac{6}{16} \times 2$$
$$\mu_{1|2} = 3 + 0.375 \times 2$$
$$\mu_{1|2} = 3 + 0.75$$
$$\mu_{1|2} = 3.75$$

Next, we calculate the conditional variance:
$$\sigma_{1|2}^2 = \sigma_{11} - \frac{\sigma_{12}^2}{\sigma_{22}}$$
$$\sigma_{1|2}^2 = 9 - \frac{6^2}{16}$$
$$\sigma_{1|2}^2 = 9 - \frac{36}{16}$$
$$\sigma_{1|2}^2 = 9 - 2.25$$
$$\sigma_{1|2}^2 = 6.75$$
$$\sigma_{1|2} = \sqrt{6.75} \approx 2.60$$

Therefore, the conditional distribution is:
$$X_1 | (X_2 = 7) \sim \mathcal{N}(3.75, 6.75)$$

#### Part b: Best prediction for $X_1$

The best prediction for $X_1$ given $X_2 = 7$ is the conditional mean:
$$E[X_1 | X_2 = 7] = \mu_{1|2} = 3.75$$

#### Part c: Reduction in variance

The unconditional variance of $X_1$ is $\sigma_{11} = 9$.
The conditional variance after observing $X_2$ is $\sigma_{1|2}^2 = 6.75$.

The reduction in variance is:
$$\sigma_{11} - \sigma_{1|2}^2 = 9 - 6.75 = 2.25$$

This represents a reduction of $\frac{2.25}{9} \times 100\% = 25\%$ in the variance, indicating that knowledge of $X_2$ reduces our uncertainty about $X_1$ by 25%.

The regression equation for predicting $X_1$ from any value of $X_2$ is:
$$E[X_1|X_2=x_2] = 3 + 0.375 \times (x_2 - 5)$$
$$E[X_1|X_2=x_2] = 1.125 + 0.375 \times x_2$$

![Bivariate Normal Distribution](../Images/Conditional_Distributions/example1_bivariate_normal.png)
![Conditional Distribution of X1 given X2=7](../Images/Conditional_Distributions/example1_conditional_x2_7.png)
![Prediction of X1 based on X2](../Images/Conditional_Distributions/example1_prediction_interval.png)

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
- $\boldsymbol{\mu}_1 = 5$ (mean of $X_1$)
- $\boldsymbol{\mu}_2 = (7, 10)$ (mean of $(X_2, X_3)$)
- $\boldsymbol{\Sigma}_{11} = 4$ (variance of $X_1$)
- $\boldsymbol{\Sigma}_{12} = (2, 1)$ (covariance between $X_1$ and $(X_2, X_3)$)
- $\boldsymbol{\Sigma}_{21} = (2, 1)^T$ (transpose of $\boldsymbol{\Sigma}_{12}$)
- $\boldsymbol{\Sigma}_{22} = \begin{bmatrix} 9 & 3 \\ 3 & 5 \end{bmatrix}$ (covariance matrix of $(X_2, X_3)$)

Step 1: Calculate $\boldsymbol{\Sigma}_{22}^{-1}$

First, we find the determinant of $\boldsymbol{\Sigma}_{22}$:
$$|\boldsymbol{\Sigma}_{22}| = 9 \times 5 - 3 \times 3 = 45 - 9 = 36$$

Then, we find the adjugate matrix:
$$\text{adj}(\boldsymbol{\Sigma}_{22}) = \begin{bmatrix} 5 & -3 \\ -3 & 9 \end{bmatrix}$$

Finally, we compute the inverse:
$$\boldsymbol{\Sigma}_{22}^{-1} = \frac{1}{36} \begin{bmatrix} 5 & -3 \\ -3 & 9 \end{bmatrix} = \begin{bmatrix} 0.1389 & -0.0833 \\ -0.0833 & 0.2500 \end{bmatrix}$$

Step 2: Calculate $(\mathbf{x}_2 - \boldsymbol{\mu}_2)$
$$\mathbf{x}_2 - \boldsymbol{\mu}_2 = \begin{pmatrix} 8 \\ 11 \end{pmatrix} - \begin{pmatrix} 7 \\ 10 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

Step 3: Calculate the conditional mean $\boldsymbol{\mu}_{1|2}$
$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

First, calculate $\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$:
$$\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = \begin{bmatrix} 0.1389 & -0.0833 \\ -0.0833 & 0.2500 \end{bmatrix} \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 0.0556 \\ 0.1667 \end{pmatrix}$$

Next, calculate $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = \begin{pmatrix} 2 & 1 \end{pmatrix} \begin{pmatrix} 0.0556 \\ 0.1667 \end{pmatrix}$$
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = 2 \times 0.0556 + 1 \times 0.1667 = 0.1111 + 0.1667 = 0.2778$$

Finally, calculate the complete expression:
$$\boldsymbol{\mu}_{1|2} = 5 + 0.2778 = 5.2778$$

Step 4: Calculate the conditional variance $\boldsymbol{\Sigma}_{1|2}$
$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

First, calculate $\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$:
$$\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = \begin{bmatrix} 0.1389 & -0.0833 \\ -0.0833 & 0.2500 \end{bmatrix} \begin{pmatrix} 2 \\ 1 \end{pmatrix} = \begin{pmatrix} 0.1944 \\ 0.0833 \end{pmatrix}$$

Next, calculate $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = \begin{pmatrix} 2 & 1 \end{pmatrix} \begin{pmatrix} 0.1944 \\ 0.0833 \end{pmatrix}$$
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = 2 \times 0.1944 + 1 \times 0.0833 = 0.3889 + 0.0833 = 0.4722$$

Finally, calculate the complete expression:
$$\boldsymbol{\Sigma}_{1|2} = 4 - 0.4722 = 3.5278$$
$$\sigma_{1|2} = \sqrt{3.5278} \approx 1.8782$$

Therefore, the conditional distribution is:
$$X_1 | (X_2 = 8, X_3 = 11) \sim \mathcal{N}(5.2778, 3.5278)$$

#### Part b: Best prediction for $X_1$ given only $X_2 = 8$

When we observe only $X_2 = 8$, we use a simpler bivariate conditioning:

- $\mu_1 = 5$ (mean of $X_1$)
- $\mu_2 = 7$ (mean of $X_2$)
- $\sigma_{11} = 4$ (variance of $X_1$)
- $\sigma_{12} = \sigma_{21} = 2$ (covariance between $X_1$ and $X_2$)
- $\sigma_{22} = 9$ (variance of $X_2$)

The conditional mean is:
$$\mu_{1|2} = \mu_1 + \frac{\sigma_{12}}{\sigma_{22}}(x_2 - \mu_2)$$
$$\mu_{1|2} = 5 + \frac{2}{9}(8 - 7)$$
$$\mu_{1|2} = 5 + \frac{2}{9} \times 1$$
$$\mu_{1|2} = 5 + 0.2222$$
$$\mu_{1|2} = 5.2222$$

The conditional variance is:
$$\sigma_{1|2}^2 = \sigma_{11} - \frac{\sigma_{12}^2}{\sigma_{22}}$$
$$\sigma_{1|2}^2 = 4 - \frac{2^2}{9}$$
$$\sigma_{1|2}^2 = 4 - \frac{4}{9}$$
$$\sigma_{1|2}^2 = 4 - 0.4444$$
$$\sigma_{1|2}^2 = 3.5556$$
$$\sigma_{1|2} = \sqrt{3.5556} \approx 1.8856$$

Therefore, the conditional distribution considering only $X_2$ is:
$$X_1 | (X_2 = 8) \sim \mathcal{N}(5.2222, 3.5556)$$

The best prediction of $X_1$ given only $X_2 = 8$ is:
$$E[X_1 | X_2 = 8] = 5.2222$$

#### Part c: Reduction in variance

Let's analyze the reduction in variance step by step:

1. Unconditional variance of $X_1$: $\sigma_{11} = 4$
2. Variance when conditioning on $X_2$ only: $\sigma_{1|2}^2 = 3.5556$
3. Variance when conditioning on both $X_2$ and $X_3$: $\sigma_{1|2,3}^2 = 3.5278$

Reduction from unconditional to conditioning on $X_2$:
$$4 - 3.5556 = 0.4444$$
Percentage reduction: $\frac{0.4444}{4} \times 100\% = 11.11\%$

Additional reduction when also conditioning on $X_3$:
$$3.5556 - 3.5278 = 0.0278$$
Percentage additional reduction: $\frac{0.0278}{3.5556} \times 100\% = 0.78\%$

Total reduction from unconditional to conditioning on both $X_2$ and $X_3$:
$$4 - 3.5278 = 0.4722$$
Percentage total reduction: $\frac{0.4722}{4} \times 100\% = 11.81\%$

This small additional reduction in variance (only 0.78%) indicates that knowing $X_3$ in addition to $X_2$ provides relatively little additional information about $X_1$. This makes sense given the covariance structure, where $X_1$ has a stronger correlation with $X_2$ (covariance of 2) than with $X_3$ (covariance of 1).

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

First, we partition the parameters:
- $\boldsymbol{\mu}_1 = 78$ (mean of final exam, $Y$)
- $\boldsymbol{\mu}_2 = (75, 80)$ (mean of midterm and homework, $(X_1, X_2)$)
- $\boldsymbol{\Sigma}_{11} = 81$ (variance of final exam)
- $\boldsymbol{\Sigma}_{12} = (70, 48)$ (covariance between final exam and (midterm, homework))
- $\boldsymbol{\Sigma}_{21} = (70, 48)^T$ (transpose of $\boldsymbol{\Sigma}_{12}$)
- $\boldsymbol{\Sigma}_{22} = \begin{bmatrix} 100 & 60 \\ 60 & 64 \end{bmatrix}$ (covariance matrix of midterm and homework)

Step 1: Calculate $\boldsymbol{\Sigma}_{22}^{-1}$

First, we find the determinant of $\boldsymbol{\Sigma}_{22}$:
$$|\boldsymbol{\Sigma}_{22}| = 100 \times 64 - 60 \times 60 = 6400 - 3600 = 2800$$

Then, we find the adjugate matrix:
$$\text{adj}(\boldsymbol{\Sigma}_{22}) = \begin{bmatrix} 64 & -60 \\ -60 & 100 \end{bmatrix}$$

Finally, we compute the inverse:
$$\boldsymbol{\Sigma}_{22}^{-1} = \frac{1}{2800} \begin{bmatrix} 64 & -60 \\ -60 & 100 \end{bmatrix} = \begin{bmatrix} 0.0229 & -0.0214 \\ -0.0214 & 0.0357 \end{bmatrix}$$

Step 2: Calculate $(\mathbf{x}_2 - \boldsymbol{\mu}_2)$
$$\mathbf{x}_2 - \boldsymbol{\mu}_2 = \begin{pmatrix} 85 \\ 90 \end{pmatrix} - \begin{pmatrix} 75 \\ 80 \end{pmatrix} = \begin{pmatrix} 10 \\ 10 \end{pmatrix}$$

Step 3: Calculate the conditional mean (predicted final exam score)
$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

First, calculate $\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$:
$$\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = \begin{bmatrix} 0.0229 & -0.0214 \\ -0.0214 & 0.0357 \end{bmatrix} \begin{pmatrix} 10 \\ 10 \end{pmatrix}$$

First row calculation: $0.0229 \times 10 + (-0.0214) \times 10 = 0.2286 + (-0.2143) = 0.0143$
Second row calculation: $(-0.0214) \times 10 + 0.0357 \times 10 = (-0.2143) + 0.3571 = 0.1429$

$$\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = \begin{pmatrix} 0.0143 \\ 0.1429 \end{pmatrix}$$

Next, calculate $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = \begin{pmatrix} 70 & 48 \end{pmatrix} \begin{pmatrix} 0.0143 \\ 0.1429 \end{pmatrix}$$
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) = 70 \times 0.0143 + 48 \times 0.1429 = 1.0000 + 6.8571 = 7.8571$$

Finally, calculate the complete expression:
$$\boldsymbol{\mu}_{1|2} = 78 + 7.8571 = 85.8571$$

Therefore, the predicted final exam score for this student is approximately 85.86.

#### Part b: 95% prediction interval

To find the prediction interval, we need the conditional variance:
$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

First, calculate $\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$:
$$\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = \begin{bmatrix} 0.0229 & -0.0214 \\ -0.0214 & 0.0357 \end{bmatrix} \begin{pmatrix} 70 \\ 48 \end{pmatrix}$$

First row calculation: $0.0229 \times 70 + (-0.0214) \times 48 = 1.6000 + (-1.0286) = 0.5714$
Second row calculation: $(-0.0214) \times 70 + 0.0357 \times 48 = (-1.5000) + 1.7143 = 0.2143$

$$\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = \begin{pmatrix} 0.5714 \\ 0.2143 \end{pmatrix}$$

Next, calculate $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$:
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = \begin{pmatrix} 70 & 48 \end{pmatrix} \begin{pmatrix} 0.5714 \\ 0.2143 \end{pmatrix}$$
$$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} = 70 \times 0.5714 + 48 \times 0.2143 = 40.0000 + 10.2857 = 50.2857$$

Finally, calculate the conditional variance:
$$\boldsymbol{\Sigma}_{1|2} = 81 - 50.2857 = 30.7143$$
$$\sigma_{1|2} = \sqrt{30.7143} \approx 5.5420$$

For a 95% prediction interval, we use a normal multiplier of 1.96:

$$\text{Lower bound} = \mu_{1|2} - 1.96 \times \sigma_{1|2} = 85.8571 - 1.96 \times 5.5420 = 74.99$$
$$\text{Upper bound} = \mu_{1|2} + 1.96 \times \sigma_{1|2} = 85.8571 + 1.96 \times 5.5420 = 96.72$$

Therefore, with 95% confidence, the student's final exam score will be between 74.99 and 96.72.

#### Part c: Variance explained

The total variance in final exam scores is $\boldsymbol{\Sigma}_{11} = 81$.
The residual variance after conditioning on midterm and homework is $\boldsymbol{\Sigma}_{1|2} = 30.7143$.

The proportion of variance explained is:
$$1 - \frac{\boldsymbol{\Sigma}_{1|2}}{\boldsymbol{\Sigma}_{11}} = 1 - \frac{30.7143}{81} = 1 - 0.3792 = 0.6208$$

Therefore, approximately 62.08% of the variance in final exam scores can be explained by knowing both the midterm score and homework average.

![Student Exam Score Prediction](../Images/Conditional_Distributions/example3_student_prediction.png)
![Regression Plane for Student Scores](../Images/Conditional_Distributions/example3_regression_plane.png)

## Related Topics

- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: More examples on multivariate normal distributions
- [[L2_1_Joint_Distributions_Examples|Joint Distributions Examples]]: Working with joint probability distributions
- [[L2_1_Conditional_Probability_Examples|Conditional Probability Examples]]: Basic concepts of conditional probability
- [[L2_1_Covariance_Examples|Covariance Examples]]: Understanding covariance structures
- [[L2_1_Regression_Examples|Regression Examples]]: Applications of conditional distributions in regression 