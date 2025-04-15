# Question 6: MLE for Uniform Distribution

## Problem Statement
Suppose we have a dataset of n independent observations from a uniform distribution on the interval $[0, \theta]$, where $\theta$ is an unknown parameter. The probability density function is:

$$f(x|\theta) = \begin{cases} 
\frac{1}{\theta} & \text{for } 0 \leq x \leq \theta \\
0 & \text{otherwise}
\end{cases}$$

### Task
1. Find the maximum likelihood estimator $\hat{\theta}_{MLE}$ for $\theta$
2. Show that this estimator is biased
3. Propose an unbiased estimator based on $\hat{\theta}_{MLE}$

## Understanding the Probability Model

The uniform distribution is a continuous probability distribution where all values in a specified range are equally likely to occur. Some key properties include:

- All points in the range $[0, \theta]$ have equal probability density (1/θ)
- It has a rectangular shape with constant height
- The mean is θ/2 and the variance is θ²/12
- It represents maximum uncertainty over a bounded interval
- It's often used as a prior distribution in Bayesian analysis
- Common applications include random number generation, rounding errors, and modeling scenarios with equal probability over a range

## Solution

The uniform distribution is a continuous probability distribution where all values in a range are equally likely. In this problem, we're working with a uniform distribution over the interval $[0, \theta]$, where $\theta$ is an unknown parameter that represents the upper boundary of the distribution.

### Step 1: Formulate the likelihood function
For n independent observations $x_1, x_2, \ldots, x_n$, the likelihood function is:

$$L(\theta) = \prod_{i=1}^n f(x_i|\theta) = \prod_{i=1}^n \frac{1}{\theta} \cdot I(0 \leq x_i \leq \theta) = \frac{1}{\theta^n} \cdot \prod_{i=1}^n I(0 \leq x_i \leq \theta)$$

where $I(\cdot)$ is the indicator function that equals 1 if the condition is true and 0 otherwise.

Note that for the likelihood to be non-zero, all observations must be in the range $[0, \theta]$, which means $\theta \geq \max(x_1, x_2, \ldots, x_n)$.

### Step 2: Take the logarithm to get the log-likelihood
Taking the natural logarithm, we get the log-likelihood function:

$$\ell(\theta) = \log L(\theta) = -n \log \theta + \sum_{i=1}^n \log I(0 \leq x_i \leq \theta)$$

The second term is 0 if all observations are in $[0, \theta]$ and $-\infty$ otherwise.

### Step 3: Find the critical points by taking the derivative
Since the log-likelihood is a strictly decreasing function of $\theta$ when all observations are in $[0, \theta]$, the maximum is achieved at the smallest possible value of $\theta$ that ensures all observations are in the range. This is:

$$\hat{\theta}_{MLE} = \max(x_1, x_2, \ldots, x_n)$$

Therefore, the MLE for the uniform distribution parameter $\theta$ is the maximum of the observed data.

### Step 4: Verify it's a maximum
There's no need to take the second derivative in this case, as we've determined that the log-likelihood is a strictly decreasing function of θ when θ ≥ max(x₁, x₂, ..., xₙ), and it's equal to -∞ when θ < max(x₁, x₂, ..., xₙ). Therefore, the maximum must occur at exactly θ = max(x₁, x₂, ..., xₙ).

### Step 5: Show that this estimator is biased
To show bias, we need to calculate $E[\hat{\theta}_{MLE}]$ and show that it's not equal to $\theta$.

For a uniform distribution on $[0, \theta]$, the CDF of the maximum of n independent observations is:

$$F_{\max}(x) = P(\max(X_1, \ldots, X_n) \leq x) = \prod_{i=1}^n P(X_i \leq x) = \left(\frac{x}{\theta}\right)^n$$

for $0 \leq x \leq \theta$. The PDF is the derivative of the CDF:

$$f_{\max}(x) = \frac{n}{\theta} \left(\frac{x}{\theta}\right)^{n-1}$$

Now, the expected value is:

$$E[\hat{\theta}_{MLE}] = \int_0^{\theta} x \cdot f_{\max}(x) \, dx = \int_0^{\theta} x \cdot \frac{n}{\theta} \left(\frac{x}{\theta}\right)^{n-1} \, dx = \frac{n}{\theta^n} \int_0^{\theta} x^n \, dx = \frac{n}{\theta^n} \cdot \frac{\theta^{n+1}}{n+1} = \frac{n}{n+1} \cdot \theta$$

The bias of the estimator is:

$$\text{Bias}(\hat{\theta}_{MLE}) = E[\hat{\theta}_{MLE}] - \theta = \frac{n}{n+1} \cdot \theta - \theta = \theta \left(\frac{n}{n+1} - 1\right) = -\frac{\theta}{n+1}$$

Since the bias is negative, the MLE systematically underestimates the true parameter value. The bias approaches zero as the sample size n increases, which is a property of asymptotic unbiasedness.

### Step 6: Propose an unbiased estimator
Based on the calculated bias, we can derive an unbiased estimator by adjusting the MLE:

$$\hat{\theta}_{unbiased} = \frac{n+1}{n} \cdot \hat{\theta}_{MLE} = \frac{n+1}{n} \cdot \max(x_1, x_2, \ldots, x_n)$$

Let's verify that this is indeed unbiased:

$$E[\hat{\theta}_{unbiased}] = E\left[\frac{n+1}{n} \cdot \hat{\theta}_{MLE}\right] = \frac{n+1}{n} \cdot E[\hat{\theta}_{MLE}] = \frac{n+1}{n} \cdot \frac{n}{n+1} \cdot \theta = \theta$$

Therefore, $\hat{\theta}_{unbiased}$ is an unbiased estimator for $\theta$.

## Visual Explanations

### Uniform PDFs for Different θ Values
![Uniform PDFs](Images/L2_4_Quiz_6/uniform_pdfs.png)

This figure shows how the distribution shape changes with different θ values and illustrates the constant density within the range $[0, \theta]$.

### Likelihood Surface
![Likelihood Surface](Images/L2_4_Quiz_6/likelihood_surface.png)

This visualization of the log-likelihood function shows how it increases as θ decreases, until it hits the maximum of the data, followed by a sharp drop when θ is less than the maximum observation.

### MLE Fit to Data
![MLE Fit](Images/L2_4_Quiz_6/mle_fit.png)

This figure shows how well the MLE estimate fits the observed data and highlights the maximum observation as the MLE.

### Bias Demonstration
![Bias Demonstration](Images/L2_4_Quiz_6/bias_demonstration.png)

This visualization shows the distribution of MLE estimates for different sample sizes and illustrates how the MLE systematically underestimates the true parameter.

### Bias Curve
![Bias Curve](Images/L2_4_Quiz_6/bias_curve.png)

This figure compares the average MLE estimates with the unbiased estimator and shows how the unbiased estimator correctly centers on the true parameter value.

## Key Insights

### MLE Properties
- The MLE for the uniform distribution is the maximum of the observed data
- This is an intuitive result: if the data comes from a uniform distribution on $[0, \theta]$, then all observations must be less than or equal to θ
- The smallest valid value for θ is the maximum observation
- The MLE is consistent but biased

### Bias Analysis
- The MLE systematically underestimates the true parameter value
- The bias is $-\frac{\theta}{n+1}$, which decreases with increasing sample size
- The bias occurs because the true maximum of the distribution is almost always larger than the maximum observed in a finite sample

### Unbiased Estimation
- A simple adjustment of the MLE yields an unbiased estimator: $\frac{n+1}{n} \cdot \max(x_1, x_2, \ldots, x_n)$
- This estimator correctly accounts for the systematic underestimation in the MLE
- The adjustment factor approaches 1 as the sample size increases

## Conclusion

The maximum likelihood estimator for the uniform distribution parameter $\theta$ is:
- Straightforward to compute: it's simply the maximum of the observed data
- Biased: it systematically underestimates the true parameter
- Easily corrected: a simple scaling by $\frac{n+1}{n}$ yields an unbiased estimator

This example illustrates an important principle in statistical estimation: the maximum likelihood estimator is not always unbiased, but it often has other desirable properties such as consistency (convergence to the true parameter as sample size increases) and efficiency (achieving the Cramér-Rao lower bound asymptotically). 