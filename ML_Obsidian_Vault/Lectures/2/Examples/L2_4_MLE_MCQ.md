# Maximum Likelihood Estimation Multiple Choice Questions

This document provides practice multiple-choice questions on Maximum Likelihood Estimation (MLE) to test your understanding of key concepts.

## Questions

### Question 1
What does Maximum Likelihood Estimation (MLE) aim to find?

**Options:**
A) The parameter values that minimize the likelihood function
B) The parameter values that maximize the likelihood function
C) The parameter values that minimize the mean squared error
D) The parameter values that maximize the posterior probability

**Answer:** B

**Explanation:** Maximum Likelihood Estimation aims to find the parameter values that maximize the likelihood function, which represents how likely the observed data is under different parameter values.

### Question 2
Which of the following is the fundamental MLE formula?

**Options:**
A) $\hat{\theta}_{MLE} = \underset{\theta}{\operatorname{argmin}}\ p(D|\theta)$
B) $\hat{\theta}_{MLE} = \underset{\theta}{\operatorname{argmax}}\ p(\theta|D)$
C) $\hat{\theta}_{MLE} = \underset{\theta}{\operatorname{argmax}}\ p(D|\theta)$
D) $\hat{\theta}_{MLE} = \underset{\theta}{\operatorname{argmin}}\ p(\theta|D)$

**Answer:** C

**Explanation:** The fundamental MLE formula is $\hat{\theta}_{MLE} = \underset{\theta}{\operatorname{argmax}}\ p(D|\theta)$, where we find the parameter values that maximize the likelihood of observing the data given those parameters.

### Question 3
For computational convenience, MLE is often performed by maximizing which of the following?

**Options:**
A) The product of probability distributions
B) The posterior probability
C) The log-likelihood function
D) The prior distribution

**Answer:** C

**Explanation:** For computational convenience, MLE is often performed by maximizing the log-likelihood function instead of the likelihood function itself. This transforms products into sums, which are easier to work with mathematically.

### Question 4
For a normal distribution, what is the MLE estimate of the mean?

**Options:**
A) The sample median
B) The sample mean
C) The sample mode
D) The sample variance

**Answer:** B

**Explanation:** For a normal distribution, the MLE estimate of the mean is the sample mean: $\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}x_i$, which is the average of all observations.

### Question 5
For a normal distribution, what is the MLE estimate of the variance?

**Options:**
A) $\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu}_{MLE})^2$
B) $\hat{\sigma}^2_{MLE} = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \hat{\mu}_{MLE})^2$
C) $\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}x_i^2$
D) $\hat{\sigma}^2_{MLE} = \frac{1}{n-1}\sum_{i=1}^{n}x_i^2$

**Answer:** A

**Explanation:** For a normal distribution, the MLE estimate of the variance is $\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu}_{MLE})^2$, which uses $n$ in the denominator rather than $n-1$ (which would be the unbiased estimator).

### Question 6
For a Bernoulli distribution with parameter $p$, what is the MLE estimate of $p$ given a sample of $n$ trials with $k$ successes?

**Options:**
A) $\hat{p}_{MLE} = \frac{k}{n-1}$
B) $\hat{p}_{MLE} = \frac{k+1}{n+2}$
C) $\hat{p}_{MLE} = \frac{k}{n}$
D) $\hat{p}_{MLE} = \frac{k+1}{n}$

**Answer:** C

**Explanation:** For a Bernoulli distribution, the MLE estimate of the parameter $p$ is simply the proportion of successes: $\hat{p}_{MLE} = \frac{k}{n}$, where $k$ is the number of successes and $n$ is the total number of trials.

### Question 7
For a Poisson distribution with parameter $\lambda$, what is the MLE estimate of $\lambda$?

**Options:**
A) The sample median
B) The sample mode
C) The sample mean
D) The sample variance

**Answer:** C

**Explanation:** For a Poisson distribution, the MLE estimate of the parameter $\lambda$ is the sample mean: $\hat{\lambda}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}x_i$, which is the average of all observations.

### Question 8
When does the MLE estimate not have a closed-form solution?

**Options:**
A) When using normal distributions
B) When using discrete distributions
C) When using continuous distributions
D) When the likelihood function is complex and cannot be maximized analytically

**Answer:** D

**Explanation:** The MLE estimate does not have a closed-form solution when the likelihood function is complex and cannot be maximized analytically. In such cases, numerical optimization methods like gradient descent must be used.

### Question 9
Which of the following is NOT a property of Maximum Likelihood Estimators?

**Options:**
A) Consistency
B) Asymptotic normality
C) Unbiasedness
D) Efficiency

**Answer:** C

**Explanation:** Maximum Likelihood Estimators are not necessarily unbiased, especially for small sample sizes. However, they do possess the properties of consistency, asymptotic normality, and efficiency under certain conditions.

### Question 10
In the context of linear regression, what does MLE assume about the errors?

**Options:**
A) Errors follow a uniform distribution
B) Errors follow a normal distribution
C) Errors follow a Poisson distribution
D) Errors follow a Bernoulli distribution

**Answer:** B

**Explanation:** In the context of linear regression, MLE typically assumes that the errors follow a normal distribution, which leads to the familiar least squares estimation method.

## Related Topics

- [[L2_4_MLE_Examples|MLE Examples]]: Main examples document for Maximum Likelihood Estimation
