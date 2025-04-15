# Question 5: MLE for Poisson Distribution

## Problem Statement
Consider a dataset of n independent observations $X_1, X_2, \ldots, X_n$ from a Poisson distribution with unknown rate parameter $\lambda$. The probability mass function is:

$$f(x|\lambda) = \frac{\lambda^x e^{-\lambda}}{x!} \text{ for } x = 0, 1, 2, \ldots$$

### Task
1. Derive the maximum likelihood estimator $\hat{\lambda}_{MLE}$ for $\lambda$
2. Show that this estimator is efficient
3. Calculate the asymptotic distribution of $\hat{\lambda}_{MLE}$

## Understanding the Probability Model

The Poisson distribution is a discrete probability distribution that describes the number of events occurring in a fixed time or space interval, given that these events occur with a known average rate and independently of each other. Some key properties include:

- The distribution has a single parameter $\lambda$ that represents both the mean and variance
- It models rare events where the probability of occurrence is small but the number of opportunities is large
- Events occur independently of the time since the last event
- The Poisson distribution approaches a normal distribution for large values of $\lambda$
- Common applications include modeling call center arrivals, website traffic, radioactive decay, and defects in manufacturing

## Solution

The Poisson distribution is a discrete probability distribution that models count data in situations where events occur randomly but at a known average rate. It has a single parameter $\lambda$ that represents both the mean and variance of the distribution.

### Step 1: Formulate the likelihood function
For n independent observations $x_1, x_2, \ldots, x_n$, the likelihood function is:

$$L(\lambda) = \prod_{i=1}^n f(x_i|\lambda) = \prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!} = \frac{\lambda^{\sum_{i=1}^n x_i} e^{-n\lambda}}{\prod_{i=1}^n x_i!}$$

### Step 2: Take the logarithm to get the log-likelihood
Taking the natural logarithm, we get the log-likelihood function:

$$\ell(\lambda) = \log L(\lambda) = \sum_{i=1}^n x_i \log \lambda - n\lambda - \sum_{i=1}^n \log(x_i!)$$

Since the last term does not depend on $\lambda$, we can simplify:

$$\ell(\lambda) = \sum_{i=1}^n x_i \log \lambda - n\lambda + C$$

where $C$ is a constant that doesn't affect the maximization.

### Step 3: Find the critical points by taking the derivative
To find the maximum, we take the derivative with respect to $\lambda$ and set it to zero:

$$\frac{d\ell}{d\lambda} = \frac{\sum_{i=1}^n x_i}{\lambda} - n = 0$$

### Step 4: Solve for the MLE estimate
Solving for $\lambda$:

$$\frac{\sum_{i=1}^n x_i}{\lambda} = n$$

$$\lambda = \frac{\sum_{i=1}^n x_i}{n} = \bar{X}$$

Therefore, the MLE is:

$$\hat{\lambda}_{MLE} = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$$

### Step 5: Verify it's a maximum
The second derivative of the log-likelihood is:

$$\frac{d^2\ell}{d\lambda^2} = -\frac{\sum_{i=1}^n x_i}{\lambda^2} < 0$$

Since the second derivative is negative for all $\lambda > 0$, we confirm that our critical point is indeed a maximum.

### Step 6: Show that the estimator is efficient
An estimator is efficient if it achieves the minimum possible variance among all unbiased estimators, as given by the Cramér-Rao lower bound.

First, we calculate the Fisher information for a Poisson distribution:

$$I(\lambda) = E\left[\left(\frac{\partial \log f(X|\lambda)}{\partial \lambda}\right)^2\right] = E\left[\left(\frac{X}{\lambda} - 1\right)^2\right]$$

Since $E[X] = \lambda$ and $Var(X) = \lambda$ for a Poisson distribution:

$$I(\lambda) = \frac{1}{\lambda^2}E[X^2] - \frac{2}{\lambda}E[X] + 1 = \frac{\lambda + \lambda^2}{\lambda^2} - \frac{2\lambda}{\lambda} + 1 = \frac{1}{\lambda}$$

The Cramér-Rao lower bound (CRLB) for an unbiased estimator is:

$$Var(\hat{\lambda}) \geq \frac{1}{nI(\lambda)} = \frac{\lambda}{n}$$

For the sample mean of Poisson random variables:

$$Var(\hat{\lambda}_{MLE}) = Var\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n Var(X_i) = \frac{1}{n^2} \cdot n\lambda = \frac{\lambda}{n}$$

Since $Var(\hat{\lambda}_{MLE}) = \frac{\lambda}{n}$, which exactly equals the CRLB, the MLE is efficient.

### Step 7: Calculate the asymptotic distribution
By the Central Limit Theorem, for large n, the distribution of $\hat{\lambda}_{MLE}$ approaches a normal distribution:

$$\hat{\lambda}_{MLE} \sim \mathcal{N}\left(\lambda, \frac{\lambda}{n}\right)$$

More formally:

$$\sqrt{n}(\hat{\lambda}_{MLE} - \lambda) \stackrel{d}{\rightarrow} \mathcal{N}(0, \lambda)$$

This means that $\hat{\lambda}_{MLE}$ is asymptotically normal with mean $\lambda$ and variance $\frac{\lambda}{n}$.

## Visual Explanations

### Poisson PMFs for Different λ Values
![Poisson PMFs](Images/L2_4_Quiz_5/poisson_pmfs.png)

This figure shows how the distribution shape changes with different λ values and illustrates the discrete nature of the distribution.

### Likelihood Surface
![Likelihood Surface](Images/L2_4_Quiz_5/likelihood_surface.png)

This visualization of the log-likelihood function shows the maximum point corresponding to the MLE and demonstrates the concavity of the log-likelihood function.

### MLE Fit to Data
![MLE Fit](Images/L2_4_Quiz_5/mle_fit.png)

This figure shows how well the MLE estimate fits the observed data by comparing the estimated PMF with the observed frequencies.

### Efficiency Demonstration
![Efficiency](Images/L2_4_Quiz_5/efficiency.png)

This visualization compares the variance of the MLE with other estimators, shows how the MLE achieves the Cramér-Rao lower bound, and demonstrates that alternative estimators have higher variance.

### Asymptotic Distribution
![Asymptotic Distribution](Images/L2_4_Quiz_5/asymptotic_distribution.png)

This figure visualizes how the standardized MLE estimates converge to a standard normal distribution and demonstrates the Central Limit Theorem in action.

## Key Insights

### MLE Properties
- The MLE for the Poisson distribution is simply the sample mean
- The estimator is unbiased: $E[\hat{\lambda}_{MLE}] = \lambda$
- The estimator is efficient: it achieves the minimum possible variance
- The estimator is asymptotically normal: its distribution converges to normal for large samples

### Practical Considerations
- The MLE is easy to compute and interpret
- It provides a direct relationship to the data
- The efficiency is maintained for all sample sizes
- The variance decreases at a rate proportional to 1/n

## Conclusion

The maximum likelihood estimator for the Poisson distribution parameter $\lambda$ is:
- Simple: it's just the arithmetic mean of the observations
- Efficient: it achieves the minimum possible variance for an unbiased estimator
- Asymptotically normal: its distribution converges to a normal distribution for large samples

These properties make it an ideal estimator for practical applications involving count data, such as modeling the number of events, arrivals, or occurrences in a fixed time interval or region. 