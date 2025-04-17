# Question 4: Normal Distribution Parameter Estimation

## Problem Statement
Consider the problem of estimating the mean $\mu$ of a normal distribution with known variance $\sigma^2 = 4$. We observe data $X = \{10.2, 8.7, 9.5, 11.3, 10.8\}$.

## Task
1. If we use a normal prior $N(9, 1)$ for $\mu$, derive the posterior distribution
2. Calculate the posterior mean and variance
3. Find the 90% credible interval for $\mu$
4. Compare the Bayesian estimate with the maximum likelihood estimate (MLE) and discuss the differences between the two approaches

## Solution

### Step 1: Understanding the Problem

We have:
- Data $X = \{10.2, 8.7, 9.5, 11.3, 10.8\}$ from a normal distribution
- Known variance $\sigma^2 = 4$
- Normal prior for $\mu$: $N(9, 1)$
- Sample size: $n = 5$
- Sample mean: $\bar{x} = 10.1$
- Sample variance: $s^2 = 1.065$

We need to apply Bayesian inference to estimate the mean parameter $\mu$, taking advantage of the conjugate relationship between normal likelihood and normal prior.

### Step 2: Normal-Normal Conjugate Prior Relationship

For a normal likelihood with known variance $\sigma^2$:
- Likelihood: $p(x|\mu) = N(x|\mu, \sigma^2)$
- Prior: $p(\mu) = N(\mu|\mu_0, \sigma_0^2)$

The normal distribution is self-conjugate for its mean parameter when the variance is known. After observing data $X$, the posterior distribution is:
$$p(\mu|X) = N(\mu|\mu_n, \sigma_n^2)$$

Where:
$$\mu_n = \frac{\mu_0/\sigma_0^2 + n\cdot\bar{x}/\sigma^2}{1/\sigma_0^2 + n/\sigma^2}$$
$$\sigma_n^2 = \frac{1}{1/\sigma_0^2 + n/\sigma^2}$$

The following figure shows the prior distribution alongside the likelihood function based on our data:

![Prior and Likelihood](../Images/L2_5_Quiz_4/prior_likelihood.png)

### Step 3: Deriving the Posterior Distribution

Using Bayes' theorem:
$$p(\mu|X) \propto p(X|\mu) \cdot p(\mu)$$

For $n$ independent observations from a normal distribution:
$$p(X|\mu) = \prod_{i=1}^n p(x_i|\mu) = (2\pi\sigma^2)^{-n/2} \cdot \exp\left[-\frac{\sum(x_i-\mu)^2}{2\sigma^2}\right]$$

This can be rewritten as:
$$p(X|\mu) = (2\pi\sigma^2)^{-n/2} \cdot \exp\left[-\frac{n(\bar{x}-\mu)^2 + \sum(x_i-\bar{x})^2}{2\sigma^2}\right]$$

Multiplying by the prior and focusing on terms involving $\mu$:
$$p(\mu|X) \propto \exp\left[-\frac{n(\bar{x}-\mu)^2}{2\sigma^2}\right] \cdot \exp\left[-\frac{(\mu-\mu_0)^2}{2\sigma_0^2}\right]$$
$$\propto \exp\left[-\frac{1}{2}\left(\frac{n}{\sigma^2}(\bar{x}-\mu)^2 + \frac{1}{\sigma_0^2}(\mu-\mu_0)^2\right)\right]$$

After completing the square and rearranging, we get a normal distribution with:
$$\sigma_n^2 = \frac{1}{1/\sigma_0^2 + n/\sigma^2}$$
$$\mu_n = \sigma_n^2 \cdot \left(\frac{\mu_0}{\sigma_0^2} + \frac{n\cdot\bar{x}}{\sigma^2}\right)$$

### Step 4: Calculating the Posterior Distribution

With our specific values:
- Prior: $N(\mu_0=9, \sigma_0^2=1)$
- Data: $X = \{10.2, 8.7, 9.5, 11.3, 10.8\}$
- Sample mean: $\bar{x} = 10.1$
- Known variance: $\sigma^2 = 4$
- Sample size: $n = 5$

We can calculate:
- Prior precision: $1/\sigma_0^2 = 1$
- Data precision: $n/\sigma^2 = 5/4 = 1.25$
- Posterior precision: $1 + 1.25 = 2.25$
- Posterior variance: $\sigma_n^2 = 1/2.25 = 0.4444$
- Posterior mean: $\mu_n = 0.4444 \cdot (9 \cdot 1 + 10.1 \cdot 1.25) = 9.6111$

Therefore, our posterior distribution is:
$$p(\mu|X) = N(9.6111, 0.4444)$$

The 90% credible interval for $\mu$ is $[8.5145, 10.7077]$.

The following figure shows the prior, likelihood, and posterior distributions:

![Posterior Distribution](../Images/L2_5_Quiz_4/posterior_distribution.png)

### Step 5: Posterior Mean as a Weighted Average

An intuitive way to understand the posterior mean is as a precision-weighted average of the prior mean and the sample mean:
$$\mu_n = w_1 \cdot \mu_0 + w_2 \cdot \bar{x}$$

Where:
$$w_1 = \frac{1/\sigma_0^2}{1/\sigma_0^2 + n/\sigma^2} = 0.4444 \text{ (44.4\% weight to prior)}$$
$$w_2 = \frac{n/\sigma^2}{1/\sigma_0^2 + n/\sigma^2} = 0.5556 \text{ (55.6\% weight to data)}$$

So:
$$\mu_n = 0.4444 \times 9.0 + 0.5556 \times 10.1 = 4.0000 + 5.6111 = 9.6111$$

This weighted average interpretation is visualized below:

![Weighted Average](../Images/L2_5_Quiz_4/weighted_average.png)

The more precise source of information (lower variance) gets more weight in determining the posterior mean.

### Step 6: Comparing Bayesian Estimate with MLE

The Maximum Likelihood Estimate (MLE) for the mean of a normal distribution is simply the sample mean:
$$\text{MLE} = \bar{x} = 10.1$$

Comparing the Bayesian and frequentist approaches:

1. Point Estimates:
   - MLE: $\hat{\mu} = 10.1$
   - Bayesian posterior mean: $9.6111$

2. Measures of Uncertainty:
   - Standard error of MLE: $\text{SE}(\hat{\mu}) = \sigma/\sqrt{n} = 0.8944$
   - Posterior standard deviation: $0.6667$

3. Intervals:
   - 90% Confidence interval (frequentist): $[8.1932, 12.0068]$
   - 90% Credible interval (Bayesian): $[8.5145, 10.7077]$

![MLE vs Bayesian](../Images/L2_5_Quiz_4/mle_vs_bayesian.png)

Key differences:
1. The Bayesian approach incorporates prior information, pulling the estimate toward the prior mean ($9.0$).
2. The Bayesian credible interval is narrower than the frequentist confidence interval, reflecting the additional information from the prior.
3. The credible interval has a direct probability interpretation: "There is a 90% probability that $\mu$ lies within this interval, given our data and prior."
4. The confidence interval has a frequency interpretation: "If we repeated the experiment many times, 90% of the computed intervals would contain the true $\mu$."

### Step 7: Effect of Different Priors

The choice of prior can significantly impact the posterior distribution, especially with small sample sizes:

![Effect of Different Priors](../Images/L2_5_Quiz_4/prior_effect.png)

This figure compares posteriors resulting from four different priors:
- A strong prior centered at $\mu = 9$ (higher precision/smaller variance)
- Our original prior $N(9, 1)$
- A weak prior centered at $\mu = 9$ (lower precision/larger variance)
- A strong prior centered at a different value ($\mu = 11$)

As the prior becomes more informative (smaller variance), it exerts more influence on the posterior distribution. A prior centered far from the data will pull the posterior mean away from the MLE.

### Step 8: Effect of Sample Size

As sample size increases, the influence of the prior diminishes:

![Effect of Sample Size](../Images/L2_5_Quiz_4/sample_size_effect.png)

With just one observation, the posterior is heavily influenced by the prior. With 100 observations, the posterior is virtually centered at the sample mean, and the prior has minimal impact. This demonstrates how Bayesian inference naturally transitions from prior-dominated to data-dominated as more evidence is accumulated.

## Key Insights

1. **Conjugate Prior**: The normal distribution is self-conjugate for its mean parameter when the variance is known, leading to a normal posterior distribution.

2. **Precision Weighting**: The posterior mean is a precision-weighted average of the prior mean and the sample mean, where precision is the reciprocal of variance $(1/\sigma^2)$.

3. **Uncertainty Reduction**: The posterior variance is always smaller than both the prior variance and the variance of the MLE, reflecting the combined information from both sources.

4. **Prior Influence**: The impact of the prior depends on:
   - Its precision relative to the data precision
   - The sample size (more data = less prior influence)
   - The discrepancy between the prior mean and the sample mean

5. **Interval Interpretation**: Bayesian credible intervals have direct probability interpretations about the parameter, unlike frequentist confidence intervals which make statements about the procedure.

6. **Convergence**: As sample size increases, Bayesian and frequentist approaches converge - the posterior mean approaches the MLE, and credible intervals approach confidence intervals.

## Conclusion

In this Bayesian estimation problem, we've derived a posterior distribution $N(9.6111, 0.4444)$ for the mean of a normal distribution. This posterior combines our prior belief $N(9, 1)$ with the observed data (sample mean $10.1$), giving slightly more weight to the data (55.6%) than to the prior (44.4%). The 90% credible interval $[8.5145, 10.7077]$ is narrower than the corresponding frequentist confidence interval, reflecting the additional information incorporated from the prior. 

The Bayesian approach allows us to formally incorporate prior knowledge and provides a more intuitive framework for making probability statements about parameters. However, the results are sensitive to the choice of prior, particularly with small sample sizes. As sample size increases, both approaches converge to similar conclusions, with the data eventually dominating over any reasonable prior. 