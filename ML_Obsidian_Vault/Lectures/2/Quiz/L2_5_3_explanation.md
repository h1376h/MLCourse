# Question 3: Poisson-Gamma Bayesian Model

## Problem Statement
Consider a Poisson likelihood with parameter λ for count data. We observe data X = {3, 5, 2, 4, 6, 3, 4, 5, 2, 3}.

## Task
1. If we use a Gamma(α, β) prior for λ, derive the posterior distribution
2. Assuming a Gamma(2, 1) prior, calculate the posterior distribution
3. Find the posterior mean, mode, and variance of λ
4. Calculate the predictive distribution for a new observation X_new

## Solution

### Step 1: Understanding the Problem

We have:
- Count data X = {3, 5, 2, 4, 6, 3, 4, 5, 2, 3} assumed to follow a Poisson distribution with parameter λ
- A Gamma(α, β) prior for the Poisson rate parameter λ
- Sample size n = 10
- Sum of observations: sum(x_i) = 37
- Sample mean: x̄ = 3.70

We need to leverage the conjugate relationship between the Poisson likelihood and Gamma prior to derive the posterior distribution, calculate its properties, and determine the predictive distribution for new observations.

### Step 2: Poisson-Gamma Conjugate Prior Relationship

The Poisson distribution describes the probability of observing x events in a fixed interval when events occur at a constant rate λ:
- Likelihood: p(x|λ) = e^(-λ) * λ^x / x!

The Gamma distribution is the conjugate prior for the Poisson likelihood:
- Prior: p(λ) = Gamma(α, β) = β^α * λ^(α-1) * e^(-βλ) / Γ(α)
  where α is the shape parameter and β is the rate parameter

The following figure shows Poisson distributions with different λ values, including our sample mean λ = 3.7:

![Poisson Distributions with Different λ Values](../Images/L2_5_Quiz_3/poisson_distributions.png)

### Step 3: Deriving the Posterior Distribution

Using Bayes' theorem, the posterior distribution is proportional to the product of the likelihood and the prior:
p(λ|X) ∝ p(X|λ) * p(λ)

For our Poisson model with n independent observations, the likelihood is:
p(X|λ) = ∏ p(x_i|λ) = ∏ [e^(-λ) * λ^(x_i) / x_i!] = e^(-nλ) * λ^(sum(x_i)) / ∏(x_i!)

Multiplying by the Gamma prior:
p(λ|X) ∝ [e^(-nλ) * λ^(sum(x_i))] * [β^α * λ^(α-1) * e^(-βλ) / Γ(α)]
∝ e^(-nλ) * λ^(sum(x_i)) * λ^(α-1) * e^(-βλ)
∝ e^(-(n+β)λ) * λ^(α+sum(x_i)-1)

This expression has the form of a Gamma distribution with updated parameters:
p(λ|X) = Gamma(α + sum(x_i), β + n)

Therefore, the posterior distribution is:
p(λ|X) = Gamma(α + sum(x_i), β + n)

This elegant result is why Gamma is considered the conjugate prior for the Poisson likelihood - the posterior has the same functional form as the prior, just with updated parameters.

### Step 4: Calculating the Posterior with Gamma(2, 1) Prior

Given our specific prior Gamma(α=2, β=1) and observed data X = {3, 5, 2, 4, 6, 3, 4, 5, 2, 3}:

Posterior parameters:
- α' = α + sum(x_i) = 2 + 37 = 39
- β' = β + n = 1 + 10 = 11

Therefore, the posterior distribution is Gamma(39, 11).

We can calculate the following posterior statistics:
- Posterior mean: E[λ|X] = α'/β' = 39/11 = 3.5455
- Posterior mode: (α'-1)/β' = 38/11 = 3.4545
- Posterior variance: α'/β'^2 = 39/121 = 0.3223
- Posterior standard deviation: 0.5677

The following figure shows the prior, likelihood, and posterior distributions:

![Prior, Likelihood, and Posterior Distributions](../Images/L2_5_Quiz_3/prior_likelihood_posterior.png)

This figure illustrates how the posterior distribution (red line) combines information from both the prior (blue dashed line) and the likelihood function (green line). The posterior mean (3.55) is between the prior mean (2.0) and the sample mean (3.7), but closer to the sample mean because we have a substantial amount of data.

### Step 5: Calculating the Predictive Distribution

The predictive distribution gives the probability of observing a new data point x_new given our current data X:
p(x_new|X) = ∫ p(x_new|λ) p(λ|X) dλ

For the Poisson-Gamma model, this integral has a closed-form solution - the Negative Binomial distribution:
p(x_new|X) = NegBin(r=α', p=β'/(β'+1))

With our posterior parameters:
p(x_new|X) = NegBin(r=39, p=11/12 = 0.9167)

The predictive distribution has:
- Mean: r(1-p)/p = 39(1-0.9167)/0.9167 = 3.5455 (same as posterior mean)
- Variance: r(1-p)/p² = 39(1-0.9167)/0.9167² = 3.8678 (greater than posterior variance)

The probabilities for specific new observations are:
- P(x_new = 0) = 0.0336
- P(x_new = 1) = 0.1092
- P(x_new = 2) = 0.1820
- P(x_new = 3) = 0.2072
- P(x_new = 4) = 0.1813
- P(x_new = 5) = 0.1300

The following figure compares the predictive distribution (bars) with a Poisson distribution using λ = posterior mean (red line):

![Predictive Distribution for a New Observation](../Images/L2_5_Quiz_3/predictive_distribution.png)

Notice that the predictive distribution is wider than the Poisson distribution with λ fixed at the posterior mean. This is because the predictive distribution incorporates uncertainty in the parameter λ, while the Poisson distribution assumes a fixed parameter value.

### Step 6: Effect of Different Priors

The choice of prior can influence the posterior distribution, especially with limited data. The following figure shows how different priors affect the resulting posterior:

![Effect of Different Priors on Posterior](../Images/L2_5_Quiz_3/prior_comparison.png)

This figure compares posteriors resulting from four different priors:
- Gamma(1, 1) - a relatively uninformative prior
- Gamma(2, 1) - our original prior
- Gamma(10, 2) - a more informative prior centered at λ=5
- Gamma(20, 10) - a very informative prior centered at λ=2

As shown, more informative priors pull the posterior means toward their respective prior means. However, with 10 observations, the data still significantly influences all posteriors.

### Step 7: Posterior Mean vs MLE and Credible Interval

The Maximum Likelihood Estimate (MLE) for a Poisson distribution is simply the sample mean: λ̂ = 3.7.

The Bayesian approach gives us:
- Posterior mean: 3.5455
- 95% Credible interval: [2.52, 4.74]

![Posterior Distribution with Credible Interval](../Images/L2_5_Quiz_3/credible_interval.png)

The 95% credible interval represents the range within which we are 95% certain the true value of λ lies, given our data and prior. Unlike frequentist confidence intervals, Bayesian credible intervals have a direct probability interpretation.

## Key Insights

1. **Conjugate Priors**: The Gamma distribution is the conjugate prior for the Poisson likelihood, resulting in a Gamma posterior. This mathematical convenience greatly simplifies Bayesian inference.

2. **Parameter Updating**: The posterior parameters α' = α + sum(x_i) and β' = β + n show how Bayesian updating combines prior information (α, β) with data information (sum(x_i), n).

3. **Posterior Mean Interpretation**: The posterior mean can be written as a weighted average:
   E[λ|X] = (α/β + sum(x_i)/n) / (1/β + 1) = w·(α/β) + (1-w)·(sum(x_i)/n)
   where w = 1/(β+1) represents the weight given to the prior mean, and (1-w) is the weight given to the sample mean.

4. **Predictive Distribution**: The Negative Binomial predictive distribution has greater variance than a simple Poisson with λ = posterior mean, as it accounts for both the inherent randomness in the Poisson process and the uncertainty in estimating λ.

5. **Prior Influence**: With small sample sizes, the prior has a stronger influence on the posterior. As sample size increases, the data dominates and the posterior converges to the frequentist result.

6. **Uncertainty Quantification**: The Bayesian approach naturally quantifies uncertainty through the full posterior distribution and provides interpretable credible intervals.

## Conclusion

The Poisson-Gamma model demonstrates the elegance of Bayesian inference with conjugate priors. For our count data, the posterior distribution is Gamma(39, 11), with a mean of 3.55, mode of 3.45, and variance of 0.32. The predictive distribution for future observations follows a Negative Binomial distribution, which accounts for both parameter uncertainty and inherent randomness. This Bayesian approach provides a comprehensive framework for inference about the Poisson rate parameter λ, incorporating prior knowledge and systematically updating our beliefs based on observed data. 