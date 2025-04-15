# MLE for Binomial Distribution

This document provides key concepts and formulas for Maximum Likelihood Estimation (MLE) applied to the Binomial distribution.

## Key Concepts and Formulas

### Binomial Distribution

The Binomial distribution is characterized by two parameters:
- $n$ - the number of trials (typically known)
- $p$ - the success probability for each trial

Its probability mass function (PMF) is:

$$P(X = k|n,p) = \binom{n}{k} p^k (1-p)^{n-k}$$

for $k \in \{0, 1, 2, \ldots, n\}$ and $0 \leq p \leq 1$.

### MLE for Binomial Distribution Parameters

For a single observation $x$ from a Binomial distribution with known $n$:

#### MLE for Success Probability ($p$)

$$\hat{p}_{MLE} = \frac{x}{n}$$

For multiple observations $X = \{x_1, x_2, \ldots, x_m\}$ with the same $n$:

$$\hat{p}_{MLE} = \frac{\sum_{i=1}^{m}x_i}{m \cdot n}$$

**Derivation**:
1. **Likelihood function** (single observation):
   $L(p|x) = \binom{n}{x} p^x (1-p)^{n-x}$

2. **Log-likelihood**:
   $\ell(p|x) = \ln\binom{n}{x} + x\ln(p) + (n-x)\ln(1-p)$

3. **Differentiate with respect to $p$**:
   $\frac{d\ell(p|x)}{dp} = \frac{x}{p} - \frac{n-x}{1-p}$

4. **Set derivative to zero**:
   $\frac{x}{p} = \frac{n-x}{1-p}$
   
   Rearranging:
   $x(1-p) = p(n-x)$
   $x - xp = np - xp$
   $x = np$
   $p = \frac{x}{n}$

5. **Verify it's a maximum** using the second derivative:
   $\frac{d^2\ell(p|x)}{dp^2} = -\frac{x}{p^2} - \frac{n-x}{(1-p)^2} < 0$, confirming it's a maximum.

For multiple observations, the derivation is similar, summing the log-likelihood for each observation.

### Properties of Binomial Distribution MLEs

1. **Mean Estimation**: 
   For a Binomial distribution, the mean equals $np$:
   $$\hat{\mu}_{MLE} = n\hat{p}_{MLE} = \frac{\sum_{i=1}^{m}x_i}{m}$$

2. **Variance Estimation**: 
   For a Binomial distribution, the variance equals $np(1-p)$:
   $$\hat{\sigma}^2_{MLE} = n\hat{p}_{MLE}(1-\hat{p}_{MLE})$$

3. **Standard Deviation Estimation**:
   $$\hat{\sigma}_{MLE} = \sqrt{n\hat{p}_{MLE}(1-\hat{p}_{MLE})}$$

4. **Efficiency**: 
   The MLE for the Binomial distribution parameter is efficient, achieving the Cramér-Rao lower bound.

5. **Sufficiency**: 
   The sum of observations is a sufficient statistic for the Binomial distribution parameter.

6. **Consistency**: 
   As sample size increases, the MLE converges to the true parameter value.

7. **Bias**: 
   The MLE for the Binomial distribution parameter is unbiased:
   $$E[\hat{p}_{MLE}] = p$$

8. **Skewness Estimation**:
   $$\hat{\text{Skewness}}_{MLE} = \frac{1-2\hat{p}_{MLE}}{\sqrt{n\hat{p}_{MLE}(1-\hat{p}_{MLE})}}$$

9. **Kurtosis Estimation**:
   $$\hat{\text{Kurtosis}}_{MLE} = \frac{1-6\hat{p}_{MLE}(1-\hat{p}_{MLE})}{n\hat{p}_{MLE}(1-\hat{p}_{MLE})}$$

## Applications

The Binomial distribution is commonly used to model:
- Number of successes in a fixed number of independent trials
- Number of defective items in quality control sampling
- Number of patients responding to a treatment in clinical trials
- Number of customers making a purchase after a marketing campaign
- Election polling and survey data analysis

## Related Topics

- [[L2_1_Bernoulli_Binomial|Bernoulli and Binomial Distributions]]
- [[L2_4_MLE|Maximum Likelihood Estimation]]
- [[L2_3_Parameter_Estimation|Parameter Estimation]] 