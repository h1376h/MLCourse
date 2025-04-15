# MLE for Bernoulli Distribution

This document provides key concepts and formulas for Maximum Likelihood Estimation (MLE) applied to the Bernoulli distribution.

## Key Concepts and Formulas

### Bernoulli Distribution

The Bernoulli distribution is characterized by one parameter:
- $p$ - the success probability

Its probability mass function (PMF) is:

$$P(X = x|p) = p^x (1-p)^{1-x}$$

for $x \in \{0, 1\}$ and $0 \leq p \leq 1$.

### MLE for Bernoulli Distribution Parameters

For data $X = \{x_1, x_2, \ldots, x_n\}$ following a Bernoulli distribution:

#### MLE for Success Probability ($p$)

$$\hat{p}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}x_i$$

This is simply the proportion of successes (1's) in the sample.

**Derivation**:
1. **Likelihood function**:
   $L(p|X) = \prod_{i=1}^{n} p^{x_i} (1-p)^{1-x_i}$

2. **Log-likelihood**:
   $\ell(p|X) = \sum_{i=1}^{n} [x_i \ln(p) + (1-x_i) \ln(1-p)]$

3. **Differentiate with respect to $p$**:
   $\frac{d\ell(p|X)}{dp} = \sum_{i=1}^{n} \frac{x_i}{p} - \sum_{i=1}^{n} \frac{1-x_i}{1-p}$

4. **Set derivative to zero**:
   $\sum_{i=1}^{n} \frac{x_i}{p} = \sum_{i=1}^{n} \frac{1-x_i}{1-p}$
   
   Rearranging:
   $\frac{\sum_{i=1}^{n} x_i}{p} = \frac{n - \sum_{i=1}^{n} x_i}{1-p}$
   
   Solving for $p$:
   $p = \frac{\sum_{i=1}^{n} x_i}{n}$

5. **Verify it's a maximum** using the second derivative:
   $\frac{d^2\ell(p|X)}{dp^2} = -\sum_{i=1}^{n} \frac{x_i}{p^2} - \sum_{i=1}^{n} \frac{1-x_i}{(1-p)^2} < 0$, confirming it's a maximum.

### Properties of Bernoulli Distribution MLEs

1. **Mean Estimation**: 
   For a Bernoulli distribution, the mean equals $p$:
   $$\hat{\mu}_{MLE} = \hat{p}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}x_i$$

2. **Variance Estimation**: 
   For a Bernoulli distribution, the variance equals $p(1-p)$:
   $$\hat{\sigma}^2_{MLE} = \hat{p}_{MLE}(1-\hat{p}_{MLE})$$

3. **Standard Deviation Estimation**:
   $$\hat{\sigma}_{MLE} = \sqrt{\hat{p}_{MLE}(1-\hat{p}_{MLE})}$$

4. **Efficiency**: 
   The MLE for the Bernoulli distribution parameter is efficient, achieving the Cramér-Rao lower bound.

5. **Sufficiency**: 
   The sum of observations is a sufficient statistic for the Bernoulli distribution parameter.

6. **Consistency**: 
   As sample size increases, the MLE converges to the true parameter value.

7. **Bias**: 
   The MLE for the Bernoulli distribution parameter is unbiased:
   $$E[\hat{p}_{MLE}] = p$$

8. **Skewness Estimation**:
   $$\hat{\text{Skewness}}_{MLE} = \frac{1-2\hat{p}_{MLE}}{\sqrt{\hat{p}_{MLE}(1-\hat{p}_{MLE})}}$$

9. **Kurtosis Estimation**:
   $$\hat{\text{Kurtosis}}_{MLE} = \frac{1-6\hat{p}_{MLE}(1-\hat{p}_{MLE})}{\hat{p}_{MLE}(1-\hat{p}_{MLE})}$$

## Applications

The Bernoulli distribution is commonly used to model:
- Binary outcomes (success/failure)
- Coin flips (heads/tails)
- Pass/fail scenarios
- Yes/no responses
- Present/absent features

## Related Topics

- [[L2_1_Bernoulli_Binomial|Bernoulli and Binomial Distributions]]
- [[L2_4_MLE|Maximum Likelihood Estimation]]
- [[L2_3_Parameter_Estimation|Parameter Estimation]] 