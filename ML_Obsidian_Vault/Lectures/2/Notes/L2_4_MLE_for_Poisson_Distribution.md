# MLE for Poisson Distribution

This document provides key concepts and formulas for Maximum Likelihood Estimation (MLE) applied to the Poisson distribution.

## Key Concepts and Formulas

### Poisson Distribution

The Poisson distribution is characterized by one parameter:
- $\lambda$ - the rate parameter (mean number of events in the interval)

Its probability mass function (PMF) is:

$$P(X = k|\lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$$

for $k = 0, 1, 2, \ldots$ and $\lambda > 0$.

### MLE for Poisson Distribution Parameters

For data $X = \{x_1, x_2, \ldots, x_n\}$ following a Poisson distribution:

#### MLE for Rate Parameter ($\lambda$)

$$\hat{\lambda}_{MLE} = \bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$$

**Derivation**:
1. **Likelihood function**:
   $L(\lambda|X) = \prod_{i=1}^{n} \frac{\lambda^{x_i} e^{-\lambda}}{x_i!}$

2. **Log-likelihood**:
   $\ell(\lambda|X) = \sum_{i=1}^{n}x_i\ln(\lambda) - n\lambda - \sum_{i=1}^{n}\ln(x_i!)$

3. **Differentiate with respect to $\lambda$**:
   $\frac{d\ell(\lambda|X)}{d\lambda} = \frac{1}{\lambda}\sum_{i=1}^{n}x_i - n$

4. **Set derivative to zero**:
   $\frac{1}{\lambda}\sum_{i=1}^{n}x_i = n$, which gives $\lambda = \frac{1}{n}\sum_{i=1}^{n}x_i$

5. **Verify it's a maximum** using the second derivative:
   $\frac{d^2\ell(\lambda|X)}{d\lambda^2} = -\frac{1}{\lambda^2}\sum_{i=1}^{n}x_i < 0$, confirming it's a maximum.

### Properties of Poisson Distribution MLEs

1. **Variance Estimation**: 
   Since for a Poisson distribution, the variance equals the mean:
   $$\hat{\sigma}^2_{MLE} = \hat{\lambda}_{MLE} = \bar{x}$$

2. **Standard Deviation Estimation**:
   $$\hat{\sigma}_{MLE} = \sqrt{\hat{\lambda}_{MLE}} = \sqrt{\bar{x}}$$

3. **Efficiency**: 
   The MLE for the Poisson distribution parameter is efficient, achieving the Cramér-Rao lower bound.

4. **Sufficiency**: 
   The sample mean (sum of observations) is a sufficient statistic for the Poisson distribution parameter.

5. **Consistency**: 
   As sample size increases, the MLE converges to the true parameter value.

6. **Bias**: 
   The MLE for the Poisson distribution parameter is unbiased:
   $$E[\hat{\lambda}_{MLE}] = \lambda$$

## Applications

The Poisson distribution is commonly used to model:
- Number of events occurring in a fixed time interval
- Number of defects in manufacturing
- Number of arrivals at a service point
- Number of rare events in large populations
- Count data in various scientific fields

## Related Topics

- [[L2_1_Discrete_Distributions|Discrete Distributions]]
- [[L2_4_MLE|Maximum Likelihood Estimation]]
- [[L2_3_Parameter_Estimation|Parameter Estimation]] 