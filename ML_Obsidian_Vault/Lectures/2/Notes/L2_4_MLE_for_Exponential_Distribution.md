# MLE for Exponential Distribution

This document provides key concepts and formulas for Maximum Likelihood Estimation (MLE) applied to the Exponential distribution.

## Key Concepts and Formulas

### Exponential Distribution

The exponential distribution is characterized by one parameter:
- $\theta$ - the mean parameter (also called the rate parameter, sometimes denoted as $\lambda = 1/\theta$)

Its probability density function (PDF) is:

$$f(x|\theta) = \frac{1}{\theta} \exp\left(-\frac{x}{\theta}\right)$$

for $x \geq 0$ and $\theta > 0$.

### MLE for Exponential Distribution Parameters

For data $X = \{x_1, x_2, \ldots, x_n\}$ following an exponential distribution:

#### MLE for Mean Parameter ($\theta$)

$$\hat{\theta}_{MLE} = \bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$$

**Derivation**:
1. **Likelihood function**:
   $L(\theta|X) = \prod_{i=1}^{n} \frac{1}{\theta} \exp\left(-\frac{x_i}{\theta}\right)$

2. **Log-likelihood**:
   $\ell(\theta|X) = -n\ln(\theta) - \frac{1}{\theta}\sum_{i=1}^{n}x_i$

3. **Differentiate with respect to $\theta$**:
   $\frac{d\ell(\theta|X)}{d\theta} = -\frac{n}{\theta} + \frac{1}{\theta^2}\sum_{i=1}^{n}x_i$

4. **Set derivative to zero**:
   $-\frac{n}{\theta} + \frac{1}{\theta^2}\sum_{i=1}^{n}x_i = 0$, which gives $\theta = \frac{1}{n}\sum_{i=1}^{n}x_i$

5. **Verify it's a maximum** using the second derivative:
   $\frac{d^2\ell(\theta|X)}{d\theta^2} = \frac{n}{\theta^2} - \frac{2}{\theta^3}\sum_{i=1}^{n}x_i$
   
   Substituting the critical point $\theta = \frac{1}{n}\sum_{i=1}^{n}x_i$, we get $\frac{d^2\ell(\theta|X)}{d\theta^2} = -\frac{n}{\theta^2} < 0$, confirming it's a maximum.

#### MLE for Rate Parameter ($\lambda = 1/\theta$)

When parameterized with rate parameter $\lambda$, where the PDF is $f(x|\lambda) = \lambda e^{-\lambda x}$:

$$\hat{\lambda}_{MLE} = \frac{1}{\bar{x}} = \frac{n}{\sum_{i=1}^{n}x_i}$$

### Properties of Exponential Distribution MLEs

1. **Standard Deviation Estimation**: 
   Since for an exponential distribution, the standard deviation equals the mean:
   $$\hat{\sigma}_{MLE} = \hat{\theta}_{MLE} = \bar{x}$$

2. **Efficiency**: 
   The MLE for the exponential distribution parameter is efficient, achieving the Cramér-Rao lower bound asymptotically.

3. **Sufficiency**: 
   The sample mean is a sufficient statistic for the exponential distribution parameter.

4. **Consistency**: 
   As sample size increases, the MLE converges to the true parameter value.

5. **Bias**: 
   Unlike the normal distribution's variance estimator, the MLE for the exponential distribution parameter is unbiased:
   $$E[\hat{\theta}_{MLE}] = \theta$$

## Applications

The exponential distribution is commonly used to model:
- Time between events in a Poisson process
- Waiting times in queuing theory
- Survival times in reliability engineering
- Lifetime of components in failure analysis

## Related Topics

- [[L2_1_Exponential_Distribution|Exponential Distribution]]
- [[L2_4_MLE|Maximum Likelihood Estimation]]
- [[L2_3_Parameter_Estimation|Parameter Estimation]] 