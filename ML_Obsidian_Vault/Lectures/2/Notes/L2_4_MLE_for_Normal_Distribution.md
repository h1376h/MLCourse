# MLE for Normal Distribution

This document provides key concepts and formulas for Maximum Likelihood Estimation (MLE) applied to the Normal distribution.

## Key Concepts and Formulas

### Normal Distribution

The normal (or Gaussian) distribution is characterized by two parameters:
- $\mu$ - the mean parameter
- $\sigma^2$ - the variance parameter

Its probability density function (PDF) is:

$$f(x|\mu,\sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

### MLE for Normal Distribution Parameters

For data $X = \{x_1, x_2, \ldots, x_n\}$ following a normal distribution:

#### MLE for Mean ($\mu$)

$$\hat{\mu}_{MLE} = \bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$$

**Derivation**:
1. **Likelihood function**:
   $L(\mu, \sigma^2|X) = \prod_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$

2. **Log-likelihood**:
   $\ell(\mu, \sigma^2|X) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2$

3. **Differentiate with respect to $\mu$**:
   $\frac{d\ell(\mu|X)}{d\mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i-\mu)$

4. **Set derivative to zero**:
   $\sum_{i=1}^{n}(x_i-\mu) = 0$, which gives $\mu = \frac{1}{n}\sum_{i=1}^{n}x_i$

5. **Verify it's a maximum** using the second derivative:
   $\frac{d^2\ell(\mu|X)}{d\mu^2} = -\frac{n}{\sigma^2} < 0$

#### MLE for Variance ($\sigma^2$)

If $\mu$ is unknown and estimated by its MLE (sample mean $\bar{x}$):
$$\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2$$

If $\mu$ is known:
$$\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2$$

**Derivation**:
1. **Log-likelihood**:
   $\ell(\mu, \sigma^2|X) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2$

2. **Differentiate with respect to $\sigma^2$**:
   $\frac{d\ell(\mu, \sigma^2|X)}{d\sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^{n}(x_i-\mu)^2$

3. **Set derivative to zero**:
   $-\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^{n}(x_i-\mu)^2 = 0$, which gives $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2$

#### MLE for Standard Deviation ($\sigma$)

$$\hat{\sigma}_{MLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2}$$

### Note on Bias

The MLE for variance is **biased**:
- It underestimates the true population variance on average
- $E[\hat{\sigma}^2_{MLE}] = \frac{n-1}{n}\sigma^2$

For an unbiased estimator of variance, we use:
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2$$

This is known as the sample variance with Bessel's correction.

## Properties of Normal MLE Estimators

1. **Efficiency**: The MLE estimators for normal distribution are efficient, achieving the Cramér-Rao lower bound.

2. **Sufficiency**: The sample mean and sample variance are sufficient statistics for the normal distribution parameters.

3. **Consistency**: As sample size increases, the MLEs converge to the true parameter values.

4. **Asymptotic Normality**: For large samples, the estimators are approximately normally distributed.

## Related Topics

- [[L2_1_Normal_Distribution|Normal Distribution]]
- [[L2_4_MLE|Maximum Likelihood Estimation]]
- [[L2_3_Parameter_Estimation|Parameter Estimation]] 