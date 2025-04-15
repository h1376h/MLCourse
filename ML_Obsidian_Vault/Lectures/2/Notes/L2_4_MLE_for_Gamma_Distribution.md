# MLE for Gamma Distribution

This document provides key concepts and formulas for Maximum Likelihood Estimation (MLE) applied to the Gamma distribution.

## Key Concepts and Formulas

### Gamma Distribution

The Gamma distribution is characterized by two parameters:
- $k$ - the shape parameter (sometimes denoted as $\alpha$)
- $\theta$ - the scale parameter (sometimes the rate parameter $\beta = 1/\theta$ is used instead)

Its probability density function (PDF) is:

$$f(x|k,\theta) = \frac{1}{\Gamma(k)\theta^k}x^{k-1}e^{-x/\theta}$$

for $x > 0$, $k > 0$, and $\theta > 0$, where $\Gamma(k)$ is the gamma function.

When parameterized with the rate parameter $\beta = 1/\theta$, the PDF becomes:

$$f(x|k,\beta) = \frac{\beta^k}{\Gamma(k)}x^{k-1}e^{-\beta x}$$

### MLE for Gamma Distribution Parameters

For data $X = \{x_1, x_2, \ldots, x_n\}$ following a Gamma distribution:

#### MLE for Scale Parameter ($\theta$)

Assuming the shape parameter $k$ is known:

$$\hat{\theta}_{MLE} = \frac{1}{k}\sum_{i=1}^{n}\frac{x_i}{n} = \frac{\bar{x}}{k}$$

#### MLE for Shape Parameter ($k$)

There is no closed-form solution for the MLE of the shape parameter. It requires solving the following equation numerically:

$$\ln(k) - \psi(k) = \ln\left(\frac{1}{n}\sum_{i=1}^{n}x_i\right) - \frac{1}{n}\sum_{i=1}^{n}\ln(x_i)$$

where $\psi$ is the digamma function (the derivative of the logarithm of the gamma function).

**Derivation**:
1. **Likelihood function**:
   $L(k,\theta|X) = \prod_{i=1}^{n} \frac{1}{\Gamma(k)\theta^k}x_i^{k-1}e^{-x_i/\theta}$

2. **Log-likelihood**:
   $\ell(k,\theta|X) = -n\ln\Gamma(k) - nk\ln\theta + (k-1)\sum_{i=1}^{n}\ln(x_i) - \frac{1}{\theta}\sum_{i=1}^{n}x_i$

3. **Differentiate with respect to $\theta$**:
   $\frac{\partial\ell(k,\theta|X)}{\partial\theta} = -\frac{nk}{\theta} + \frac{1}{\theta^2}\sum_{i=1}^{n}x_i$

4. **Set derivative to zero**:
   $-\frac{nk}{\theta} + \frac{1}{\theta^2}\sum_{i=1}^{n}x_i = 0$
   
   Solving for $\theta$:
   $\theta = \frac{1}{nk}\sum_{i=1}^{n}x_i = \frac{\bar{x}}{k}$

5. **Differentiate with respect to $k$**:
   $\frac{\partial\ell(k,\theta|X)}{\partial k} = -n\psi(k) - n\ln\theta + \sum_{i=1}^{n}\ln(x_i)$

6. **Set derivative to zero**:
   $-n\psi(k) - n\ln\theta + \sum_{i=1}^{n}\ln(x_i) = 0$
   
   Substituting the MLE for $\theta$:
   $-n\psi(k) - n\ln\left(\frac{\bar{x}}{k}\right) + \sum_{i=1}^{n}\ln(x_i) = 0$
   
   Simplifying:
   $-n\psi(k) - n\ln(\bar{x}) + n\ln(k) + \sum_{i=1}^{n}\ln(x_i) = 0$
   
   Dividing by $n$:
   $-\psi(k) - \ln(\bar{x}) + \ln(k) + \frac{1}{n}\sum_{i=1}^{n}\ln(x_i) = 0$
   
   Rearranging:
   $\ln(k) - \psi(k) = \ln(\bar{x}) - \frac{1}{n}\sum_{i=1}^{n}\ln(x_i)$

This equation cannot be solved analytically for $k$ and requires numerical methods.

### Properties of Gamma Distribution MLEs

1. **Mean Estimation**:
   $$\hat{\mu}_{MLE} = \hat{k}_{MLE}\hat{\theta}_{MLE} = \hat{k}_{MLE}\frac{\bar{x}}{\hat{k}_{MLE}} = \bar{x}$$

2. **Variance Estimation**:
   $$\hat{\sigma}^2_{MLE} = \hat{k}_{MLE}\hat{\theta}_{MLE}^2 = \hat{k}_{MLE}\left(\frac{\bar{x}}{\hat{k}_{MLE}}\right)^2 = \frac{\bar{x}^2}{\hat{k}_{MLE}}$$

3. **Standard Deviation Estimation**:
   $$\hat{\sigma}_{MLE} = \sqrt{\hat{k}_{MLE}}\hat{\theta}_{MLE} = \sqrt{\hat{k}_{MLE}}\frac{\bar{x}}{\hat{k}_{MLE}} = \frac{\bar{x}}{\sqrt{\hat{k}_{MLE}}}$$

4. **Consistency**:
   The MLEs for the Gamma distribution parameters are consistent—they converge to the true parameter values as the sample size approaches infinity.

5. **Asymptotic Efficiency**:
   For large samples, the MLEs for Gamma distribution parameters achieve the Cramér-Rao lower bound.

6. **Skewness Estimation**:
   $$\hat{\text{Skewness}}_{MLE} = \frac{2}{\sqrt{\hat{k}_{MLE}}}$$

7. **Kurtosis Estimation**:
   $$\hat{\text{Kurtosis}}_{MLE} = \frac{6}{\hat{k}_{MLE}}$$

## Method of Moments Alternative

When numerical solutions for MLE are challenging, the Method of Moments provides alternative estimates:

$$\hat{k}_{MM} = \frac{(\bar{x})^2}{s^2}$$

$$\hat{\theta}_{MM} = \frac{s^2}{\bar{x}}$$

where $s^2$ is the sample variance.

## Applications

The Gamma distribution is commonly used to model:
- Waiting times between Poisson events
- Rainfall amounts
- Insurance claim sizes
- Lifetime data in reliability engineering
- Service times in queuing theory
- Financial return distributions

## Related Topics

- [[L2_1_Gamma_Distribution|Gamma Distribution]]
- [[L2_4_MLE|Maximum Likelihood Estimation]]
- [[L2_3_Parameter_Estimation|Parameter Estimation]] 