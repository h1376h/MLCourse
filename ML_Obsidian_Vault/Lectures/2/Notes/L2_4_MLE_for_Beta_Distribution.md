# MLE for Beta Distribution

This document provides key concepts and formulas for Maximum Likelihood Estimation (MLE) applied to the Beta distribution.

## Key Concepts and Formulas

### Beta Distribution

The Beta distribution is characterized by two parameters:
- $\alpha$ - the first shape parameter 
- $\beta$ - the second shape parameter

Its probability density function (PDF) is:

$$f(x|\alpha,\beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$$

for $0 \leq x \leq 1$, $\alpha > 0$, and $\beta > 0$, where $B(\alpha,\beta)$ is the beta function defined as:

$$B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$$

and $\Gamma$ is the gamma function.

### MLE for Beta Distribution Parameters

For data $X = \{x_1, x_2, \ldots, x_n\}$ following a Beta distribution:

#### MLE for Shape Parameters ($\alpha$ and $\beta$)

There is no closed-form solution for the MLEs of the Beta distribution parameters. They require solving the following system of equations numerically:

$$\psi(\alpha) - \psi(\alpha + \beta) = \frac{1}{n}\sum_{i=1}^{n}\ln(x_i)$$

$$\psi(\beta) - \psi(\alpha + \beta) = \frac{1}{n}\sum_{i=1}^{n}\ln(1-x_i)$$

where $\psi$ is the digamma function (the derivative of the logarithm of the gamma function).

**Derivation**:
1. **Likelihood function**:
   $L(\alpha,\beta|X) = \prod_{i=1}^{n} \frac{x_i^{\alpha-1}(1-x_i)^{\beta-1}}{B(\alpha,\beta)}$

2. **Log-likelihood**:
   $\ell(\alpha,\beta|X) = -n\ln B(\alpha,\beta) + (\alpha-1)\sum_{i=1}^{n}\ln(x_i) + (\beta-1)\sum_{i=1}^{n}\ln(1-x_i)$

3. **Differentiate with respect to $\alpha$**:
   $\frac{\partial\ell(\alpha,\beta|X)}{\partial\alpha} = -n\frac{\partial\ln B(\alpha,\beta)}{\partial\alpha} + \sum_{i=1}^{n}\ln(x_i)$
   
   Using the property $\frac{\partial\ln B(\alpha,\beta)}{\partial\alpha} = \psi(\alpha) - \psi(\alpha+\beta)$:
   $\frac{\partial\ell(\alpha,\beta|X)}{\partial\alpha} = -n[\psi(\alpha) - \psi(\alpha+\beta)] + \sum_{i=1}^{n}\ln(x_i)$

4. **Differentiate with respect to $\beta$**:
   $\frac{\partial\ell(\alpha,\beta|X)}{\partial\beta} = -n\frac{\partial\ln B(\alpha,\beta)}{\partial\beta} + \sum_{i=1}^{n}\ln(1-x_i)$
   
   Using the property $\frac{\partial\ln B(\alpha,\beta)}{\partial\beta} = \psi(\beta) - \psi(\alpha+\beta)$:
   $\frac{\partial\ell(\alpha,\beta|X)}{\partial\beta} = -n[\psi(\beta) - \psi(\alpha+\beta)] + \sum_{i=1}^{n}\ln(1-x_i)$

5. **Set derivatives to zero**:
   $-n[\psi(\alpha) - \psi(\alpha+\beta)] + \sum_{i=1}^{n}\ln(x_i) = 0$
   $-n[\psi(\beta) - \psi(\alpha+\beta)] + \sum_{i=1}^{n}\ln(1-x_i) = 0$
   
   This gives:
   $\psi(\alpha) - \psi(\alpha+\beta) = \frac{1}{n}\sum_{i=1}^{n}\ln(x_i)$
   $\psi(\beta) - \psi(\alpha+\beta) = \frac{1}{n}\sum_{i=1}^{n}\ln(1-x_i)$

These equations must be solved numerically to find $\hat{\alpha}_{MLE}$ and $\hat{\beta}_{MLE}$.

### Method of Moments Alternative

When numerical solutions for MLE are challenging, the Method of Moments provides alternative estimates:

$$\hat{\alpha}_{MM} = \bar{x}\left(\frac{\bar{x}(1-\bar{x})}{s^2} - 1\right)$$

$$\hat{\beta}_{MM} = (1-\bar{x})\left(\frac{\bar{x}(1-\bar{x})}{s^2} - 1\right)$$

where $\bar{x}$ is the sample mean and $s^2$ is the sample variance.

### Properties of Beta Distribution MLEs

1. **Mean Estimation**:
   $$\hat{\mu}_{MLE} = \frac{\hat{\alpha}_{MLE}}{\hat{\alpha}_{MLE} + \hat{\beta}_{MLE}}$$

2. **Variance Estimation**:
   $$\hat{\sigma}^2_{MLE} = \frac{\hat{\alpha}_{MLE}\hat{\beta}_{MLE}}{(\hat{\alpha}_{MLE} + \hat{\beta}_{MLE})^2(\hat{\alpha}_{MLE} + \hat{\beta}_{MLE} + 1)}$$

3. **Standard Deviation Estimation**:
   $$\hat{\sigma}_{MLE} = \sqrt{\frac{\hat{\alpha}_{MLE}\hat{\beta}_{MLE}}{(\hat{\alpha}_{MLE} + \hat{\beta}_{MLE})^2(\hat{\alpha}_{MLE} + \hat{\beta}_{MLE} + 1)}}$$

4. **Consistency**:
   The MLEs for the Beta distribution parameters are consistent—they converge to the true parameter values as the sample size approaches infinity.

5. **Asymptotic Efficiency**:
   For large samples, the MLEs for Beta distribution parameters achieve the Cramér-Rao lower bound.

6. **Skewness Estimation**:
   $$\hat{\text{Skewness}}_{MLE} = \frac{2(\hat{\beta}_{MLE}-\hat{\alpha}_{MLE})\sqrt{\hat{\alpha}_{MLE}+\hat{\beta}_{MLE}+1}}{(\hat{\alpha}_{MLE}+\hat{\beta}_{MLE}+2)\sqrt{\hat{\alpha}_{MLE}\hat{\beta}_{MLE}}}$$

7. **Kurtosis Estimation**:
   $$\hat{\text{Kurtosis}}_{MLE} = \frac{6[(\hat{\alpha}_{MLE}-\hat{\beta}_{MLE})^2(\hat{\alpha}_{MLE}+\hat{\beta}_{MLE}+1)-\hat{\alpha}_{MLE}\hat{\beta}_{MLE}(\hat{\alpha}_{MLE}+\hat{\beta}_{MLE}+2)]}{\hat{\alpha}_{MLE}\hat{\beta}_{MLE}(\hat{\alpha}_{MLE}+\hat{\beta}_{MLE}+2)(\hat{\alpha}_{MLE}+\hat{\beta}_{MLE}+3)}$$

## Applications

The Beta distribution is commonly used to model:
- Probabilities and proportions
- Random variables bounded between 0 and 1
- Success rates
- Prior distributions in Bayesian inference
- Time allocation in project management
- Variation in percentages

## Related Topics

- [[L2_1_Beta_Distribution|Beta Distribution]]
- [[L2_4_MLE|Maximum Likelihood Estimation]]
- [[L2_3_Parameter_Estimation|Parameter Estimation]] 