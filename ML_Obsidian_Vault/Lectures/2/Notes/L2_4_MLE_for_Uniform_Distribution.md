# MLE for Uniform Distribution

This document provides key concepts and formulas for Maximum Likelihood Estimation (MLE) applied to the Uniform distribution.

## Key Concepts and Formulas

### Uniform Distribution

The Uniform distribution is characterized by two parameters:
- $a$ - the lower bound
- $b$ - the upper bound

Its probability density function (PDF) is:

$$f(x|a,b) = \frac{1}{b-a}$$

for $a \leq x \leq b$.

### MLE for Uniform Distribution Parameters

For data $X = \{x_1, x_2, \ldots, x_n\}$ following a Uniform distribution on $[a, b]$:

#### MLE for Lower Bound ($a$)

$$\hat{a}_{MLE} = \min(x_1, x_2, \ldots, x_n)$$

#### MLE for Upper Bound ($b$)

$$\hat{b}_{MLE} = \max(x_1, x_2, \ldots, x_n)$$

**Derivation**:
1. **Likelihood function**:
   $L(a,b|X) = \prod_{i=1}^{n} \frac{1}{b-a} = \frac{1}{(b-a)^n}$
   
   This assumes that all $x_i$ values are within the interval $[a,b]$. If any $x_i < a$ or $x_i > b$, then the likelihood becomes zero.

2. **Log-likelihood**:
   $\ell(a,b|X) = -n \ln(b-a)$

3. **To maximize the log-likelihood**:
   - We need to minimize $b-a$
   - $a$ must be ≤ the smallest $x_i$
   - $b$ must be ≥ the largest $x_i$
   
   To minimize $b-a$ while satisfying these constraints, we set:
   $a = \min(x_1, x_2, \ldots, x_n)$ and $b = \max(x_1, x_2, \ldots, x_n)$

### Properties of Uniform Distribution MLEs

1. **Mean Estimation**:
   $$\hat{\mu}_{MLE} = \frac{\hat{a}_{MLE} + \hat{b}_{MLE}}{2}$$

2. **Variance Estimation**:
   $$\hat{\sigma}^2_{MLE} = \frac{(\hat{b}_{MLE} - \hat{a}_{MLE})^2}{12}$$

3. **Standard Deviation Estimation**:
   $$\hat{\sigma}_{MLE} = \frac{\hat{b}_{MLE} - \hat{a}_{MLE}}{2\sqrt{3}}$$

4. **Bias**:
   The MLEs for the uniform distribution boundaries are biased:
   - $E[\hat{a}_{MLE}] = a + \frac{b-a}{n+1}$
   - $E[\hat{b}_{MLE}] = b - \frac{b-a}{n+1}$
   
   As sample size increases, this bias becomes negligible.

5. **Consistency**:
   Despite being biased for finite samples, the MLEs are consistent—they converge to the true parameter values as the sample size approaches infinity.

6. **Efficiency**:
   The MLEs are the minimum variance unbiased estimators (MVUEs) for the uniform distribution parameters.

7. **Skewness**:
   For the uniform distribution, the skewness is 0 regardless of the parameter values.

8. **Kurtosis**:
   For the uniform distribution, the excess kurtosis is -1.2 regardless of the parameter values.

## Applications

The Uniform distribution is commonly used to model:
- Random number generation
- Quantization error in digital signal processing
- Round-off errors in numerical calculations
- Prior distributions in Bayesian inference when no prior information is available
- Waiting times with fixed minimum and maximum durations

## Related Topics

- [[L2_1_Uniform_Distribution|Uniform Distribution]]
- [[L2_4_MLE|Maximum Likelihood Estimation]]
- [[L2_3_Parameter_Estimation|Parameter Estimation]] 