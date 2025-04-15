# Maximum Likelihood

Maximum Likelihood Estimation (MLE) is a fundamental statistical method for estimating the parameters of a probability distribution from observed data. It finds parameter values that maximize the likelihood function.

## Core Concept

- The likelihood function measures how probable the observed data is under different parameter values
- MLE finds parameter values that make the observed data most probable
- Mathematically: $\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta|X) = \arg\max_{\theta} P(X|\theta)$

## Key Properties

- **Consistency**: As sample size increases, MLE converges to the true parameter value
- **Asymptotic normality**: For large samples, MLEs are approximately normally distributed
- **Efficiency**: MLEs achieve the Cramér-Rao lower bound asymptotically
- **Invariance**: If $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$

## MLE Process

1. **Formulate the likelihood function**: $L(\theta|X) = P(X|\theta)$
2. **Take the logarithm** (simplifies computation): $\ell(\theta|X) = \log L(\theta|X)$
3. **Find critical points**: Take derivative and set equal to zero: $\frac{d\ell(\theta|X)}{d\theta} = 0$
4. **Verify it's a maximum**: Check the second derivative is negative
5. **Solve for parameters**: Find $\hat{\theta}$ that maximizes likelihood

## Advantages

- No prior knowledge required
- Often has closed-form solutions
- Consistent estimator (converges to true value with more data)
- Basis for many statistical tests and metrics

## Limitations

- May lead to overfitting with small samples
- No incorporation of prior knowledge (unlike Bayesian methods)
- Point estimates without uncertainty quantification
- May fail for complex models with many parameters

## Applications

- **Regression**: Finding coefficients that maximize likelihood of observed data
- **Classification**: Estimating class probabilities from training examples
- **Density Estimation**: Fitting distributions to data
- **Time Series**: Estimating parameters for forecasting models

## Examples

- [[L2_4_MLE_Examples|MLE_Examples]]: More practical examples of MLE

## Related Topics

- [[L2_3_Likelihood|Likelihood]]: The mathematical function maximized by MLE
- [[L2_3_Probability_vs_Likelihood|Probability_vs_Likelihood]]: Understanding the distinction
- [[L2_3_Parameter_Estimation|Parameter_Estimation]]: Overview of estimation methods
