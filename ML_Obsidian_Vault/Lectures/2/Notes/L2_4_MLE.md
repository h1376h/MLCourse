# Maximum Likelihood Estimation (MLE)

Maximum Likelihood Estimation (MLE) provides a systematic approach to estimating parameters of statistical distributions by finding the parameter values that maximize the probability of observing the given data. This note covers the principles of MLE and references specific notes on MLE for various common distributions.

## Key Concepts and Formulas

### The Maximum Likelihood Principle

The general approach to maximum likelihood estimation follows these steps:

1. Define the likelihood function for the observed data given the parameters
2. Take the logarithm to obtain the log-likelihood (simplifies calculations by converting products to sums)
3. Find the parameter values that maximize the log-likelihood by taking derivatives
4. Verify that these values correspond to a maximum using the second derivative test (Hessian matrix for multivariate cases)

### The Likelihood Function

For independent and identically distributed (i.i.d.) data $X = \{x_1, x_2, \ldots, x_n\}$ and parameters $\theta$, the likelihood function is:

$$L(\theta|X) = P(X|\theta) = \prod_{i=1}^{n} f(x_i|\theta)$$

Where $f(x_i|\theta)$ is the probability density function (PDF) for continuous distributions or probability mass function (PMF) for discrete distributions, evaluated at observation $x_i$.

### The Log-Likelihood Function

Taking the logarithm of the likelihood function (which is monotonically increasing and thus preserves maxima):

$$\ell(\theta|X) = \log L(\theta|X) = \sum_{i=1}^{n} \log f(x_i|\theta)$$

The log transformation converts the product into a sum, which is typically easier to differentiate and optimize.

### Maximum Likelihood Estimator

The maximum likelihood estimator $\hat{\theta}_{MLE}$ is the value of $\theta$ that maximizes the log-likelihood:

$$\hat{\theta}_{MLE} = \arg\max_{\theta} \ell(\theta|X)$$

To find this value, we typically:
1. Take the partial derivatives of $\ell(\theta|X)$ with respect to each component of $\theta$
2. Set each derivative equal to zero (finding critical points)
3. Solve the resulting system of equations for $\theta$
4. Verify it's a maximum by examining the negative definiteness of the Hessian matrix

## MLE for Common Distributions

For detailed derivations, formulas, and properties of MLE for specific distributions, please refer to the following notes:

- [[L2_4_MLE_for_Normal_Distribution|MLE for Normal Distribution]]: Estimating mean ($\mu$) and variance ($\sigma^2$) parameters of Gaussian distributions
- [[L2_4_MLE_for_Exponential_Distribution|MLE for Exponential Distribution]]: Estimating the rate parameter ($\lambda$) of exponential distributions
- [[L2_4_MLE_for_Poisson_Distribution|MLE for Poisson Distribution]]: Estimating the rate parameter ($\lambda$) of Poisson count data
- [[L2_4_MLE_for_Bernoulli_Distribution|MLE for Bernoulli Distribution]]: Estimating the success probability ($p$) of binary outcomes
- [[L2_4_MLE_for_Binomial_Distribution|MLE for Binomial Distribution]]: Estimating the success probability ($p$) with fixed number of trials ($n$)
- [[L2_4_MLE_for_Uniform_Distribution|MLE for Uniform Distribution]]: Estimating the boundaries ($a$ and $b$) of uniform distributions
- [[L2_4_MLE_for_Gamma_Distribution|MLE for Gamma Distribution]]: Estimating shape ($\alpha$) and scale ($\beta$) parameters of gamma distributions
- [[L2_4_MLE_for_Beta_Distribution|MLE for Beta Distribution]]: Estimating shape parameters ($\alpha$ and $\beta$) of beta distributions

For practical examples and additional exercises on MLE, see [[L2_4_MLE_Examples|MLE Examples]].

## Key Insights

### Theoretical Properties
- **Consistency**: The MLE is consistent; as sample size increases, it converges to the true parameter value
- **Asymptotic Efficiency**: For large samples, MLE has minimum variance among all consistent estimators
- **Asymptotic Normality**: For large samples, MLE follows a normal distribution with mean at the true parameter value and variance given by the inverse Fisher Information
- **Invariance**: If $\hat{\theta}$ is the MLE of $\theta$, then for any function $g$, the MLE of $g(\theta)$ is $g(\hat{\theta})$
- **Sufficiency**: MLE depends only on sufficient statistics when they exist

### Practical Considerations
- MLEs may be biased for small samples (e.g., variance estimator for normal distribution is biased by a factor of $(n-1)/n$)
- MLEs may not exist in some cases, or there may be multiple maxima (non-identifiability)
- When closed-form solutions don't exist, numerical methods are required
- For large samples, the bias of MLEs usually becomes negligible
- MLE can be sensitive to outliers in some distributions

### Implementation Notes
1. For distributions without closed-form MLE solutions (like Beta and Gamma), numerical optimization methods like Newton-Raphson, gradient descent, or EM algorithm are typically used
2. For small samples, bias-corrected versions of MLEs may be preferred (e.g., using $n-1$ instead of $n$ in variance estimation)
3. Some distributions require specialized algorithms for parameter estimation
4. In practice, optimization constraints may be necessary to ensure parameters remain in valid ranges
5. Modern ML frameworks (PyTorch, TensorFlow, JAX) provide automatic differentiation tools that simplify MLE computation

## Extended Topics

For more comprehensive coverage of MLE-related topics, please refer to these specialized notes:

- [[L2_4_MLE_Applications|Applications of MLE]]: Detailed coverage of MLE applications across different domains and ML techniques
- [[L2_4_MLE_vs_MAP|Comparison of MLE and MAP]]: Analysis of differences between MLE and MAP estimation approaches

## Related Topics

- [[L2_1_Normal_Distribution|Normal_Distribution]]: Properties of the normal distribution
- [[L2_3_Parameter_Estimation|Parameter_Estimation]]: Overview of different estimation approaches
- [[L2_7_MAP_Estimation|MAP_Estimation]]: Bayesian alternative to MLE
- [[L2_5_Bayesian_Inference|Bayesian_Inference]]: Broader framework for statistical inference
- [[L2_2_Information_Theory|Information_Theory]]: Connection between MLE and information-theoretic principles
- [[L2_2_KL_Divergence|KL_Divergence]]: Relation to MLE in distribution fitting

## Quiz
- [[L2_4_Quiz]]: Test your understanding of maximum likelihood estimation 