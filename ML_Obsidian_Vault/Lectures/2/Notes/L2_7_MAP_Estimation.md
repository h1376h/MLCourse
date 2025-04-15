# Maximum A Posteriori (MAP) Estimation

Maximum A Posteriori (MAP) estimation is a Bayesian method for parameter estimation that combines the likelihood function with prior knowledge. It represents a middle ground between pure maximum likelihood estimation and full Bayesian inference.

## Core Concept

MAP estimation finds the mode (most likely value) of the posterior distribution:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|X) = \arg\max_{\theta} \frac{P(X|\theta) \cdot P(\theta)}{P(X)}$$

Since $P(X)$ doesn't depend on $\theta$, this simplifies to:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(X|\theta) \cdot P(\theta)$$

Or in log form:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} \log P(X|\theta) + \log P(\theta)$$

## Relationship to MLE

- **MLE**: $\hat{\theta}_{MLE} = \arg\max_{\theta} P(X|\theta)$
- **MAP**: $\hat{\theta}_{MAP} = \arg\max_{\theta} P(X|\theta) \cdot P(\theta)$

When the prior is uniform (i.e., $P(\theta) \propto$ constant), MAP becomes equivalent to MLE.

## Advantages of MAP

1. **Incorporates Prior Knowledge**: Uses domain expertise or previously observed data
2. **Regularization Effect**: Prior distributions can prevent overfitting
3. **Works with Limited Data**: More stable than MLE when sample size is small
4. **Computational Simplicity**: Provides point estimates, unlike full Bayesian inference

## Common Prior Choices

1. **Gaussian Prior**: $P(\theta) \propto \exp(-\lambda\|\theta\|^2)$ 
   - Leads to L2 regularization (Ridge regression)
   
2. **Laplace Prior**: $P(\theta) \propto \exp(-\lambda\|\theta\|_1)$
   - Leads to L1 regularization (Lasso regression)
   
3. **Beta Prior**: For parameters representing probabilities
   - Controls strength of regularization via shape parameters

4. **Conjugate Priors**: Makes computation analytically tractable
   - See [[L2_5_Conjugate_Priors|Conjugate_Priors]] for details

## MAP vs. Full Bayesian Inference

- **MAP**: Provides a point estimate (mode of posterior)
- **Full Bayesian**: Uses entire posterior distribution
- **Tradeoff**: Computational simplicity vs complete uncertainty quantification

## Applications in Machine Learning

- **Regularized Regression**: Ridge, Lasso, Elastic Net
- **Bayesian Networks**: Parameter estimation
- **Image Processing**: Image restoration and reconstruction
- **Natural Language Processing**: Topic models

## Implementation Considerations

- Often solved using optimization algorithms (gradient descent, Newton's method)
- Convex optimization when both likelihood and prior are log-concave
- May require numerical methods for complex models

## Related Concepts

- [[L2_4_Maximum_Likelihood|Maximum_Likelihood]]
- [[L2_5_Bayesian_Inference|Bayesian_Inference]]
- [[L2_7_Full_Bayesian_Inference|Full_Bayesian_Inference]]
- [[L2_7_MAP_Examples|MAP_Examples]]
- [[L2_7_MAP_Formula_Explanation|MAP_Formula_Explanation]] 