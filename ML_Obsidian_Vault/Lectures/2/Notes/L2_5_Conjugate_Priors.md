# Conjugate Priors

## Definition

In Bayesian statistics, a **conjugate prior** is a prior distribution that, when combined with a likelihood function, yields a posterior distribution of the same family as the prior. This mathematical convenience simplifies Bayesian calculations significantly.

## Key Properties

- Allows for closed-form posterior computation
- Enables sequential updating of beliefs as new data arrives
- Provides interpretable hyperparameters as "prior observations"
- Simplifies the mathematical complexity in Bayesian inference

## Important Conjugate Prior Pairs

| Likelihood Model | Parameter | Conjugate Prior | Posterior |
|------------------|-----------|-----------------|-----------|
| Bernoulli/Binomial | p (probability) | Beta(α, β) | Beta(α + successes, β + failures) |
| Poisson | λ (rate) | Gamma(α, β) | Gamma(α + sum(x), β + n) |
| Normal (known variance σ²) | μ (mean) | Normal(μ₀, σ₀²) | Normal(μ', σ'²) |
| Normal (known mean μ) | σ² (variance) | Inverse-Gamma(α, β) | Inverse-Gamma(α + n/2, β + sum((x-μ)²)/2) |
| Multinomial | p (probability vector) | Dirichlet(α) | Dirichlet(α + counts) |

## Why Use Conjugate Priors?

1. **Computational Efficiency**: Closed-form solutions avoid the need for complex numerical integration or sampling methods
2. **Interpretability**: Hyperparameters can often be interpreted as "pseudo-counts" or "prior observations"
3. **Sequential Learning**: Allows for easy updating as new data arrives
4. **Mathematical Tractability**: Simplifies derivations and analysis

## Relationship to MAP Estimation

Maximum A Posteriori (MAP) estimation can be viewed as finding the mode of the posterior distribution. When using conjugate priors:

1. The MAP estimate often has a simple closed-form expression
2. The influence of the prior diminishes as more data is observed
3. MAP estimation with conjugate priors provides a regularized alternative to Maximum Likelihood Estimation

See [[L2_5_Conjugate_Priors_Examples|Conjugate_Priors_Examples]] for practical implementations and visualizations of these concepts.

## References

- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer. Chapter 2.
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press. Chapter 3. 