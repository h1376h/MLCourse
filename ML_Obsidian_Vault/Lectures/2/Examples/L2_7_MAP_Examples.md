# Maximum A Posteriori (MAP) Estimation Examples

This document provides examples and key concepts on Maximum A Posteriori (MAP) estimation to help you understand this important Bayesian approach to parameter estimation.

## Key Concepts and Formulas

Maximum A Posteriori (MAP) estimation is a Bayesian approach that combines prior knowledge with observed data to estimate parameters. Unlike Maximum Likelihood Estimation (which only considers the data), MAP incorporates prior beliefs about parameters before updating with observed data.

### The MAP Estimation Formula

$$\hat{\theta}_{\text{MAP}} = \underset{\theta}{\operatorname{argmax}}\ p(\theta|D)$$

Using Bayes' theorem, this can be expressed as:

$$\hat{\theta}_{\text{MAP}} = \underset{\theta}{\operatorname{argmax}}\ \frac{p(D|\theta)p(\theta)}{p(D)} = \underset{\theta}{\operatorname{argmax}}\ p(D|\theta)p(\theta)$$

Where:
- $\hat{\theta}_{\text{MAP}}$ = The MAP estimate (most probable parameter value)
- $\theta$ = Parameter being estimated
- $D$ = Observed data
- $p(\theta|D)$ = Posterior probability (probability of parameter given data)
- $p(D|\theta)$ = Likelihood (probability of data given parameter)
- $p(\theta)$ = Prior probability (initial belief about parameter)
- $p(D)$ = Evidence (marginal likelihood, constant with respect to θ)

## Practice Questions

For practice multiple-choice questions on Maximum A Posteriori estimation, see:
- [[L2_7_MCQ|MAP Multiple Choice Questions]]

## Examples

1. [[L2_7_MAP_Normal|Normal Distribution MAP]]: Examples of MAP for Normal distributions
2. [[L2_7_MAP_Bernoulli|Bernoulli Distribution MAP]]: Examples of MAP for Bernoulli distributions
3. [[L2_7_MAP_Formula|MAP Formula Examples]]: Detailed applications of MAP formulas with calculations
4. [[L2_7_MAP_Special_Cases|MAP Special Cases]]: Theoretical edge cases and special situations
