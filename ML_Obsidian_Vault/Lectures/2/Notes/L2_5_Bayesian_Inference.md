# Bayesian Inference

Bayesian inference is a method of statistical inference that updates the probability of a hypothesis as more evidence becomes available. It applies Bayes' theorem to incorporate prior knowledge when drawing conclusions from data.

## Core Components

1. **Prior Probability**: Initial belief about the parameter before observing data
2. **Likelihood Function**: How likely the observed data is given different parameter values
3. **Posterior Probability**: Updated belief after incorporating new evidence
4. **Evidence**: Normalizing factor that ensures the posterior is a valid probability distribution

## Bayes' Theorem

$$P(\theta|X) = \frac{P(X|\theta) \cdot P(\theta)}{P(X)}$$

Where:
- $P(\theta|X)$ is the posterior probability
- $P(X|\theta)$ is the likelihood
- $P(\theta)$ is the prior probability
- $P(X)$ is the evidence

## Applications in Machine Learning

- **Parameter Estimation**: See [[L2_3_Parameter_Estimation|Parameter_Estimation]] for details on MLE, MAP, and Full Bayesian approaches
- **Probabilistic Models**: Bayesian Networks, Hidden Markov Models
- **Classification**: Naive Bayes classifiers
- **Regression**: Bayesian Linear Regression

## Advantages

- Incorporates prior knowledge
- Provides full probability distributions as output
- Handles uncertainty naturally
- Allows incremental updating as more data arrives

## Common Prior Distributions

### Continuous Parameters
- **Normal Distribution**: For location parameters like means
- **Gamma Distribution**: For positive parameters like variances
- **Inverse-Gamma**: Alternative prior for variances

### Discrete Parameters
- **Beta Distribution**: For probability parameters (0-1 range)
- **Dirichlet Distribution**: Multivariate generalization of Beta
- **Poisson Distribution**: For count data

## Computational Methods

- **Analytical Solutions**: Available for conjugate prior-likelihood pairs
- **Markov Chain Monte Carlo (MCMC)**: For complex posteriors
- **Variational Inference**: Approximates posterior with simpler distributions
- **Laplace Approximation**: Approximates posterior with a normal distribution

## Related Concepts

- [[L2_1_Beta_Distribution|Beta_Distribution]]
- [[L2_1_Normal_Distribution|Normal_Distribution]]
- [[L2_3_Parameter_Estimation|Parameter_Estimation]]
- [[L2_7_Full_Bayesian_Inference|Full_Bayesian_Inference]] 