# Maximum Likelihood Estimation Examples

This document provides examples and key concepts on Maximum Likelihood Estimation (MLE) to help you understand this important statistical method.

## Key Concepts and Formulas

Maximum Likelihood Estimation is a statistical method for estimating the parameters of a probability distribution by maximizing the likelihood function, which measures how likely the observed data is under different parameter values.

### The MLE Formula

$$\hat{\theta}_{MLE} = \underset{\theta}{\operatorname{argmax}}\ p(D|\theta)$$

Where:
- $\hat{\theta}_{MLE}$ = The MLE estimate
- $\theta$ = The parameter(s) being estimated
- $D$ = The observed data
- $p(D|\theta)$ = The likelihood function

For a normal distribution, the MLE formulas for mean and variance are:

$$\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}x_i$$

$$\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu}_{MLE})^2$$

Where:
- $\hat{\mu}_{MLE}$ = MLE estimate of mean
- $\hat{\sigma}^2_{MLE}$ = MLE estimate of variance
- $x_i$ = Individual observations
- $n$ = Number of observations

## Practice Questions

For practice multiple-choice questions on Maximum Likelihood Estimation, see:
- [[L2_4_MCQ|MLE Multiple Choice Questions]]

## Examples

1. [[L2_4_MLE_Power_Law|Power Law Distribution MLE]]: Examples of MLE for Power Law distributions
2. [[L2_4_MLE_Linear|Linear Distribution MLE]]: Examples of MLE for Linear distributions
3. [[L2_4_MLE_Normal|Normal Distribution MLE]]: Examples of MLE for Normal distributions
4. [[L2_4_MLE_Bernoulli|Bernoulli Distribution MLE]]: Examples of MLE for Bernoulli distributions
5. [[L2_4_MLE_Poisson|Poisson Distribution MLE]]: Examples of MLE for Poisson distributions
6. [[L2_4_MLE_Multinomial|Multinomial Distribution MLE]]: Examples of MLE for Multinomial distributions
