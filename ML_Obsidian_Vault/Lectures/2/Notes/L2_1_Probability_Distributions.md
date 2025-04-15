# Probability Distributions

Probability distributions describe how probabilities are distributed over the possible values a random variable can take.

## Discrete Probability Distributions

For detailed information about discrete distributions, see [[L2_1_Discrete_Distributions|Discrete Distributions]].

### Bernoulli and Binomial Distribution
- Models binary outcomes and their counts
- See [[L2_1_Bernoulli_Binomial|Bernoulli and Binomial Distributions]] for details

### Poisson Distribution
- Models the number of events occurring in a fixed interval
- Parameter: $\lambda$ (average rate of occurrence)
- PMF: $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$
- Mean: $E[X] = \lambda$
- Variance: $\text{Var}(X) = \lambda$
- Used in: Rare event modeling, queuing theory, network traffic

### Geometric Distribution
- Models the number of trials until the first success
- Parameter: $p$ (probability of success)
- PMF: $P(X = k) = (1-p)^{k-1} p$
- Mean: $E[X] = \frac{1}{p}$
- Variance: $\text{Var}(X) = \frac{1-p}{p^2}$
- Used in: Reliability testing, waiting time problems

### Uniform Discrete Distribution
- Equal probability for all values in a range
- Parameters: $a$, $b$ (minimum and maximum values)
- PMF: $P(X = x) = \frac{1}{b-a+1}$ for $a \leq x \leq b$
- Mean: $E[X] = \frac{a+b}{2}$
- Variance: $\text{Var}(X) = \frac{(b-a+1)^2-1}{12}$
- Used in: Randomization, simple modeling

## Continuous Probability Distributions

For detailed information about continuous distributions, see [[L2_1_Continuous_Distributions|Continuous Distributions]].

### Uniform Distribution
- Constant probability density over an interval
- See [[L2_1_Uniform_Distribution|Uniform Distribution]] for details

### Normal (Gaussian) Distribution
- Bell-shaped curve, fundamental in statistics
- See [[L2_1_Normal_Distribution|Normal Distribution]] for details

### Exponential Distribution
- Models time between events in a Poisson process
- See [[L2_1_Exponential_Distribution|Exponential Distribution]] for details

### Beta Distribution
- Models probabilities or proportions
- See [[L2_1_Beta_Distribution|Beta Distribution]] for details

### Gamma Distribution
- Generalizes exponential distribution
- See [[L2_1_Gamma_Distribution|Gamma Distribution]] for details

## Properties of Distributions

### Moments
- **Mean/Expected Value**: First moment, average value
- **Variance**: Second central moment, spread of distribution
- **Skewness**: Third standardized moment, asymmetry
- **Kurtosis**: Fourth standardized moment, tailedness

### Transformations
- Linear transformation: $Y = aX + b$
  - $E[Y] = aE[X] + b$
  - $\text{Var}(Y) = a^2\text{Var}(X)$
- Function transformation: $Y = g(X)$
  - For non-linear transformations, distribution shape changes

## Related Topics
- [[L2_1_Basic_Probability|Basic Probability]]: Foundation concepts
- [[L2_1_Normal_Distribution|Normal Distribution]]: Detailed exploration of Gaussian distribution
- [[L2_1_Beta_Distribution|Beta Distribution]]: Detailed exploration of Beta distribution
- [[L2_1_Joint_Probability|Joint Probability]]: Distributions over multiple random variables
- [[L2_1_Expectation|Expectation]]: Expected values and their properties
- [[L2_1_Discrete_Distributions|Discrete Distributions]]: Detailed exploration of discrete distributions
- [[L2_1_Continuous_Distributions|Continuous Distributions]]: Detailed exploration of continuous distributions
- [[L2_1_PMF_PDF_CDF|PMF, PDF, and CDF]]: Understanding probability distribution functions
- [[L2_1_Transformations|Transformations]]: Understanding transformations of random variables 