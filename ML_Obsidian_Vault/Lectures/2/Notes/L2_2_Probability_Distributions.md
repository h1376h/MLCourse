# Probability Distributions

Probability distributions describe how probabilities are distributed over the possible values a random variable can take.

## Discrete Probability Distributions

### Bernoulli Distribution
- Models a binary outcome (success/failure)
- Parameter: p (probability of success)
- PMF: P(X = 1) = p, P(X = 0) = 1-p
- Mean: p
- Variance: p(1-p)
- Used in: Binary classification, coin flips

### Binomial Distribution
- Models the number of successes in n independent Bernoulli trials
- Parameters: n (number of trials), p (probability of success)
- PMF: P(X = k) = (n choose k) p^k (1-p)^(n-k)
- Mean: np
- Variance: np(1-p)
- Used in: Quality control, testing, survey analysis

### Poisson Distribution
- Models the number of events occurring in a fixed interval
- Parameter: λ (average rate of occurrence)
- PMF: P(X = k) = (λ^k e^(-λ))/k!
- Mean: λ
- Variance: λ
- Used in: Rare event modeling, queuing theory, network traffic

### Geometric Distribution
- Models the number of trials until the first success
- Parameter: p (probability of success)
- PMF: P(X = k) = (1-p)^(k-1) p
- Mean: 1/p
- Variance: (1-p)/p²
- Used in: Reliability testing, waiting time problems

### Uniform Discrete Distribution
- Equal probability for all values in a range
- Parameters: a, b (minimum and maximum values)
- PMF: P(X = x) = 1/(b-a+1) for a ≤ x ≤ b
- Mean: (a+b)/2
- Variance: ((b-a+1)²-1)/12
- Used in: Randomization, simple modeling

## Continuous Probability Distributions

### Uniform Continuous Distribution
- Constant probability density over an interval
- Parameters: a, b (interval limits)
- PDF: f(x) = 1/(b-a) for a ≤ x ≤ b
- Mean: (a+b)/2
- Variance: (b-a)²/12
- Used in: Random number generation, uninformative priors

### Normal (Gaussian) Distribution
- Bell-shaped curve, fundamental in statistics
- Parameters: μ (mean), σ² (variance)
- PDF: f(x) = (1/(σ√(2π))) e^(-(x-μ)²/(2σ²))
- Mean: μ
- Variance: σ²
- Used in: Natural phenomena, measurement errors, central limit theorem applications
- See also: [[L2_1_Normal_Distribution|Normal Distribution]]

### Exponential Distribution
- Models time between events in a Poisson process
- Parameter: λ (rate parameter)
- PDF: f(x) = λe^(-λx) for x ≥ 0
- Mean: 1/λ
- Variance: 1/λ²
- Used in: Lifetime modeling, waiting times, reliability analysis

### Beta Distribution
- Models probabilities or proportions
- Parameters: α, β (shape parameters)
- PDF: f(x) = (x^(α-1)(1-x)^(β-1))/B(α,β) for 0 ≤ x ≤ 1
- Mean: α/(α+β)
- Variance: αβ/((α+β)²(α+β+1))
- Used in: Bayesian inference, modeling probabilities
- See also: [[L2_1_Beta_Distribution|Beta Distribution]]

### Gamma Distribution
- Generalizes exponential distribution
- Parameters: k (shape), θ (scale)
- PDF: f(x) = (x^(k-1)e^(-x/θ))/(Γ(k)θ^k) for x > 0
- Mean: kθ
- Variance: kθ²
- Used in: Waiting times, rainfall amounts, insurance claims

## Properties of Distributions

### Moments
- **Mean/Expected Value**: First moment, average value
- **Variance**: Second central moment, spread of distribution
- **Skewness**: Third standardized moment, asymmetry
- **Kurtosis**: Fourth standardized moment, tailedness

### Transformations
- Linear transformation: Y = aX + b
  - E[Y] = aE[X] + b
  - Var(Y) = a²Var(X)
- Function transformation: Y = g(X)
  - For non-linear transformations, distribution shape changes

## Related Topics
- [[L2_1_Basic_Probability|Basic Probability]]: Foundation concepts
- [[L2_1_Normal_Distribution|Normal Distribution]]: Detailed exploration of Gaussian distribution
- [[L2_1_Beta_Distribution|Beta Distribution]]: Detailed exploration of Beta distribution
- [[L2_1_Joint_Probability|Joint Probability]]: Distributions over multiple random variables
- [[L2_1_Expectation|Expectation]]: Expected values and their properties 