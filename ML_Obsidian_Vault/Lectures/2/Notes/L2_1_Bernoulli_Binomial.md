# Bernoulli and Binomial Distributions

Bernoulli and Binomial distributions are fundamental probability distributions that model binary outcomes and form the basis for many machine learning algorithms.

## Bernoulli Distribution

### Definition
- Models a single binary outcome (success/failure, 0/1, heads/tails)
- Governed by a single parameter: probability of success $p$ $(0 \leq p \leq 1)$
- Random variable $X$ can take only two values: $X \in \{0, 1\}$

### Probability Mass Function (PMF)
- $P(X = 1) = p$
- $P(X = 0) = 1 - p$
- Compact form: $P(X = x) = p^x \cdot (1-p)^{(1-x)}$ for $x \in \{0, 1\}$

### Properties
- **Mean (Expected Value)**: $E[X] = p$
- **Variance**: $\text{Var}(X) = p(1-p)$
- **Standard Deviation**: $\sigma = \sqrt{p(1-p)}$
- **Moment Generating Function**: $M_X(t) = (1-p) + pe^t$

### Application in Machine Learning
- Binary classification problems
- Modeling presence/absence of features
- Bernoulli trials in reinforcement learning
- Foundation for logistic regression

## Binomial Distribution

### Definition
- Extension of the Bernoulli distribution to multiple independent trials
- Models the number of successes in $n$ independent Bernoulli trials
- Governed by two parameters:
  - $n$: number of trials (positive integer)
  - $p$: probability of success in each trial $(0 \leq p \leq 1)$

### Probability Mass Function (PMF)
- For $X \sim \text{Bin}(n, p)$, the probability of $k$ successes is:
- $P(X = k) = \binom{n}{k} p^k (1-p)^{(n-k)}$ for $k \in \{0, 1, 2, \ldots, n\}$
- Where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient

### Properties
- **Mean (Expected Value)**: $E[X] = np$
- **Variance**: $\text{Var}(X) = np(1-p)$
- **Standard Deviation**: $\sigma = \sqrt{np(1-p)}$
- **Skewness**: $\frac{1-2p}{\sqrt{np(1-p)}}$
- **Kurtosis**: $3 + \frac{1-6p(1-p)}{np(1-p)}$

### Relationship to Other Distributions
- Sum of $n$ independent Bernoulli random variables with same parameter $p$
- Approximates the normal distribution for large $n$ (Central Limit Theorem)
- When $n$ is large and $p$ is small, can be approximated by Poisson distribution

### Application in Machine Learning
- Classification with multiple independent features
- Ensemble methods (e.g., bagging, random forests)
- A/B testing and experimental design
- Modeling count data with fixed number of trials

## Related Topics
- [[L2_1_Basic_Probability|Basic Probability]]
- [[L2_1_Probability_Distributions|Probability Distributions]]
- [[L2_1_Multinomial_Distribution|Multinomial Distribution]] 