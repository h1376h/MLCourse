# Discrete Probability Distributions

Discrete probability distributions describe the probability of outcomes for discrete random variables. These distributions are fundamental in machine learning for modeling categorical data and discrete events.

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

## Poisson Distribution

### Definition
- Models the number of events occurring in a fixed interval
- Governed by a single parameter: $\lambda$ (average rate of occurrence)
- Used for modeling rare events

### Probability Mass Function (PMF)
- $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$ for $k = 0, 1, 2, \ldots$
- Where $k!$ is the factorial of $k$

### Properties
- **Mean (Expected Value)**: $E[X] = \lambda$
- **Variance**: $\text{Var}(X) = \lambda$
- **Standard Deviation**: $\sigma = \sqrt{\lambda}$
- **Moment Generating Function**: $M_X(t) = e^{\lambda(e^t - 1)}$

## Geometric Distribution

### Definition
- Models the number of trials until the first success
- Governed by a single parameter: $p$ (probability of success)
- Memoryless property: $P(X > m + n | X > m) = P(X > n)$

### Probability Mass Function (PMF)
- $P(X = k) = (1-p)^{k-1}p$ for $k = 1, 2, 3, \ldots$

### Properties
- **Mean (Expected Value)**: $E[X] = \frac{1}{p}$
- **Variance**: $\text{Var}(X) = \frac{1-p}{p^2}$
- **Standard Deviation**: $\sigma = \sqrt{\frac{1-p}{p^2}}$
- **Moment Generating Function**: $M_X(t) = \frac{pe^t}{1 - (1-p)e^t}$

## Negative Binomial Distribution

### Definition
- Models the number of trials until $r$ successes
- Governed by two parameters:
  - $r$: number of successes required
  - $p$: probability of success in each trial

### Probability Mass Function (PMF)
- $P(X = k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}$ for $k = r, r+1, r+2, \ldots$

### Properties
- **Mean (Expected Value)**: $E[X] = \frac{r}{p}$
- **Variance**: $\text{Var}(X) = \frac{r(1-p)}{p^2}$
- **Standard Deviation**: $\sigma = \sqrt{\frac{r(1-p)}{p^2}}$
- **Moment Generating Function**: $M_X(t) = \left(\frac{pe^t}{1 - (1-p)e^t}\right)^r$

## Applications in Machine Learning

1. **Classification Problems**
   - Bernoulli for binary classification
   - Binomial for multi-class classification
   - Poisson for count data

2. **Natural Language Processing**
   - Poisson for word counts
   - Geometric for word lengths
   - Negative binomial for overdispersed counts

3. **Computer Vision**
   - Bernoulli for binary images
   - Binomial for pixel intensities
   - Poisson for photon counting

4. **Reinforcement Learning**
   - Geometric for first success
   - Negative binomial for multiple successes
   - Poisson for event counting

## Related Topics
- [[L2_1_Basic_Probability|Basic Probability]]: Foundation concepts
- [[L2_1_Continuous_Distributions|Continuous Distributions]]: For continuous random variables
- [[L2_1_Examples|Probability Examples]]: Practical applications 