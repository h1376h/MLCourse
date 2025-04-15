# Multinomial Distribution

The multinomial distribution is a generalization of the binomial distribution to multiple categories, making it fundamental for modeling categorical data in machine learning.

## Definition
- Models the probability of counts for each category in a fixed number of independent trials
- Each trial results in exactly one of $k$ possible categories
- Governed by parameters:
  - $n$: total number of trials (positive integer)
  - $p_1, p_2, \ldots, p_k$: probabilities of each category $(0 \leq p_i \leq 1, \sum p_i = 1)$

## Probability Mass Function (PMF)
- For random vector $\mathbf{X} = (X_1, X_2, \ldots, X_k)$ where $X_i$ represents the count of category $i$:
- $P(X_1 = x_1, X_2 = x_2, \ldots, X_k = x_k) = \frac{n!}{x_1! \times x_2! \times \ldots \times x_k!} \times p_1^{x_1} \times p_2^{x_2} \times \ldots \times p_k^{x_k}$
- Where $x_1 + x_2 + \ldots + x_k = n$

## Notation
- $\mathbf{X} \sim \text{Multinomial}(n, \mathbf{p})$
- Where $\mathbf{p} = (p_1, p_2, \ldots, p_k)$ is the vector of probabilities

## Properties
- **Constraint**: $\sum X_i = n$
- **Expected Value**: $E[X_i] = n \times p_i$
- **Variance**: $\text{Var}(X_i) = n \times p_i \times (1 - p_i)$
- **Covariance**: $\text{Cov}(X_i, X_j) = -n \times p_i \times p_j$ (for $i \neq j$)

## Special Cases
- When $k = 2$, the multinomial distribution reduces to the binomial distribution
- Each individual category follows a binomial distribution: $X_i \sim \text{Binomial}(n, p_i)$

## Relationship to Other Distributions
- **Categorical Distribution**: The multinomial is a sum of $n$ independent categorical random variables
- **Dirichlet Distribution**: The conjugate prior for the multinomial distribution
- **Multivariate Normal**: For large $n$, approximates a multivariate normal distribution

## Application in Machine Learning
- **Text Classification**: Modeling word frequencies in documents
- **Multi-class Classification**: Probabilistic prediction of multiple classes
- **Topic Modeling**: Distribution over topics in documents
- **Natural Language Processing**: Language models and text generation
- **Image Recognition**: Pixel intensity distributions across categories

## Mathematical Formulation

The general form of the multinomial coefficient is:

$$\binom{n}{x_1, x_2, \ldots, x_k} = \frac{n!}{x_1! \times x_2! \times \ldots \times x_k!}$$

The complete probability mass function is:

$$P(X_1 = x_1, X_2 = x_2, \ldots, X_k = x_k) = \binom{n}{x_1, x_2, \ldots, x_k} \prod_{i=1}^k p_i^{x_i}$$

## Related Topics
- [[L2_1_Bernoulli_Binomial|Bernoulli and Binomial Distributions]]
- [[L2_1_Probability_Distributions|Probability Distributions]]
- [[L2_1_Independence|Independence]] 