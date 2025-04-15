# Variance

Variance is a statistical measure that quantifies the spread or dispersion of a random variable around its expected value. It's a fundamental concept in probability theory and statistics that helps characterize probability distributions and understand data variability.

## Definition

The variance of a random variable $X$, denoted $\text{Var}(X)$ or $\sigma^2$, is the expected value of the squared deviation from the mean:

- $$\text{Var}(X) = E[(X - E[X])^2]$$
- Alternative formula: $$\text{Var}(X) = E[X^2] - (E[X])^2$$

## Properties

### Non-negativity
- $\text{Var}(X) \geq 0$ for any random variable $X$
- $\text{Var}(X) = 0$ if and only if $X$ is constant (has no randomness)

### Scale Property
- $\text{Var}(aX) = a^2 \cdot \text{Var}(X)$ for any constant $a$

### Additivity for Independent Variables
- If $X$ and $Y$ are independent, $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$
- In general: $$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2 \cdot \text{Cov}(X,Y)$$

### Standard Deviation
- $\sigma = \sqrt{\text{Var}(X)}$, measured in the same units as $X$
- Used when we need a measure of dispersion in the original units

## Variance for Common Distributions

| Distribution | Variance |
|--------------|----------|
| Bernoulli$(p)$ | $p(1-p)$ |
| Binomial$(n,p)$ | $np(1-p)$ |
| Poisson$(\lambda)$ | $\lambda$ |
| Uniform$(a,b)$ | $(b-a)^2/12$ |
| Normal$(\mu,\sigma^2)$ | $\sigma^2$ |
| Exponential$(\lambda)$ | $1/\lambda^2$ |

## Applications in ML

### Model Evaluation
- Variance measures prediction consistency
- Low variance indicates stable predictions

### Bias-Variance Tradeoff
- Fundamental concept in model complexity
- $$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$
- Underfitting (high bias) vs. Overfitting (high variance)

### Feature Selection
- Features with high variance often contain more information
- Variance thresholding as a feature selection method

### Ensemble Methods
- Techniques like bagging reduce prediction variance
- Random forests combine trees to reduce overfitting

## Statistical Moments

Variance is the second central moment of a probability distribution. Statistical moments provide a way to characterize distribution shapes:

1. First Moment: Mean (Expected Value)
2. Second Central Moment: Variance
3. Third Standardized Moment: Skewness (asymmetry)
4. Fourth Standardized Moment: Kurtosis (tail heaviness)

## Covariance and Correlation

- **Covariance**: $\text{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])]$
- **Correlation**: $\rho = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$
- Correlation standardizes covariance to be between -1 and 1

## Related Topics

- [[L2_1_Expectation|Expectation]]: The foundation for variance calculation
- [[L2_1_Basic_Probability|Basic Probability]]: The theoretical basis for variance
- [[L2_1_Variance_Examples|Variance_Examples]]: Practical examples of calculating and interpreting variance
- [[L2_1_Normal_Distribution|Normal_Distribution]]: A key distribution defined by mean and variance
- [[L2_3_Parameter_Estimation|Parameter_Estimation]]: Methods to estimate variance from data 