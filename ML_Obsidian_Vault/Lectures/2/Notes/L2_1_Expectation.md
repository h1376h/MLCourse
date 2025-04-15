# Expectation

Expectation (or Expected Value) is a fundamental concept in probability theory that represents the long-run average value of a random variable. It's a central measure that helps quantify the center of a probability distribution.

## Definition

The expected value of a random variable $X$, denoted $E[X]$, is the probability-weighted average of all possible values of $X$:

- **Discrete Case**: $$E[X] = \sum_x x \cdot P(X=x)$$
- **Continuous Case**: $$E[X] = \int x \cdot f(x)dx$$

Where $P(X=x)$ is the probability mass function and $f(x)$ is the probability density function.

## Properties

### Linearity
- $E[X + Y] = E[X] + E[Y]$
- $E[aX] = a \cdot E[X]$ for any constant $a$

### Independence
- If $X$ and $Y$ are independent, $E[XY] = E[X] \cdot E[Y]$

### Law of the Unconscious Statistician
- $E[g(X)] = \sum g(x) \cdot P(X=x)$ for discrete random variables
- $E[g(X)] = \int g(x) \cdot f(x)dx$ for continuous random variables

## Expectation for Common Distributions

| Distribution | Expected Value |
|--------------|---------------|
| Bernoulli$(p)$ | $p$ |
| Binomial$(n,p)$ | $np$ |
| Poisson$(\lambda)$ | $\lambda$ |
| Uniform$(a,b)$ | $(a+b)/2$ |
| Normal$(\mu,\sigma^2)$ | $\mu$ |
| Exponential$(\lambda)$ | $1/\lambda$ |

## Applications in ML

### Loss Function Minimization
- Empirical Risk Minimization aims to minimize the expected value of a loss function
- $$E[L(\theta,X)] = \sum L(\theta,x) \cdot P(X=x)$$

### Parameter Estimation
- Expected values often correspond to parameters we want to estimate
- Sample mean estimates the population mean: $E[X]$

### Prediction
- Expected value provides a central prediction point
- In regression, we predict $E[Y|X]$

### Feature Engineering
- Moment-based features use expected values
- Mean, variance, skewness all involve expected values

## Relationship to Other Concepts

- **Mean**: The expected value of a random variable
- **Variance**: $E[(X - E[X])^2]$ - measures dispersion around the expected value
- **Covariance**: $E[(X - E[X])(Y - E[Y])]$ - measures joint variability
- **Higher Moments**: Used to characterize distribution shapes beyond mean and variance

## Related Topics

- [[L2_1_Basic_Probability|Basic Probability]]: The foundation for expected value calculation
- [[L2_1_Variance|Variance]]: A measure of dispersion derived from expectation
- [[L2_1_Expectation_Examples|Expectation_Examples]]: Practical examples of calculating and interpreting expected values
- [[L2_1_Normal_Distribution|Normal_Distribution]]: A key distribution defined by expectation and variance
- [[L2_3_Parameter_Estimation|Parameter_Estimation]]: Methods to estimate expected values 