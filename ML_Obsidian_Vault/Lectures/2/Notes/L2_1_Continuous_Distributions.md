# Continuous Distributions

Continuous probability distributions are essential in machine learning for modeling real-valued data, time series, and continuous features.

## Common Continuous Distributions

### [[L2_1_Normal_Distribution|Normal (Gaussian) Distribution]]
- **Definition**: Symmetric, bell-shaped distribution
- **PDF**: $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- **CDF**: $F(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]$
- **Parameters**: $\mu$ (mean), $\sigma^2$ (variance)
- **Mean**: $\mu$
- **Variance**: $\sigma^2$
- **MGF**: $M_X(t) = e^{\mu t + \frac{1}{2}\sigma^2 t^2}$
- **Applications**: Error modeling, feature distributions, central limit theorem

### [[L2_1_Uniform_Distribution|Uniform Distribution]]
- **Definition**: Equal probability over a finite interval
- **PDF**: $f(x) = \frac{1}{b-a}$ for $x \in [a,b]$
- **CDF**: $F(x) = \frac{x-a}{b-a}$ for $x \in [a,b]$
- **Parameters**: $a$ (minimum), $b$ (maximum)
- **Mean**: $\mu = \frac{a+b}{2}$
- **Variance**: $\sigma^2 = \frac{(b-a)^2}{12}$
- **MGF**: $M_X(t) = \frac{e^{bt} - e^{at}}{t(b-a)}$
- **Applications**: Random number generation, prior distributions

### [[L2_1_Exponential_Distribution|Exponential Distribution]]
- **Definition**: Models waiting times between events in a Poisson process
- **PDF**: $f(x) = \lambda e^{-\lambda x}$ for $x \geq 0$
- **CDF**: $F(x) = 1 - e^{-\lambda x}$ for $x \geq 0$
- **Parameters**: $\lambda$ (rate parameter)
- **Mean**: $\mu = \frac{1}{\lambda}$
- **Variance**: $\sigma^2 = \frac{1}{\lambda^2}$
- **MGF**: $M_X(t) = \frac{\lambda}{\lambda - t}$ for $t < \lambda$
- **Applications**: Survival analysis, reliability, queuing theory

### [[L2_1_Gamma_Distribution|Gamma Distribution]]
- **Definition**: Generalization of exponential distribution
- **PDF**: $f(x) = \frac{\lambda^\alpha x^{\alpha-1} e^{-\lambda x}}{\Gamma(\alpha)}$ for $x > 0$
- **CDF**: $F(x) = \frac{\gamma(\alpha, \lambda x)}{\Gamma(\alpha)}$ where $\gamma$ is the lower incomplete gamma function
- **Parameters**: $\alpha$ (shape), $\lambda$ (rate)
- **Mean**: $\mu = \frac{\alpha}{\lambda}$
- **Variance**: $\sigma^2 = \frac{\alpha}{\lambda^2}$
- **MGF**: $M_X(t) = \left(\frac{\lambda}{\lambda - t}\right)^\alpha$ for $t < \lambda$
- **Applications**: Waiting times, reliability analysis

### [[L2_1_Beta_Distribution|Beta Distribution]]
- **Definition**: Models probabilities and proportions
- **PDF**: $f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$ for $x \in [0,1]$
- **CDF**: $F(x) = \frac{B(x; \alpha, \beta)}{B(\alpha, \beta)}$ where $B(x; \alpha, \beta)$ is the incomplete beta function
- **Parameters**: $\alpha, \beta$ (shape parameters)
- **Mean**: $\mu = \frac{\alpha}{\alpha+\beta}$
- **Variance**: $\sigma^2 = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$
- **MGF**: No closed form
- **Applications**: Modeling probabilities and proportions

## Properties and Relationships

### Memoryless Property
- Exponential distribution is memoryless
- $P(X > s + t | X > s) = P(X > t)$

### Scaling and Shifting
- Linear transformations of normal random variables remain normal
- $Y = aX + b \rightarrow Y \sim N(a\mu + b, a^2\sigma^2)$
- Standardization: $Z = \frac{X - \mu}{\sigma} \sim N(0,1)$

## Applications in Machine Learning

1. **Regression**: Modeling continuous target variables
2. **Generative Models**: Data generation and density estimation
3. **Anomaly Detection**: Modeling normal behavior
4. **Time Series Analysis**: Modeling temporal dependencies

## Parameter Estimation

### Maximum Likelihood Estimation
- Normal: $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$
- Exponential: $\hat{\lambda} = \frac{1}{\bar{x}}$
- Uniform: $\hat{a} = \min(x_i)$, $\hat{b} = \max(x_i)$
- Beta: Requires numerical methods (e.g., method of moments)

## Transformations and Operations

### Sum of Random Variables
- Sum of independent normals is normal: $X_i \sim N(\mu_i, \sigma_i^2) \Rightarrow \sum X_i \sim N(\sum \mu_i, \sum \sigma_i^2)$
- Sum of independent exponentials is gamma: $X_i \sim \text{Exp}(\lambda) \Rightarrow \sum_{i=1}^n X_i \sim \text{Gamma}(n, \lambda)$
- Sum of independent uniforms approaches normal (CLT)

### Central Limit Theorem
- Sum of many independent random variables approaches normal distribution
- $\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} N(0,1)$ as $n \rightarrow \infty$

## Related Topics
- [[L2_1_Discrete_Distributions|Discrete Distributions]]: Complementary to continuous distributions
- [[L2_1_Examples|Probability Examples]]: Practical applications of continuous distributions 