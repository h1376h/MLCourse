# Likelihood

Likelihood is a fundamental statistical concept that measures how well a statistical model explains observed data. It's a key component in parameter estimation, model selection, and statistical inference.

## Core Concepts

### Likelihood Function
- **Definition**: $L(\theta|X)$ - likelihood of parameters θ given observed data X
- **Mathematical Form**: The same function as probability, but viewed differently:

$$L(\theta|X) = P(X|\theta)$$

- **Perspective**: Function of parameters (θ) for fixed observations (X)
- **Property**: Does NOT need to sum/integrate to 1 over parameter space

> **Note on Notation:** Throughout literature, likelihood functions may be written using either:
> - Semicolon notation: L(θ;X)
> - Conditional notation: L(θ|X)
> 
> Both notations represent the same concept. The key distinction is conceptual: probability describes data given parameters, while likelihood assesses parameters given data.

### Key Characteristics
- Likelihood values only have relative meaning (not absolute)
- Often worked with in log form (log-likelihood) for computational convenience
- Central to both frequentist and Bayesian statistical methods
- Provides a basis for comparing different parameter values or models

## Likelihood Principles

1. **Likelihood Principle**: All information about parameters from an experiment is contained in the likelihood function
2. **Sufficiency Principle**: If a statistic t(X) is sufficient for θ, then the likelihood based on t(X) is proportional to the likelihood based on X
3. **Invariance Principle**: Likelihood-based inferences are invariant to one-to-one transformations of parameters

## Common Uses

### Maximum Likelihood Estimation (MLE)
- Finds parameters that maximize the likelihood:

$$\hat{\theta} = \arg\max_{\theta} L(\theta|X)$$

- Provides point estimates with desirable asymptotic properties
- See [[L2_4_MLE_Examples|MLE_Examples]] for detailed examples

### Likelihood Ratio Tests
- Compares likelihoods of nested models:

$$\Lambda = \frac{L(\theta_0|X)}{L(\theta|X)}$$

- Used for hypothesis testing and model comparison
- Under certain conditions, $-2\log(\Lambda)$ follows a chi-squared distribution

### Likelihood in Bayesian Methods
- Used to update prior beliefs:

$$P(\theta|X) \propto L(\theta|X) \times P(\theta)$$

- Serves as the updating mechanism in Bayesian inference
- See [[L2_5_Bayesian_Inference|Bayesian_Inference]] for theoretical foundations

## Distinction from Probability

Likelihood is the complementary perspective to probability:

- **Probability**: $P(X|\theta)$ - probability of data X given parameters θ
- **Likelihood**: $L(\theta|X)$ - likelihood of parameters θ given data X

The key difference is in what varies and what's fixed:
- For probability, parameters (θ) are fixed and we vary the data (X)
- For likelihood, data (X) is fixed and we vary the parameters (θ)

See [[L2_3_Probability_vs_Likelihood|Probability_vs_Likelihood]] for a detailed comparison.

## Likelihood in Machine Learning

### Model Training
- Most ML training objectives are based on likelihood maximization
- Cross-entropy loss is derived from negative log-likelihood
- Regularization can be viewed as adding prior information to likelihood

### Model Evaluation
- Log-likelihood on test data evaluates predictive performance
- Area under ROC curve relates to likelihood ratios of binary classifiers
- Model selection criteria like AIC and BIC are based on likelihood

### Feature Selection
- Likelihood ratio tests can identify informative features
- Information criteria based on likelihood help avoid overfitting
- Forward/backward selection procedures often use likelihood-based metrics

## Examples of Likelihood Functions

- **Bernoulli**:

$$L(\theta|x) = \theta^x(1-\theta)^{1-x}$$

- **Binomial**:

$$L(\theta|x) = \binom{n}{x}\theta^x(1-\theta)^{n-x}$$

- **Normal**:

$$L(\mu,\sigma|x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- **Poisson**:

$$L(\lambda|x) = \frac{e^{-\lambda}\lambda^x}{x!}$$

See [[L2_3_Likelihood_Examples|Likelihood_Examples]] for detailed applications with code.

## Related Topics

- [[L2_1_Basic_Probability|Basic Probability]]: The complementary perspective on data and models
- [[L2_3_Probability_vs_Likelihood|Probability_vs_Likelihood]]: Detailed comparison between these concepts
- [[L2_3_Likelihood_Examples|Likelihood_Examples]]: Practical examples of likelihood calculations
- [[L2_4_MLE_Examples|MLE_Examples]]: Maximum likelihood parameter estimation
- [[L2_5_Bayesian_Inference|Bayesian_Inference]]: Using likelihood to update prior beliefs
- [[L2_3_Parameter_Estimation|Parameter_Estimation]]: Methods for estimating distribution parameters 