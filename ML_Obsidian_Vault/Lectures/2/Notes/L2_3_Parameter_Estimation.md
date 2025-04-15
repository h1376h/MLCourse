# Parameter Estimation

Parameter estimation is the process of using data to estimate the parameters of a statistical model. It's a fundamental task in statistics and machine learning.

## Main Methods
Methods to estimate model parameters from data:

### Frequentist Approach
- **Maximum Likelihood Estimation (MLE)**: Maximizes the likelihood of observing the data

### Bayesian Approaches
- **Maximum A Posteriori (MAP)**: Incorporates prior knowledge with observed data
- **Full Bayesian Estimation**: Treats parameters as distributions rather than point estimates

## Common Estimation Methods

### Maximum Likelihood Estimation (MLE)

- Finds parameter values that maximize the likelihood of observing the given data
- Ignores prior information
- Formula: $\hat{\theta}_{MLE} = \arg\max_{\theta} P(X|\theta)$
- See [[L2_4_MLE_Examples|MLE_Examples]] for examples

### Maximum A Posteriori (MAP) Estimation

- Finds parameter values that maximize the posterior probability
- Incorporates prior information
- Formula: $\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|X) = \arg\max_{\theta} P(X|\theta)P(\theta)$
- See [[L2_7_MAP_Examples|MAP Examples]] for examples

### Full Bayesian Estimation

- Treats parameters as random variables with distributions
- Computes the full posterior distribution rather than a point estimate
- Uses the posterior for inference and prediction
- Considers the entire posterior distribution rather than just the mode
- Provides complete uncertainty quantification
- Enables prediction that accounts for parameter uncertainty
- See [[L2_7_Full_Bayesian_Inference|Full_Bayesian_Inference]] for detailed examples
- See [[L2_5_Bayesian_Inference|Bayesian_Inference]] for the theoretical foundation

## Comparison of Methods

| Method | Prior Information | Result | Computational Complexity |
|--------|-------------------|--------|--------------------------|
| MLE    | No                | Point estimate | Lower |
| MAP    | Yes               | Point estimate | Medium |
| Full Bayesian | Yes       | Distribution | Higher |

## Common Distributions for Estimation

- [[L2_1_Beta_Distribution|Beta_Distribution]]: For probability parameters (e.g., Bernoulli trials)
- [[L2_1_Normal_Distribution|Normal_Distribution]]: For continuous parameters with uncertainty
- Dirichlet Distribution: For multinomial parameters
- Gamma Distribution: For scale parameters

## Applications

- Classification: Estimating class probabilities
- Regression: Estimating coefficients
- Time Series: Estimating trend and seasonality parameters
- Clustering: Estimating cluster centers and variances

## Related Topics

- [[L2_5_Bayesian_Inference|Bayesian_Inference]]
- [[Likelihood_Function]] 