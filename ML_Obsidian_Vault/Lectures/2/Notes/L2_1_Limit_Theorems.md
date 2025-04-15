# Limit Theorems

Limit theorems are fundamental results in probability theory that describe the asymptotic behavior of sequences of random variables. They are crucial for understanding large-scale behavior in machine learning and statistics.

## Basic Limit Theorems

### Law of Large Numbers (LLN)
- **Weak LLN**: For i.i.d. random variables X₁,X₂,... with mean μ,
  X̄ₙ → μ in probability
- **Strong LLN**: For i.i.d. random variables with finite mean μ,
  X̄ₙ → μ almost surely
- **Applications**:
  - Sample mean convergence
  - Monte Carlo methods
  - Empirical risk minimization
  - Convergence of gradient descent

### Central Limit Theorem (CLT)
- **Statement**: For i.i.d. random variables with mean μ and variance σ²,
  √n(X̄ₙ - μ) → N(0,σ²) in distribution
- **Applications**:
  - Confidence intervals
  - Hypothesis testing
  - Asymptotic normality
  - Bootstrap methods in ML

## Advanced Limit Theorems

### Berry-Esseen Theorem
- **Statement**: Provides rate of convergence in CLT
  |Fₙ(x) - Φ(x)| ≤ Cρ/σ³√n
- **Applications**:
  - Finite sample approximations
  - Error bounds
  - Rate of convergence analysis
  - Generalization error estimation

### Multivariate CLT
- **Statement**: For i.i.d. random vectors with mean μ and covariance Σ,
  √n(X̄ₙ - μ) → N(0,Σ) in distribution
- **Applications**:
  - Multivariate statistics
  - Principal component analysis
  - High-dimensional inference
  - Covariance estimation in deep learning

## Functional Limit Theorems

### Donsker's Theorem
- **Statement**: Weak convergence of empirical processes
- **Applications**:
  - Kolmogorov-Smirnov test
  - Goodness-of-fit tests
  - Empirical process theory
  - Neural network convergence analysis

### Functional CLT
- **Statement**: Convergence of stochastic processes
- **Applications**:
  - Time series analysis
  - Stochastic calculus
  - Brownian motion
  - Sequential learning algorithms

## Applications in Machine Learning

1. **Statistical Inference**
   - Confidence intervals
   - Hypothesis testing
   - Parameter estimation
   - Uncertainty quantification in predictions

2. **Algorithm Analysis**
   - Convergence rates
   - Asymptotic behavior
   - Performance guarantees
   - Stochastic gradient descent convergence

3. **Large-Scale Learning**
   - Distributed learning
   - Stochastic optimization
   - Online learning
   - Mini-batch processing guarantees

4. **Model Selection**
   - Information criteria
   - Cross-validation
   - Model complexity
   - Neural architecture search

5. **Deep Learning Theory**
   - Infinite width network limits
   - Neural tangent kernel
   - Random feature approximations
   - Optimization dynamics

## Practical Considerations

### Finite Sample Behavior
- Sample size requirements
- Convergence rates
- Approximation quality
- Small data regime performance

### Implementation
- Numerical stability
- Computational efficiency
- Error analysis
- Practical algorithm design

## Related Topics
- [[L2_1_Probability_Inequalities|Probability Inequalities]]: Bounds on probabilities
- [[L2_1_Concentration_Inequalities|Concentration Inequalities]]: Non-asymptotic results
- [[L2_9_Stochastic_Processes|Stochastic Processes]]: Markov chains and random walks
- [[L2_1_Examples|Probability Examples]]: Practical applications 