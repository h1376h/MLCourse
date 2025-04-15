# Lecture 2: Probability and Statistical Foundations for ML

## Overview
This module provides essential probability and statistical foundations required for machine learning, presented in a logical sequence from basic concepts to advanced techniques. Understanding these concepts is crucial for developing robust ML models.

### Lecture 2.1: Probability Fundamentals
- [[L2_1_Basic_Probability|Basic Probability]]: Core concepts of probability theory (random variables, probability axioms)
- [[L2_1_Combinatorial_Probability|Combinatorial Probability]]: Counting principles, permutations, combinations, and sampling applications
- [[L2_1_PMF_PDF_CDF|PMF, PDF, and CDF]]: Probability mass, density, and cumulative distribution functions
- [[L2_1_Expectation|Expectation]]: Expected values, Law of Total Expectation, Law of Total Variance
- [[L2_1_Variance|Variance and Moments]]: Measures of dispersion and higher moments
- [[L2_1_Covariance_Correlation|Covariance and Correlation]]: Relationship measures between random variables
- [[L2_1_Discrete_Distributions|Discrete Distributions]]: Bernoulli, Binomial, Multinomial, Poisson, Geometric distributions
- [[L2_1_Continuous_Distributions|Continuous Distributions]]: Normal, Uniform, Exponential, Gamma, Beta distributions
- Multivariate Distributions: Joint distributions and transformations
- [[L2_1_Conditional_Probability|Conditional Probability]]: Dependent events, Bayes' theorem, Chain Rule
- [[L2_1_Independence|Independence]]: Independent events, Conditional Independence, Pairwise vs Mutual
- [[L2_1_Joint_Probability|Joint Probability]]: Multiple random variables, Marginal Distributions
- [[L2_1_Probability_Inequalities|Probability Inequalities]]: Markov, Chebyshev, and Hoeffding inequalities
- [[L2_1_Concentration_Inequalities|Concentration Inequalities]]: Data concentration in high dimensions
- [[L2_1_Limit_Theorems|Limit Theorems]]: Law of Large Numbers and Central Limit Theorem
- [[L2_1_Examples|Examples]]: Comprehensive set of examples covering all topics in Lecture 2.1
- Required Reading: Chapters 2.1-2.3 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L2_1_Quiz]]: Test your understanding of probability fundamentals

### Lecture 2.2: Information Theory and Entropy
- [[L2_2_Information_Theory|Information Theory]]: Origins and relevance to ML
- Self-Information: Quantifying information content of events
- [[L2_2_Entropy|Entropy]]: Uncertainty and information content in distributions
- Joint and Conditional Entropy: Entropy for multiple random variables
- Relative Entropy: Understanding divergences between distributions
- [[L2_2_Cross_Entropy|Cross Entropy]]: Information-theoretic loss function for classification
- Cross Entropy Loss: Implementation in ML algorithms
- [[L2_2_KL_Divergence|KL Divergence]]: Properties, interpretations, and asymmetry considerations
- Jensen-Shannon Divergence: Symmetric measure of distribution similarity
- [[L2_2_Mutual_Information|Mutual Information]]: Shared information between random variables
- Conditional Mutual Information: Effect of additional variables
- Information Gain: Decision trees and feature selection
- Maximum Entropy Principle: Least-biased distribution estimation
- Channel Capacity: Information transmission limits
- Minimum Description Length: Model selection using information principles
- Entropy in Data Compression: Huffman and arithmetic coding
- Information Bottleneck Method: Balancing compression and prediction
- [[L2_2_Information_Theory_Applications|Information Theory Applications]]: Feature selection and model evaluation
- Examples: Comprehensive set of examples covering all topics in Lecture 2.2
- Required Reading: Chapter 1.6 of "Pattern Recognition and Machine Learning" by Bishop
- Supplementary Reading: Chapter 2 of "Elements of Information Theory" by Cover and Thomas
- Quiz: [[L2_2_Quiz]]: Test your understanding of information theory concepts

### Lecture 2.3: Statistical Estimation Basics
- [[L2_3_Parameter_Estimation|Parameter Estimation / Statistical Inference]]: Foundations of parameter estimation
- Point Estimation: Methods for single-value parameter estimates
- Interval Estimation: Confidence and credible intervals
- Estimator Properties: Bias, consistency, efficiency, sufficiency
- MSE Decomposition: Understanding bias-variance tradeoff
- Cramer-Rao Bound: Theoretical limits on estimator performance
- Method of Moments: Classical estimation technique
- [[L2_3_Likelihood|Likelihood Function]]: Construction and interpretation
- Log-Likelihood: Computational advantages and properties
- [[L2_3_Likelihood_Examples|Likelihood Examples]]: Common distribution likelihood functions
- [[L2_3_Probability_vs_Likelihood|Probability vs. Likelihood]]: Critical distinctions
- Fisher Information: Information about parameters
- Sufficient Statistics: Data reduction without information loss
- Examples: Statistical estimation in ML contexts
- Required Reading: Chapter 2.1-2.3 of "Pattern Recognition and Machine Learning" by Bishop
- Supplementary Reading: Chapter 7.1-7.3 of "Statistical Inference" by Casella and Berger
- Quiz: [[L2_3_Quiz]]: Test your understanding of statistical estimation

### Lecture 2.4: Maximum Likelihood Estimation
- [[L2_4_MLE_Introduction|MLE Introduction]]: Fundamental principles and intuition
- [[L2_4_MLE_Theory|MLE Theory]]: Theoretical foundations and mathematical derivation 
- MLE Properties: Consistency, asymptotic normality, efficiency
- [[L2_4_MLE_Common_Distributions|MLE for Common Distributions]]: Bernoulli, Poisson, normal, exponential
- MLE for Multinomial Models: Maximum likelihood for categorical data
- Exponential Family MLE: Unified approach to maximum likelihood
- Optimization for MLE: Analytical and numerical approaches
- Expectation-Maximization: MLE for latent variable models
- Limitations of MLE: Overfitting and bias in small samples
- [[L2_4_MLE_Examples|MLE Numerical Examples]]: Step-by-step examples with Python
- [[L2_4_MLE_Applications|MLE Applications]]: Practical applications in statistics
- Required Reading: Chapter 2.4-2.5 of "Pattern Recognition and Machine Learning" by Bishop
- Supplementary Reading: Chapter 9.1-9.3 of "The Elements of Statistical Learning" by Hastie et al.
- Quiz: [[L2_4_Quiz]]: Test your understanding of maximum likelihood estimation

### Lecture 2.5: Bayesian Approach to ML
- Bayesian Paradigm: Philosophical foundations of Bayesian thinking
- [[L2_5_Bayesian_Inference|Bayesian Inference]]: Core principles of incorporating prior knowledge
- Bayes' Rule in ML: Application of Bayes' theorem to ML problems
- Prior Distributions: Types, selection strategies, mathematical formulation
- Informative Priors: Incorporating domain knowledge
- Noninformative Priors: Maximum entropy and reference priors
- [[L2_5_Conjugate_Priors|Conjugate Priors]]: Mathematical convenience in Bayesian analysis
- Common Conjugate Pairs: Beta-Bernoulli, Gamma-Poisson, Normal-Normal
- Posterior Distributions: Derivation, interpretation, and usage
- Posterior Predictive Distributions: Making Bayesian predictions
- Bayesian Credible Intervals: Quantifying uncertainty
- Hierarchical Bayesian Models: Multi-level modeling approaches
- Empirical Bayes: Data-driven prior specification
- [[L2_5_Bayesian_vs_Frequentist|Bayesian vs. Frequentist]]: Comparative analysis of paradigms
- Computational Challenges: Integration and sampling issues
- [[L2_5_Bayesian_Examples|Bayesian Examples]]: Practical implementations of concepts
- Required Reading: Chapter 2.6 of "Pattern Recognition and Machine Learning" by Bishop
- Supplementary Reading: Chapters 3-4 of "Bayesian Data Analysis" by Gelman et al.
- Quiz: [[L2_5_Quiz]]: Test your understanding of the Bayesian approach

### Lecture 2.6: Monte Carlo Methods and Sampling
- Monte Carlo Introduction: Sampling approaches to approximation
- Random Number Generation: Pseudo-random number algorithms
- Direct Sampling: Sampling from standard distributions
- Inverse Transform Sampling: Generating samples from CDF
- Rejection Sampling: Accept-reject methods for complex distributions
- Importance Sampling: Variance reduction techniques
- Monte Carlo Integration: Numerical integration via sampling
- MCMC Theory: Theoretical foundations of Markov Chain Monte Carlo
- Metropolis-Hastings Algorithm: General-purpose MCMC method
- Gibbs Sampling: Sampling complex posteriors by components
- Hamiltonian Monte Carlo: Physics-inspired efficient sampling
- Slice Sampling: Alternative MCMC method for efficient sampling
- Sequential Monte Carlo: Particle filtering methods
- Bootstrap Methods: Resampling for statistical inference
- Sampling Diagnostics: Assessing convergence and efficiency
- Monte Carlo Examples: Applications in ML and statistics
- Required Reading: Chapter 11 of "Pattern Recognition and Machine Learning" by Bishop
- Supplementary Reading: Chapter 29 of "Information Theory, Inference, and Learning Algorithms" by MacKay
- Quiz: L2_6_Quiz: Test your understanding of Monte Carlo methods

### Lecture 2.7: Maximum A Posteriori and Full Bayesian Inference
- [[L2_7_MAP_Estimation|MAP Estimation Theory]]: Mathematical foundations and derivation
- MAP vs MLE: Detailed comparison of approaches
- MAP Point Estimates: Finding the mode of the posterior
- Log Posterior Optimization: Computational techniques for MAP
- Regularization as MAP: Bayesian interpretation of regularization
- [[L2_7_MAP_Examples|MAP Implementation Examples]]: Worked examples with code
- [[L2_7_Full_Bayesian_Inference|Full Bayesian Inference]]: Beyond point estimates to full posterior
- Posterior Sampling: Using Monte Carlo methods for inference
- Bayesian Model Selection: Comparing model evidence
- Bayesian Model Averaging: Combining predictions
- Bayes Factors: Quantifying evidence for model comparison
- Bayesian Information Criterion: Approximating Bayes factors
- Marginal Likelihood Computation: Model evidence calculation
- Variational Inference: Approximation methods for posteriors
- Computational Considerations: Implementation challenges
- [[L2_7_MAP_Full_Bayesian_Examples|Comprehensive Examples]]: Case studies comparing approaches
- Required Reading: Chapter 3.1-3.3 of "Pattern Recognition and Machine Learning" by Bishop
- Supplementary Reading: Chapter 5 of "Bayesian Data Analysis" by Gelman et al.
- Quiz: [[L2_7_Quiz]]: Test your understanding of MAP and Full Bayesian Inference

### Lecture 2.8: Statistical Hypothesis Testing for ML
- Hypothesis Testing Fundamentals: Basic concepts and methodology
- Type I and Type II Errors: Understanding error types in hypothesis testing
- P-Values: Interpretation and limitations
- Statistical Power: Sample size and effect size considerations
- Multiple Testing Problem: Corrections for multiple comparisons
- Parametric Tests: t-tests, F-tests, z-tests
- Nonparametric Tests: Distribution-free methods
- Analysis of Variance: Comparing multiple groups
- Chi-Square Tests: Testing categorical variables
- Goodness of Fit Tests: Model validation
- Independence Tests: Testing relationships between variables
- Resampling-Based Tests: Permutation and bootstrap tests
- Statistical Tests for Model Comparison: Comparing ML models
- Confidence Intervals: Construction and interpretation
- Hypothesis Testing Examples: Practical applications in ML
- Required Reading: Chapter 2.5 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: L2_8_Quiz: Test your understanding of statistical hypothesis testing

### Lecture 2.9: Advanced Probabilistic Topics in ML
- Probabilistic Graphical Models: Bayesian networks and Markov random fields
- Causality vs Correlation: Causal inference fundamentals
- Probabilistic Programming: Languages and frameworks
- Information Geometry: Geometric view of statistical manifolds
- High-Dimensional Probability: Concentration phenomena
- Robust Statistics: Handling outliers and model misspecification
- Uncertainty Quantification: Methods beyond variance
- Probabilistic Numerics: Uncertainty in numerical computations
- [[L2_9_Stochastic_Processes|Stochastic Processes]]: Markov chains, random walks, and their applications
- Advanced Probability Examples: Cutting-edge applications
- Required Reading: Chapter 8 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: L2_9_Quiz: Test your understanding of advanced probabilistic topics

## Related Slides
*(not included in the repo)*
- Probability_Fundamentals.pdf
- Distributions_in_ML.pdf
- Information_Theory.pdf
- Statistical_Estimation.pdf
- Maximum_Likelihood.pdf
- Bayesian_Statistics.pdf
- Monte_Carlo_Methods.pdf
- MAP_and_Full_Bayesian.pdf
- Hypothesis_Testing.pdf
- Advanced_Probability.pdf

## Related Videos
- [Probability Theory Fundamentals](https://www.youtube.com/watch?v=1uW3qMFA9Ho)
- [Information Theory for Machine Learning](https://www.youtube.com/watch?v=ErfnhcEV1O8)
- [In Statistics, Probability is not Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4)
- [Maximum Likelihood Estimation](https://www.youtube.com/watch?v=XepXtl9YKwc)
- [Bayesian Inference](https://www.youtube.com/watch?v=HZGCoVF3YvM)
- [Monte Carlo Methods Explained](https://www.youtube.com/watch?v=AyrRoKN_kvg)
- [Hypothesis Testing in Machine Learning](https://www.youtube.com/watch?v=tTsMDLwQUa4)


## All Quizzes
Test your understanding with these quizzes:
- [[L2_1_Quiz]]: Probability Fundamentals
- [[L2_2_Quiz]]: Information Theory and Entropy
- [[L2_3_Quiz]]: Statistical Estimation Basics
- [[L2_4_Quiz]]: Maximum Likelihood Estimation
- [[L2_5_Quiz]]: Bayesian Approach to ML
- L2_6_Quiz: Monte Carlo Methods and Sampling
- [[L2_7_Quiz]]: Maximum A Posteriori and Full Bayesian Inference
- L2_8_Quiz: Statistical Hypothesis Testing for ML
- L2_9_Quiz: Advanced Probabilistic Topics in ML