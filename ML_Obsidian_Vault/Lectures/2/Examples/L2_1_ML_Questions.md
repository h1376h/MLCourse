# Machine Learning Probability Examples

This document provides examples and key concepts on probability applied to machine learning to help you understand these important concepts in data analysis and model development.

## Key Concepts and Formulas

Probability theory forms the mathematical foundation of many machine learning algorithms. Understanding probability concepts helps in developing intuition for how these models work, how to diagnose issues, and how to improve performance.

### Key Probabilistic ML Formulas

$$P(y|x) = \frac{P(x|y)P(y)}{P(x)}$$ (Bayes' theorem - foundation of many ML models)

$$P(x) = \sum_{y} P(x|y)P(y)$$ (Law of Total Probability - evidence calculation)

$$\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X) \text{Var}(Y)}}$$ (Correlation coefficient - relationship strength)

$$I(X;Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$ (Mutual Information - information shared between variables)

Where:
- $P(y|x)$ = Posterior probability of class $y$ given features $x$
- $P(x|y)$ = Likelihood of features $x$ given class $y$
- $P(y)$ = Prior probability of class $y$
- $P(x)$ = Evidence (marginal likelihood)
- $\rho_{X,Y}$ = Pearson correlation coefficient between X and Y
- $I(X;Y)$ = Mutual information between X and Y

## Practice Questions

For practice multiple-choice questions on probability in machine learning, see:
- [[L2_1_Quiz|Probability Fundamentals Quiz]]

## Examples

1. [[L2_1_Conditional_Bayes_ML|Conditional Probability & Bayes' Theorem]]: Applications of Bayes' theorem in ML classification tasks, naive Bayes classifiers, spam filtering, and medical diagnosis. Includes chain rule applications in sequential data modeling and marginalization techniques for handling latent variables.
2. [[L2_1_Distributions_ML|Probability Distributions in ML]]: How discrete distributions (Bernoulli, Binomial, Multinomial, Poisson, Geometric) and continuous distributions (Normal, Uniform, Exponential, Gamma, Beta) are used to model data and errors in machine learning. Includes practical examples of PMFs, PDFs, and CDFs in feature engineering and model development.
3. [[L2_1_Expectation_Variance_ML|Expectation, Variance & Moments]]: Applications of expected value, variance, and statistical moments in ML model evaluation, ensemble methods, and uncertainty quantification. Covers the Law of Total Expectation and Law of Total Variance for handling complex ML scenarios with multiple conditions.
4. [[L2_1_Covariance_Correlation_ML|Covariance, Correlation & Independence]]: Techniques for measuring feature relationships, detecting multicollinearity, performing dimensionality reduction (PCA), and understanding feature importance. Contrasts correlation vs. causation and examines independence assumptions in various ML algorithms.
5. [[L2_1_Advanced_Probability_ML|Advanced Probability Concepts]]: Applications of probability inequalities (Markov, Chebyshev, Hoeffding), concentration inequalities, limit theorems (Law of Large Numbers, Central Limit Theorem), and Monte Carlo methods in modern machine learning algorithms.