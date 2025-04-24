# Probability Examples in Machine Learning

This document provides examples and key concepts on probability to help you understand this important concept in machine learning and data analysis.

## Key Concepts and Formulas

Probability is a measure of the likelihood of an event occurring. It is a numerical value between 0 and 1, where 0 indicates impossibility and 1 indicates certainty.

### Basic Probability Axioms

1. $P(A) \geq 0$ for any event $A$
2. $P(\Omega) = 1$ where $\Omega$ is the sample space
3. If $A$ and $B$ are mutually exclusive events, then $P(A \cup B) = P(A) + P(B)$

### Important Formulas

- **Complement Rule**: $P(A^c) = 1 - P(A)$
- **Inclusion-Exclusion**: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- **Conditional Probability**: $P(A|B) = \frac{P(A \cap B)}{P(B)}$
- **Chain Rule**: $P(A_1, A_2, ..., A_n) = P(A_1)P(A_2|A_1)...P(A_n|A_1,...,A_{n-1})$
- **Marginalization**: $P(A) = \sum_B P(A,B)$
- **Independence**: $P(A \cap B) = P(A) \times P(B)$ (for independent events)
- **Conditional Independence**: $P(A \cap B|C) = P(A|C) \times P(B|C)$
- **Bayes' Theorem**: $P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$
- **Law of Total Expectation**: $E[X] = E[E[X|Y]]$
- **Law of Total Variance**: $\text{Var}(X) = E[\text{Var}(X|Y)] + \text{Var}(E[X|Y])$
- **Expectation**: $E[X] = \sum_i x_i P(X = x_i)$ for discrete random variables
- **Variance**: $\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$
- **Covariance**: $\text{Cov}(X,Y) = E[(X - E[X])(Y - E[Y])]$
- **Correlation**: $\rho(X,Y) = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X) \times \text{Var}(Y)}}$
- **Partial Correlation**: $\rho_{XY|Z} = \frac{\rho_{XY} - \rho_{XZ}\rho_{YZ}}{\sqrt{(1-\rho_{XZ}^2)(1-\rho_{YZ}^2)}}$

## Practice Questions

For practice multiple-choice questions on probability, see:
- [[L2_1_Quiz|Probability Fundamentals Quiz]]

## Examples

### Basic Probability
- [[L2_1_Basic_Probability_Examples|Basic Probability Examples]]: Fundamental probability problems and applications
- [[L2_1_Basic_Combinatorial_Probability_Examples|Basic Combinatorial Probability Examples]]: Fundamental counting principles and basic combinatorial problems
- [[L2_1_Combinatorial_Probability_Examples|Combinatorial Probability Examples]]: Advanced combinatorial problems and techniques
- [[L2_1_PMF_PDF_CDF_Examples|PMF, PDF, and CDF Examples]]: Working with probability mass functions, density functions, and cumulative distribution functions
- [[L2_1_Geometric_Probability_Examples|Geometric Probability Examples]]: Probability in spatial contexts and geometric settings

### Expectation, Variance, and Correlation
- [[L2_1_Expectation_Examples|Expectation Examples]]: Calculating expected values in various scenarios
- [[L2_1_Variance_Examples|Variance Examples]]: Problems on variance calculation and statistical dispersion
- [[L2_1_Covariance_Examples|Covariance Examples]]: Problems on measuring linear relationships between random variables
- [[L2_1_Correlation_Examples|Correlation Examples]]: Examples of Pearson correlation coefficient and its applications

### Discrete Distributions
- [[L2_1_Discrete_Probability_Examples|Discrete Probability Examples]]: Advanced applications of discrete probability including dice games, card problems, and combinatorial counting
- [[L2_1_Bernoulli_Binomial_Examples|Bernoulli and Binomial Examples]]: Modeling binary outcomes and trials
- [[L2_1_Multinomial_Examples|Multinomial Examples]]: Problems with categorical outcomes
- [[L2_1_Poisson_Examples|Poisson Examples]]: Modeling count data
- [[L2_1_Geometric_Examples|Geometric Examples]]: Modeling waiting times
- [[L2_1_Negative_Binomial_Examples|Negative Binomial Examples]]: Modeling overdispersed count data

### Continuous Distributions
- [[L2_1_Continuous_Probability_Examples|Continuous Probability Examples]]: Covers normal, exponential, and uniform distributions
- [[L2_1_Normal_Distribution_Examples|Normal Distribution Examples]]: Working with Gaussian distributions
- [[L2_1_Uniform_Distribution_Examples|Uniform Distribution Examples]]: Equal probability problems
- [[L2_1_Exponential_Examples|Exponential Examples]]: Modeling waiting times and decay
- [[L2_1_Gamma_Examples|Gamma Examples]]: Working with positive continuous data
- [[L2_1_Beta_Examples|Beta Examples]]: Working with distributions over probabilities

### Conditional Probability and Independence
- [[L2_1_Basic_Conditional_Probability_Examples|Basic Conditional Probability Examples]]: Simple conditional probability without Bayes' theorem
- [[L2_1_Conditional_Probability_Examples|Conditional Probability Examples]]: Medical diagnosis, spam filtering, and Naive Bayes examples
- [[L2_1_Independence_Examples|Independence Examples]]: Problems illustrating independence between events and random variables
- [[L2_1_Conditional_Independence_Examples|Conditional Independence Examples]]: Understanding conditional independence in ML models

### Joint and Multivariate Analysis
- [[L2_1_Joint_Probability_Examples|Joint Probability Examples]]: Working with multiple events or random variables simultaneously
- [[L2_1_Joint_Distributions_Examples|Joint Distributions Examples]]: Discrete and continuous joint distributions with applications
- [[L2_1_Multivariate_Analysis_Examples|Multivariate Analysis Examples]]: Mean vectors, PCA, and Mahalanobis distance
- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: Working with multivariate normal distributions
- [[L2_1_Multivariate_Exam_Problems|Multivariate Exam Problems]]: Practice problems and solutions covering density functions, multivariate Gaussian, transformation techniques, and common exam questions
- [[L2_1_Contour_Plot_Examples|Contour Plot Examples]]: Visualizing multivariate distributions using contour plots

### Advanced Topics
- Probability Inequality Examples: Applications of Markov, Chebyshev, and Hoeffding inequalities
- Concentration Inequality Examples: Understanding data concentration
- Limit Theorem Examples: Applications of LLN and CLT in ML

### Real-world Applications
- [[L2_1_Probability_Application_Examples|Application Examples]]: Real-world problems applying probability concepts in ML
- [[L2_1_ML_Questions|ML Probability Examples]]: Specific applications in machine learning models