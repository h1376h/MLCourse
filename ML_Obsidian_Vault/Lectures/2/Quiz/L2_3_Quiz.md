# Lecture 2.3: Statistical Estimation Basics Quiz

## Overview
This quiz contains 13 questions from different topics covered in section 2.3 of the lectures on Likelihood and Estimation.

## Question 1

### Problem Statement
Consider a random sample $X_1, X_2, \ldots, X_{10}$ from a normal distribution with unknown mean $\mu$ and known standard deviation $\sigma = 2$. Suppose we observe the following values:
$\{4.2, 3.8, 5.1, 4.5, 3.2, 4.9, 5.3, 4.0, 4.7, 3.6\}$

#### Task
1. Write down the likelihood function $L(\mu)$ for this sample
2. Write down the log-likelihood function $\ell(\mu)$
3. Calculate the maximum likelihood estimate for $\mu$
4. Find the likelihood ratio for testing $H_0: \mu = 5$ versus $H_1: \mu \neq 5$

For a detailed explanation of this question, see [Question 1: Likelihood Function and MLE](L2_3_1_explanation.md).

## Question 2

### Problem Statement
Let $X_1, X_2, \ldots, X_{20}$ be a random sample from a distribution with PDF:

$$f(x|\theta) = \theta x^{\theta-1}, \quad 0 < x < 1, \theta > 0$$

#### Task
1. Derive the likelihood function $L(\theta)$
2. Derive the log-likelihood function $\ell(\theta)$
3. Find the score function (the derivative of the log-likelihood with respect to $\theta$)
4. Suppose the observed data has geometric mean 0.8. Find the maximum likelihood estimate for $\theta$

For a detailed explanation of this question, see [Question 2: Likelihood and Score Function](L2_3_2_explanation.md).

## Question 3

### Problem Statement
Consider estimating the parameter $\lambda$ of a Poisson distribution based on $n$ independent observations $X_1, X_2, \ldots, X_n$.

#### Task
1. Derive the maximum likelihood estimator (MLE) for $\lambda$
2. Is this estimator unbiased? If not, calculate its bias
3. Calculate the variance of the MLE
4. Find the Cramér-Rao lower bound for the variance of any unbiased estimator of $\lambda$ and determine whether the MLE achieves this bound

For a detailed explanation of this question, see [Question 3: Properties of Poisson MLE](L2_3_3_explanation.md).

## Question 4

### Problem Statement
Consider the estimation of a parameter $\theta$ with two different estimators:

Estimator A: $\hat{\theta}_A$ with bias $b_A(\theta) = 0.1\theta$ and variance $\text{Var}(\hat{\theta}_A) = 0.5$
Estimator B: $\hat{\theta}_B$ with bias $b_B(\theta) = 0$ and variance $\text{Var}(\hat{\theta}_B) = 0.8$

#### Task
1. Calculate the Mean Squared Error (MSE) for each estimator when $\theta = 2$
2. Which estimator would you prefer when $\theta = 2$, and why?
3. Is there a value of $\theta$ for which estimator A has lower MSE than estimator B? If yes, find the range of $\theta$ values for which this is true
4. Discuss the bias-variance tradeoff in the context of these two estimators

For a detailed explanation of this question, see [Question 4: MSE and Bias-Variance Tradeoff](L2_3_4_explanation.md).

## Question 5

### Problem Statement
Is the sample mean a sufficient statistic for the parameter $\mu$ of a normal distribution with known variance?

### Task
1. Determine if the sample mean $\bar{X}$ is a sufficient statistic for $\mu$ in a normal distribution with known variance $\sigma^2$.
2. Prove your answer using the factorization theorem.
3. Explain the implications of your findings.

For a detailed explanation of this question, see [Question 5: Sufficient Statistics](L2_3_5_explanation.md).

## Question 6

### Problem Statement
For a Bernoulli distribution with parameter $p$, write down the Fisher Information $I(p)$.

### Task
1. Define the Fisher Information and its significance.
2. Derive the Fisher Information $I(p)$ for a Bernoulli distribution.
3. Interpret the meaning of the result in practical terms.

For a detailed explanation of this question, see [Question 6: Fisher Information](L2_3_6_explanation.md).

## Question 7

### Problem Statement
Consider the bias-variance tradeoff in estimator selection.

#### Task
If an estimator has bias $b(\theta)$ and variance $v(\theta)$, write the formula for its Mean Squared Error (MSE) in terms of $b(\theta)$ and $v(\theta)$.

For a detailed explanation of this question, see [Question 7: MSE Formula](L2_3_7_explanation.md).

## Question 8

### Problem Statement
What is the fundamental difference between probability and likelihood?

#### Task
Explain in one sentence the key distinction between probability and likelihood in the context of statistical estimation.

For a detailed explanation of this question, see [Question 8: Probability vs Likelihood](L2_3_8_explanation.md).

## Question 9

### Problem Statement
For a random sample $X_1, X_2, \ldots, X_n$ from a distribution with parameter $\theta$, the Fisher Information $I(\theta)$ quantifies how much information the sample contains about $\theta$.

#### Task
If an estimator $\hat{\theta}$ is unbiased and has variance $\text{Var}(\hat{\theta})$, write the Cramér-Rao inequality that establishes the lower bound for the variance of $\hat{\theta}$.

For a detailed explanation of this question, see [Question 9: Cramér-Rao Bound](L2_3_9_explanation.md).

## Question 10

### Problem Statement
Consider the Method of Moments for parameter estimation.

#### Task
If a random variable $X$ has a distribution with parameter $\theta$ where $E[X] = \frac{\theta}{1+\theta}$, derive the Method of Moments estimator for $\theta$ based on a sample $X_1, X_2, \ldots, X_n$.

For a detailed explanation of this question, see [Question 10: Method of Moments Estimation](L2_3_10_explanation.md).

## Question 11

### Problem Statement
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

#### Task
1. The likelihood function represents the probability of observing the data given the parameters.
2. If two estimators have the same variance, the one with lower bias will always have lower Mean Squared Error (MSE).
3. An estimator that achieves the Cramér-Rao lower bound is considered efficient.

For a detailed explanation of this question, see [Question 11: Statistical Estimation Fundamentals](L2_3_11_explanation.md).

## Question 12

### Problem Statement
Consider a random sample from a Bernoulli distribution with parameter $p$.

#### Task
What is the sufficient statistic for estimating the parameter $p$?

A) The sample median
B) The sample mean
C) The sample variance
D) The sample size

For a detailed explanation of this question, see [Question 12: Sufficient Statistics](L2_3_12_explanation.md).

## Question 13

### Problem Statement
For a random sample $X_1, X_2, \ldots, X_n$ from a distribution with unknown mean $\mu$ and known variance $\sigma^2$, the Point Estimator $\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n}X_i$ is used.

#### Task
Calculate the bias and variance of this estimator.

For a detailed explanation of this question, see [Question 13: Bias and Variance of Estimators](L2_3_13_explanation.md).