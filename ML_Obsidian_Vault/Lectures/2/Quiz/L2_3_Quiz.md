# Lecture 2.3: Statistical Estimation Basics Quiz

## Overview
This quiz contains 4 questions from different topics covered in section 2.3 of the lectures on Likelihood and Estimation.

## Question 1

### Problem Statement
Consider a random sample X₁, X₂, ..., X₁₀ from a normal distribution with unknown mean μ and known standard deviation σ = 2. Suppose we observe the following values:
{4.2, 3.8, 5.1, 4.5, 3.2, 4.9, 5.3, 4.0, 4.7, 3.6}

#### Task
1. Write down the likelihood function L(μ) for this sample
2. Write down the log-likelihood function ℓ(μ)
3. Calculate the maximum likelihood estimate for μ
4. Find the likelihood ratio for testing H₀: μ = 5 versus H₁: μ ≠ 5

## Question 2

### Problem Statement
Let X₁, X₂, ..., X₂₀ be a random sample from a distribution with PDF:

$$f(x|\theta) = \theta x^{\theta-1}, \quad 0 < x < 1, \theta > 0$$

#### Task
1. Derive the likelihood function L(θ)
2. Derive the log-likelihood function ℓ(θ)
3. Find the score function (the derivative of the log-likelihood with respect to θ)
4. Suppose the observed data has geometric mean 0.8. Find the maximum likelihood estimate for θ

## Question 3

### Problem Statement
Consider estimating the parameter λ of a Poisson distribution based on n independent observations X₁, X₂, ..., X_n.

#### Task
1. Derive the maximum likelihood estimator (MLE) for λ
2. Is this estimator unbiased? If not, calculate its bias
3. Calculate the variance of the MLE
4. Find the Cramér-Rao lower bound for the variance of any unbiased estimator of λ and determine whether the MLE achieves this bound

## Question 4

### Problem Statement
Consider the estimation of a parameter θ with two different estimators:

Estimator A: $\hat{\theta}_A$ with bias b_A(θ) = 0.1θ and variance Var($\hat{\theta}_A$) = 0.5
Estimator B: $\hat{\theta}_B$ with bias b_B(θ) = 0 and variance Var($\hat{\theta}_B$) = 0.8

#### Task
1. Calculate the Mean Squared Error (MSE) for each estimator when θ = 2
2. Which estimator would you prefer when θ = 2, and why?
3. Is there a value of θ for which estimator A has lower MSE than estimator B? If yes, find the range of θ values for which this is true
4. Discuss the bias-variance tradeoff in the context of these two estimators 