# Lecture 2.7: Maximum A Posteriori and Full Bayesian Inference Quiz

## Overview
This quiz contains 4 questions from different topics covered in section 2.6 of the lectures on Maximum A Posteriori (MAP) Estimation and Full Bayesian Inference.

## Question 1

### Problem Statement
Consider a binomial likelihood with parameter θ representing the probability of success. We observe data D = 8 successes out of n = 20 trials.

#### Task
1. Using a Beta(2, 2) prior for θ, derive the posterior distribution
2. Calculate the Maximum A Posteriori (MAP) estimate for θ
3. Calculate the Maximum Likelihood Estimate (MLE) for θ
4. Compare the MAP and MLE estimates and explain why they differ

## Question 2

### Problem Statement
Consider a normal distribution with unknown mean μ and known variance σ² = 4. We observe data X = {7.2, 6.8, 8.3, 7.5, 6.9}.

#### Task
1. If we use a normal prior N(7, 1) for μ, derive the posterior distribution
2. Calculate the MAP estimate for μ
3. Derive the full Bayesian posterior predictive distribution for a new observation X_{new}
4. Calculate the 95% prediction interval for a new observation

## Question 3

### Problem Statement
Consider two competing models for a dataset:
- Model 1: Normal distribution with unknown mean μ₁ and known variance σ₁² = 2
- Model 2: Normal distribution with unknown mean μ₂ and known variance σ₂² = 4

We use the following priors:
- μ₁ ~ N(0, 1)
- μ₂ ~ N(0, 2)

We observe data X = {1.5, 2.3, 1.8, 2.5, 1.9}.

#### Task
1. Calculate the posterior distribution for μ₁ under Model 1
2. Calculate the posterior distribution for μ₂ under Model 2
3. Calculate the marginal likelihood (evidence) for each model
4. Calculate the Bayes factor and interpret the result for model comparison

## Question 4

### Problem Statement
Consider a linear regression model y = βx + ε where ε ~ N(0, σ²) with known σ² = 1. We observe the following data points (x, y):
{(1, 2.1), (2, 3.8), (3, 5.2), (4, 6.9), (5, 8.3)}

#### Task
1. If we use a normal prior β ~ N(1, 0.5) for the slope parameter, derive the posterior distribution
2. Calculate the MAP estimate for β
3. Calculate the MLE for β
4. Derive the posterior predictive distribution for a new observation y_{new} given x_{new} = 6 