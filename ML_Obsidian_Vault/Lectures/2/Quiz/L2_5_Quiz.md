# Lecture 2.5: Bayesian Approach to ML Quiz

## Overview
This quiz contains 4 questions from different topics covered in section 2.5 of the lectures on Bayesian Framework.

## Question 1

### Problem Statement
Suppose we have a coin with an unknown probability θ of landing heads. We want to estimate θ using Bayesian inference.

1. Assume a prior distribution for θ as Beta(2, 2)
2. We toss the coin 10 times and observe 7 heads and 3 tails

#### Task
1. Write down the likelihood function for the observed data
2. Calculate the posterior distribution for θ
3. Find the posterior mean, mode, and variance of θ
4. Calculate the 95% credible interval for θ

## Question 2

### Problem Statement
Consider a diagnostic test for a disease that affects 1% of the population. The test has a true positive rate (sensitivity) of 95% and a true negative rate (specificity) of 90%.

#### Task
1. Using Bayes' theorem, calculate the probability that a person has the disease given a positive test result
2. If we repeat the test on the same person and it comes back positive again, what is the updated probability that the person has the disease?
3. How many consecutive positive test results would be needed to have at least a 95% probability that the person has the disease?
4. Discuss how the prevalence of the disease affects the interpretation of test results

## Question 3

### Problem Statement
Consider a Poisson likelihood with parameter λ for count data. We observe data X = {3, 5, 2, 4, 6, 3, 4, 5, 2, 3}.

#### Task
1. If we use a Gamma(α, β) prior for λ, derive the posterior distribution
2. Assuming a Gamma(2, 1) prior, calculate the posterior distribution
3. Find the posterior mean, mode, and variance of λ
4. Calculate the predictive distribution for a new observation X_{new}

## Question 4

### Problem Statement
Consider the problem of estimating the mean μ of a normal distribution with known variance σ² = 4. We observe data X = {10.2, 8.7, 9.5, 11.3, 10.8}.

#### Task
1. If we use a normal prior N(9, 1) for μ, derive the posterior distribution
2. Calculate the posterior mean and variance
3. Find the 90% credible interval for μ
4. Compare the Bayesian estimate with the maximum likelihood estimate (MLE) and discuss the differences between the two approaches 