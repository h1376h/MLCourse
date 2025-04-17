# Lecture 2.5: Bayesian Approach to ML Quiz

## Overview
This quiz contains 13 questions from different topics covered in section 2.5 of the lectures on Bayesian Framework.

## Question 1

### Problem Statement
Suppose we have a coin with an unknown probability $\theta$ of landing heads. We want to estimate $\theta$ using Bayesian inference.

1. Assume a prior distribution for $\theta$ as $\text{Beta}(2, 2)$
2. We toss the coin 10 times and observe 7 heads and 3 tails

#### Task
1. Write down the likelihood function for the observed data
2. Calculate the posterior distribution for $\theta$
3. Find the posterior mean, mode, and variance of $\theta$
4. Calculate the 95% credible interval for $\theta$

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Bayesian Coin Flip Analysis](L2_5_1_explanation.md).

## Question 2

### Problem Statement
Consider a diagnostic test for a disease that affects 1% of the population. The test has a true positive rate (sensitivity) of 95% and a true negative rate (specificity) of 90%.

#### Task
1. Using Bayes' theorem, calculate the probability that a person has the disease given a positive test result
2. If we repeat the test on the same person and it comes back positive again, what is the updated probability that the person has the disease?
3. How many consecutive positive test results would be needed to have at least a 95% probability that the person has the disease?
4. Discuss how the prevalence of the disease affects the interpretation of test results

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Bayesian Medical Diagnostics](L2_5_2_explanation.md).

## Question 3

### Problem Statement
Consider a Poisson likelihood with parameter $\lambda$ for count data. We observe data $X = \{3, 5, 2, 4, 6, 3, 4, 5, 2, 3\}$.

#### Task
1. If we use a $\text{Gamma}(\alpha, \beta)$ prior for $\lambda$, derive the posterior distribution
2. Assuming a $\text{Gamma}(2, 1)$ prior, calculate the posterior distribution
3. Find the posterior mean, mode, and variance of $\lambda$
4. Calculate the predictive distribution for a new observation $X_{\text{new}}$

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Poisson-Gamma Bayesian Model](L2_5_3_explanation.md).

## Question 4

### Problem Statement
Consider the problem of estimating the mean $\mu$ of a normal distribution with known variance $\sigma^2 = 4$. We observe data $X = \{10.2, 8.7, 9.5, 11.3, 10.8\}$.

#### Task
1. If we use a normal prior $N(9, 1)$ for $\mu$, derive the posterior distribution
2. Calculate the posterior mean and variance
3. Find the 90% credible interval for $\mu$
4. Compare the Bayesian estimate with the maximum likelihood estimate (MLE) and discuss the differences between the two approaches

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Normal Distribution Parameter Estimation](L2_5_4_explanation.md).

## Question 5

### Problem Statement
For a simple classification problem, we want to estimate the probability $p$ of an observation belonging to class 1. We have observed 5 instances belonging to class 1 out of 20 total observations.

#### Task
1. If we use a uniform prior ($\text{Beta}(1,1)$) for $p$, what is the posterior distribution?
2. What is the posterior mean of $p$?
3. How would the posterior change if we had used an informative prior $\text{Beta}(10,30)$?
4. Explain the practical significance of using an informative prior versus a uniform prior in this context.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Bayesian Classification Probability](L2_5_5_explanation.md).

## Question 6

### Problem Statement
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

#### Task
1. In Bayesian statistics, the posterior distribution represents our updated belief about a parameter after observing data.
2. Conjugate priors always lead to the most accurate Bayesian inference results.
3. Bayesian credible intervals and frequentist confidence intervals have identical interpretations.
4. The posterior predictive distribution incorporates both the uncertainty in the parameter estimates and the inherent randomness in generating new data.
5. Hierarchical Bayesian models are useful only when we have a large amount of data.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Bayesian Statistics Concepts](L2_5_6_explanation.md).

## Question 7

### Problem Statement
Consider a scenario where we are modeling the number of defects in a manufacturing process using a Poisson distribution with parameter $\lambda$.

#### Task
1. What is the conjugate prior for a Poisson likelihood?
2. If our prior for $\lambda$ is $\text{Gamma}(3, 2)$ and we observe the following defect counts in 5 batches: $\{1, 0, 2, 1, 1\}$, what is the resulting posterior distribution?
3. Calculate the posterior mean of $\lambda$.
4. What is the advantage of using a conjugate prior in this specific scenario?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Bayesian Manufacturing Process Analysis](L2_5_7_explanation.md).

## Question 8

### Problem Statement
Consider a Bayesian inference problem where we want to determine if a coin is fair.

#### Task
1. If our prior belief is represented by a Beta(3,3) distribution and we observe 5 heads out of 8 coin flips, what is the posterior distribution?
2. What is the posterior mean probability of the coin showing heads?
3. How does this posterior mean compare to the maximum likelihood estimate (5/8)?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Prior and Posterior Comparison](L2_5_8_explanation.md).

## Question 9

### Problem Statement
For each of the following likelihoods, identify the corresponding conjugate prior:

#### Task
1. Bernoulli likelihood (for binary outcomes)
2. Normal likelihood with known variance (for the mean parameter)
3. Poisson likelihood (for the rate parameter)

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Conjugate Prior Identification](L2_5_9_explanation.md).

## Question 10

### Problem Statement
Consider a simple hierarchical Bayesian model for analyzing students' test scores across different schools.

#### Task
1. Describe the basic structure of a two-level hierarchical Bayesian model for this scenario
2. Explain one advantage of using a hierarchical model versus a non-hierarchical model in this context
3. Identify a scenario where empirical Bayes might be used instead of a fully Bayesian approach

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Hierarchical Bayesian Modeling](L2_5_10_explanation.md).

## Question 11

### Problem Statement
A soil scientist is developing a Bayesian model to predict soil nutrient content. Based on previous studies at similar sites, the nutrient concentration follows a normal distribution with unknown mean μ.

#### Task
1. If the scientist uses a normal prior for μ with mean 25 ppm and variance 4, and then collects 6 samples with measurements {22, 27, 24, 23, 26, 25} ppm and known measurement variance σ² = 9, what is the posterior distribution for μ?
2. Calculate the posterior mean and variance.
3. Compare how the posterior would differ if an uninformative prior had been used instead.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: Informative vs Noninformative Priors](L2_5_11_explanation.md).

## Question 12

### Problem Statement
In astronomy, researchers are modeling the number of exoplanets in different star systems. They believe the count follows a Poisson distribution with an unknown rate parameter λ.

#### Task
1. Given that Gamma is the conjugate prior for the Poisson distribution, express the posterior distribution if the prior is Gamma(α=2, β=0.5) and observations from 5 star systems show {3, 1, 4, 2, 5} exoplanets.
2. What is the posterior mean of λ?
3. What is the posterior predictive probability of finding exactly 3 exoplanets in the next observed star system?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: Posterior Predictive Distribution](L2_5_12_explanation.md).

## Question 13

### Problem Statement
A music streaming service is analyzing user preferences across different genres. They want to apply a hierarchical Bayesian model to understand listening patterns.

#### Task
1. Explain how a two-level hierarchical Bayesian model could be structured for this scenario, where individual users are grouped by geographical regions.
2. If the service has limited computational resources, how could empirical Bayes be used as an alternative to full Bayesian inference?
3. Describe one key difference between Bayesian credible intervals and frequentist confidence intervals in interpreting user preference data.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 13: Hierarchical Models and Empirical Bayes](L2_5_13_explanation.md). 