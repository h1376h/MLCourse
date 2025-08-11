# Lecture 8.5: Model-Based Clustering Quiz

## Overview
This quiz contains 5 questions covering different topics from section 8.5 of the lectures on Model-Based Clustering, including Gaussian Mixture Models, EM algorithm, model selection, and soft clustering.

## Question 1

### Problem Statement
Consider a Gaussian Mixture Model (GMM) with 2 components in 1D space.

#### Task
1. [🔍] What are the parameters of a 1D Gaussian distribution?
2. [🔍] What additional parameter does a GMM have compared to a single Gaussian?
3. [🔍] How many total parameters does this 2-component GMM have?
4. [🔍] What does the mixing coefficient represent in a GMM?

For a detailed explanation of this question, see [Question 1: GMM Parameters and Structure](L8_5_1_explanation.md).

## Question 2

### Problem Statement
The Expectation-Maximization (EM) algorithm is used to fit GMMs to data.

#### Task
1. [📚] What are the two main steps of the EM algorithm?
2. [📚] What happens during the "E-step" (Expectation step)?
3. [📚] What happens during the "M-step" (Maximization step)?
4. [📚] Why is the EM algorithm iterative?

For a detailed explanation of this question, see [Question 2: EM Algorithm Steps](L8_5_2_explanation.md).

## Question 3

### Problem Statement
Consider the problem of choosing the optimal number of components in a GMM.

#### Task
1. [📚] What is the Akaike Information Criterion (AIC)?
2. [📚] What is the Bayesian Information Criterion (BIC)?
3. [📚] How do these criteria help in model selection?
4. [📚] What is the tradeoff between model complexity and fit?

For a detailed explanation of this question, see [Question 3: GMM Model Selection](L8_5_3_explanation.md).

## Question 4

### Problem Statement
GMMs provide "soft clustering" where each point has a probability of belonging to each cluster.

#### Task
1. [📚] What is the difference between hard clustering and soft clustering?
2. [📚] How do you calculate the probability that a point belongs to cluster $k$?
3. [📚] What is the advantage of soft clustering over hard clustering?
4. [📚] How would you convert soft clustering to hard clustering?

For a detailed explanation of this question, see [Question 4: Soft vs Hard Clustering](L8_5_4_explanation.md).

## Question 5

### Problem Statement
Consider the advantages of model-based clustering approaches like GMM.

#### Task
1. [📚] **Advantage 1**: How do GMMs handle uncertainty in cluster assignments?
2. [📚] **Advantage 2**: What statistical properties do GMMs provide?
3. [📚] **Advantage 3**: How do GMMs compare to K-means in terms of cluster shapes?
4. [📚] **Advantage 4**: What types of data are GMMs particularly well-suited for?

For a detailed explanation of this question, see [Question 5: GMM Advantages](L8_5_5_explanation.md).
