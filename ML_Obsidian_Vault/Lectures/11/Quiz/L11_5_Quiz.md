# Lecture 11.5: Model-Based Clustering Quiz

## Overview
This quiz contains 19 questions covering different topics from section 11.5 of the lectures on Model-Based Clustering, including probabilistic clustering approaches, Gaussian Mixture Models, Expectation-Maximization algorithm, model selection, and soft clustering.

## Question 1

### Problem Statement
Model-based clustering takes a probabilistic approach to clustering by assuming data is generated from a mixture of probability distributions.

#### Task
1. [🔍] What is the fundamental assumption behind model-based clustering?
2. [📚] How does model-based clustering differ from distance-based clustering methods?
3. [🔍] What does it mean for clustering to be "probabilistic"?
4. [📚] List three advantages of using a model-based approach to clustering

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Model-Based Clustering Fundamentals](L11_5_1_explanation.md).

## Question 2

### Problem Statement
Gaussian Mixture Models (GMM) are the most common form of model-based clustering.

#### Task
1. [📚] What is a Gaussian Mixture Model mathematically?
2. [📚] Write the formula for a mixture of K Gaussian distributions
3. [🔍] What do the parameters μ, Σ, and π represent in a GMM?
4. [📚] How many parameters need to be estimated for a GMM with K components in d dimensions?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Gaussian Mixture Models](L11_5_2_explanation.md).

## Question 3

### Problem Statement
The Expectation-Maximization (EM) algorithm is used to learn GMM parameters from data.

#### Task
1. [📚] Describe the two steps of the EM algorithm
2. [📚] What is computed in the E-step (Expectation step)?
3. [📚] What is computed in the M-step (Maximization step)?
4. [🔍] Why is EM called an iterative algorithm, and what guarantees its convergence?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Expectation-Maximization Algorithm](L11_5_3_explanation.md).

## Question 4

### Problem Statement
Consider a simple 1D dataset with points: {1, 2, 8, 9, 10} and assume K=2 Gaussian components.

Initial parameters: μ₁=1.5, σ₁²=1, π₁=0.5, μ₂=9, σ₂²=1, π₂=0.5

#### Task
1. [📚] Calculate the responsibility γᵢⱼ for each point belonging to each component
2. [📚] Perform the M-step to update μ₁, μ₂, σ₁², σ₂², π₁, π₂
3. [📚] Calculate the log-likelihood after this iteration
4. [🔍] Interpret the clustering results - which points belong to which cluster?
5. [📚] Continue the EM algorithm for a second iteration and calculate the change in log-likelihood. Determine if the algorithm has converged using a threshold of 0.01 for the log-likelihood change.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Manual EM Calculation](L11_5_4_explanation.md).

## Question 5

### Problem Statement
Model selection in GMM involves choosing the optimal number of components K.

#### Task
1. [🔍] Why can't you simply choose K to maximize likelihood?
2. [📚] Explain how the Akaike Information Criterion (AIC) addresses overfitting
3. [📚] Explain how the Bayesian Information Criterion (BIC) differs from AIC
4. [📚] Given log-likelihood = -100, K=3, n=200, d=2, calculate AIC and BIC
5. [📚] For model comparison, you have three GMMs with the following results: Model 1 (K=2): LL=-150, Model 2 (K=3): LL=-120, Model 3 (K=4): LL=-115. Calculate AIC and BIC for each (n=100, d=2) and determine which model each criterion selects.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Model Selection for GMM](L11_5_5_explanation.md).

## Question 6

### Problem Statement
Soft clustering provides probabilistic cluster assignments rather than hard assignments.

#### Task
1. [🔍] What is the difference between hard and soft clustering?
2. [📚] How do you interpret the responsibility values γᵢⱼ in GMM?
3. [📚] Convert soft clustering probabilities [0.8, 0.2] and [0.3, 0.7] to hard cluster assignments
4. [🔍] When would soft clustering be preferred over hard clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Soft Clustering Concepts](L11_5_6_explanation.md).

## Question 7

### Problem Statement
GMM can handle clusters with different shapes and orientations through covariance matrices.

#### Task
1. [📚] How does the covariance matrix Σ affect cluster shape?
2. [📚] What cluster shapes result from: (a) diagonal covariance, (b) spherical covariance, (c) full covariance?
3. [🔍] Why might you choose to constrain covariance matrices in practice?
4. [📚] How many parameters are needed for a full covariance matrix in d dimensions?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Covariance Structures in GMM](L11_5_7_explanation.md).

## Question 8

### Problem Statement
Initialization strategies for EM algorithm can significantly affect the final results.

#### Task
1. [🔍] Why is initialization important for the EM algorithm?
2. [📚] Describe three common initialization strategies for GMM
3. [📚] How can K-Means be used to initialize GMM parameters?
4. [🔍] What are the advantages and disadvantages of random initialization?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: EM Initialization Strategies](L11_5_8_explanation.md).

## Question 9

### Problem Statement
Consider a 2D dataset representing customer purchasing behavior with features: frequency and monetary value.

#### Task
1. [📚] How would you set up a GMM for this customer segmentation problem?
2. [🔍] What would each Gaussian component represent in business terms?
3. [📚] How would you interpret the mixing coefficients π for different customer segments?
4. [🔍] How could soft clustering help in marketing strategy development?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Customer Segmentation with GMM](L11_5_9_explanation.md).

## Question 10

### Problem Statement
Model-based clustering provides uncertainty quantification that other methods lack.

#### Task
1. [🔍] How does GMM quantify uncertainty in cluster assignments?
2. [📚] What does it mean if a point has responsibilities [0.6, 0.4] for two clusters?
3. [📚] How would you identify points that are difficult to classify?
4. [🔍] How can uncertainty information be used in decision-making?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Uncertainty Quantification](L11_5_10_explanation.md).

## Question 11

### Problem Statement
The EM algorithm is guaranteed to converge but may find local optima.

#### Task
1. [📚] Prove that the likelihood increases (or stays the same) at each EM iteration
2. [🔍] What is meant by "local optimum" in the context of EM?
3. [📚] How can multiple random initializations help find better solutions?
4. [🔍] What are the convergence criteria typically used for EM?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: EM Convergence Properties](L11_5_11_explanation.md).

## Question 12

### Problem Statement
Regularization techniques can improve GMM performance and prevent overfitting.

#### Task
1. [🔍] What problems can arise with unrestricted covariance matrices?
2. [📚] How does adding a regularization term to the covariance help?
3. [📚] Explain the concept of "regularized EM" for GMM
4. [🔍] When would you use tied covariances across components?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: Regularization in GMM](L11_5_12_explanation.md).

## Question 13

### Problem Statement
Compare GMM with K-Means clustering on the same dataset.

#### Task
1. [📚] How are GMM and K-Means related mathematically?
2. [📚] Under what conditions does GMM reduce to K-Means?
3. [🔍] What additional information does GMM provide that K-Means doesn't?
4. [📚] Compare computational complexity of GMM vs K-Means

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 13: GMM vs K-Means Comparison](L11_5_13_explanation.md).

## Question 14

### Problem Statement
High-dimensional data poses special challenges for GMM clustering.

#### Task
1. [🔍] What is the "curse of dimensionality" for GMM?
2. [📚] How does the number of parameters grow with dimensionality?
3. [📚] What regularization strategies are used for high-dimensional GMM?
4. [🔍] When might dimensionality reduction be applied before GMM?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 14: High-Dimensional GMM](L11_5_14_explanation.md).

## Question 15

### Problem Statement
Missing data can be handled naturally within the EM framework.

#### Task
1. [🔍] How can EM handle missing data values during clustering?
2. [📚] Modify the E-step to account for missing features
3. [📚] How does missing data affect parameter estimation in the M-step?
4. [🔍] What assumptions are made about the missing data mechanism?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 15: GMM with Missing Data](L11_5_15_explanation.md).

## Question 16

### Problem Statement
Mixture models can be extended beyond Gaussian distributions.

#### Task
1. [📚] When would you use a mixture of Bernoulli distributions?
2. [📚] How would you set up a mixture of Poisson distributions?
3. [🔍] What are the advantages of non-Gaussian mixture models?
4. [📚] How does the EM algorithm change for different distribution families?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 16: Non-Gaussian Mixture Models](L11_5_16_explanation.md).

## Question 17

### Problem Statement
Bayesian approaches to mixture modeling provide additional benefits over maximum likelihood.

#### Task
1. [🔍] What are the advantages of Bayesian mixture modeling?
2. [📚] How do priors affect the clustering results?
3. [📚] Explain the concept of the Dirichlet Process Mixture Model
4. [🔍] How does Bayesian model selection differ from AIC/BIC?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 17: Bayesian Mixture Models](L11_5_17_explanation.md).

## Question 18

### Problem Statement
Semi-supervised clustering combines labeled and unlabeled data in mixture models.

#### Task
1. [🔍] How can partial label information be incorporated into GMM?
2. [📚] Modify the EM algorithm to handle semi-supervised learning
3. [📚] What are the benefits of using labeled data in clustering?
4. [🔍] How do you balance the influence of labeled vs unlabeled data?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 18: Semi-Supervised GMM](L11_5_18_explanation.md).

## Question 19

### Problem Statement
Consider an image segmentation problem where you want to cluster pixels based on color values.

#### Task
1. [📚] How would you set up a GMM for RGB color clustering?
2. [🔍] What would each mixture component represent in image segmentation?
3. [📚] How would you determine the optimal number of color clusters?
4. [🔍] How could spatial information be incorporated into the model?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 19: Image Segmentation with GMM](L11_5_19_explanation.md).
