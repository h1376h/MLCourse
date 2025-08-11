# Lecture 8.6: Clustering Evaluation and Validation Quiz

## Overview
This quiz contains 5 questions covering different topics from section 8.6 of the lectures on Clustering Evaluation and Validation, including internal indices, external indices, elbow method, and cross-validation for clustering.

## Question 1

### Problem Statement
Consider the following clustering results with 3 clusters:

| Point | Cluster | $x_1$ | $x_2$ |
|-------|---------|-------|-------|
| A     | 1       | 1     | 1     |
| B     | 1       | 1     | 2     |
| C     | 1       | 2     | 1     |
| D     | 2       | 8     | 8     |
| E     | 2       | 8     | 9     |
| F     | 3       | 15    | 15    |

#### Task
1. [🔍] Calculate the within-cluster sum of squares (WCSS) for cluster 1
2. [🔍] Calculate the between-cluster sum of squares (BCSS)
3. [🔍] What is the Silhouette coefficient for point A?
4. [🔍] How would you interpret a high Silhouette coefficient?

For a detailed explanation of this question, see [Question 1: Internal Clustering Indices](L8_6_1_explanation.md).

## Question 2

### Problem Statement
You have ground truth labels for a clustering problem and want to evaluate clustering quality.

#### Task
1. [📚] What is the Adjusted Rand Index (ARI)?
2. [📚] What is the Normalized Mutual Information (NMI)?
3. [📚] What range of values do these metrics take?
4. [📚] When would you use external vs internal indices?

For a detailed explanation of this question, see [Question 2: External Clustering Indices](L8_6_2_explanation.md).

## Question 3

### Problem Statement
The elbow method is a common technique for choosing the optimal number of clusters.

#### Task
1. [📚] What metric do you plot on the y-axis in the elbow method?
2. [📚] What does the "elbow" represent in the plot?
3. [📚] What are the limitations of the elbow method?
4. [📚] How would you implement the elbow method computationally?

For a detailed explanation of this question, see [Question 3: Elbow Method for Cluster Selection](L8_6_3_explanation.md).

## Question 4

### Problem Statement
Cross-validation can be adapted for clustering evaluation using stability measures.

#### Task
1. [📚] How do you perform cross-validation for clustering?
2. [📚] What is clustering stability and how do you measure it?
3. [📚] What are the challenges of cross-validation for clustering?
4. [📚] How does stability relate to cluster quality?

For a detailed explanation of this question, see [Question 4: Cross-Validation for Clustering](L8_6_4_explanation.md).

## Question 5

### Problem Statement
Consider different evaluation scenarios for clustering algorithms.

#### Task
1. [📚] **Scenario A**: You have ground truth labels and want to compare algorithms
2. [📚] **Scenario B**: You have no ground truth and want to choose the best K
3. [📚] **Scenario C**: You want to evaluate clustering robustness
4. [📚] **Scenario D**: You want to compare clustering with different distance metrics

For each scenario, suggest the most appropriate evaluation approach.

For a detailed explanation of this question, see [Question 5: Clustering Evaluation Strategies](L8_6_5_explanation.md).
