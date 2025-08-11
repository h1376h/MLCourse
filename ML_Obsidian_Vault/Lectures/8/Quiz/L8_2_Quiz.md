# Lecture 8.2: K-Means Clustering Quiz

## Overview
This quiz contains 5 questions covering different topics from section 8.2 of the lectures on K-Means Clustering, including the algorithm steps, initialization strategies, convergence properties, and limitations.

## Question 1

### Problem Statement
Consider the following dataset with 8 points in 2D space:

| Point | $x_1$ | $x_2$ |
|-------|-------|-------|
| A     | 1     | 1     |
| B     | 1     | 2     |
| C     | 2     | 1     |
| D     | 2     | 2     |
| E     | 8     | 8     |
| F     | 8     | 9     |
| G     | 9     | 8     |
| H     | 9     | 9     |

#### Task
1. [🔍] If you want to cluster this data into 2 groups, suggest initial cluster centers
2. [🔍] Perform one iteration of K-means: assign each point to the nearest center
3. [🔍] Calculate the new cluster centers after the first assignment
4. [🔍] Would you expect this algorithm to converge quickly? Explain why

For a detailed explanation of this question, see [Question 1: K-Means Initialization and First Iteration](L8_2_1_explanation.md).

## Question 2

### Problem Statement
The K-means algorithm follows these steps: initialization, assignment, update, and repeat until convergence.

#### Task
1. [📚] Explain what happens during the "assignment" step
2. [📚] Explain what happens during the "update" step
3. [📚] What is the mathematical formula for updating cluster centers?
4. [📚] How do you determine when the algorithm has converged?

For a detailed explanation of this question, see [Question 2: K-Means Algorithm Steps](L8_2_2_explanation.md).

## Question 3

### Problem Statement
Consider the following scenario: You run K-means with $k=3$ on a dataset and get different results each time.

#### Task
1. [📚] Why does K-means give different results on different runs?
2. [📚] What initialization strategy could help make results more consistent?
3. [📚] How would you implement K-means++ initialization?
4. [📚] What are the advantages of K-means++ over random initialization?

For a detailed explanation of this question, see [Question 3: K-Means Initialization Strategies](L8_2_3_explanation.md).

## Question 4

### Problem Statement
You have a dataset with 1000 points and want to find the optimal number of clusters using the elbow method.

#### Task
1. [📚] What is the elbow method and how does it work?
2. [📚] What metric would you plot on the y-axis when using the elbow method?
3. [📚] How would you interpret the "elbow" in the plot?
4. [📚] What are the limitations of the elbow method?

For a detailed explanation of this question, see [Question 4: Elbow Method for Optimal K](L8_2_4_explanation.md).

## Question 5

### Problem Statement
Consider the limitations of K-means clustering in different scenarios.

#### Task
1. [📚] **Scenario A**: Data with clusters of different sizes
2. [📚] **Scenario B**: Data with clusters of different shapes (non-spherical)
3. [📚] **Scenario C**: Data with outliers
4. [📚] **Scenario D**: Data with clusters of different densities

For each scenario, explain why K-means might struggle and suggest an alternative clustering approach.

For a detailed explanation of this question, see [Question 5: K-Means Limitations and Alternatives](L8_2_5_explanation.md).
