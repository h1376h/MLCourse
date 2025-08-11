# Lecture 8.3: Hierarchical Clustering Quiz

## Overview
This quiz contains 5 questions covering different topics from section 8.3 of the lectures on Hierarchical Clustering, including agglomerative clustering, linkage methods, dendrogram interpretation, and advantages/disadvantages.

## Question 1

### Problem Statement
Consider the following dataset with 5 points in 2D space:

| Point | $x_1$ | $x_2$ |
|-------|-------|-------|
| A     | 1     | 1     |
| B     | 1     | 2     |
| C     | 4     | 4     |
| D     | 4     | 5     |
| E     | 8     | 8     |

#### Task
1. [🔍] Calculate the Euclidean distance matrix between all pairs of points
2. [🔍] Using single linkage, which two points would be merged first?
3. [🔍] Using complete linkage, which two points would be merged first?
4. [🔍] Explain why the results might differ between single and complete linkage

For a detailed explanation of this question, see [Question 1: Distance Matrix and Linkage Methods](L8_3_1_explanation.md).

## Question 2

### Problem Statement
Hierarchical clustering can be either agglomerative (bottom-up) or divisive (top-down).

#### Task
1. [📚] Explain the difference between agglomerative and divisive clustering
2. [📚] Which approach is more commonly used and why?
3. [📚] What is the computational complexity of agglomerative clustering?
4. [📚] How many clusters do you start with in agglomerative clustering?

For a detailed explanation of this question, see [Question 2: Agglomerative vs Divisive Clustering](L8_3_2_explanation.md).

## Question 3

### Problem Statement
Different linkage methods produce different clustering results. Consider single, complete, and average linkage.

#### Task
1. [📚] How does single linkage measure distance between clusters?
2. [📚] How does complete linkage measure distance between clusters?
3. [📚] How does average linkage measure distance between clusters?
4. [📚] Which linkage method is most sensitive to outliers and why?

For a detailed explanation of this question, see [Question 3: Linkage Method Differences](L8_3_3_explanation.md).

## Question 4

### Problem Statement
A dendrogram shows the hierarchical structure of clusters. Consider the following dendrogram heights:

| Merge Step | Height |
|------------|--------|
| 1          | 0.5    |
| 2          | 1.2    |
| 3          | 2.8    |
| 4          | 4.5    |

#### Task
1. [📚] How many clusters were there after step 2?
2. [📚] What does the height represent in a dendrogram?
3. [📚] If you want 3 clusters, where would you "cut" the dendrogram?
4. [📚] How would you interpret a large jump in height between merge steps?

For a detailed explanation of this question, see [Question 4: Dendrogram Interpretation](L8_4_4_explanation.md).

## Question 5

### Problem Statement
Consider the advantages and disadvantages of hierarchical clustering compared to K-means.

#### Task
1. [📚] **Advantage**: What unique information does hierarchical clustering provide?
2. [📚] **Advantage**: How does hierarchical clustering handle the choice of number of clusters?
3. [📚] **Disadvantage**: What is the computational complexity limitation?
4. [📚] **Disadvantage**: Why can't hierarchical clustering handle large datasets efficiently?

For a detailed explanation of this question, see [Question 5: Hierarchical Clustering Pros and Cons](L8_3_5_explanation.md).
