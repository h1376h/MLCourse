# Lecture 11.3: Hierarchical Clustering Quiz

## Overview
This quiz contains 16 questions covering different topics from section 11.3 of the lectures on Hierarchical Clustering, including tree-based clustering structure, agglomerative and divisive approaches, linkage methods, dendrogram interpretation, and advantages/disadvantages.

## Question 1

### Problem Statement
Hierarchical clustering creates a tree-like structure that shows relationships between data points at different levels of granularity.

#### Task
1. [🔍] What is the main difference between hierarchical and partitional clustering (like K-Means)?
2. [📚] Explain the difference between agglomerative and divisive hierarchical clustering
3. [🔍] Why is hierarchical clustering called "hierarchical"?
4. [📚] What is a dendrogram and what information does it provide?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Hierarchical Clustering Fundamentals](L11_3_1_explanation.md).

## Question 2

### Problem Statement
Agglomerative clustering builds clusters from the bottom up by merging the closest pairs of points or clusters.

#### Task
1. [📚] Describe the agglomerative clustering algorithm step by step
2. [📚] What data structure is used to efficiently find the closest pair of clusters?
3. [🔍] How does the algorithm determine when to stop merging?
4. [📚] What is the time complexity of agglomerative clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Agglomerative Clustering Algorithm](L11_3_2_explanation.md).

## Question 3

### Problem Statement
Consider the following distance matrix for points A, B, C, D:

|   | A | B | C | D |
|---|---|---|---|---|
| A | 0 | 2 | 6 | 8 |
| B | 2 | 0 | 5 | 7 |
| C | 6 | 5 | 0 | 3 |
| D | 8 | 7 | 3 | 0 |

#### Task
1. [📚] Perform agglomerative clustering using single linkage, showing each merge step
2. [📚] Draw the resulting dendrogram
3. [📚] What clusters would you get if you cut the dendrogram at distance 4?
4. [📚] How many different clusterings are possible with this dendrogram?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Manual Agglomerative Clustering](L11_3_3_explanation.md).

## Question 4

### Problem Statement
Linkage criteria determine how the distance between clusters is measured in hierarchical clustering.

#### Task
1. [📚] Define and explain single linkage (minimum linkage)
2. [📚] Define and explain complete linkage (maximum linkage)
3. [📚] Define and explain average linkage
4. [🔍] When would you prefer each linkage method?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Linkage Methods Overview](L11_3_4_explanation.md).

## Question 5

### Problem Statement
Compare different linkage methods using the same dataset with clusters A={1,2}, B={8,9}, C={5}:

#### Task
1. [📚] Calculate the distance between clusters A and B using single linkage (assume Euclidean distance)
2. [📚] Calculate the distance using complete linkage
3. [📚] Calculate the distance using average linkage
4. [🔍] Which linkage method would merge A and C first, and why?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Linkage Methods Comparison](L11_3_5_explanation.md).

## Question 6

### Problem Statement
Ward linkage is a special linkage method that minimizes within-cluster variance.

#### Task
1. [📚] Explain the principle behind Ward linkage
2. [📚] How does Ward linkage differ from other linkage methods in terms of objective function?
3. [🔍] What types of cluster shapes does Ward linkage tend to produce?
4. [📚] Calculate the Ward distance between clusters {(1,1), (2,2)} and {(5,5)}
5. [📚] For three clusters C₁={(0,0), (1,0)}, C₂={(3,0)}, C₃={(0,3), (1,3)}, calculate the Ward distances for all possible merges (C₁∪C₂, C₁∪C₃, C₂∪C₃) and determine which merge minimizes the increase in total within-cluster sum of squares. 

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Ward Linkage Method](L11_3_6_explanation.md).

## Question 7

### Problem Statement
Dendrograms provide a visual representation of the hierarchical clustering process.

#### Task
1. [📚] What information can you extract from a dendrogram?
2. [🔍] How do you determine the optimal number of clusters from a dendrogram?
3. [📚] What does the height of a merge point in a dendrogram represent?
4. [🔍] How do you interpret the leaf nodes and internal nodes in a dendrogram?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Dendrogram Interpretation](L11_3_7_explanation.md).

## Question 8

### Problem Statement
Consider the following dendrogram for points A, B, C, D, E:

```
      ┌─ A
   ┌─┤
   │  └─ B
───┤
   │     ┌─ C
   └────┤
        │  ┌─ D
        └─┤
           └─ E
```

(Heights: A-B merge at 2, C merge at 4, D-E merge at 3, final merge at 6)

#### Task
1. [📚] What is the distance between clusters {A,B} and {C}?
2. [📚] What clusters would result from cutting at height 3.5?
3. [📚] Which two points are most similar according to this dendrogram?
4. [🔍] How many possible clusterings can be obtained from this dendrogram?
5. [📚] Calculate the cophenetic distance between all pairs of points (A,B), (A,C), (A,D), (A,E), (B,C), (B,D), (B,E), (C,D), (C,E), (D,E) based on this dendrogram. The cophenetic distance is the height at which two points are first joined in the same cluster.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Dendrogram Analysis](L11_3_8_explanation.md).

## Question 9

### Problem Statement
Divisive clustering takes a top-down approach, starting with all points in one cluster.

#### Task
1. [📚] Describe the divisive clustering algorithm
2. [🔍] What are the main challenges in divisive clustering compared to agglomerative?
3. [📚] How do you decide which cluster to split and where to split it?
4. [🔍] Why is divisive clustering less commonly used than agglomerative clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Divisive Clustering](L11_3_9_explanation.md).

## Question 10

### Problem Statement
Hierarchical clustering has distinct advantages and disadvantages compared to other clustering methods.

#### Task
1. [📚] **Advantage 1**: How does hierarchical clustering handle the choice of number of clusters?
2. [📚] **Advantage 2**: What makes hierarchical clustering results more interpretable?
3. [📚] **Disadvantage 1**: What is the main computational limitation of hierarchical clustering?
4. [📚] **Disadvantage 2**: How sensitive is hierarchical clustering to outliers and noise?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Advantages and Disadvantages](L11_3_10_explanation.md).

## Question 11

### Problem Statement
The choice of distance metric significantly affects hierarchical clustering results.

#### Task
1. [📚] How does using Manhattan distance vs Euclidean distance affect the clustering?
2. [📚] For the points (0,0), (1,1), (0,2), calculate both Euclidean and Manhattan distance matrices
3. [🔍] When might you choose correlation distance for hierarchical clustering?
4. [📚] How do you handle mixed data types in hierarchical clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: Distance Metrics in Hierarchical Clustering](L11_3_11_explanation.md).

## Question 12

### Problem Statement
Consider a gene expression dataset where you want to cluster genes based on their expression patterns across different conditions.

#### Task
1. [🔍] Why is hierarchical clustering particularly suitable for gene expression analysis?
2. [📚] What distance metric would be appropriate for gene expression data?
3. [📚] How would you interpret a dendrogram of gene clusters in biological terms?
4. [🔍] What advantages does hierarchical clustering offer over K-Means for this application?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: Gene Expression Clustering](L11_3_12_explanation.md).

## Question 13

### Problem Statement
Phylogenetic trees in biology are a form of hierarchical clustering showing evolutionary relationships.

#### Task
1. [🔍] How do phylogenetic trees relate to hierarchical clustering?
2. [📚] What does the distance represent in a phylogenetic context?
3. [📚] How would you interpret branch lengths and topology in a phylogenetic tree?
4. [🔍] What special considerations apply to phylogenetic clustering that don't apply to general clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 13: Phylogenetic Trees and Clustering](L11_3_13_explanation.md).

## Question 14

### Problem Statement
Cutting a dendrogram at different heights produces different clustering solutions.

#### Task
1. [📚] How do you choose the optimal cutting height for a dendrogram?
2. [🔍] What is the relationship between cutting height and number of clusters?
3. [📚] How can you use the "elbow method" with hierarchical clustering?
4. [🔍] What happens if you cut too high vs too low in the dendrogram?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 14: Optimal Dendrogram Cutting](L11_3_14_explanation.md).

## Question 15

### Problem Statement
Large datasets pose computational challenges for traditional hierarchical clustering.

#### Task
1. [📚] What is the space and time complexity of agglomerative clustering?
2. [🔍] How can you make hierarchical clustering more scalable for large datasets?
3. [📚] What are the trade-offs between accuracy and scalability in approximate methods?
4. [🔍] When would you choose hierarchical clustering over K-Means for large datasets?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 15: Scalability Challenges](L11_3_15_explanation.md).

## Question 16

### Problem Statement
Consider a market research scenario where you want to cluster customers based on their purchasing behavior across different product categories.

#### Task
1. [📚] What advantages would hierarchical clustering offer for customer segmentation?
2. [📚] How would you prepare transaction data for hierarchical clustering?
3. [🔍] How could the resulting dendrogram help in marketing strategy decisions?
4. [📚] What linkage method would be most appropriate and why?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 16: Customer Segmentation with Hierarchical Clustering](L11_3_16_explanation.md).
