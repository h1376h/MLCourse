# Lecture 11.6: Clustering Evaluation and Validation Quiz

## Overview
This quiz contains 20 questions covering different topics from section 11.6 of the lectures on Clustering Evaluation and Validation, including internal and external validation measures, elbow method, gap statistic, cross-validation for clustering, and stability-based evaluation.

## Question 1

### Problem Statement
Evaluating clustering quality is challenging because there are often no ground truth labels in unsupervised learning.

#### Task
1. Why is clustering evaluation more difficult than supervised learning evaluation?
2. What are the two main categories of clustering evaluation measures?
3. When would you use external vs internal evaluation measures?
4. List three criteria that define a "good" clustering result

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Clustering Evaluation Fundamentals](L11_6_1_explanation.md).

## Question 2

### Problem Statement
Internal validation measures assess clustering quality using only the data and clustering results.

#### Task
1. Define the silhouette coefficient for a single data point
2. Calculate the silhouette score for a point with $a(i) = 0.3$ and $b(i) = 0.7$
3. What does a negative silhouette score indicate?
4. How do you compute the overall silhouette score for a clustering?
5. Given three points in a clustering: $P_1$ has intra-cluster distances $[0.2, 0.4, 0.3]$ and nearest other cluster distances $[0.8, 0.9, 0.7]$, $P_2$ has intra-cluster distances $[0.1, 0.2]$ and nearest other cluster distances $[0.6, 0.5]$, $P_3$ has intra-cluster distances $[0.5]$ and nearest other cluster distances $[0.4]$. Calculate the silhouette coefficient for each point and the overall silhouette score.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Silhouette Analysis](L11_6_2_explanation.md).

## Question 3

### Problem Statement
The Calinski-Harabasz Index measures the ratio of between-cluster to within-cluster variance.

#### Task
1. Write the formula for the Calinski-Harabasz Index
2. Calculate CH index for: Between-cluster SS $= 120$, Within-cluster SS $= 80$, $n = 100$, $k = 3$
3. What does a higher CH index value indicate?
4. How does the CH index relate to the F-statistic in ANOVA?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Calinski-Harabasz Index](L11_6_3_explanation.md).

## Question 4

### Problem Statement
The Davies-Bouldin Index measures the average similarity between clusters.

#### Task
1. Explain the principle behind the Davies-Bouldin Index
2. Calculate DB index for two clusters: Cluster 1 (avg intra-distance $= 2$, centroid distance $= 8$), Cluster 2 (avg intra-distance $= 3$)
3. What does a lower DB index value indicate?
4. Compare the interpretability of DB index vs silhouette coefficient

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Davies-Bouldin Index](L11_6_4_explanation.md).

## Question 5

### Problem Statement
External validation measures compare clustering results to known ground truth labels.

#### Task
1. When are external validation measures applicable in clustering?
2. What is the limitation of using external measures for unsupervised learning evaluation?
3. List four common external validation measures
4. How do external measures help in algorithm comparison and parameter tuning?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: External Validation Overview](L11_6_5_explanation.md).

## Question 6

### Problem Statement
The Adjusted Rand Index (ARI) measures agreement between two clustering assignments.

#### Task
1. Calculate the ARI for the confusion matrix: $[[50, 5], [10, 35]]$
2. What does an ARI of 0 indicate? An ARI of 1?
3. Why is the "adjusted" version preferred over the regular Rand Index?
4. How does ARI handle different cluster sizes and numbers?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Adjusted Rand Index](L11_6_6_explanation.md).

## Question 7

### Problem Statement
Normalized Mutual Information (NMI) measures the information shared between two clustering assignments.

#### Task
1. Explain the concept of mutual information in clustering context
2. Why is normalization necessary in mutual information measures?
3. Compare NMI with ARI - when might you prefer one over the other?
4. What are the range and interpretation of NMI values?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Normalized Mutual Information](L11_6_7_explanation.md).

## Question 8

### Problem Statement
The elbow method is a popular heuristic for determining the optimal number of clusters.

#### Task
1. Describe the elbow method procedure step by step
2. Given WCSS values: $K=1: 1000$, $K=2: 600$, $K=3: 400$, $K=4: 350$, $K=5: 320$, identify the elbow
3. What are the limitations of the elbow method?
4. How do you automate elbow detection instead of visual inspection?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Elbow Method](L11_6_8_explanation.md).

## Question 9

### Problem Statement
The Gap Statistic provides a statistical approach to choosing the number of clusters.

#### Task
1. Explain the principle behind the Gap Statistic
2. How is the reference distribution generated in Gap Statistic?
3. What does it mean when Gap$(k)$ is maximized?
4. Calculate Gap$(k)$ if $\log(W_k) = 5.2$ and $E[\log(W_k)] = 6.1$

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Gap Statistic](L11_6_9_explanation.md).

## Question 10

### Problem Statement
Cross-validation for clustering requires special approaches since there are no labels to predict.

#### Task
1. Why can't standard cross-validation be directly applied to clustering?
2. Describe stability-based cross-validation for clustering
3. How do you measure clustering stability across different subsamples?
4. What does high stability indicate about the clustering quality?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Cross-Validation for Clustering](L11_6_10_explanation.md).

## Question 11

### Problem Statement
Consider evaluating K-Means clustering on a customer dataset with the following results:

| $K$ | WCSS | Silhouette | CH Index | DB Index |
|---|------|------------|----------|----------|
| 2 | 150  | 0.65       | 89.2     | 0.85     |
| 3 | 100  | 0.55       | 95.1     | 0.75     |
| 4 | 80   | 0.45       | 88.7     | 0.90     |
| 5 | 70   | 0.35       | 82.3     | 1.10     |

#### Task
1. Which $K$ would each metric suggest as optimal?
2. Why do different metrics give different recommendations?
3. How would you make a final decision on the optimal $K$?
4. What additional information would help in this decision?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: Multi-Metric Evaluation Example](L11_6_11_explanation.md).

## Question 12

### Problem Statement
Clustering validation becomes more complex with different types of data and distance metrics.

#### Task
1. How do evaluation metrics change for non-Euclidean distance measures?
2. What special considerations apply to categorical data clustering evaluation?
3. How do you evaluate clustering quality for mixed data types?
4. What metrics are appropriate for density-based clustering like DBSCAN?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: Evaluation for Different Data Types](L11_6_12_explanation.md).

## Question 13

### Problem Statement
Stability-based evaluation measures how consistent clustering results are across different samples or initializations.

#### Task
1. Define clustering stability and why it's important
2. Describe the bootstrap resampling approach for stability assessment
3. How do you interpret high vs low stability scores?
4. Compare stability-based evaluation with other validation approaches

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 13: Stability-Based Evaluation](L11_6_13_explanation.md).

## Question 14

### Problem Statement
The choice of clustering algorithm affects which evaluation metrics are most appropriate.

#### Task
1. Which evaluation metrics are most suitable for K-Means clustering?
2. How do you evaluate hierarchical clustering results?
3. What special considerations apply to DBSCAN evaluation?
4. How do you compare algorithms that produce different types of clustering (hard vs soft)?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 14: Algorithm-Specific Evaluation](L11_6_14_explanation.md).

## Question 15

### Problem Statement
Ground truth labels may be available for some clustering problems, enabling external validation.

#### Task
1. In which scenarios might ground truth be available for clustering?
2. Calculate precision, recall, and F1-score for cluster assignment vs true labels
3. How do you handle the cluster label correspondence problem?
4. What are the limitations of external validation measures?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 15: External Validation with Ground Truth](L11_6_15_explanation.md).

## Question 16

### Problem Statement
Visual evaluation methods complement quantitative metrics in clustering assessment.

#### Task
1. What visual techniques are useful for clustering evaluation?
2. How do you use scatter plots for 2D clustering evaluation?
3. What is a silhouette plot and how do you interpret it?
4. How do you visualize high-dimensional clustering results?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 16: Visual Clustering Evaluation](L11_6_16_explanation.md).

## Question 17

### Problem Statement
Consider a clustering evaluation where you have both internal metrics and some ground truth labels available.

Clustering A: Silhouette = 0.7, ARI = 0.6
Clustering B: Silhouette = 0.5, ARI = 0.8

#### Task
1. Which clustering would you prefer and why?
2. What might cause this disagreement between internal and external measures?
3. How would you investigate this discrepancy further?
4. What additional factors should influence your decision?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 17: Conflicting Evaluation Results](L11_6_17_explanation.md).

## Question 18

### Problem Statement
Parameter sensitivity analysis helps understand how clustering quality depends on algorithm parameters.

#### Task
1. How do you conduct sensitivity analysis for clustering parameters?
2. What parameters would you analyze for K-Means, DBSCAN, and hierarchical clustering?
3. How do you visualize parameter sensitivity results?
4. How does sensitivity analysis inform parameter selection?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 18: Parameter Sensitivity Analysis](L11_6_18_explanation.md).

## Question 19

### Problem Statement
Domain-specific evaluation considers the practical utility of clustering results in specific applications.

#### Task
1. How might clustering evaluation differ for customer segmentation vs gene expression analysis?
2. What business metrics might be relevant for evaluating customer clustering?
3. How would you evaluate clustering quality in image segmentation?
4. What role does domain expertise play in clustering validation?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 19: Domain-Specific Evaluation](L11_6_19_explanation.md).

## Question 20

### Problem Statement
Design a comprehensive evaluation strategy for a real-world clustering project.

**Scenario**: You're clustering customer transaction data to identify distinct customer segments for targeted marketing. You have transaction features but no existing customer labels.

#### Task
1. What internal validation metrics would you use and why?
2. How would you determine the optimal number of clusters?
3. What stability analysis would you perform?
4. How would you validate the business value of your clustering results?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 20: Comprehensive Evaluation Strategy](L11_6_20_explanation.md).
