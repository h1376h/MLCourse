# Lecture 11.1: Foundations of Unsupervised Learning Quiz

## Overview
This quiz contains 15 questions covering different topics from section 11.1 of the lectures on Foundations of Unsupervised Learning, including unsupervised learning concepts, clustering overview, distance metrics, clustering applications, and clustering challenges.

## Question 1

### Problem Statement
Unsupervised learning differs fundamentally from supervised learning in the type of data available and the learning objectives.

#### Task
1. [🔍] Define unsupervised learning in one sentence
2. [🔍] What is the main challenge in unsupervised learning compared to supervised learning?
3. [🔍] List three common types of unsupervised learning tasks
4. [📚] Explain why evaluation of unsupervised learning algorithms is more difficult than supervised learning

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Unsupervised vs Supervised Learning](L11_1_1_explanation.md).

## Question 2

### Problem Statement
Clustering is one of the most important unsupervised learning tasks, aiming to group similar data points together.

#### Task
1. [🔍] Define clustering in one sentence
2. [🔍] What makes two data points "similar" in clustering?
3. [📚] Explain the difference between hard clustering and soft clustering
4. [📚] Give three real-world examples where clustering would be useful

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Clustering Fundamentals](L11_1_2_explanation.md).

## Question 3

### Problem Statement
Distance metrics are fundamental to many clustering algorithms, as they determine how similarity between data points is measured.

#### Task
1. [📚] Calculate the Euclidean distance between points A(2, 3) and B(5, 7)
2. [📚] Calculate the Manhattan (L1) distance between the same points
3. [📚] Calculate the Cosine similarity between vectors [1, 2, 3] and [2, 4, 6]
4. [🔍] When would you choose Manhattan distance over Euclidean distance?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Distance Metrics Calculations](L11_1_3_explanation.md).

## Question 4

### Problem Statement
Consider the following 2D dataset for clustering:

| Point | X | Y |
|-------|---|---|
| A     | 1 | 2 |
| B     | 2 | 3 |
| C     | 8 | 9 |
| D     | 9 | 8 |
| E     | 1 | 3 |
| F     | 8 | 8 |

#### Task
1. [📚] Calculate the distance matrix using Euclidean distance
2. [📚] Which two points are closest to each other?
3. [📚] Which two points are farthest apart?
4. [🔍] Based on the distances, suggest a natural clustering of these points into 2 clusters

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Distance Matrix Analysis](L11_1_4_explanation.md).

## Question 5

### Problem Statement
Customer segmentation is a classic application of clustering in business analytics.

#### Task
1. [🔍] What is customer segmentation and why is it important for businesses?
2. [📚] List four customer attributes that would be useful for clustering
3. [📚] How would you measure the success of a customer segmentation clustering?
4. [🔍] What challenges might arise when clustering customer data?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Customer Segmentation](L11_1_5_explanation.md).

## Question 6

### Problem Statement
Image compression using clustering involves reducing the number of colors in an image by grouping similar colors.

#### Task
1. [🔍] Explain how clustering can be used for image compression
2. [📚] If an image has 1000 unique colors and you cluster them into 16 groups, what is the compression ratio?
3. [📚] What distance metric would be appropriate for clustering RGB color values?
4. [🔍] What are the trade-offs between compression ratio and image quality?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Image Compression with Clustering](L11_1_6_explanation.md).

## Question 7

### Problem Statement
Choosing the optimal number of clusters is one of the fundamental challenges in clustering.

#### Task
1. [🔍] Why is determining the number of clusters challenging in unsupervised learning?
2. [📚] What happens if you choose too few clusters? Too many clusters?
3. [🔍] List three methods for determining the optimal number of clusters
4. [📚] In what scenarios might the "true" number of clusters be subjective?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Choosing Number of Clusters](L11_1_7_explanation.md).

## Question 8

### Problem Statement
Noise and outliers can significantly impact clustering results.

#### Task
1. [🔍] Define what constitutes "noise" in the context of clustering
2. [📚] How do outliers affect clustering algorithms differently?
3. [🔍] What strategies can be used to handle noisy data in clustering?
4. [📚] Give an example of how a single outlier could mislead a clustering algorithm

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Handling Noise and Outliers](L11_1_8_explanation.md).

## Question 9

### Problem Statement
Feature scaling and preprocessing are crucial steps before applying clustering algorithms.

#### Task
1. [📚] Why is feature scaling important for distance-based clustering?
2. [📚] Compare the effect of scaling on Euclidean vs Manhattan distance
3. [📚] Given features: Age (20-80), Income ($20k-$200k), calculate scaled versions using min-max normalization
4. [🔍] When might you choose standardization over min-max normalization for clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Feature Scaling for Clustering](L11_1_9_explanation.md).

## Question 10

### Problem Statement
Clustering can be performed in different types of feature spaces with varying characteristics.

#### Task
1. [🔍] What challenges arise when clustering high-dimensional data?
2. [📚] How does the "curse of dimensionality" affect clustering algorithms?
3. [📚] What is the difference between clustering in numerical vs categorical feature spaces?
4. [🔍] How would you cluster mixed data types (numerical and categorical)?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Feature Space Considerations](L11_1_10_explanation.md).

## Question 11

### Problem Statement
Consider a document clustering scenario where you want to group similar news articles.

#### Task
1. [📚] What features would you extract from text documents for clustering?
2. [🔍] Why might Cosine similarity be preferred over Euclidean distance for text clustering?
3. [📚] How would you handle documents of different lengths?
4. [🔍] What preprocessing steps would be important for text clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: Document Clustering](L11_1_11_explanation.md).

## Question 12

### Problem Statement
Clustering quality assessment requires different approaches since there are no ground truth labels.

#### Task
1. [🔍] What makes evaluating clustering results challenging?
2. [📚] Explain the difference between internal and external validation measures
3. [📚] When would you use external validation if you have some labeled data?
4. [🔍] How can domain expertise help in evaluating clustering results?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: Clustering Evaluation Challenges](L11_1_12_explanation.md).

## Question 13

### Problem Statement
Clustering algorithms make different assumptions about the underlying data structure.

#### Task
1. [🔍] What assumption does a centroid-based clustering algorithm make about cluster shape?
2. [📚] What types of clusters are difficult for distance-based algorithms to find?
3. [🔍] How do density-based algorithms differ in their assumptions?
4. [📚] Give an example of data where hierarchical clustering might be preferred over centroid-based clustering

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 13: Clustering Algorithm Assumptions](L11_1_13_explanation.md).

## Question 14

### Problem Statement
Gene expression analysis often uses clustering to identify groups of genes with similar expression patterns.

#### Task
1. [🔍] Why is clustering useful in bioinformatics and gene expression analysis?
2. [📚] What distance metric would be appropriate for gene expression data?
3. [📚] How would you interpret a cluster of genes that are highly correlated?
4. [🔍] What challenges arise when clustering high-dimensional biological data?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 14: Bioinformatics Clustering Applications](L11_1_14_explanation.md).

## Question 15

### Problem Statement
Social network analysis uses clustering to identify communities and groups within networks.

#### Task
1. [🔍] How is clustering applied to social network data?
2. [📚] What would constitute a "distance" between two users in a social network?
3. [📚] How would you represent social network data for clustering algorithms?
4. [🔍] What insights can be gained from clustering social network data?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 15: Social Network Clustering](L11_1_15_explanation.md).
