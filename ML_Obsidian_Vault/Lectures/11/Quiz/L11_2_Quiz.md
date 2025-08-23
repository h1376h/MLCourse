# Lecture 11.2: K-Means Clustering Quiz

## Overview
This quiz contains 18 questions covering different topics from section 11.2 of the lectures on K-Means Clustering, including the algorithm steps, initialization strategies, convergence properties, limitations, and implementation details.

## Question 1

### Problem Statement
The K-Means algorithm follows a specific iterative process to find optimal cluster centroids.

#### Task
1. [📚] List the three main steps of the K-Means algorithm
2. [🔍] Why is K-Means called an "iterative" algorithm?
3. [📚] What is the stopping criterion for K-Means?
4. [🔍] How do you initialize the centroids at the beginning of the algorithm?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: K-Means Algorithm Steps](L11_2_1_explanation.md).

## Question 2

### Problem Statement
Consider the following 2D dataset and apply one iteration of K-Means with K=2:

Initial centroids: C1=(1, 1), C2=(4, 4)
Data points: A(0, 0), B(1, 2), C(3, 1), D(4, 5), E(5, 4)

#### Task
1. [📚] Assign each data point to the nearest centroid using Euclidean distance
2. [📚] Calculate the new centroid positions after the assignment step
3. [📚] Show your distance calculations for point A to both centroids
4. [📚] How much did each centroid move from its initial position?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: K-Means Manual Calculation](L11_2_2_explanation.md).

## Question 3

### Problem Statement
The objective function of K-Means is to minimize the within-cluster sum of squares (WCSS).

#### Task
1. [📚] Write the mathematical formula for the WCSS objective function
2. [📚] Calculate the WCSS for the clustering: Cluster 1: {(1,1), (2,2)} with centroid (1.5, 1.5), Cluster 2: {(4,4), (5,5)} with centroid (4.5, 4.5)
3. [🔍] Why does K-Means minimize WCSS rather than maximize between-cluster distance?
4. [📚] How does the WCSS typically change over iterations?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: K-Means Objective Function](L11_2_3_explanation.md).

## Question 4

### Problem Statement
K-Means++ is an improved initialization strategy that addresses some problems with random initialization.

#### Task
1. [🔍] What problem does K-Means++ solve compared to random initialization?
2. [📚] Describe the K-Means++ initialization procedure step by step
3. [📚] Calculate the probability of selecting each point as the second centroid using K-Means++ for points: A(0,0), B(2,0), C(4,0) with first centroid at (0,0)
4. [🔍] Why does K-Means++ lead to better clustering results?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: K-Means++ Initialization](L11_2_4_explanation.md).

## Question 5

### Problem Statement
K-Means convergence properties determine when and why the algorithm stops.

#### Task
1. [📚] Prove that K-Means always converges (will not run indefinitely)
2. [🔍] What does it mean for K-Means to converge to a "local optimum"?
3. [📚] Give an example of how different initializations can lead to different final clusterings
4. [🔍] How many possible clustering solutions exist for n points and k clusters?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: K-Means Convergence](L11_2_5_explanation.md).

## Question 6

### Problem Statement
K-Means has several important limitations that affect its applicability.

#### Task
1. [🔍] Why does K-Means struggle with non-spherical clusters?
2. [📚] Show an example of data where K-Means would fail to find the "natural" clustering
3. [🔍] How does K-Means perform with clusters of different sizes?
4. [📚] What happens when clusters have different densities?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: K-Means Limitations](L11_2_6_explanation.md).

## Question 7

### Problem Statement
Consider implementing K-Means from scratch with the following dataset:

Data points: (1,1), (1,2), (2,1), (8,8), (8,9), (9,8)
K = 2, Initial centroids: (0,0), (10,10)

#### Task
1. [📚] Implement one complete iteration of K-Means showing all calculations
2. [📚] Calculate the WCSS before and after this iteration
3. [📚] Continue for one more iteration and show the centroid movements
4. [🔍] How many more iterations would you expect before convergence?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: K-Means Implementation Example](L11_2_7_explanation.md).

## Question 8

### Problem Statement
The choice of distance metric affects K-Means clustering results.

#### Task
1. [📚] How would K-Means change if you used Manhattan distance instead of Euclidean?
2. [📚] Calculate centroids for points {(0,0), (2,0), (1,1)} using both L1 and L2 norms
3. [🔍] When might Manhattan distance be preferred for K-Means?
4. [📚] What happens to the cluster shapes with different distance metrics?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Distance Metrics in K-Means](L11_2_8_explanation.md).

## Question 9

### Problem Statement
Feature scaling significantly impacts K-Means clustering results.

#### Task
1. [📚] Why is feature scaling crucial for K-Means?
2. [📚] Consider features: Age (20-80), Income ($20k-$200k). Calculate the effect of not scaling on distance calculations
3. [📚] Apply standardization (z-score) to transform the data: [(25, 30000), (45, 60000), (65, 90000)]
4. [🔍] How do the clustering results differ before and after scaling?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Feature Scaling in K-Means](L11_2_9_explanation.md).

## Question 10

### Problem Statement
Choosing the optimal value of K is a critical decision in K-Means clustering.

#### Task
1. [🔍] Why can't you simply choose K to minimize WCSS?
2. [📚] Describe the "elbow method" for choosing K
3. [📚] Given WCSS values: K=1: 100, K=2: 60, K=3: 40, K=4: 35, K=5: 32, which K would you choose using the elbow method?
4. [🔍] What are the limitations of the elbow method?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Choosing Optimal K](L11_2_10_explanation.md).

## Question 11

### Problem Statement
K-Means computational complexity affects its scalability to large datasets.

#### Task
1. [📚] What is the time complexity of K-Means per iteration?
2. [📚] How does the complexity scale with the number of data points, clusters, and dimensions?
3. [🔍] Why is K-Means considered efficient for large datasets?
4. [📚] Compare the space complexity requirements of K-Means

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: K-Means Complexity Analysis](L11_2_11_explanation.md).

## Question 12

### Problem Statement
Empty clusters can occur during K-Means execution and need special handling.

#### Task
1. [🔍] How can a cluster become empty during K-Means execution?
2. [📚] Give a specific example of data and initialization that would cause an empty cluster
3. [🔍] What strategies can be used to handle empty clusters?
4. [📚] How does K-Means++ initialization help prevent empty clusters?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: Handling Empty Clusters](L11_2_12_explanation.md).

## Question 13

### Problem Statement
K-Means can be adapted for different types of data and constraints.

#### Task
1. [🔍] How would you modify K-Means for categorical data?
2. [📚] What distance metric would be appropriate for binary features?
3. [🔍] How can you incorporate constraints (e.g., must-link, cannot-link) into K-Means?
4. [📚] Describe how "fuzzy K-Means" differs from standard K-Means

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 13: K-Means Variants](L11_2_13_explanation.md).

## Question 14

### Problem Statement
Consider a customer segmentation problem with the following data:

| Customer | Age | Income | Spending |
|----------|-----|--------|----------|
| A        | 25  | 40k    | High     |
| B        | 35  | 60k    | Medium   |
| C        | 45  | 80k    | Low      |
| D        | 30  | 50k    | High     |

#### Task
1. [📚] How would you prepare this mixed data type for K-Means?
2. [📚] What value of K would you choose and why?
3. [🔍] How would you interpret the resulting clusters for business decisions?
4. [📚] What additional features might improve the clustering quality?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 14: Customer Segmentation with K-Means](L11_2_14_explanation.md).

## Question 15

### Problem Statement
Outliers can significantly impact K-Means clustering results.

#### Task
1. [🔍] Why is K-Means sensitive to outliers?
2. [📚] Show how a single outlier can distort cluster centroids with an example
3. [🔍] What preprocessing steps can help mitigate outlier effects?
4. [📚] How do robust variants of K-Means handle outliers?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 15: Outlier Effects in K-Means](L11_2_15_explanation.md).

## Question 16

### Problem Statement
Mini-batch K-Means is a scalable variant for large datasets.

#### Task
1. [🔍] How does Mini-batch K-Means differ from standard K-Means?
2. [📚] What are the advantages and trade-offs of using mini-batches?
3. [📚] Describe the update rule for centroids in Mini-batch K-Means
4. [🔍] When would you choose Mini-batch K-Means over standard K-Means?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 16: Mini-batch K-Means](L11_2_16_explanation.md).

## Question 17

### Problem Statement
K-Means clustering quality can be evaluated using various metrics.

#### Task
1. [📚] Calculate the silhouette score for a point at distance 2 from its cluster center and distance 5 from the nearest other cluster center
2. [📚] What does a negative silhouette score indicate?
3. [🔍] How do you interpret cluster compactness vs separation?
4. [📚] Compare internal vs external evaluation measures for K-Means

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 17: K-Means Evaluation Metrics](L11_2_17_explanation.md).

## Question 18

### Problem Statement
Image color quantization is a classic application of K-Means clustering.

#### Task
1. [🔍] Explain how K-Means can be used for image color quantization
2. [📚] If an image has RGB colors [(255,0,0), (200,50,50), (180,30,30)], how would K-Means group them with K=2?
3. [📚] Calculate the compression ratio when reducing from 256 colors to 16 colors
4. [🔍] What are the trade-offs between compression ratio and image quality?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 18: Image Color Quantization](L11_2_18_explanation.md).
