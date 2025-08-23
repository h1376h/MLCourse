# Lecture 11.7: Advanced Clustering Techniques Quiz

## Overview
This quiz contains 18 questions covering different topics from section 11.7 of the lectures on Advanced Clustering Techniques, including spectral clustering, mean shift clustering, affinity propagation, scalability considerations, and online clustering algorithms.

## Question 1

### Problem Statement
Spectral clustering uses eigendecomposition of similarity matrices to find clusters in data.

#### Task
1. [🔍] What is the fundamental principle behind spectral clustering?
2. [📚] How does spectral clustering differ from traditional distance-based methods?
3. [📚] What are the main steps of the spectral clustering algorithm?
4. [🔍] Why is spectral clustering effective for non-convex clusters?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Spectral Clustering Fundamentals](L11_7_1_explanation.md).

## Question 2

### Problem Statement
The similarity matrix (affinity matrix) is central to spectral clustering performance.

#### Task
1. [📚] How do you construct a similarity matrix from distance measurements?
2. [📚] Compare Gaussian (RBF) kernel vs k-nearest neighbor similarity matrices
3. [📚] Given distances d₁₂=2, d₁₃=5, d₂₃=3, calculate similarity using Gaussian kernel with σ=1
4. [🔍] How does the choice of similarity function affect clustering results?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Similarity Matrices in Spectral Clustering](L11_7_2_explanation.md).

## Question 3

### Problem Statement
The graph Laplacian matrix is key to understanding spectral clustering theory.

#### Task
1. [📚] Define the unnormalized graph Laplacian matrix L
2. [📚] Calculate the Laplacian for the adjacency matrix: [[0,1,0],[1,0,1],[0,1,0]]
3. [📚] What do the eigenvalues and eigenvectors of L represent?
4. [🔍] Why do we use the smallest eigenvectors for clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Graph Laplacian and Eigendecomposition](L11_7_3_explanation.md).

## Question 4

### Problem Statement
Mean shift clustering finds clusters by iteratively shifting points toward regions of higher density.

#### Task
1. [📚] Describe the mean shift algorithm step by step
2. [📚] What is the kernel function's role in mean shift?
3. [🔍] How does the bandwidth parameter affect clustering results?
4. [📚] Compare mean shift with DBSCAN in terms of density-based clustering

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Mean Shift Algorithm](L11_7_4_explanation.md).

## Question 5

### Problem Statement
Consider applying mean shift to the 1D dataset: [1, 2, 2.5, 8, 8.5, 9] with Gaussian kernel and bandwidth h=1.

#### Task
1. [📚] Calculate the mean shift vector for point x=2
2. [📚] Show one iteration of mean shift starting from x=2
3. [📚] Identify the modes (cluster centers) this algorithm would find
4. [🔍] How would changing bandwidth to h=3 affect the results?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Mean Shift Manual Calculation](L11_7_5_explanation.md).

## Question 6

### Problem Statement
Affinity Propagation uses message passing to identify exemplars and form clusters.

#### Task
1. [📚] What are "exemplars" in affinity propagation?
2. [📚] Describe the responsibility and availability messages
3. [🔍] How does affinity propagation automatically determine the number of clusters?
4. [📚] What is the preference parameter and how does it affect clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Affinity Propagation Algorithm](L11_7_6_explanation.md).

## Question 7

### Problem Statement
Large datasets pose scalability challenges that require specialized approaches.

#### Task
1. [📚] What are the main computational bottlenecks in traditional clustering algorithms?
2. [🔍] How do sampling-based approaches address scalability?
3. [📚] Explain the concept of "mini-batch" clustering algorithms
4. [🔍] Compare the trade-offs between accuracy and scalability in approximate clustering

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Scalability Challenges in Clustering](L11_7_7_explanation.md).

## Question 8

### Problem Statement
Online clustering algorithms process data streams where all data is not available at once.

#### Task
1. [🔍] What makes online clustering different from batch clustering?
2. [📚] Describe the challenges of clustering streaming data
3. [📚] How does online K-Means adapt to new data points?
4. [🔍] What are the trade-offs between memory usage and clustering quality in online algorithms?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Online Clustering Algorithms](L11_7_8_explanation.md).

## Question 9

### Problem Statement
CURE (Clustering Using REpresentatives) handles clusters of different shapes and sizes.

#### Task
1. [📚] How does CURE represent clusters differently from centroid-based methods?
2. [📚] What is the role of "representative points" in CURE?
3. [🔍] How does CURE handle outliers and noise?
4. [📚] Compare CURE with hierarchical clustering in terms of scalability

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: CURE Clustering Algorithm](L11_7_9_explanation.md).

## Question 10

### Problem Statement
BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) is designed for large datasets.

#### Task
1. [📚] What is the CF-tree data structure in BIRCH?
2. [📚] Define a Clustering Feature (CF) and its components
3. [🔍] How does BIRCH achieve memory efficiency?
4. [📚] Describe the two-phase approach of BIRCH

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: BIRCH Algorithm](L11_7_10_explanation.md).

## Question 11

### Problem Statement
Subspace clustering finds clusters in different subspaces of high-dimensional data.

#### Task
1. [🔍] Why is subspace clustering important for high-dimensional data?
2. [📚] What is the difference between subspace clustering and feature selection?
3. [📚] Describe the CLIQUE algorithm for subspace clustering
4. [🔍] How do you evaluate clustering quality in different subspaces?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: Subspace Clustering](L11_7_11_explanation.md).

## Question 12

### Problem Statement
Consider a scenario where you need to cluster a large social network with 1 million users and their connection patterns.

#### Task
1. [🔍] Which advanced clustering technique would be most appropriate and why?
2. [📚] How would you represent the social network data for clustering?
3. [📚] What scalability considerations are important for this problem?
4. [🔍] How would you evaluate the quality of social network clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: Social Network Clustering](L11_7_12_explanation.md).

## Question 13

### Problem Statement
Kernel methods can be applied to clustering for handling non-linearly separable data.

#### Task
1. [📚] How do kernel methods extend traditional clustering algorithms?
2. [📚] Describe kernel K-Means and how it differs from standard K-Means
3. [🔍] What types of kernels are commonly used in clustering?
4. [📚] What are the computational implications of using kernels in clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 13: Kernel-Based Clustering](L11_7_13_explanation.md).

## Question 14

### Problem Statement
Ensemble clustering combines multiple clustering solutions to create a more robust result.

#### Task
1. [🔍] What are the motivations for ensemble clustering?
2. [📚] Describe three approaches for generating diverse clusterings
3. [📚] How do you combine multiple clustering solutions into a consensus?
4. [🔍] When might ensemble clustering perform worse than individual methods?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 14: Ensemble Clustering Methods](L11_7_14_explanation.md).

## Question 15

### Problem Statement
Multi-view clustering deals with data that has multiple representations or views.

#### Task
1. [🔍] What is multi-view data and why does it require special clustering approaches?
2. [📚] Compare early fusion vs late fusion approaches in multi-view clustering
3. [📚] How does co-training apply to clustering multiple views?
4. [🔍] What are the benefits and challenges of multi-view clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 15: Multi-View Clustering](L11_7_15_explanation.md).

## Question 16

### Problem Statement
Deep clustering combines neural networks with clustering objectives.

#### Task
1. [🔍] How do deep learning methods enhance traditional clustering?
2. [📚] Describe the concept of joint representation learning and clustering
3. [📚] What is a deep embedding clustering (DEC) algorithm?
4. [🔍] What advantages do deep clustering methods offer for complex data?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 16: Deep Clustering Methods](L11_7_16_explanation.md).

## Question 17

### Problem Statement
Time series clustering requires special consideration of temporal dependencies.

#### Task
1. [📚] What makes time series clustering different from static data clustering?
2. [📚] Compare shape-based vs feature-based approaches to time series clustering
3. [🔍] How do you handle time series of different lengths in clustering?
4. [📚] What distance measures are appropriate for time series clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 17: Time Series Clustering](L11_7_17_explanation.md).

## Question 18

### Problem Statement
Design an advanced clustering solution for a real-world big data problem.

**Scenario**: You need to cluster 10 million documents from news articles, social media posts, and scientific papers. The goal is to identify emerging topics and trends over time.

#### Task
1. [📚] Which advanced clustering techniques would you combine and why?
2. [🔍] How would you handle the scalability requirements?
3. [📚] What preprocessing and feature extraction steps would be necessary?
4. [🔍] How would you incorporate the temporal aspect of emerging trends?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 18: Advanced Clustering System Design](L11_7_18_explanation.md).
