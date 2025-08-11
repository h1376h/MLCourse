# Lecture 8: Clustering and Unsupervised Learning

## Overview
This module introduces unsupervised learning methods, with a focus on clustering algorithms. You'll learn about different clustering approaches, distance metrics, evaluation methods, and applications of unsupervised learning in machine learning.

### Lecture 8.1: Foundations of Unsupervised Learning
- [[L8_1_Unsupervised_Learning_Concept|Unsupervised Learning Concept]]: Learning without labels
- [[L8_1_Clustering_Overview|Clustering Overview]]: Grouping similar data points
- [[L8_1_Distance_Metrics|Distance Metrics]]: Euclidean, Manhattan, Cosine similarity
- [[L8_1_Clustering_Applications|Clustering Applications]]: Customer segmentation, image compression
- [[L8_1_Clustering_Challenges|Clustering Challenges]]: Choosing number of clusters, handling noise
- [[L8_1_Examples|Basic Examples]]: Simple clustering demonstrations
- Required Reading: Chapter 9 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L8_1_Quiz]]: Test your understanding of unsupervised learning foundations

### Lecture 8.2: K-Means Clustering
- [[L8_2_K_Means_Algorithm|K-Means Algorithm]]: Iterative clustering approach
- [[L8_2_Algorithm_Steps|Algorithm Steps]]: Initialization, assignment, update
- [[L8_2_Initialization_Strategies|Initialization Strategies]]: K-means++, random initialization
- [[L8_2_Convergence|Convergence Properties]]: When and why K-means converges
- [[L8_2_K_Means_Limitations|K-Means Limitations]]: Local optima, cluster shapes
- [[L8_2_Implementation|K-Means Implementation]]: Code examples and pseudocode
- [[L8_2_Examples|K-Means Examples]]: Applications and visualizations
- Required Reading: Chapter 9.1 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L8_2_Quiz]]: Test your understanding of K-means clustering

### Lecture 8.3: Hierarchical Clustering
- [[L8_3_Hierarchical_Clustering|Hierarchical Clustering]]: Tree-based clustering structure
- [[L8_3_Agglomerative_Clustering|Agglomerative Clustering]]: Bottom-up approach
- [[L8_3_Divisive_Clustering|Divisive Clustering]]: Top-down approach
- [[L8_3_Linkage_Methods|Linkage Methods]]: Single, complete, average, Ward
- [[L8_3_Dendrogram_Interpretation|Dendrogram Interpretation]]: Reading cluster hierarchies
- [[L8_3_Advantages_Disadvantages|Advantages and Disadvantages]]: When to use hierarchical clustering
- [[L8_3_Examples|Hierarchical Examples]]: Implementation and applications
- Required Reading: Chapter 9.2 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L8_3_Quiz]]: Test your understanding of hierarchical clustering

### Lecture 8.4: Density-Based Clustering
- [[L8_4_Density_Based_Clustering|Density-Based Clustering]]: Clustering based on data density
- [[L8_4_DBSCAN_Algorithm|DBSCAN Algorithm]]: Density-Based Spatial Clustering
- [[L8_4_Core_Border_Noise|Core, Border, and Noise Points]]: DBSCAN point classification
- [[L8_4_DBSCAN_Parameters|DBSCAN Parameters]]: Epsilon and MinPts selection
- [[L8_4_DBSCAN_Advantages|DBSCAN Advantages]]: Handling irregular shapes, noise
- [[L8_4_DBSCAN_Limitations|DBSCAN Limitations]]: Parameter sensitivity, high-dimensional data
- [[L8_4_Examples|DBSCAN Examples]]: Implementation and case studies
- Required Reading: "A Density-Based Algorithm for Discovering Clusters" by Ester et al.
- Quiz: [[L8_4_Quiz]]: Test your understanding of density-based clustering

### Lecture 8.5: Model-Based Clustering
- [[L8_5_Model_Based_Clustering|Model-Based Clustering]]: Probabilistic clustering approaches
- [[L8_5_Gaussian_Mixture_Models|Gaussian Mixture Models]]: GMM clustering
- [[L8_5_Expectation_Maximization|Expectation-Maximization]]: EM algorithm for GMM
- [[L8_5_Model_Selection|Model Selection]]: Choosing number of components
- [[L8_5_Soft_Clustering|Soft Clustering]]: Probabilistic cluster assignments
- [[L8_5_Advantages_Model_Based|Advantages of Model-Based Approaches]]: Uncertainty quantification
- [[L8_5_Examples|Model-Based Examples]]: GMM implementation and applications
- Required Reading: Chapter 9.3 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L8_5_Quiz]]: Test your understanding of model-based clustering

### Lecture 8.6: Clustering Evaluation and Validation
- [[L8_6_Clustering_Evaluation|Clustering Evaluation]]: Measuring clustering quality
- [[L8_6_Internal_Indices|Internal Indices]]: Silhouette, Calinski-Harabasz, Davies-Bouldin
- [[L8_6_External_Indices|External Indices]]: Adjusted Rand Index, Normalized Mutual Information
- [[L8_6_Elbow_Method|Elbow Method]]: Choosing optimal number of clusters
- [[L8_6_Gap_Statistic|Gap Statistic]]: Statistical approach to cluster number selection
- [[L8_6_Cross_Validation_Clustering|Cross-Validation for Clustering]]: Stability-based evaluation
- [[L8_6_Examples|Evaluation Examples]]: Implementation and interpretation
- Required Reading: Chapter 9.4 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L8_6_Quiz]]: Test your understanding of clustering evaluation

### Lecture 8.7: Advanced Clustering Techniques
- [[L8_7_Spectral_Clustering|Spectral Clustering]]: Graph-based clustering approach
- [[L8_7_Mean_Shift|Mean Shift Clustering]]: Mode-seeking algorithm
- [[L8_7_Affinity_Propagation|Affinity Propagation]]: Message-passing clustering
- [[L8_7_Clustering_Large_Datasets|Clustering Large Datasets]]: Scalability considerations
- [[L8_7_Online_Clustering|Online Clustering]]: Incremental clustering algorithms
- [[L8_7_Examples|Advanced Examples]]: Implementation and applications
- Required Reading: Chapter 9.5 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L8_7_Quiz]]: Test your understanding of advanced clustering

### Lecture 8.8: Clustering Applications and Case Studies
- [[L8_8_Image_Segmentation|Image Segmentation]]: Clustering in computer vision
- [[L8_8_Customer_Segmentation|Customer Segmentation]]: Marketing applications
- [[L8_8_Document_Clustering|Document Clustering]]: Text mining applications
- [[L8_8_Anomaly_Detection|Anomaly Detection]]: Finding outliers using clustering
- [[L8_8_Real_World_Applications|Real-World Applications]]: Case studies and examples
- [[L8_8_Clustering_Challenges|Practical Challenges]]: Implementation considerations
- Required Reading: Chapter 9.6 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L8_8_Quiz]]: Test your understanding of clustering applications

## Programming Resources
- [[L8_Clustering_Python_Guide|Python Implementation Guide]]: Step-by-step implementation
- [[L8_K_Means_From_Scratch|Building K-Means from Scratch]]: Code tutorial
- [[L8_DBSCAN_Implementation|DBSCAN Algorithm Implementation]]: Density-based clustering
- [[L8_Scikit_Learn_Clustering|Using Scikit-learn for Clustering]]: Library tutorial
- [[L8_Clustering_Visualization|Clustering Visualization]]: Plotting techniques
- [[L8_GMM_Implementation|Gaussian Mixture Models]]: Probabilistic clustering
- [[L8_Clustering_Evaluation_Code|Clustering Evaluation Methods]]: Implementation of metrics

## Related Slides
*(not included in the repo)*
- Unsupervised_Learning_Foundations.pdf
- K_Means_Algorithm_Deep_Dive.pdf
- Hierarchical_Clustering_Methods.pdf
- Density_Based_Clustering.pdf
- Model_Based_Clustering.pdf
- Clustering_Evaluation.pdf
- Advanced_Clustering_Techniques.pdf
- Clustering_Applications.pdf

## Related Videos
- [Introduction to Unsupervised Learning](https://www.youtube.com/watch?v=8yZMXCaFshs)
- [K-Means Clustering Algorithm](https://www.youtube.com/watch?v=4b5d3muPnMA)
- [Hierarchical Clustering Methods](https://www.youtube.com/watch?v=7xHsRkOdVlE)
- [DBSCAN Density-Based Clustering](https://www.youtube.com/watch?v=MEs1ufJm92w)
- [Gaussian Mixture Models](https://www.youtube.com/watch?v=JNlEIEwe-Cg)
- [Clustering Evaluation Methods](https://www.youtube.com/watch?v=8yZMXCaFshs)
- [Advanced Clustering Techniques](https://www.youtube.com/watch?v=8yZMXCaFshs)

## All Quizzes
Test your understanding with these quizzes:
- [[L8_1_Quiz]]: Foundations of Unsupervised Learning
- [[L8_2_Quiz]]: K-Means Clustering
- [[L8_3_Quiz]]: Hierarchical Clustering
- [[L8_4_Quiz]]: Density-Based Clustering
- [[L8_5_Quiz]]: Model-Based Clustering
- [[L8_6_Quiz]]: Clustering Evaluation and Validation
- [[L8_7_Quiz]]: Advanced Clustering Techniques
- [[L8_8_Quiz]]: Clustering Applications and Case Studies
