# Lecture 11: Clustering and Unsupervised Learning

## Overview
This module introduces unsupervised learning methods, with a focus on clustering algorithms. You'll learn about different clustering approaches, distance metrics, evaluation methods, and applications of unsupervised learning in machine learning.

### Lecture 11.1: Foundations of Unsupervised Learning
- [[L11_1_Unsupervised_Learning_Concept|Unsupervised Learning Concept]]: Learning without labels
- [[L11_1_Clustering_Overview|Clustering Overview]]: Grouping similar data points
- [[L11_1_Distance_Metrics|Distance Metrics]]: Euclidean, Manhattan, Cosine similarity
- [[L11_1_Clustering_Applications|Clustering Applications]]: Customer segmentation, image compression
- [[L11_1_Clustering_Challenges|Clustering Challenges]]: Choosing number of clusters, handling noise
- [[L11_1_Examples|Basic Examples]]: Simple clustering demonstrations
- Required Reading: Chapter 9 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L11_1_Quiz]]: Test your understanding of unsupervised learning foundations

### Lecture 11.2: K-Means Clustering
- [[L11_2_K_Means_Algorithm|K-Means Algorithm]]: Iterative clustering approach
- [[L11_2_Algorithm_Steps|Algorithm Steps]]: Initialization, assignment, update
- [[L11_2_Initialization_Strategies|Initialization Strategies]]: K-means++, random initialization
- [[L11_2_Convergence|Convergence Properties]]: When and why K-means converges
- [[L11_2_K_Means_Limitations|K-Means Limitations]]: Local optima, cluster shapes
- [[L11_2_Implementation|K-Means Implementation]]: Code examples and pseudocode
- [[L11_2_Examples|K-Means Examples]]: Applications and visualizations
- Required Reading: Chapter 9.1 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L11_2_Quiz]]: Test your understanding of K-means clustering

### Lecture 11.3: Hierarchical Clustering
- [[L11_3_Hierarchical_Clustering|Hierarchical Clustering]]: Tree-based clustering structure
- [[L11_3_Agglomerative_Clustering|Agglomerative Clustering]]: Bottom-up approach
- [[L11_3_Divisive_Clustering|Divisive Clustering]]: Top-down approach
- [[L11_3_Linkage_Methods|Linkage Methods]]: Single, complete, average, Ward
- [[L11_3_Dendrogram_Interpretation|Dendrogram Interpretation]]: Reading cluster hierarchies
- [[L11_3_Advantages_Disadvantages|Advantages and Disadvantages]]: When to use hierarchical clustering
- [[L11_3_Examples|Hierarchical Examples]]: Implementation and applications
- Required Reading: Chapter 9.2 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L11_3_Quiz]]: Test your understanding of hierarchical clustering

### Lecture 11.4: Density-Based Clustering
- [[L11_4_Density_Based_Clustering|Density-Based Clustering]]: Clustering based on data density
- [[L11_4_DBSCAN_Algorithm|DBSCAN Algorithm]]: Density-Based Spatial Clustering
- [[L11_4_Core_Border_Noise|Core, Border, and Noise Points]]: DBSCAN point classification
- [[L11_4_DBSCAN_Parameters|DBSCAN Parameters]]: Epsilon and MinPts selection
- [[L11_4_DBSCAN_Advantages|DBSCAN Advantages]]: Handling irregular shapes, noise
- [[L11_4_DBSCAN_Limitations|DBSCAN Limitations]]: Parameter sensitivity, high-dimensional data
- [[L11_4_Examples|DBSCAN Examples]]: Implementation and case studies
- Required Reading: "A Density-Based Algorithm for Discovering Clusters" by Ester et al.
- Quiz: [[L11_4_Quiz]]: Test your understanding of density-based clustering

### Lecture 11.5: Model-Based Clustering
- [[L11_5_Model_Based_Clustering|Model-Based Clustering]]: Probabilistic clustering approaches
- [[L11_5_Gaussian_Mixture_Models|Gaussian Mixture Models]]: GMM clustering
- [[L11_5_Expectation_Maximization|Expectation-Maximization]]: EM algorithm for GMM
- [[L11_5_Model_Selection|Model Selection]]: Choosing number of components
- [[L11_5_Soft_Clustering|Soft Clustering]]: Probabilistic cluster assignments
- [[L11_5_Advantages_Model_Based|Advantages of Model-Based Approaches]]: Uncertainty quantification
- [[L11_5_Examples|Model-Based Examples]]: GMM implementation and applications
- Required Reading: Chapter 9.3 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L11_5_Quiz]]: Test your understanding of model-based clustering

### Lecture 11.6: Clustering Evaluation and Validation
- [[L11_6_Clustering_Evaluation|Clustering Evaluation]]: Measuring clustering quality
- [[L11_6_Internal_Indices|Internal Indices]]: Silhouette, Calinski-Harabasz, Davies-Bouldin
- [[L11_6_External_Indices|External Indices]]: Adjusted Rand Index, Normalized Mutual Information
- [[L11_6_Elbow_Method|Elbow Method]]: Choosing optimal number of clusters
- [[L11_6_Gap_Statistic|Gap Statistic]]: Statistical approach to cluster number selection
- [[L11_6_Cross_Validation_Clustering|Cross-Validation for Clustering]]: Stability-based evaluation
- [[L11_6_Examples|Evaluation Examples]]: Implementation and interpretation
- Required Reading: Chapter 9.4 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L11_6_Quiz]]: Test your understanding of clustering evaluation

### Lecture 11.7: Advanced Clustering Techniques
- [[L11_7_Spectral_Clustering|Spectral Clustering]]: Graph-based clustering approach
- [[L11_7_Mean_Shift|Mean Shift Clustering]]: Mode-seeking algorithm
- [[L11_7_Affinity_Propagation|Affinity Propagation]]: Message-passing clustering
- [[L11_7_Clustering_Large_Datasets|Clustering Large Datasets]]: Scalability considerations
- [[L11_7_Online_Clustering|Online Clustering]]: Incremental clustering algorithms
- [[L11_7_Examples|Advanced Examples]]: Implementation and applications
- Required Reading: Chapter 9.5 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L11_7_Quiz]]: Test your understanding of advanced clustering

### Lecture 11.8: Clustering Applications and Case Studies
- [[L11_8_Image_Segmentation|Image Segmentation]]: Clustering in computer vision
- [[L11_8_Customer_Segmentation|Customer Segmentation]]: Marketing applications
- [[L11_8_Document_Clustering|Document Clustering]]: Text mining applications
- [[L11_8_Anomaly_Detection|Anomaly Detection]]: Finding outliers using clustering
- [[L11_8_Real_World_Applications|Real-World Applications]]: Case studies and examples
- [[L11_8_Clustering_Challenges|Practical Challenges]]: Implementation considerations
- Required Reading: Chapter 9.6 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L11_8_Quiz]]: Test your understanding of clustering applications

## Programming Resources
- [[L11_Clustering_Python_Guide|Python Implementation Guide]]: Step-by-step implementation
- [[L11_K_Means_From_Scratch|Building K-Means from Scratch]]: Code tutorial
- [[L11_DBSCAN_Implementation|DBSCAN Algorithm Implementation]]: Density-based clustering
- [[L11_Scikit_Learn_Clustering|Using Scikit-learn for Clustering]]: Library tutorial
- [[L11_Clustering_Visualization|Clustering Visualization]]: Plotting techniques
- [[L11_GMM_Implementation|Gaussian Mixture Models]]: Probabilistic clustering
- [[L11_Clustering_Evaluation_Code|Clustering Evaluation Methods]]: Implementation of metrics

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
- [[L11_1_Quiz]]: Foundations of Unsupervised Learning
- [[L11_2_Quiz]]: K-Means Clustering
- [[L11_3_Quiz]]: Hierarchical Clustering
- [[L11_4_Quiz]]: Density-Based Clustering
- [[L11_5_Quiz]]: Model-Based Clustering
- [[L11_6_Quiz]]: Clustering Evaluation and Validation
- [[L11_7_Quiz]]: Advanced Clustering Techniques
- [[L11_8_Quiz]]: Clustering Applications and Case Studies
