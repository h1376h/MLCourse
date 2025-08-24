# Lecture 8: Feature Engineering and Selection

## Overview
This module provides a comprehensive introduction to feature selection, a critical step in the machine learning pipeline for improving model performance, efficiency, and interpretability. We will explore the differences between feature selection and extraction, delve into various selection methodologies such as filters, wrappers, and embedded methods, and examine the search strategies required to navigate the complex space of feature subsets.

### Lecture 8.1: Foundations of Feature Selection
- [[L8_1_Why_Feature_Selection|Why Feature Selection?]]: Improving accuracy, speed, and interpretability
- [[L8_1_Curse_of_Dimensionality|The Curse of Dimensionality]]: Alleviating issues with high-dimensional data
- [[L8_1_Supervised_Selection|Supervised Feature Selection]]: Using labeled data for selection
- [[L8_1_Unsupervised_Selection|Unsupervised Feature Selection]]: Selection without class labels
- ⭐ Quiz: [[L8_1_Quiz]]
- 📚 Examples: Coming Soon

### Lecture 8.2: Univariate Feature Selection Methods
- [[L8_2_Univariate_Concept|Univariate Approach]]: Considering one feature at a time
- [[L8_2_Filter_Scoring|Univariate Filter Scoring]]: Ranking individual features
- [[L8_2_Defining_Irrelevance|Criteria: Defining Feature Irrelevance]]: Using conditional probabilities and KL Divergence
- [[L8_2_Pearson_Correlation|Criteria: Pearson Correlation]]: Measuring linear relationships
- [[L8_2_Mutual_Information|Criteria: Mutual Information]]: Measuring feature-label dependence
- [[L8_2_Chi_Square|Criteria: Chi-Square Test]]: Testing independence for categorical features
- [[L8_2_Select_K_Features|Determining the Number of Features to Select]]: Using cross-validation to find the optimal 'k'
- [[L8_2_Pros_Cons|Advantages and Disadvantages]]: Scalability vs. ignoring feature interactions and redundancy
- ⭐ Quiz: [[L8_2_Quiz]]
- 📚 Examples: Coming Soon

### Lecture 8.3: Multivariate Feature Selection Methods
- [[L8_3_Multivariate_Concept|Multivariate Approach]]: Considering subsets of features together
- [[L8_3_Univariate_Failure|When Univariate Methods Fail]]: The need for multivariate analysis
- [[L8_3_Redundancy|Handling Feature Redundancy]]: Why multivariate is necessary
- [[L8_3_Search_Space|The Search Space Problem]]: Navigating $2^d$ feature subsets
- [[L8_3_Feature_Clustering|Feature Clustering and Grouping]]: Identifying feature clusters
- ⭐ Quiz: [[L8_3_Quiz]]
- 📚 Examples: Coming Soon

### Lecture 8.4: Evaluation Criteria for Subsets
- [[L8_4_Distance_Criteria|Distance Measures]]: Euclidean distance for class separability
- [[L8_4_Information_Criteria|Information Measures]]: Information Gain
- [[L8_4_Dependency_Criteria|Dependency Measures]]: Selecting features highly correlated with the class, yet uncorrelated with each other
- [[L8_4_Consistency_Criteria|Consistency Measures]]: Min-features bias
- [[L8_4_Stability_Measures|Stability Measures]]: Consistency across different data samples
- ⭐ Quiz: [[L8_4_Quiz]]
- 📚 Examples: Coming Soon

### Lecture 8.5: Filter Methods In-Depth
- [[L8_5_Filter_Concept|Filter Method Overview]]: Preprocessing independent of classifiers
- [[L8_5_Univariate_vs_Multivariate_Filters|Univariate vs. Multivariate Filters]]
- [[L8_5_Relief_Algorithm|Relief Algorithm]]: Instance-based feature weighting
- [[L8_5_Pros_Cons|Advantages and Disadvantages]]: Speed and generality vs. monotonic objectives leading to large subsets
- ⭐ Quiz: [[L8_5_Quiz]]
- 📚 Examples: Coming Soon

### Lecture 8.6: Wrapper and Embedded Methods
- [[L8_6_Wrapper_Concept|Wrapper Methods]]: Using classifier performance for evaluation
- [[L8_6_Forward_Selection|Forward Selection]]: Greedy forward search
- [[L8_6_Backward_Elimination|Backward Elimination]]: Greedy backward search
- [[L8_6_Recursive_Feature_Elimination|Recursive Feature Elimination (RFE)]]: Iterative feature removal
- [[L8_6_Wrapper_Pros_Cons|Wrapper Advantages and Disadvantages]]: High accuracy vs. computational cost, lack of generality, and overfitting avoidance
- [[L8_6_Filter_vs_Wrapper|Comparison: Filters vs. Wrappers]]
- [[L8_6_Examples|Wrapper Examples]]: Implementation and case studies
- [[L8_6_Embedded_Concept|Embedded Methods]]: Selection integrated into model training (e.g., L1 Regularization)
- Required Reading: Chapter 3.5 of "Feature Engineering for Machine Learning" by Alice Zheng
- ⭐ Quiz: [[L8_6_Quiz]]
- 📚 Examples: Coming Soon

### Lecture 8.7: Search Strategies
- [[L8_7_General_Procedure|General Search Procedure]]: Subset Generation, Evaluation, Stopping Criterion, and Validation
- [[L8_7_Stopping_Criteria|Stopping Criteria]]: When to stop the search process
- [[L8_7_Search_Strategies_Types|Search Strategies Overview]]: Complete, Heuristic, and Random approaches
- [[L8_7_Exhaustive_Search|Exhaustive (Complete) Search]]: Evaluating all feature subsets
- [[L8_7_Heuristic_Search|Heuristic Search]]: Sequential Forward Selection (SFS), Backward Elimination (SBE), and Floating Selection
- [[L8_7_Greedy_Algorithms|Greedy Algorithms]]: Hill climbing, best-first search
- [[L8_7_Branch_and_Bound|Branch and Bound Search]]: Optimal search with pruning
- [[L8_7_Random_Search|Random (Non-deterministic) Search]]: Genetic Algorithms and Simulated Annealing
- [[L8_7_Search_Comparison|Search Method Comparison]]: Tradeoffs and selection criteria
- [[L8_7_Examples|Search Examples]]: Implementation and applications
- Required Reading: Chapter 3.6 of "Feature Engineering for Machine Learning" by Alice Zheng
- ⭐ Quiz: [[L8_7_Quiz]]
- 📚 Examples: Coming Soon

### Lecture 8.8: Feature Extraction vs. Selection
- [[L8_8_Dimensionality_Reduction|Dimensionality Reduction Overview]]
- [[L8_8_Feature_Selection|Feature Selection Review]]: Selecting a subset of original features
- [[L8_8_Feature_Extraction|Feature Extraction]]: Transforming features into a new space (e.g., PCA, LDA)
- [[L8_8_Principal_Component_Analysis|Principal Component Analysis]]: PCA for dimensionality reduction
- [[L8_8_Linear_Discriminant_Analysis|Linear Discriminant Analysis]]: LDA for supervised reduction
- [[L8_8_Feature_Construction|Feature Construction and Engineering]]: Creating new features from existing ones
- [[L8_8_Extraction_vs_Selection|Extraction vs Selection]]: When to use each approach
- [[L8_8_Examples|Extraction Examples]]: Implementation and applications
- Required Reading: Chapter 4 of "Feature Engineering for Machine Learning" by Alice Zheng
- ⭐ Quiz: [[L8_8_Quiz]]
- 📚 Examples: Coming Soon

## Related Slides
*(not included in the repo)*
- Feature_Selection_Foundations.pdf
- Filter_Methods.pdf
- Wrapper_Methods_and_Search.pdf
- Embedded_Methods_and_Dimensionality_Reduction.pdf

## Related Videos
- [Introduction to Feature Selection](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Filter Methods Explained](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Wrapper Methods Explained](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Search Strategies in ML](https://www.youtube.com/watch?v=YaKMeAlHgqQ)

## All Quizzes
Test your understanding with these quizzes:
- [[L8_1_Quiz]]: Foundations of Feature Engineering
- [[L8_2_Quiz]]: Univariate Feature Selection
- [[L8_3_Quiz]]: Multivariate Feature Selection
- [[L8_4_Quiz]]: Correlation Criteria and Analysis
- [[L8_5_Quiz]]: Filter Methods
- [[L8_6_Quiz]]: Wrapper Methods
- [[L8_7_Quiz]]: Search Strategies and Methods
- [[L8_8_Quiz]]: Feature Extraction and Dimensionality Reduction