# Lecture 9: Feature Engineering and Selection

## Overview
This module covers feature engineering and selection techniques, including univariate and multivariate methods, correlation analysis, filter methods, wrapper methods, and search strategies. You'll learn how to identify relevant features, reduce dimensionality, and improve model performance through intelligent feature selection.

### Lecture 9.1: Foundations of Feature Engineering
- [[L9_1_Feature_Engineering_Concept|Feature Engineering Concept]]: Creating and transforming features
- [[L9_1_Feature_Types|Feature Types]]: Numerical, categorical, ordinal, binary
- [[L9_1_Feature_Quality|Feature Quality]]: Relevance, redundancy, noise
- [[L9_1_Dimensionality_Curse|Curse of Dimensionality]]: Problems with high-dimensional data
- [[L9_1_Feature_Engineering_Process|Feature Engineering Process]]: Systematic approach to feature creation
- [[L9_1_Examples|Basic Examples]]: Simple feature engineering demonstrations
- Required Reading: Chapter 1.4 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L9_1_Quiz]]: Test your understanding of feature engineering foundations

### Lecture 9.2: Univariate Feature Selection
- [[L9_2_Univariate_Selection|Univariate Feature Selection]]: Individual feature evaluation
- [[L9_2_Statistical_Tests|Statistical Tests]]: Chi-square, ANOVA, F-test
- [[L9_2_Correlation_Analysis|Correlation Analysis]]: Pearson, Spearman correlation
- [[L9_2_Mutual_Information|Mutual Information]]: Information-theoretic feature selection
- [[L9_2_Univariate_Methods|Univariate Methods]]: SelectKBest, SelectPercentile
- [[L9_2_Advantages_Limitations|Advantages and Limitations]]: When to use univariate selection
- [[L9_2_Examples|Univariate Examples]]: Implementation and applications
- Required Reading: Chapter 3.1 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L9_2_Quiz]]: Test your understanding of univariate feature selection

### Lecture 9.3: Multivariate Feature Selection
- [[L9_3_Multivariate_Selection|Multivariate Feature Selection]]: Feature subset evaluation
- [[L9_3_Feature_Subset_Evaluation|Feature Subset Evaluation]]: Evaluating feature combinations
- [[L9_3_Redundancy_Detection|Redundancy Detection]]: Identifying correlated features
- [[L9_3_Feature_Interaction|Feature Interaction]]: Capturing feature relationships
- [[L9_3_Multivariate_Methods|Multivariate Methods]]: Recursive feature elimination, genetic algorithms
- [[L9_3_Examples|Multivariate Examples]]: Implementation and case studies
- Required Reading: Chapter 3.2 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L9_3_Quiz]]: Test your understanding of multivariate feature selection

### Lecture 9.4: Correlation Criteria and Analysis
- [[L9_4_Correlation_Criteria|Correlation Criteria]]: Using correlation for feature selection
- [[L9_4_Pearson_Correlation|Pearson Correlation]]: Linear correlation coefficient
- [[L9_4_Spearman_Correlation|Spearman Correlation]]: Rank-based correlation
- [[L9_4_Correlation_Thresholds|Correlation Thresholds]]: Setting correlation limits
- [[L9_4_Multicollinearity|Multicollinearity]]: Handling highly correlated features
- [[L9_4_Correlation_Examples|Correlation Examples]]: Implementation and interpretation
- Required Reading: Chapter 3.3 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L9_4_Quiz]]: Test your understanding of correlation criteria

### Lecture 9.5: Filter Methods
- [[L9_5_Filter_Methods|Filter Methods]]: Pre-processing feature selection
- [[L9_5_Filter_Algorithms|Filter Algorithms]]: Variance threshold, correlation filters
- [[L9_5_Information_Gain|Information Gain]]: Entropy-based feature selection
- [[L9_5_Chi_Square_Test|Chi-Square Test]]: Categorical feature selection
- [[L9_5_Filter_Advantages|Filter Method Advantages]]: Speed, independence from learning algorithm
- [[L9_5_Filter_Limitations|Filter Method Limitations]]: Ignoring feature interactions
- [[L9_5_Examples|Filter Examples]]: Implementation and applications
- Required Reading: Chapter 3.4 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L9_5_Quiz]]: Test your understanding of filter methods

### Lecture 9.6: Wrapper Methods
- [[L9_6_Wrapper_Methods|Wrapper Methods]]: Learning algorithm-based selection
- [[L9_6_Forward_Selection|Forward Selection]]: Greedy forward search
- [[L9_6_Backward_Elimination|Backward Elimination]]: Greedy backward search
- [[L9_6_Recursive_Feature_Elimination|Recursive Feature Elimination]]: RFE algorithm
- [[L9_6_Wrapper_Advantages|Wrapper Method Advantages]]: Feature interaction consideration
- [[L9_6_Wrapper_Limitations|Wrapper Method Limitations]]: Computational cost, overfitting
- [[L9_6_Examples|Wrapper Examples]]: Implementation and case studies
- Required Reading: Chapter 3.5 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L9_6_Quiz]]: Test your understanding of wrapper methods

### Lecture 9.7: Search Strategies and Methods
- [[L9_7_Search_Strategies|Search Strategies Overview]]: Different search approaches
- [[L9_7_Exhaustive_Search|Exhaustive Search]]: Complete feature subset evaluation
- [[L9_7_Greedy_Algorithms|Greedy Algorithms]]: Hill climbing, best-first search
- [[L9_7_Genetic_Algorithms|Genetic Algorithms]]: Evolutionary feature selection
- [[L9_7_Simulated_Annealing|Simulated Annealing]]: Stochastic optimization
- [[L9_7_Search_Comparison|Search Method Comparison]]: Tradeoffs and selection criteria
- [[L9_7_Examples|Search Examples]]: Implementation and applications
- Required Reading: Chapter 3.6 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L9_7_Quiz]]: Test your understanding of search strategies

### Lecture 9.8: Feature Extraction and Dimensionality Reduction
- [[L9_8_Feature_Extraction|Feature Extraction]]: Creating new features from existing ones
- [[L9_8_Principal_Component_Analysis|Principal Component Analysis]]: PCA for dimensionality reduction
- [[L9_8_Linear_Discriminant_Analysis|Linear Discriminant Analysis]]: LDA for supervised reduction
- [[L9_8_Feature_Construction|Feature Construction]]: Mathematical combinations and transformations
- [[L9_8_Feature_Extraction_vs_Selection|Extraction vs Selection]]: When to use each approach
- [[L9_8_Examples|Extraction Examples]]: Implementation and applications
- Required Reading: Chapter 4 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L9_8_Quiz]]: Test your understanding of feature extraction

## Programming Resources
- [[L9_Feature_Engineering_Python_Guide|Python Implementation Guide]]: Step-by-step implementation
- [[L9_Feature_Selection_Implementation|Feature Selection Implementation]]: Code tutorial
- [[L9_Correlation_Analysis_Code|Correlation Analysis Implementation]]: Statistical methods
- [[L9_Scikit_Learn_Features|Using Scikit-learn for Feature Selection]]: Library tutorial
- [[L9_Filter_Wrapper_Implementation|Filter and Wrapper Methods]]: Implementation guide
- [[L9_Search_Strategies_Code|Search Strategy Implementation]]: Optimization algorithms
- [[L9_Feature_Extraction_Code|Feature Extraction Methods]]: PCA, LDA implementation

## Related Slides
*(not included in the repo)*
- Feature_Engineering_Foundations.pdf
- Univariate_Multivariate_Selection.pdf
- Correlation_Criteria_Analysis.pdf
- Filter_vs_Wrapper_Methods.pdf
- Search_Strategies_Overview.pdf
- Feature_Extraction_Methods.pdf
- Dimensionality_Reduction.pdf
- Feature_Selection_Applications.pdf

## Related Videos
- [Introduction to Feature Engineering](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Feature Selection Methods](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Correlation Analysis for Features](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Filter vs Wrapper Methods](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Search Strategies in Feature Selection](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Feature Extraction Techniques](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Dimensionality Reduction Methods](https://www.youtube.com/watch?v=YaKMeAlHgqQ)

## All Quizzes
Test your understanding with these quizzes:
- [[L9_1_Quiz]]: Foundations of Feature Engineering
- [[L9_2_Quiz]]: Univariate Feature Selection
- [[L9_3_Quiz]]: Multivariate Feature Selection
- [[L9_4_Quiz]]: Correlation Criteria and Analysis
- [[L9_5_Quiz]]: Filter Methods
- [[L9_6_Quiz]]: Wrapper Methods
- [[L9_7_Quiz]]: Search Strategies and Methods
- [[L9_8_Quiz]]: Feature Extraction and Dimensionality Reduction
