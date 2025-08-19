# Lecture 8: Feature Engineering and Selection

## Overview
This module covers feature engineering and selection techniques, including univariate and multivariate methods, correlation analysis, filter methods, wrapper methods, and search strategies. You'll learn how to identify relevant features, reduce dimensionality, and improve model performance through intelligent feature selection.

### Lecture 8.1: Foundations of Feature Engineering
- [[L8_1_Feature_Engineering_Concept|Feature Engineering Concept]]: Creating and transforming features
- [[L8_1_Feature_Types|Feature Types]]: Numerical, categorical, ordinal, binary
- [[L8_1_Feature_Quality|Feature Quality]]: Relevance, redundancy, noise
- [[L8_1_Dimensionality_Curse|Curse of Dimensionality]]: Problems with high-dimensional data
- [[L8_1_Feature_Engineering_Process|Feature Engineering Process]]: Systematic approach to feature creation
- [[L8_1_Examples|Basic Examples]]: Simple feature engineering demonstrations
- Required Reading: Chapter 1.4 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L8_1_Quiz]]: Test your understanding of feature engineering foundations

### Lecture 8.2: Univariate Feature Selection
- [[L8_2_Univariate_Selection|Univariate Feature Selection]]: Individual feature evaluation
- [[L8_2_Statistical_Tests|Statistical Tests]]: Chi-square, ANOVA, F-test
- [[L8_2_Correlation_Analysis|Correlation Analysis]]: Pearson, Spearman correlation
- [[L8_2_Mutual_Information|Mutual Information]]: Information-theoretic feature selection
- [[L8_2_Univariate_Methods|Univariate Methods]]: SelectKBest, SelectPercentile
- [[L8_2_Advantages_Limitations|Advantages and Limitations]]: When to use univariate selection
- [[L8_2_Examples|Univariate Examples]]: Implementation and applications
- Required Reading: Chapter 3.1 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L8_2_Quiz]]: Test your understanding of univariate feature selection

### Lecture 8.3: Multivariate Feature Selection
- [[L8_3_Multivariate_Selection|Multivariate Feature Selection]]: Feature subset evaluation
- [[L8_3_Feature_Subset_Evaluation|Feature Subset Evaluation]]: Evaluating feature combinations
- [[L8_3_Redundancy_Detection|Redundancy Detection]]: Identifying correlated features
- [[L8_3_Feature_Interaction|Feature Interaction]]: Capturing feature relationships
- [[L8_3_Multivariate_Methods|Multivariate Methods]]: Recursive feature elimination, genetic algorithms
- [[L8_3_Examples|Multivariate Examples]]: Implementation and case studies
- Required Reading: Chapter 3.2 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L8_3_Quiz]]: Test your understanding of multivariate feature selection

### Lecture 8.4: Correlation Criteria and Analysis
- [[L8_4_Correlation_Criteria|Correlation Criteria]]: Using correlation for feature selection
- [[L8_4_Pearson_Correlation|Pearson Correlation]]: Linear correlation coefficient
- [[L8_4_Spearman_Correlation|Spearman Correlation]]: Rank-based correlation
- [[L8_4_Correlation_Thresholds|Correlation Thresholds]]: Setting correlation limits
- [[L8_4_Multicollinearity|Multicollinearity]]: Handling highly correlated features
- [[L8_4_Correlation_Examples|Correlation Examples]]: Implementation and interpretation
- Required Reading: Chapter 3.3 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L8_4_Quiz]]: Test your understanding of correlation criteria

### Lecture 8.5: Filter Methods
- [[L8_5_Filter_Methods|Filter Methods]]: Pre-processing feature selection
- [[L8_5_Filter_Algorithms|Filter Algorithms]]: Variance threshold, correlation filters
- [[L8_5_Information_Gain|Information Gain]]: Entropy-based feature selection
- [[L8_5_Chi_Square_Test|Chi-Square Test]]: Categorical feature selection
- [[L8_5_Filter_Advantages|Filter Method Advantages]]: Speed, independence from learning algorithm
- [[L8_5_Filter_Limitations|Filter Method Limitations]]: Ignoring feature interactions
- [[L8_5_Examples|Filter Examples]]: Implementation and applications
- Required Reading: Chapter 3.4 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L8_5_Quiz]]: Test your understanding of filter methods

### Lecture 8.6: Wrapper Methods
- [[L8_6_Wrapper_Methods|Wrapper Methods]]: Learning algorithm-based selection
- [[L8_6_Forward_Selection|Forward Selection]]: Greedy forward search
- [[L8_6_Backward_Elimination|Backward Elimination]]: Greedy backward search
- [[L8_6_Recursive_Feature_Elimination|Recursive Feature Elimination]]: RFE algorithm
- [[L8_6_Wrapper_Advantages|Wrapper Method Advantages]]: Feature interaction consideration
- [[L8_6_Wrapper_Limitations|Wrapper Method Limitations]]: Computational cost, overfitting
- [[L8_6_Examples|Wrapper Examples]]: Implementation and case studies
- Required Reading: Chapter 3.5 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L8_6_Quiz]]: Test your understanding of wrapper methods

### Lecture 8.7: Search Strategies and Methods
- [[L8_7_Search_Strategies|Search Strategies Overview]]: Different search approaches
- [[L8_7_Exhaustive_Search|Exhaustive Search]]: Complete feature subset evaluation
- [[L8_7_Greedy_Algorithms|Greedy Algorithms]]: Hill climbing, best-first search
- [[L8_7_Genetic_Algorithms|Genetic Algorithms]]: Evolutionary feature selection
- [[L8_7_Simulated_Annealing|Simulated Annealing]]: Stochastic optimization
- [[L8_7_Search_Comparison|Search Method Comparison]]: Tradeoffs and selection criteria
- [[L8_7_Examples|Search Examples]]: Implementation and applications
- Required Reading: Chapter 3.6 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L8_7_Quiz]]: Test your understanding of search strategies

### Lecture 8.8: Feature Extraction and Dimensionality Reduction
- [[L8_8_Feature_Extraction|Feature Extraction]]: Creating new features from existing ones
- [[L8_8_Principal_Component_Analysis|Principal Component Analysis]]: PCA for dimensionality reduction
- [[L8_8_Linear_Discriminant_Analysis|Linear Discriminant Analysis]]: LDA for supervised reduction
- [[L8_8_Feature_Construction|Feature Construction]]: Mathematical combinations and transformations
- [[L8_8_Feature_Extraction_vs_Selection|Extraction vs Selection]]: When to use each approach
- [[L8_8_Examples|Extraction Examples]]: Implementation and applications
- Required Reading: Chapter 4 of "Feature Engineering for Machine Learning" by Alice Zheng
- Quiz: [[L8_8_Quiz]]: Test your understanding of feature extraction

## Programming Resources
- [[L8_Feature_Engineering_Python_Guide|Python Implementation Guide]]: Step-by-step implementation
- [[L8_Feature_Selection_Implementation|Feature Selection Implementation]]: Code tutorial
- [[L8_Correlation_Analysis_Code|Correlation Analysis Implementation]]: Statistical methods
- [[L8_Scikit_Learn_Features|Using Scikit-learn for Feature Selection]]: Library tutorial
- [[L8_Filter_Wrapper_Implementation|Filter and Wrapper Methods]]: Implementation guide
- [[L8_Search_Strategies_Code|Search Strategy Implementation]]: Optimization algorithms
- [[L8_Feature_Extraction_Code|Feature Extraction Methods]]: PCA, LDA implementation

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
- [[L8_1_Quiz]]: Foundations of Feature Engineering
- [[L8_2_Quiz]]: Univariate Feature Selection
- [[L8_3_Quiz]]: Multivariate Feature Selection
- [[L8_4_Quiz]]: Correlation Criteria and Analysis
- [[L8_5_Quiz]]: Filter Methods
- [[L8_6_Quiz]]: Wrapper Methods
- [[L8_7_Quiz]]: Search Strategies and Methods
- [[L8_8_Quiz]]: Feature Extraction and Dimensionality Reduction
