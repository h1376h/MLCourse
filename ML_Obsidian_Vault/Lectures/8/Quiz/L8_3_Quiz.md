# Lecture 8.3: Multivariate Feature Selection Methods Quiz

## Overview
This quiz contains 15 questions covering multivariate feature selection methods, including when univariate methods fail, handling feature redundancy, search space problems, and feature clustering approaches. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Consider a dataset with 4 features: A, B, C, and D. Features A and B are individually weak predictors but when combined, they perfectly predict the target. Features C and D are individually strong predictors but redundant.

#### Task
1. Calculate the number of possible feature subsets (including empty set)
2. If univariate selection picks the top 2 features, which would it select and why?
3. If multivariate selection considers all subsets, which subset would be optimal?
4. What is the main advantage of multivariate methods in this scenario?
5. Calculate the search space size for 4 features vs 10 features

For a detailed explanation of this question, see [Question 1: Multivariate Approach](L8_3_1_explanation.md).

## Question 2

### Problem Statement
In a loan default prediction dataset, features include: income, debt_ratio, credit_score, and age. Income and debt_ratio together determine debt-to-income ratio, which is a strong predictor. Credit_score and age are moderately correlated (r = 0.6).

#### Task
1. Design a scenario where univariate selection would miss the income-debt_ratio interaction
2. If income and debt_ratio have correlation r = 0.3, calculate the redundancy score
3. How many independent feature combinations exist if you group correlated features?
4. Calculate the reduction in search space if you treat correlated features as groups
5. Design a multivariate selection strategy for this dataset

For a detailed explanation of this question, see [Question 2: Feature Interactions](L8_3_2_explanation.md).

## Question 3

### Problem Statement
You have a dataset with 20 features and want to find the optimal subset of 5-10 features.

#### Task
1. Calculate the total number of possible feature subsets
2. Calculate the number of subsets with exactly 7 features
3. If each subset evaluation takes 0.1 seconds, how long would exhaustive search take?
4. If you use forward selection starting with 1 feature, how many evaluations are needed?
5. Design a heuristic search strategy that evaluates at most 1000 subsets

For a detailed explanation of this question, see [Question 3: Search Space Problem](L8_3_3_explanation.md).

## Question 4

### Problem Statement
A dataset has 100 features that can be grouped into 5 clusters based on correlation. Each cluster contains 20 features with average within-cluster correlation of 0.8.

#### Task
1. Calculate the effective number of independent features after clustering
2. If you select one feature from each cluster, how does this reduce the search space?
3. Calculate the reduction in search space size (as a percentage)
4. If features within clusters have correlation > 0.7, how many clusters would you expect?
5. Design a clustering strategy that maximizes feature independence

For a detailed explanation of this question, see [Question 4: Feature Clustering](L8_3_4_explanation.md).

## Question 5

### Problem Statement
You want to maximize model accuracy while minimizing the number of features. Your current model has 50 features with 85% accuracy.

#### Task
1. Formulate this as a multi-objective optimization problem
2. If accuracy increases by 0.5% for each additional feature up to 20, then decreases by 0.1% per feature, find the optimal number
3. Design a penalty function that balances accuracy and feature count
4. If you have a budget of 30 features maximum, how do you modify the objective?
5. Compare greedy vs exhaustive search for this optimization problem

For a detailed explanation of this question, see [Question 5: Optimization Formulation](L8_3_5_explanation.md).

## Question 6

### Problem Statement
You have 25 features and limited computational time. You can evaluate at most 500 feature subsets.

#### Task
1. Compare the number of evaluations needed for exhaustive search vs forward selection
2. If forward selection adds one feature at a time, how many evaluations are needed?
3. Design a random search strategy that samples 500 subsets efficiently
4. Calculate the probability of finding the optimal subset with random sampling
5. Design a hybrid strategy combining forward selection and random sampling

For a detailed explanation of this question, see [Question 6: Search Strategies](L8_3_6_explanation.md).

## Question 7

### Problem Statement
Consider three algorithms: linear regression, decision tree, and neural network. You have 100 features where 20 are truly relevant and 80 are noise.

#### Task
1. How would multivariate selection affect linear regression performance?
2. How would it affect decision tree performance?
3. How would it affect neural network training time?
4. Which algorithm benefits most from multivariate selection and why?
5. Calculate the expected performance improvement for each algorithm

For a detailed explanation of this question, see [Question 7: Algorithm Effects](L8_3_7_explanation.md).

## Question 8

### Problem Statement
You have 1 hour to perform multivariate feature selection on a dataset with 50 features. Each subset evaluation takes 2 seconds.

#### Task
1. Calculate how many feature subsets you can evaluate in 1 hour
2. If you use forward selection, how many features can you select in the time limit?
3. Design a time-efficient selection strategy for this constraint
4. What's the trade-off between evaluation time and selection quality?
5. If you can parallelize evaluations, how does this change your strategy?

For a detailed explanation of this question, see [Question 8: Resource Constraints](L8_3_8_explanation.md).

## Question 9

### Problem Statement
You're building a machine learning pipeline with preprocessing, feature selection, and model training.

#### Task
1. When should you perform multivariate selection relative to preprocessing?
2. If you select features before scaling, what problems might occur?
3. How does selection timing affect cross-validation results?
4. Design a pipeline that integrates selection at the optimal stage
5. Compare the workflow with early vs late feature selection

For a detailed explanation of this question, see [Question 9: Selection Timing](L8_3_9_explanation.md).

## Question 10

### Problem Statement
You're using cross-validation to evaluate feature subsets. Training accuracy improves with more features, but validation accuracy peaks at 15 features.

#### Task
1. What does this pattern suggest about overfitting?
2. How do you modify your evaluation criterion to prevent overfitting?
3. If you use nested cross-validation, how many folds would you recommend?
4. Calculate the optimal number of features based on validation performance
5. Design an evaluation strategy that balances bias and variance

For a detailed explanation of this question, see [Question 10: Evaluation Criteria](L8_3_10_explanation.md).

## Question 11

### Problem Statement
A model with 100 features shows high variance in cross-validation results. You want to improve stability through feature selection.

#### Task
1. How does reducing features from 100 to 20 affect model stability?
2. If cross-validation variance decreases by 30% with feature selection, what does this suggest?
3. Design a stability-based feature selection criterion
4. How do you measure feature subset stability across different data splits?
5. Compare stability metrics before and after feature selection

For a detailed explanation of this question, see [Question 11: Model Stability](L8_3_11_explanation.md).

## Question 12

### Problem Statement
You're working on medical diagnosis with 200 features including lab results, symptoms, and patient demographics.

#### Task
1. What are the key considerations for medical feature selection?
2. If you need interpretability, how does this affect your selection strategy?
3. How do you handle missing data in multivariate selection?
4. Design a selection strategy that prioritizes medical interpretability
5. Compare selection approaches for medical vs financial applications

For a detailed explanation of this question, see [Question 12: Domain Requirements](L8_3_12_explanation.md).

## Question 13

### Problem Statement
You have a target variable and want to measure the relationship with feature subsets of size 3-8.

#### Task
1. How do you measure the relationship between a feature subset and target?
2. If features have non-linear relationships with target, what measures would you use?
3. Calculate the number of possible 5-feature subsets from 20 total features
4. Design a test to detect non-linear feature subset-target relationships
5. Compare correlation, mutual information, and other relationship measures

For a detailed explanation of this question, see [Question 13: Feature-Target Relationships](L8_3_13_explanation.md).

## Question 14

### Problem Statement
Feature selection affects the entire machine learning workflow from data preparation to deployment.

#### Task
1. How does multivariate selection impact data preprocessing steps?
2. How does it affect model validation and testing?
3. What changes are needed in the deployment pipeline?
4. Design a workflow that integrates feature selection seamlessly
5. Compare the workflow complexity with and without multivariate selection

For a detailed explanation of this question, see [Question 14: Workflow Impact](L8_3_14_explanation.md).

## Question 15

### Problem Statement
You've created 50 new features through feature engineering and now need to select the best subset.

#### Task
1. How do you coordinate feature creation and selection?
2. If new features are combinations of original features, how does this affect selection?
3. Design a pipeline that creates and selects features iteratively
4. Calculate the total search space including original and engineered features
5. Design a comprehensive feature engineering and selection strategy

For a detailed explanation of this question, see [Question 15: Feature Engineering Integration](L8_3_15_explanation.md).
