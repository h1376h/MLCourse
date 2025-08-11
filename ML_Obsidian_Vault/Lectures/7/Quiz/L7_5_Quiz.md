# Topic 7.5: Random Forest Deep Dive Quiz

## Overview
This quiz tests your understanding of Random Forest as an ensemble method, including tree diversity, feature subsampling, bagging integration, and voting strategies.

## Question 1

### Problem Statement
Random Forest combines bagging with feature subsampling.

#### Task
1. How does Random Forest create diversity among trees?
2. What is the relationship between Random Forest and bagging?
3. Why is feature subsampling important in Random Forest?
4. How does Random Forest differ from a simple bagging of decision trees?

**Answer**:
1. Random Forest creates diversity through bootstrap sampling and random feature selection at each split
2. Random Forest is an extension of bagging that adds feature subsampling for additional diversity
3. Feature subsampling prevents all trees from using the same features, creating more diverse trees
4. Random Forest uses feature subsampling while simple bagging uses all available features

## Question 2

### Problem Statement
Feature subsampling in Random Forest affects tree diversity.

#### Task
1. If you have 20 features and consider 5 at each split, what is the probability a specific feature is used?
2. How does this probability change if you increase the number of features considered?
3. What is the tradeoff between feature subsampling and tree performance?
4. How do you choose the optimal number of features to consider?

**Answer**:
1. Probability = 5/20 = 0.25 or 25%
2. Higher number of features considered increases the probability but reduces diversity
3. Tradeoff: fewer features = more diversity but potentially worse individual tree performance
4. Optimal choice: typically √(number of features) for classification, number of features/3 for regression

## Question 3

### Problem Statement
Random Forest uses different voting strategies for predictions.

#### Task
1. What is the difference between hard voting and soft voting?
2. When would you prefer soft voting over hard voting?
3. How does Random Forest handle probability estimates?
4. What is the advantage of ensemble voting over single tree predictions?

**Answer**:
1. Hard voting counts class predictions, soft voting averages class probabilities
2. Prefer soft voting when you need probability estimates or confidence scores
3. Random Forest averages probability estimates from all trees for each class
4. Ensemble voting reduces variance, is more robust to individual tree errors, and provides better generalization

## Question 4

### Problem Statement
Out-of-bag estimation provides internal validation for Random Forest.

#### Task
1. How does out-of-bag estimation work?
2. What is the advantage of OOB estimation over cross-validation?
3. When might OOB estimation not be reliable?
4. How does OOB estimation help with model selection?

**Answer**:
1. OOB estimation uses trees where a sample was not in the bootstrap sample to make predictions for that sample
2. OOB is faster (no separate validation) and uses all data for training
3. OOB might not be reliable with very small datasets or when the number of trees is small
4. OOB helps select optimal parameters (number of trees, features per split) without external validation

## Question 5

### Problem Statement
Feature importance in Random Forest measures variable significance.

#### Task
1. How is feature importance calculated in Random Forest?
2. Why is Random Forest feature importance more reliable than single tree importance?
3. What are the limitations of feature importance measures?
4. How can you use feature importance for feature selection?

**Answer**:
1. Feature importance is calculated by averaging impurity reduction across all trees when that feature is used for splitting
2. Random Forest importance is more reliable because it averages across many trees, reducing individual tree biases
3. Limitations: correlation between features can inflate importance, importance doesn't indicate causality
4. Feature importance can be used to select top features or remove low-importance features to reduce dimensionality
