# Lecture 8.1: Foundations of Feature Selection Quiz

## Overview
This quiz contains 20 questions covering the foundations of feature selection, including motivations, the curse of dimensionality, different selection approaches, and practical applications. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Feature selection is a critical step in the machine learning pipeline that affects multiple aspects of model development.

#### Task
1. What are the three main benefits of feature selection?
2. How does feature selection improve model interpretability?
3. Why is feature selection important for real-time applications?
4. If a model takes 5 minutes to train with 100 features, estimate training time with 25 features (assume linear scaling)
5. Calculate the memory reduction when reducing features from 1000 to 100 (assume each feature uses 8 bytes per sample)

For a detailed explanation of this question, see [Question 1: Feature Selection Fundamentals](L8_1_1_explanation.md).

## Question 2

### Problem Statement
The curse of dimensionality affects model performance as the number of features increases, particularly for distance-based algorithms.

#### Task
1. What is the curse of dimensionality in one sentence?
2. How does the curse affect nearest neighbor algorithms?
3. What happens to the volume of a hypercube as dimensions increase?
4. If you have 1000 samples in 2D, how many samples would you need in 10D for similar density?
5. Calculate the ratio of volume to surface area for a unit hypercube in 2D vs 10D

For a detailed explanation of this question, see [Question 2: Curse of Dimensionality](L8_1_2_explanation.md).

## Question 3

### Problem Statement
Feature selection approaches can be categorized as supervised (using labels) or unsupervised (without labels), each with different advantages and applications.

#### Task
1. What is the main advantage of supervised feature selection?
2. When would you use unsupervised feature selection?
3. How do you measure feature relevance in unsupervised scenarios?
4. If you have 1000 samples with 50 features, how many possible feature subsets exist?
5. Calculate the number of feature subsets with exactly 10 features from 50 total features

For a detailed explanation of this question, see [Question 3: Supervised vs Unsupervised Selection](L8_1_3_explanation.md).

## Question 4

### Problem Statement
Feature selection and feature extraction are different approaches to dimensionality reduction with distinct trade-offs.

#### Task
1. What is the key difference between selection and extraction?
2. Which approach preserves original feature interpretability?
3. When would you choose extraction over selection?
4. If you transform features using PCA, is this selection or extraction?
5. Compare the interpretability and computational cost of both approaches

For a detailed explanation of this question, see [Question 4: Selection vs Extraction](L8_1_4_explanation.md).

## Question 5

### Problem Statement
Consider a dataset with 1000 samples and 100 features where only 20 features are truly relevant to the target variable.

#### Task
1. What percentage of features are irrelevant in this dataset?
2. If each irrelevant feature adds 1% noise, what's the total noise level?
3. How would this affect model performance and training time?
4. What's the signal-to-noise ratio with all features vs relevant features only?
5. Calculate the probability of selecting only relevant features by random chance if you pick 20 features

For a detailed explanation of this question, see [Question 5: Irrelevant Features Impact](L8_1_5_explanation.md).

## Question 6

### Problem Statement
The search space for feature selection grows exponentially with the number of features, making exhaustive search impractical for large feature sets.

#### Task
1. If you have 10 features, how many possible feature subsets exist?
2. How many subsets have exactly 5 features?
3. What's the growth rate of the search space (express as a function of n)?
4. If evaluating each subset takes 1 second, how long would exhaustive search take for 20 features?
5. Calculate the number of subsets with 3-7 features from 20 total features

For a detailed explanation of this question, see [Question 6: Search Space Complexity](L8_1_6_explanation.md).

## Question 7

### Problem Statement
Feature selection can improve model generalization by reducing overfitting and managing the bias-variance trade-off.

#### Task
1. How does having too many features lead to overfitting?
2. What is the relationship between features and model complexity?
3. How does feature selection help with the bias-variance trade-off?
4. If a model overfits with 100 features, what would happen with 10 features?
5. Compare the generalization error before and after feature selection using a concrete example

For a detailed explanation of this question, see [Question 7: Overfitting and Generalization](L8_1_7_explanation.md).

## Question 8

### Problem Statement
Different domains have different feature selection requirements based on their specific constraints and goals.

#### Task
1. What are the key considerations for medical diagnosis features?
2. How do financial applications differ from image recognition in feature selection?
3. What's important for real-time sensor data feature selection?
4. Compare feature selection needs for text vs numerical data
5. Which domain would benefit most from interpretable features and why?

For a detailed explanation of this question, see [Question 8: Domain-Specific Considerations](L8_1_8_explanation.md).

## Question 9

### Problem Statement
Feature selection affects different types of machine learning algorithms differently, requiring tailored approaches.

#### Task
1. How does feature selection affect linear models (e.g., linear regression)?
2. How does it affect tree-based models (e.g., decision trees)?
3. How does it affect neural networks?
4. Which algorithm type benefits most from feature selection and why?
5. Compare the impact on different algorithm families using specific examples

For a detailed explanation of this question, see [Question 9: Algorithm-Specific Effects](L8_1_9_explanation.md).

## Question 10

### Problem Statement
Consider a dataset with 500 samples and 50 features where features 1-10 are highly relevant, 11-30 are moderately relevant, and 31-50 are irrelevant.

#### Task
1. What percentage of features are highly relevant?
2. If you select only the top 20 features, what's the coverage of relevant information?
3. How would you measure the quality of your selection?
4. What's the optimal number of features for this dataset?
5. Calculate the information retention with different feature counts (10, 20, 30, 40, 50)

For a detailed explanation of this question, see [Question 10: Feature Relevance Analysis](L8_1_10_explanation.md).

## Question 11

### Problem Statement
Feature selection can be viewed as a search and optimization problem with multiple objectives and constraints.

#### Task
1. What is the objective function for feature selection?
2. What are the constraints in this optimization problem?
3. How do you balance multiple objectives (e.g., accuracy vs feature count)?
4. If you want to maximize accuracy while minimizing features, how do you formulate this mathematically?
5. Compare different optimization approaches (greedy, genetic, exhaustive)

For a detailed explanation of this question, see [Question 11: Optimization Formulation](L8_1_11_explanation.md).

## Question 12

### Problem Statement
The relationship between features and target variables determines selection effectiveness and requires appropriate measurement techniques.

#### Task
1. How do you measure feature-target relationships for numerical data?
2. What types of relationships are hard to detect with simple correlation?
3. How do you handle non-linear relationships in feature selection?
4. If a feature has no linear correlation but high mutual information, what does this suggest?
5. Compare different relationship measures (correlation, mutual information, chi-square) with examples

For a detailed explanation of this question, see [Question 12: Feature-Target Relationships](L8_1_12_explanation.md).

## Question 13

### Problem Statement
Feature selection affects model robustness and stability, particularly when dealing with noisy or changing data.

#### Task
1. How does feature selection improve model stability?
2. What happens to model performance with noisy features?
3. How does selection affect cross-validation results?
4. If a model is unstable with 100 features, what would happen with 20 features?
5. Compare stability before and after selection using a concrete scenario

For a detailed explanation of this question, see [Question 13: Model Stability and Robustness](L8_1_13_explanation.md).

## Question 14

### Problem Statement
The cost of feature acquisition affects selection decisions, requiring careful analysis of cost-benefit trade-offs.

#### Task
1. How do you balance feature cost vs performance improvement?
2. What's the trade-off between expensive and cheap features?
3. How do you optimize the cost-performance ratio?
4. If feature A costs $10 and improves accuracy by 2%, while feature B costs $100 and improves by 5%, which is better?
5. Calculate the cost-effectiveness (improvement per dollar) of different feature sets

For a detailed explanation of this question, see [Question 14: Feature Cost Analysis](L8_1_14_explanation.md).

## Question 15

### Problem Statement
Feature redundancy occurs when multiple features provide similar information, leading to multicollinearity and reduced model performance.

#### Task
1. What is feature redundancy and why is it problematic?
2. How do you detect multicollinearity between features?
3. What's the difference between redundancy and irrelevance?
4. If two features have correlation 0.95, what should you do and why?
5. How does redundancy affect model interpretability and performance?

For a detailed explanation of this question, see [Question 15: Feature Redundancy and Multicollinearity](L8_1_15_explanation.md).

## Question 16

### Problem Statement
Statistical significance testing helps determine if feature selection results are reliable and not due to chance.

#### Task
1. What is statistical significance in feature selection?
2. How do you test if a selected feature is truly important?
3. What's the role of p-values in feature selection?
4. If a feature improves accuracy by 0.5%, how do you know it's significant?
5. Compare different significance testing approaches (t-test, permutation test, bootstrap)

For a detailed explanation of this question, see [Question 16: Statistical Significance in Selection](L8_1_16_explanation.md).

## Question 17

### Problem Statement
Different data types require different feature selection strategies due to their unique characteristics and challenges.

#### Task
1. How does feature selection differ for text data vs numerical data?
2. What special considerations exist for image data feature selection?
3. How do you handle time series features in selection?
4. If you have mixed data types, what selection approach would you use?
5. Compare selection strategies across different data modalities with examples

For a detailed explanation of this question, see [Question 17: Data Type-Specific Strategies](L8_1_17_explanation.md).

## Question 18

### Problem Statement
Ensemble feature selection methods combine multiple selection approaches for more robust and reliable results.

#### Task
1. What is ensemble feature selection and how does it work?
2. What are the advantages of combining multiple selection methods?
3. How do you aggregate results from different selection approaches?
4. If three methods select different feature sets, how do you decide which features to keep?
5. Compare ensemble vs single method selection with pros and cons

For a detailed explanation of this question, see [Question 18: Ensemble Feature Selection](L8_1_18_explanation.md).

## Question 19

### Problem Statement
You're playing a game where you must select features to maximize model performance under specific constraints.

**Rules:**
- You have 100 total features
- Only 15 are truly useful
- Each useful feature gives +10 points
- Each useless feature gives -2 points
- You must select exactly 20 features

#### Task
1. What's your best possible score?
2. What's your worst possible score?
3. If you randomly select 20 features, what's your expected score?
4. What strategy would you use to maximize your score?
5. Calculate the probability of getting a positive score with random selection

**Hint:** For question 3, use the probability of selecting useful features: P(useful) = 15/100 = 0.15

For a detailed explanation of this question, see [Question 19: Feature Selection Strategy Game](L8_1_19_explanation.md).

## Question 20

### Problem Statement
You need to decide whether to use feature selection for your project and determine the optimal approach.

**Considerations:**
- Dataset: 500 samples, 30 features
- Goal: Interpretable model for business users
- Time constraint: 2 weeks total
- Performance requirement: 80% accuracy minimum
- Available methods: Correlation-based, mutual information, recursive feature elimination

#### Task
1. Should you use feature selection? Yes/No and why?
2. What type of selection would you choose and why?
3. How many features would you aim to keep and how do you justify this number?
4. What's your biggest risk in this decision?
5. Design a step-by-step feature selection strategy for this project

For a detailed explanation of this question, see [Question 20: Practical Feature Selection Decision](L8_1_20_explanation.md).
