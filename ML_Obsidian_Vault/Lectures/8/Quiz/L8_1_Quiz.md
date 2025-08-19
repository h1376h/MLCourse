# Lecture 8.1: Foundations of Feature Selection Quiz

## Overview
This quiz contains 25 questions covering the foundations of feature selection, including motivations, the curse of dimensionality, and different selection approaches. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Feature selection is a critical step in the machine learning pipeline.

#### Task
1. What are the three main benefits of feature selection?
2. How does feature selection improve model interpretability?
3. Why is feature selection important for real-time applications?
4. What happens to training time when you reduce features from 100 to 20?
5. If a model takes 5 minutes to train with 100 features, estimate training time with 25 features

For a detailed explanation of this question, see [Question 1: Feature Selection Benefits](L8_1_1_explanation.md).

## Question 2

### Problem Statement
The curse of dimensionality affects model performance as the number of features increases.

#### Task
1. What is the curse of dimensionality in one sentence?
2. How does the curse affect nearest neighbor algorithms?
3. What happens to the volume of a hypercube as dimensions increase?
4. If you have 1000 samples in 2D, how many samples would you need in 10D for similar density?
5. Calculate the ratio of volume to surface area for a unit hypercube in 2D vs 10D

For a detailed explanation of this question, see [Question 2: Curse of Dimensionality](L8_1_2_explanation.md).

## Question 3

### Problem Statement
Supervised feature selection uses labeled data to identify relevant features.

#### Task
1. What is the main advantage of supervised feature selection?
2. How does supervised selection differ from unsupervised selection?
3. What types of problems benefit most from supervised selection?
4. If you have 1000 samples with 50 features, how many possible feature subsets exist?
5. Calculate the number of feature subsets with exactly 10 features from 50 total features

For a detailed explanation of this question, see [Question 3: Supervised Feature Selection](L8_1_3_explanation.md).

## Question 4

### Problem Statement
Unsupervised feature selection works without class labels.

#### Task
1. When would you use unsupervised feature selection?
2. What are the main challenges of unsupervised selection?
3. How does unsupervised selection identify relevant features?
4. If you have no labels, what criteria can you use for selection?
5. Compare the computational cost of supervised vs unsupervised selection

For a detailed explanation of this question, see [Question 4: Unsupervised Feature Selection](L8_1_4_explanation.md).

## Question 5

### Problem Statement
Feature selection vs feature extraction are different approaches to dimensionality reduction.

#### Task
1. What is the key difference between selection and extraction?
2. Which approach preserves original feature interpretability?
3. When would you choose extraction over selection?
4. If you transform features using PCA, is this selection or extraction?
5. Compare the interpretability of both approaches

For a detailed explanation of this question, see [Question 5: Selection vs Extraction](L8_1_5_explanation.md).

## Question 6

### Problem Statement
Consider a dataset with 1000 samples and 100 features where only 20 features are truly relevant.

#### Task
1. What percentage of features are irrelevant in this dataset?
2. If each irrelevant feature adds 1% noise, what's the total noise level?
3. How would this affect model performance?
4. What's the signal-to-noise ratio with all features vs relevant features only?
5. Calculate the probability of selecting only relevant features by random chance

For a detailed explanation of this question, see [Question 6: Irrelevant Features Impact](L8_1_6_explanation.md).

## Question 7

### Problem Statement
Feature selection affects different aspects of the machine learning pipeline.

#### Task
1. How does feature selection impact data preprocessing time?
2. What happens to model storage requirements after selection?
3. How does selection affect prediction speed?
4. If a model takes 10ms to predict with 100 features, estimate time with 25 features
5. Compare memory usage before and after feature selection

For a detailed explanation of this question, see [Question 7: Pipeline Impact](L8_1_7_explanation.md).

## Question 8

### Problem Statement
The search space for feature selection grows exponentially with the number of features.

#### Task
1. If you have 10 features, how many possible feature subsets exist?
2. How many subsets have exactly 5 features?
3. What's the growth rate of the search space?
4. If evaluating each subset takes 1 second, how long would exhaustive search take for 20 features?
5. Calculate the number of subsets with 3-7 features from 20 total features

For a detailed explanation of this question, see [Question 8: Search Space Growth](L8_1_8_explanation.md).

## Question 9

### Problem Statement
Feature selection can improve model generalization by reducing overfitting.

#### Task
1. How does having too many features lead to overfitting?
2. What is the relationship between features and model complexity?
3. How does feature selection help with the bias-variance trade-off?
4. If a model overfits with 100 features, what would happen with 10 features?
5. Compare the generalization error before and after feature selection

For a detailed explanation of this question, see [Question 9: Overfitting Prevention](L8_1_9_explanation.md).

## Question 10

### Problem Statement
Different domains have different feature selection requirements.

#### Task
1. What are the key considerations for medical diagnosis features?
2. How do financial applications differ from image recognition?
3. What's important for real-time sensor data?
4. Compare feature selection needs for text vs numerical data
5. Which domain would benefit most from interpretable features?

For a detailed explanation of this question, see [Question 10: Domain-Specific Requirements](L8_1_10_explanation.md).

## Question 11

### Problem Statement
Feature selection affects model interpretability and explainability.

#### Task
1. How does feature selection improve model interpretability?
2. What makes a feature "interpretable"?
3. How does selection help with business decisions?
4. If you need to explain predictions to stakeholders, what features would you prefer?
5. Compare interpretability of selected vs extracted features

For a detailed explanation of this question, see [Question 11: Interpretability Benefits](L8_1_11_explanation.md).

## Question 12

### Problem Statement
Computational efficiency is crucial for large-scale feature selection.

#### Task
1. How does the number of features affect training time?
2. What's the relationship between features and memory usage?
3. How does selection impact prediction latency?
4. If training time scales as $O(n^2)$ with features, what's the speedup from 100 to 25 features?
5. Calculate the memory reduction from 1000 to 100 features

For a detailed explanation of this question, see [Question 12: Computational Efficiency](L8_1_12_explanation.md).

## Question 13

### Problem Statement
Feature selection can reveal domain knowledge and insights.

#### Task
1. How can feature selection help understand data relationships?
2. What insights can you gain from selected features?
3. How does selection help with feature engineering?
4. If certain features are consistently selected, what does this suggest?
5. Compare the insights from selection vs extraction

For a detailed explanation of this question, see [Question 13: Domain Insights](L8_1_13_explanation.md).

## Question 14

### Problem Statement
The trade-off between feature count and model performance is crucial.

#### Task
1. What happens when you select too few features?
2. What happens when you select too many features?
3. How do you find the optimal number of features?
4. If accuracy is 85% with 10 features and 87% with 20 features, is the trade-off worth it?
5. Calculate the accuracy improvement per additional feature

For a detailed explanation of this question, see [Question 14: Feature Count Trade-offs](L8_1_14_explanation.md).

## Question 15

### Problem Statement
Feature selection affects different types of machine learning algorithms differently.

#### Task
1. How does feature selection affect linear models?
2. How does it affect tree-based models?
3. How does it affect neural networks?
4. Which algorithm type benefits most from feature selection?
5. Compare the impact on different algorithm families

For a detailed explanation of this question, see [Question 15: Algorithm-Specific Effects](L8_1_15_explanation.md).

## Question 16

### Problem Statement
Consider a dataset with 500 samples and 50 features where features 1-10 are highly relevant, 11-30 are moderately relevant, and 31-50 are irrelevant.

#### Task
1. What percentage of features are highly relevant?
2. If you select only the top 20 features, what's the coverage of relevant information?
3. How would you measure the quality of your selection?
4. What's the optimal number of features for this dataset?
5. Calculate the information retention with different feature counts

For a detailed explanation of this question, see [Question 16: Feature Relevance Analysis](L8_1_16_explanation.md).

## Question 17

### Problem Statement
Feature selection can be viewed as a search and optimization problem.

#### Task
1. What is the objective function for feature selection?
2. What are the constraints in this optimization problem?
3. How do you balance multiple objectives?
4. If you want to maximize accuracy while minimizing features, how do you formulate this?
5. Compare different optimization approaches

For a detailed explanation of this question, see [Question 17: Optimization Formulation](L8_1_17_explanation.md).

## Question 18

### Problem Statement
The relationship between features and target variables determines selection effectiveness.

#### Task
1. How do you measure feature-target relationships?
2. What types of relationships are hard to detect?
3. How do you handle non-linear relationships?
4. If a feature has no linear correlation but high mutual information, what does this suggest?
5. Compare different relationship measures

For a detailed explanation of this question, see [Question 18: Feature-Target Relationships](L8_1_18_explanation.md).

## Question 19

### Problem Statement
Feature selection affects model robustness and stability.

#### Task
1. How does feature selection improve model stability?
2. What happens to model performance with noisy features?
3. How does selection affect cross-validation results?
4. If a model is unstable with 100 features, what would happen with 20 features?
5. Compare stability before and after selection

For a detailed explanation of this question, see [Question 19: Model Stability](L8_1_19_explanation.md).

## Question 20

### Problem Statement
The cost of feature acquisition affects selection decisions.

#### Task
1. How do you balance feature cost vs performance?
2. What's the trade-off between expensive and cheap features?
3. How do you optimize the cost-performance ratio?
4. If feature A costs $10 and improves accuracy by 2%, while feature B costs $100 and improves by 5%, which is better?
5. Calculate the cost-effectiveness of different feature sets

For a detailed explanation of this question, see [Question 20: Feature Cost Analysis](L8_1_20_explanation.md).

## Question 21

### Problem Statement
Feature selection affects the entire machine learning workflow.

#### Task
1. How does selection impact data preprocessing?
2. How does it affect model validation?
3. How does it impact deployment?
4. What changes in the workflow after selection?
5. Compare workflows with and without selection

For a detailed explanation of this question, see [Question 21: Workflow Impact](L8_1_21_explanation.md).

## Question 22

### Problem Statement
Consider a scenario where you have limited computational resources for feature selection.

#### Task
1. What selection strategies would you use with limited time?
2. How do you prioritize feature evaluation?
3. What's the trade-off between speed and quality?
4. If you have 1 hour to evaluate 1000 features, how many can you assess?
5. Design an efficient selection strategy

For a detailed explanation of this question, see [Question 22: Resource Constraints](L8_1_22_explanation.md).

## Question 23

### Problem Statement
Feature selection can be applied at different stages of the machine learning pipeline.

#### Task
1. When is the best time to perform feature selection?
2. How does selection timing affect results?
3. What happens if you select features before preprocessing?
4. How do you handle feature selection in online learning?
5. Compare different timing strategies

For a detailed explanation of this question, see [Question 23: Selection Timing](L8_1_23_explanation.md).

## Question 24

### Problem Statement
The success of feature selection depends on the quality of the evaluation criteria.

#### Task
1. What makes a good evaluation criterion?
2. How do you avoid overfitting in selection?
3. What's the role of cross-validation in selection?
4. If selection improves training accuracy but hurts validation accuracy, what does this suggest?
5. Compare different evaluation strategies

For a detailed explanation of this question, see [Question 24: Evaluation Criteria](L8_1_24_explanation.md).

## Question 25

### Problem Statement
Feature selection is part of a broader feature engineering strategy.

#### Task
1. How does selection complement feature creation?
2. What's the relationship between selection and transformation?
3. How do you coordinate different feature engineering steps?
4. If you create 100 new features, how do you select the best ones?
5. Design a comprehensive feature engineering pipeline

For a detailed explanation of this question, see [Question 25: Feature Engineering Integration](L8_1_25_explanation.md).
