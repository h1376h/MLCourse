# Lecture 8.4: Evaluation Criteria for Subsets Quiz

## Overview
This quiz contains 25 questions covering evaluation criteria for feature subsets, including distance measures, information measures, dependency measures, consistency measures, and stability measures. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Distance measures evaluate feature subsets based on their ability to separate different classes.

#### Task
1. What is the purpose of distance measures in feature selection?
2. How do distance measures relate to class separability?
3. What is the most common distance measure used?
4. If two classes are well-separated, what does this indicate about the features?
5. Compare distance measures vs other evaluation criteria

For a detailed explanation of this question, see [Question 1: Distance Measures](L8_4_1_explanation.md).

## Question 2

### Problem Statement
Euclidean distance is commonly used to measure class separability in feature space.

#### Task
1. What is the formula for Euclidean distance between two points?
2. How do you calculate the distance between class means?
3. What does a large distance between class means indicate?
4. If class A has mean [1, 2] and class B has mean [4, 6], what's the Euclidean distance?
5. Calculate the distance between two 3D points

For a detailed explanation of this question, see [Question 2: Euclidean Distance](L8_4_2_explanation.md).

## Question 3

### Problem Statement
Information measures evaluate how much information features provide about the target variable.

#### Task
1. What is information gain and how is it calculated?
2. How does information gain relate to entropy?
3. What does a high information gain indicate?
4. If entropy decreases from 1.0 to 0.3 after splitting, what's the information gain?
5. Compare information gain vs other information measures

For a detailed explanation of this question, see [Question 3: Information Measures](L8_4_3_explanation.md).

## Question 4

### Problem Statement
Dependency measures select features that are highly correlated with the class but uncorrelated with each other.

#### Task
1. What is the goal of dependency measures?
2. How do you measure feature-class correlation?
3. How do you measure feature-feature correlation?
4. If feature A has 0.8 correlation with class and 0.1 with feature B, is this good?
5. Design a dependency-based selection strategy

For a detailed explanation of this question, see [Question 4: Dependency Measures](L8_4_4_explanation.md).

## Question 5

### Problem Statement
Consistency measures focus on finding the minimum set of features that maintain classification consistency.

#### Task
1. What is the min-features bias in consistency measures?
2. How do you measure classification consistency?
3. What does it mean for a feature subset to be consistent?
4. If two samples have identical feature values but different labels, what does this indicate?
5. Compare consistency vs other evaluation criteria

For a detailed explanation of this question, see [Question 5: Consistency Measures](L8_4_5_explanation.md).

## Question 6

### Problem Statement
Stability measures evaluate how consistent feature selection is across different data samples.

#### Task
1. What is feature selection stability and why is it important?
2. How do you measure stability across different samples?
3. What causes instability in feature selection?
4. If a feature is selected in 8 out of 10 cross-validation folds, what's its stability score?
5. Design a stability measurement approach

For a detailed explanation of this question, see [Question 6: Stability Measures](L8_4_6_explanation.md).

## Question 7

### Problem Statement
Consider a binary classification problem with two features and the following class distributions:

| Feature Subset | Class 0 Mean | Class 1 Mean | Distance |
|----------------|--------------|--------------|----------|
| Feature 1 only | [2.0]        | [5.0]        | 3.0      |
| Feature 2 only | [1.5]        | [4.5]        | 3.0      |
| Both features  | [2.0, 1.5]   | [5.0, 4.5]   | ?        |

#### Task
1. Calculate the Euclidean distance for the "Both features" subset
2. Which feature subset provides the best class separation?
3. What's the improvement in separation when using both features?
4. If you can only use one feature, which would you choose?
5. Calculate the percentage improvement in separation

For a detailed explanation of this question, see [Question 7: Distance Measure Analysis](L8_4_7_explanation.md).

## Question 8

### Problem Statement
Information gain measures the reduction in uncertainty about the target variable.

#### Task
1. What is the formula for information gain?
2. How do you calculate entropy for a dataset?
3. What does a negative information gain indicate?
4. If the original entropy is 1.0 and the weighted average entropy after splitting is 0.6, what's the information gain?
5. Compare information gain for different feature splits

For a detailed explanation of this question, see [Question 8: Information Gain Calculation](L8_4_8_explanation.md).

## Question 9

### Problem Statement
Feature dependency can be measured using correlation coefficients and mutual information.

#### Task
1. How do you measure feature-class dependency?
2. How do you measure feature-feature dependency?
3. What's the ideal dependency pattern for feature selection?
4. If feature A has 0.9 correlation with class and 0.2 with feature B, what's the dependency score?
5. Design a dependency-based evaluation metric

For a detailed explanation of this question, see [Question 9: Dependency Measurement](L8_4_9_explanation.md).

## Question 10

### Problem Statement
Consistency measures focus on finding the minimum feature set that maintains classification accuracy.

#### Task
1. What is the min-features bias and why is it important?
2. How do you measure classification consistency?
3. What happens when you remove a feature from a consistent subset?
4. If a feature subset has 95% consistency, what does this mean?
5. Compare consistency vs accuracy as evaluation criteria

For a detailed explanation of this question, see [Question 10: Consistency Measurement](L8_4_10_explanation.md).

## Question 11

### Problem Statement
Stability measures are crucial for robust feature selection in real-world applications.

#### Task
1. Why is stability important in feature selection?
2. How do you measure stability across different data samples?
3. What factors affect feature selection stability?
4. If a feature appears in 7 out of 10 bootstrap samples, what's its stability?
5. Design a stability evaluation protocol

For a detailed explanation of this question, see [Question 11: Stability Evaluation](L8_4_11_explanation.md).

## Question 12

### Problem Statement
Different evaluation criteria may give different feature subset rankings.

#### Task
1. Why might different criteria rank feature subsets differently?
2. How do you handle conflicting rankings?
3. What's the advantage of using multiple criteria?
4. If subset A ranks 1st by distance but 3rd by information gain, what does this suggest?
5. Design a multi-criteria evaluation approach

For a detailed explanation of this question, see [Question 12: Multi-Criteria Evaluation](L8_4_12_explanation.md).

## Question 13

### Problem Statement
The choice of evaluation criteria affects the type of features selected.

#### Task
1. How do distance measures bias feature selection?
2. How do information measures bias feature selection?
3. How do dependency measures bias feature selection?
4. Which criterion would you use for interpretable features?
5. Compare the biases of different criteria

For a detailed explanation of this question, see [Question 13: Criterion Biases](L8_4_13_explanation.md).

## Question 14

### Problem Statement
Consider a dataset with 3 features and the following information gain values:

| Feature Subset | Information Gain |
|----------------|------------------|
| Feature 1      | 0.8              |
| Feature 2      | 0.6              |
| Feature 3      | 0.4              |
| Features 1+2   | 0.9              |
| Features 1+3   | 0.85             |
| Features 2+3   | 0.7              |
| All features   | 0.95             |

#### Task
1. Which single feature provides the most information?
2. Which pair of features provides the most information?
3. Is there evidence of feature interaction? Explain
4. If you can only use 2 features, which would you choose?
5. Calculate the marginal information gain for each additional feature

For a detailed explanation of this question, see [Question 14: Information Gain Analysis](L8_4_14_explanation.md).

## Question 15

### Problem Statement
Feature selection stability is important for reliable model deployment.

#### Task
1. How do you measure stability across different data samples?
2. What causes instability in feature selection?
3. How do you improve feature selection stability?
4. If a feature is selected in 6 out of 10 cross-validation folds, what's its stability?
5. Design a stability improvement strategy

For a detailed explanation of this question, see [Question 15: Stability Improvement](L8_4_15_explanation.md).

## Question 16

### Problem Statement
The curse of dimensionality affects different evaluation criteria differently.

#### Task
1. How does high dimensionality affect distance measures?
2. How does it affect information measures?
3. How does it affect dependency measures?
4. Which criterion is most robust to high dimensionality?
5. Compare the robustness of different criteria

For a detailed explanation of this question, see [Question 16: Dimensionality Impact](L8_4_16_explanation.md).

## Question 17

### Problem Statement
Feature selection can be viewed as an optimization problem with multiple objectives.

#### Task
1. What are the main objectives in feature selection?
2. How do you balance multiple objectives?
3. What's the trade-off between accuracy and interpretability?
4. If you want to maximize accuracy while minimizing features, how do you formulate this?
5. Design a multi-objective optimization approach

For a detailed explanation of this question, see [Question 17: Multi-Objective Optimization](L8_4_17_explanation.md).

## Question 18

### Problem Statement
Different domains have different requirements for feature selection evaluation.

#### Task
1. What are the key considerations for medical diagnosis features?
2. How do financial applications differ from image recognition?
3. What's important for real-time sensor data?
4. Compare evaluation needs for text vs numerical data
5. Which domain would benefit most from stability measures?

For a detailed explanation of this question, see [Question 18: Domain-Specific Evaluation](L8_4_18_explanation.md).

## Question 19

### Problem Statement
Feature selection affects different types of machine learning algorithms differently.

#### Task
1. How do different evaluation criteria affect linear models?
2. How do they affect tree-based models?
3. How do they affect neural networks?
4. Which algorithm type benefits most from distance-based selection?
5. Compare the impact on different algorithm families

For a detailed explanation of this question, see [Question 19: Algorithm-Specific Effects](L8_4_19_explanation.md).

## Question 20

### Problem Statement
Consider a scenario where you have limited computational resources for evaluation.

#### Task
1. What evaluation strategies would you use with limited time?
2. How do you prioritize different evaluation criteria?
3. What's the trade-off between speed and quality?
4. If you have 1 hour to evaluate 1000 feature subsets, how many can you assess?
5. Design an efficient evaluation strategy

For a detailed explanation of this question, see [Question 20: Resource Constraints](L8_4_20_explanation.md).

## Question 21

### Problem Statement
Feature selection can be applied at different stages of the machine learning pipeline.

#### Task
1. When is the best time to perform feature selection?
2. How does selection timing affect evaluation results?
3. What happens if you select features before preprocessing?
4. How do you handle selection in online learning?
5. Compare different timing strategies

For a detailed explanation of this question, see [Question 21: Selection Timing](L8_4_21_explanation.md).

## Question 22

### Problem Statement
The success of feature selection depends on the quality of the evaluation criteria.

#### Task
1. What makes a good evaluation criterion?
2. How do you avoid overfitting in feature selection?
3. What's the role of cross-validation in evaluation?
4. If selection improves training accuracy but hurts validation accuracy, what does this suggest?
5. Compare different evaluation strategies

For a detailed explanation of this question, see [Question 22: Evaluation Quality](L8_4_22_explanation.md).

## Question 23

### Problem Statement
Feature selection affects model robustness and stability.

#### Task
1. How does feature selection improve model stability?
2. What happens to model performance with noisy features?
3. How does selection affect cross-validation results?
4. If a model is unstable with 100 features, what would happen with 20 features?
5. Compare stability before and after selection

For a detailed explanation of this question, see [Question 23: Model Stability](L8_4_23_explanation.md).

## Question 24

### Problem Statement
Feature selection can reveal domain knowledge and insights.

#### Task
1. How can feature selection help understand data relationships?
2. What insights can you gain from selected features?
3. How does selection help with feature engineering?
4. If certain features are consistently selected, what does this suggest?
5. Compare the insights from different evaluation criteria

For a detailed explanation of this question, see [Question 24: Domain Insights](L8_4_24_explanation.md).

## Question 25

### Problem Statement
Feature selection is part of a broader feature engineering strategy.

#### Task
1. How does selection complement feature creation?
2. What's the relationship between selection and transformation?
3. How do you coordinate different feature engineering steps?
4. If you create 100 new features, how do you select the best ones?
5. Design a comprehensive feature engineering pipeline

For a detailed explanation of this question, see [Question 25: Feature Engineering Integration](L8_4_25_explanation.md).
