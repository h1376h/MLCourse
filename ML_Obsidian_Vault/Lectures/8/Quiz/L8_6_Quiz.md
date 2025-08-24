# Lecture 8.6: Wrapper and Embedded Methods Quiz

## Overview
This quiz contains 30 questions covering wrapper and embedded methods, including wrapper concepts, forward/backward selection, recursive feature elimination, and comparisons between different approaches. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Wrapper methods use the learning algorithm's performance to evaluate feature subsets.

#### Task
1. What is the main characteristic of wrapper methods?
2. How do wrapper methods differ from filter methods?
3. When are wrapper methods most appropriate?
4. What types of evaluation criteria do wrapper methods use?
5. Compare wrapper vs filter methods in terms of accuracy

For a detailed explanation of this question, see [Question 1: Wrapper Methods](L8_6_1_explanation.md).

## Question 2

### Problem Statement
Forward selection starts with an empty set and adds features one by one.

#### Task
1. How does forward selection work?
2. What are the advantages of forward selection?
3. What are the disadvantages of forward selection?
4. When would you choose forward selection?
5. Compare forward selection with other wrapper approaches

For a detailed explanation of this question, see [Question 2: Forward Selection](L8_6_2_explanation.md).

## Question 3

### Problem Statement
Backward elimination starts with all features and removes them one by one.

#### Task
1. How does backward elimination work?
2. What are the advantages of backward elimination?
3. What are the disadvantages of backward elimination?
4. When would you choose backward elimination?
5. Compare backward elimination with forward selection

For a detailed explanation of this question, see [Question 3: Backward Elimination](L8_6_3_explanation.md).

## Question 4

### Problem Statement
Recursive Feature Elimination (RFE) iteratively removes the least important features.

#### Task
1. How does RFE work?
2. What are the advantages of RFE?
3. What are the disadvantages of RFE?
4. When would you choose RFE?
5. Compare RFE with other wrapper approaches

For a detailed explanation of this question, see [Question 4: Recursive Feature Elimination](L8_6_4_explanation.md).

## Question 5

### Problem Statement
Wrapper methods have both advantages and disadvantages compared to other approaches.

#### Task
1. What are the main advantages of wrapper methods?
2. What are the main disadvantages?
3. How do wrapper methods handle feature interactions?
4. When would you choose wrappers over filters?
5. Compare the generality of different approaches

For a detailed explanation of this question, see [Question 5: Wrapper Advantages and Disadvantages](L8_6_5_explanation.md).

## Question 6

### Problem Statement
Filter and wrapper methods have different characteristics and trade-offs.

#### Task
1. What are the main differences between filters and wrappers?
2. How do they differ in computational cost?
3. How do they differ in accuracy?
4. When would you use filters vs wrappers?
5. Compare the robustness of both approaches

For a detailed explanation of this question, see [Question 6: Comparison: Filters vs Wrappers](L8_6_6_explanation.md).

## Question 7

### Problem Statement
Embedded methods integrate feature selection into the model training process.

#### Task
1. What is the main characteristic of embedded methods?
2. How do embedded methods differ from filters and wrappers?
3. What are examples of embedded methods?
4. When are embedded methods most appropriate?
5. Compare embedded methods with other approaches

For a detailed explanation of this question, see [Question 7: Embedded Methods](L8_6_7_explanation.md).

## Question 8

### Problem Statement
L1 Regularization (Lasso) is a common embedded method for feature selection.

#### Task
1. How does L1 regularization work for feature selection?
2. What happens to feature coefficients during L1 regularization?
3. How do you control the sparsity in L1 regularization?
4. What are the advantages of L1 regularization?
5. Compare L1 vs L2 regularization for feature selection

For a detailed explanation of this question, see [Question 8: L1 Regularization](L8_6_8_explanation.md).

## Question 9

### Problem Statement
Consider a dataset with 5 features and the following cross-validation accuracy scores:

| Feature Subset | CV Accuracy |
|----------------|-------------|
| Feature 1      | 0.75        |
| Feature 2      | 0.68        |
| Feature 3      | 0.72        |
| Features 1+2   | 0.82        |
| Features 1+3   | 0.79        |
| Features 2+3   | 0.76        |
| All features   | 0.85        |

#### Task
1. If you use forward selection, what would be the first feature added?
2. What would be the second feature added?
3. What's the final feature subset using forward selection?
4. If you use backward elimination, what would be the first feature removed?
5. Calculate the improvement in accuracy for each step

For a detailed explanation of this question, see [Question 9: Wrapper Method Analysis](L8_6_9_explanation.md).

## Question 10

### Problem Statement
Different wrapper methods may give different feature subset rankings.

#### Task
1. Why might forward selection and backward elimination give different results?
2. How do you handle conflicting rankings?
3. What's the advantage of using multiple wrapper approaches?
4. If forward selection selects features [1,2,3] and backward elimination selects [1,3,4], what does this suggest?
5. Design a strategy to resolve conflicts between wrapper methods

For a detailed explanation of this question, see [Question 10: Wrapper Method Conflicts](L8_6_10_explanation.md).

## Question 11

### Problem Statement
The choice of learning algorithm affects wrapper method performance.

#### Task
1. How does the learning algorithm affect feature selection?
2. Which algorithms work best with wrapper methods?
3. How do you choose the right algorithm for wrapper selection?
4. What happens if you use a different algorithm for selection vs final training?
5. Compare different algorithms for wrapper-based selection

For a detailed explanation of this question, see [Question 11: Algorithm Selection](L8_6_11_explanation.md).

## Question 12

### Problem Statement
Cross-validation is crucial for reliable wrapper method evaluation.

#### Task
1. Why is cross-validation important for wrapper methods?
2. How many folds would you recommend for wrapper selection?
3. What happens if you don't use cross-validation?
4. How do you handle overfitting in wrapper methods?
5. Design a cross-validation strategy for wrapper selection

For a detailed explanation of this question, see [Question 12: Cross-Validation in Wrappers](L8_6_12_explanation.md).

## Question 13

### Problem Statement
Wrapper methods can be computationally expensive for large datasets.

#### Task
1. How does the number of features affect wrapper method cost?
2. What strategies can you use to reduce computational cost?
3. When would you avoid wrapper methods due to cost?
4. If evaluating each feature subset takes 1 minute, how long would exhaustive search take for 20 features?
5. Design a cost-effective wrapper strategy

For a detailed explanation of this question, see [Question 13: Computational Cost](L8_6_13_explanation.md).

## Question 14

### Problem Statement
Feature selection can be viewed as an optimization problem with multiple objectives.

#### Task
1. What are the main objectives in wrapper-based selection?
2. How do you balance accuracy vs feature count?
3. What's the trade-off between performance and interpretability?
4. If you want to maximize accuracy while minimizing features, how do you formulate this?
5. Design a multi-objective wrapper approach

For a detailed explanation of this question, see [Question 14: Multi-Objective Optimization](L8_6_14_explanation.md).

## Question 15

### Problem Statement
Different search strategies can be used with wrapper methods.

#### Task
1. What are the main types of search strategies for wrappers?
2. What's the trade-off between exhaustive and heuristic search?
3. When would you use random search with wrappers?
4. If you have limited time, what wrapper search strategy would you choose?
5. Compare different search approaches for wrappers

For a detailed explanation of this question, see [Question 15: Search Strategies](L8_6_15_explanation.md).

## Question 16

### Problem Statement
Wrapper methods affect different types of machine learning algorithms differently.

#### Task
1. How do wrapper methods affect linear models?
2. How do they affect tree-based models?
3. How do they affect neural networks?
4. Which algorithm type benefits most from wrapper methods?
5. Compare the impact on different algorithm families

For a detailed explanation of this question, see [Question 16: Algorithm-Specific Effects](L8_6_16_explanation.md).

## Question 17

### Problem Statement
Consider a scenario where you have limited computational resources for wrapper-based selection.

#### Task
1. What wrapper strategies would you use with limited time?
2. How do you prioritize feature subset evaluation?
3. What's the trade-off between speed and quality?
4. If you have 1 hour to evaluate feature subsets, how many can you assess?
5. Design an efficient wrapper-based selection strategy

For a detailed explanation of this question, see [Question 17: Resource Constraints](L8_6_17_explanation.md).

## Question 18

### Problem Statement
Wrapper methods can be applied at different stages of the machine learning pipeline.

#### Task
1. When is the best time to perform wrapper-based selection?
2. How does selection timing affect results?
3. What happens if you select features before preprocessing?
4. How do you handle wrapper selection in online learning?
5. Compare different timing strategies for wrappers

For a detailed explanation of this question, see [Question 18: Selection Timing](L8_6_18_explanation.md).

## Question 19

### Problem Statement
The success of wrapper methods depends on the quality of the evaluation criteria.

#### Task
1. What makes a good evaluation criterion for wrapper methods?
2. How do you avoid overfitting in wrapper selection?
3. What's the role of cross-validation in wrapper methods?
4. If wrapper selection improves training accuracy but hurts validation accuracy, what does this suggest?
5. Compare different evaluation strategies for wrappers

For a detailed explanation of this question, see [Question 19: Evaluation Criteria](L8_6_19_explanation.md).

## Question 20

### Problem Statement
Wrapper methods affect model robustness and stability.

#### Task
1. How do wrapper methods improve model stability?
2. What happens to model performance with noisy features?
3. How do wrappers affect cross-validation results?
4. If a model is unstable with 100 features, what would happen with 20 features?
5. Compare stability before and after wrapper selection

For a detailed explanation of this question, see [Question 20: Model Stability](L8_6_20_explanation.md).

## Question 21

### Problem Statement
Different domains have different requirements for wrapper-based selection.

#### Task
1. What are the key considerations for medical diagnosis features?
2. How do financial applications differ from image recognition?
3. What's important for real-time sensor data?
4. Compare wrapper needs for text vs numerical data
5. Which domain would benefit most from wrapper methods?

For a detailed explanation of this question, see [Question 21: Domain-Specific Requirements](L8_6_21_explanation.md).

## Question 22

### Problem Statement
Wrapper methods can reveal domain knowledge and insights.

#### Task
1. How can wrapper methods help understand data relationships?
2. What insights can you gain from selected feature subsets?
3. How do wrappers help with feature engineering?
4. If certain feature combinations are consistently selected, what does this suggest?
5. Compare the insights from wrappers vs other methods

For a detailed explanation of this question, see [Question 22: Domain Insights](L8_6_22_explanation.md).

## Question 23

### Problem Statement
The relationship between features and target variables determines wrapper effectiveness.

#### Task
1. How do you measure feature subset-target relationships for wrappers?
2. What types of relationships are hard to detect with wrappers?
3. How do you handle non-linear relationships?
4. If a feature subset has no linear correlation but high mutual information, what does this suggest?
5. Compare different relationship measures for wrappers

For a detailed explanation of this question, see [Question 23: Feature-Target Relationships](L8_6_23_explanation.md).

## Question 24

### Problem Statement
Wrapper methods affect the entire machine learning workflow.

#### Task
1. How do wrapper methods impact data preprocessing?
2. How do they affect model validation?
3. How do they impact deployment?
4. What changes in the workflow after wrapper selection?
5. Compare workflows with and without wrapper selection

For a detailed explanation of this question, see [Question 24: Workflow Impact](L8_6_24_explanation.md).

## Question 25

### Problem Statement
Consider a dataset with 1000 samples and 100 features where only 20 features are truly relevant.

#### Task
1. What percentage of features are irrelevant in this dataset?
2. If each irrelevant feature adds 1% noise, what's the total noise level?
3. How would this affect wrapper method performance?
4. What's the signal-to-noise ratio with all features vs relevant features only?
5. Calculate the probability of selecting only relevant features by random chance

For a detailed explanation of this question, see [Question 25: Irrelevant Features Impact](L8_6_25_explanation.md).

## Question 26

### Problem Statement
Wrapper methods can be viewed as a search and optimization problem.

#### Task
1. What is the objective function for wrapper-based selection?
2. What are the constraints in this optimization problem?
3. How do you balance multiple objectives?
4. If you want to maximize accuracy while minimizing features, how do you formulate this?
5. Compare different optimization approaches for wrappers

For a detailed explanation of this question, see [Question 26: Optimization Formulation](L8_6_26_explanation.md).

## Question 27

### Problem Statement
The curse of dimensionality affects wrapper methods differently than other approaches.

#### Task
1. How does high dimensionality affect wrapper methods?
2. What happens to feature subset relevance as dimensions increase?
3. How do you handle the sparsity problem in wrappers?
4. If you have 10,000 features, what wrapper strategy would you use?
5. Compare the robustness of different wrapper methods

For a detailed explanation of this question, see [Question 27: High Dimensionality](L8_6_27_explanation.md).

## Question 28

### Problem Statement
Feature selection affects different types of machine learning algorithms.

#### Task
1. How do wrapper methods affect linear models?
2. How do they affect tree-based models?
3. How do they affect neural networks?
4. Which algorithm type benefits most from wrapper methods?
5. Compare the impact on different algorithm families

For a detailed explanation of this question, see [Question 28: Algorithm-Specific Effects](L8_6_28_explanation.md).

## Question 29

### Problem Statement
Feature selection can be applied at different stages of the pipeline.

#### Task
1. When is the best time to perform wrapper-based selection?
2. How does selection timing affect results?
3. What happens if you select features before preprocessing?
4. How do you handle selection in online learning?
5. Compare different timing strategies

For a detailed explanation of this question, see [Question 29: Selection Timing](L8_6_29_explanation.md).

## Question 30

### Problem Statement
Feature selection is part of a broader feature engineering strategy.

#### Task
1. How do wrapper methods complement feature creation?
2. What's the relationship between selection and transformation?
3. How do you coordinate different feature engineering steps?
4. If you create 100 new features, how do you select the best subsets?
5. Design a comprehensive feature engineering pipeline

For a detailed explanation of this question, see [Question 30: Feature Engineering Integration](L8_6_30_explanation.md).
