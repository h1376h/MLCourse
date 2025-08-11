# Lecture 6.6: Overfitting and Underfitting in Trees Quiz

## Overview
This quiz contains 18 questions covering different topics from section 6.5 of the lectures on Overfitting and Underfitting in Trees, including overfitting causes, underfitting detection, model complexity, generalization issues, learning curves, regularization techniques, advanced detection methods, and prevention strategies.

## Question 1

### Problem Statement
Consider a decision tree trained on a dataset with the following performance metrics:

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| Accuracy | 98%          | 75%      |
| Depth   | 15           | -        |
| Nodes   | 127          | -        |

#### Task
1. [🔍] What phenomenon is occurring in this tree?
2. [🔍] What is the generalization gap?
3. [🔍] Why does the training accuracy differ so much from test accuracy?
4. [🔍] What are the signs of overfitting in this tree?

For a detailed explanation of this question, see [Question 1: Overfitting Detection](L6_5_1_explanation.md).

## Question 2

### Problem Statement
Overfitting occurs when a tree becomes too complex for the data.

#### Task
1. [📚] What are the main causes of overfitting in decision trees?
2. [📚] How does tree depth relate to overfitting?
3. [📚] What happens to the bias-variance tradeoff as trees grow deeper?
4. [📚] Why do deep trees often perform poorly on unseen data?

For a detailed explanation of this question, see [Question 2: Causes of Overfitting](L6_5_2_explanation.md).

## Question 3

### Problem Statement
Underfitting occurs when a tree is too simple to capture the data patterns.

#### Task
1. [🔍] What are the signs of underfitting in a decision tree?
2. [🔍] How does underfitting affect training and test performance?
3. [🔍] What is the relationship between model complexity and underfitting?
4. [🔍] When might a very shallow tree lead to underfitting?

For a detailed explanation of this question, see [Question 3: Underfitting in Trees](L6_5_3_explanation.md).

## Question 4

### Problem Statement
The optimal tree complexity balances overfitting and underfitting.

#### Task
1. [📚] How do you find the optimal tree depth?
2. [📚] What is the relationship between training set size and optimal complexity?
3. [📚] How does cross-validation help determine optimal complexity?
4. [📚] What is the "sweet spot" in the bias-variance tradeoff?

For a detailed explanation of this question, see [Question 4: Optimal Tree Complexity](L6_5_4_explanation.md).

## Question 5

### Problem Statement
Consider different scenarios for tree complexity:

| Scenario | Training Accuracy | Test Accuracy | Tree Depth |
|----------|-------------------|---------------|------------|
| A        | 85%               | 83%           | 3          |
| B        | 95%               | 78%           | 8          |
| C        | 70%               | 68%           | 2          |

#### Task
1. [📚] Which scenario shows overfitting?
2. [📚] Which scenario shows underfitting?
3. [📚] Which scenario has the best generalization?
4. [📚] How would you adjust the tree complexity for each scenario?

For a detailed explanation of this question, see [Question 5: Complexity Analysis](L6_5_5_explanation.md).

## Question 6

### Problem Statement
Learning curves help visualize the relationship between training set size and model performance.

#### Task
1. [🔍] What does a learning curve plot show on the x-axis and y-axis?
2. [🔍] What does it mean if training and validation curves are close together but both have low accuracy?
3. [🔍] What does it mean if training accuracy is high but validation accuracy is low?
4. [🔍] How can learning curves help you decide whether to collect more data?

For a detailed explanation of this question, see [Question 6: Learning Curves](L6_5_6_explanation.md).

## Question 7

### Problem Statement
Regularization techniques help control tree complexity and prevent overfitting.

#### Task
1. [📚] What is the purpose of setting a maximum tree depth?
2. [📚] How does setting a minimum number of samples per leaf help with regularization?
3. [📚] What is the effect of setting a minimum number of samples for splitting?
4. [📚] How do these parameters relate to the bias-variance tradeoff?

For a detailed explanation of this question, see [Question 7: Regularization Techniques](L6_5_7_explanation.md).

## Question 8

### Problem Statement
Different regularization parameters have different effects on tree complexity.

#### Task
1. [📚] **Parameter 1**: If you increase max_depth from 3 to 10, what happens to bias and variance?
2. [📚] **Parameter 2**: If you increase min_samples_leaf from 1 to 10, what happens to tree size?
3. [📚] **Parameter 3**: If you increase min_samples_split from 2 to 20, what happens to overfitting?
4. [📚] How do you choose the optimal values for these regularization parameters?

For a detailed explanation of this question, see [Question 8: Regularization Parameter Effects](L6_5_8_explanation.md).

## Question 9

### Problem Statement
Advanced overfitting detection methods can identify subtle overfitting patterns.

#### Task
1. [🔍] What is the "validation curve" and how does it help detect overfitting?
2. [🔍] How can you use the "gap" between training and validation performance?
3. [🔍] What is the "stability" test for detecting overfitting?
4. [🔍] How do you distinguish between overfitting and data leakage?

For a detailed explanation of this question, see [Question 9: Advanced Overfitting Detection](L6_5_9_explanation.md).

## Question 10

### Problem Statement
Underfitting can be more subtle than overfitting and harder to detect.

#### Task
1. [📚] What are the early warning signs of underfitting?
2. [📚] How does underfitting manifest in learning curves?
3. [📚] What is the relationship between underfitting and model capacity?
4. [📚] How do you distinguish between underfitting and poor data quality?

For a detailed explanation of this question, see [Question 10: Underfitting Detection](L6_5_10_explanation.md).

## Question 11

### Problem Statement
The bias-variance tradeoff is fundamental to understanding model complexity.

#### Task
1. [🔍] What is the mathematical relationship between bias, variance, and total error?
2. [🔍] How does tree depth affect bias and variance?
3. [🔍] What is the "sweet spot" in the bias-variance tradeoff?
4. [🔍] How do you visualize the bias-variance tradeoff?

For a detailed explanation of this question, see [Question 11: Bias-Variance Mathematics](L6_5_11_explanation.md).

## Question 12

### Problem Statement
Cross-validation provides robust estimates of generalization performance.

#### Task
1. [📚] What are the different types of cross-validation for decision trees?
2. [📚] How do you choose the number of folds for cross-validation?
3. [📚] What is stratified cross-validation and when is it important?
4. [📚] How do you interpret cross-validation results?

For a detailed explanation of this question, see [Question 12: Cross-Validation Methods](L6_5_12_explanation.md).

## Question 13

### Problem Statement
Learning curves provide insights into model behavior and data requirements.

#### Task
1. [🔍] What are the different types of learning curves for decision trees?
2. [🔍] How do you interpret learning curves with different shapes?
3. [🔍] What do learning curves tell you about data collection needs?
4. [🔍] How do learning curves help with hyperparameter tuning?

For a detailed explanation of this question, see [Question 13: Learning Curve Analysis](L6_5_13_explanation.md).

## Question 14

### Problem Statement
Regularization techniques can be combined for better results.

#### Task
1. [📚] How do you combine multiple regularization parameters?
2. [📚] What is the relationship between different regularization techniques?
3. [📚] How do you tune multiple regularization parameters simultaneously?
4. [📚] What are the trade-offs of different regularization combinations?

For a detailed explanation of this question, see [Question 14: Combined Regularization](L6_5_14_explanation.md).

## Question 15

### Problem Statement
Model complexity can be measured in different ways.

#### Task
1. [🔍] What are the different measures of tree complexity?
2. [🔍] How do you measure the "effective" complexity of a tree?
3. [🔍] What is the relationship between complexity and interpretability?
4. [🔍] How do you balance complexity with performance requirements?

For a detailed explanation of this question, see [Question 15: Complexity Measurement](L6_5_15_explanation.md).

## Question 16

### Problem Statement
Early stopping can prevent overfitting during tree construction.

#### Task
1. [📚] What is early stopping and how does it work?
2. [📚] How do you implement early stopping in decision trees?
3. [📚] What are the advantages and disadvantages of early stopping?
4. [📚] How do you choose the optimal stopping point?

For a detailed explanation of this question, see [Question 16: Early Stopping](L6_5_16_explanation.md).

## Question 17

### Problem Statement
Data quality affects the overfitting-underfitting balance.

#### Task
1. [🔍] How does data quality affect the optimal model complexity?
2. [🔍] What is the relationship between noise and overfitting?
3. [🔍] How do you handle noisy data in decision trees?
4. [🔍] What is the impact of feature quality on model complexity?

For a detailed explanation of this question, see [Question 17: Data Quality Impact](L6_5_17_explanation.md).

## Question 18

### Problem Statement
Practical strategies help balance overfitting and underfitting.

#### Task
1. [📚] **Strategy 1**: How do you use validation sets effectively?
2. [📚] **Strategy 2**: How do you implement progressive complexity testing?
3. [📚] **Strategy 3**: How do you use ensemble methods to balance complexity?
4. [📚] What is the iterative process for finding optimal complexity?

For a detailed explanation of this question, see [Question 18: Practical Strategies](L6_5_18_explanation.md).
