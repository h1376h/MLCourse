# Lecture 6.5: Overfitting and Underfitting in Trees Quiz

## Overview
This quiz contains 5 questions covering different topics from section 6.5 of the lectures on Overfitting and Underfitting in Trees, including overfitting causes, underfitting detection, model complexity, and generalization issues.

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
