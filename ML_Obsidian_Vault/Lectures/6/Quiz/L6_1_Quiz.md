# Lecture 6.1: Foundations of Decision Trees Quiz

## Overview
This quiz contains 18 questions covering different topics from section 6.1 of the lectures on Foundations of Decision Trees, including tree structure, decision-making process, feature space partitioning, decision boundaries, classification vs regression trees, and advantages/disadvantages.

## Question 1

### Problem Statement
Consider a simple decision tree for classifying whether a person should play tennis based on two features:
- Outlook: {Sunny, Rainy, Cloudy}
- Temperature: {Hot, Mild, Cool}

The tree has the following structure:
- Root: Outlook
  - Sunny → Temperature
    - Hot → Don't Play
    - Mild → Play
    - Cool → Play
  - Rainy → Don't Play
  - Cloudy → Play

#### Task
1. [🔍] How many decision rules are in this tree?
2. [🔍] What is the maximum depth of this tree?
3. [🔍] If someone asks "Should I play tennis when it's sunny and hot?", what would the tree predict?
4. [🔍] Draw a simple diagram of this decision tree structure

For a detailed explanation of this question, see [Question 1: Decision Tree Structure Analysis](L6_1_1_explanation.md).

## Question 2

### Problem Statement
A decision tree is used to classify emails as "Spam" or "Not Spam" based on three binary features:

| Feature | Values |
|---------|--------|
| Contains "Free" | Yes/No |
| Contains "Money" | Yes/No |
| Length > 100 words | Yes/No |

#### Task
1. [📚] What is the maximum number of leaf nodes possible in this tree?
2. [📚] If the tree has 6 leaf nodes, what is the minimum depth required?
3. [📚] Give an example of a decision path that would classify an email as "Spam"
4. [📚] Explain why a decision tree might be a good choice for this email classification problem

For a detailed explanation of this question, see [Question 2: Email Classification Decision Tree](L6_1_2_explanation.md).

## Question 3

### Problem Statement
Consider a decision tree for predicting whether a student will pass an exam based on:

| Feature | Values |
|---------|--------|
| Study Hours | {0-2, 3-5, 6+} |
| Previous Grade | {A, B, C, D} |
| Attendance | {High, Low} |

#### Task
1. [🔍] How many different possible input combinations exist?
2. [🔍] If you want to create a tree with exactly 8 leaf nodes, how many features should you use for splitting?
3. [🔍] Which feature would you choose as the root split if you want to minimize the number of splits needed?
4. [🔍] Explain the difference between a classification tree and a regression tree

For a detailed explanation of this question, see [Question 3: Student Exam Prediction Tree](L6_1_3_explanation.md).

## Question 4

### Problem Statement
A decision tree has the following properties:

| Property | Value |
|----------|-------|
| Internal nodes (including root) | 5 |
| Leaf nodes | 6 |
| Maximum depth | 3 |

#### Task
1. [🔍] How many edges (branches) does this tree have?
2. [🔍] What is the minimum number of features needed if each feature can have at most 3 values?
3. [🔍] If you add one more feature with 2 possible values, what is the new maximum number of leaf nodes?
4. [🔍] Draw a possible tree structure that meets these specifications

For a detailed explanation of this question, see [Question 4: Tree Structure Analysis](L6_1_4_explanation.md).

## Question 5

### Problem Statement
You are designing a decision tree for a restaurant recommendation system with these features:

| Feature | Values |
|---------|--------|
| Price Range | {Low, Medium, High} |
| Cuisine Type | {Italian, Chinese, Mexican, American} |
| Distance | {Near, Far} |

#### Task
1. [📚] How many possible restaurant configurations exist?
2. [📚] If you want to recommend exactly 4 types of restaurants, what tree structure would you design?
3. [📚] Which feature would be most important for the root split and why?
4. [📚] How would you handle cases where multiple restaurants match the same criteria?

For a detailed explanation of this question, see [Question 5: Restaurant Recommendation Tree](L6_1_5_explanation.md).

## Question 6

### Problem Statement
Decision trees create decision boundaries that partition the feature space.

#### Task
1. [🔍] Draw a simple 2D feature space with two features (X, Y) and show how a decision tree with splits X > 2 and Y > 1 would partition it
2. [🔍] How many regions (partitions) would this tree create?
3. [🔍] What is the shape of the decision boundary created by this tree?
4. [🔍] Why are decision tree boundaries always parallel to the feature axes?

For a detailed explanation of this question, see [Question 6: Decision Boundaries](L6_1_6_explanation.md).

## Question 7

### Problem Statement
Decision trees can be used for both classification and regression tasks.

#### Task
1. [📚] What is the main difference between classification and regression tree outputs?
2. [📚] How does a classification tree make predictions at leaf nodes?
3. [📚] How does a regression tree make predictions at leaf nodes?
4. [📚] Give an example of when you would use each type of tree

For a detailed explanation of this question, see [Question 7: Classification vs Regression Trees](L6_1_7_explanation.md).

## Question 8

### Problem Statement
Decision trees have both advantages and disadvantages compared to other machine learning methods.

#### Task
1. [📚] **Advantage 1**: Why are decision trees easy to interpret?
2. [📚] **Advantage 2**: How do decision trees handle mixed data types?
3. [📚] **Disadvantage 1**: Why are decision trees prone to overfitting?
4. [📚] **Disadvantage 2**: What happens to decision trees when features are highly correlated?

For a detailed explanation of this question, see [Question 8: Advantages and Disadvantages](L6_1_8_explanation.md).

## Question 9

### Problem Statement
Consider a decision tree for predicting whether a customer will buy a product based on age and income.

#### Task
1. [🔍] If age has values {18-25, 26-35, 36-50, 51+}, how many possible splits can be created?
2. [🔍] What is the difference between binary and multi-way splits?
3. [🔍] How would you represent a binary split for age > 30?
4. [🔍] Why might binary splits be preferred over multi-way splits?

For a detailed explanation of this question, see [Question 9: Split Types](L6_1_9_explanation.md).

## Question 10

### Problem Statement
Decision trees can handle different types of data.

#### Task
1. [📚] What types of data can decision trees handle natively?
2. [📚] How do you handle ordinal categorical variables (e.g., Low, Medium, High)?
3. [📚] What happens if you have a feature with 1000 different values?
4. [📚] How do you handle mixed data types in the same tree?

For a detailed explanation of this question, see [Question 10: Data Type Handling](L6_1_10_explanation.md).

## Question 11

### Problem Statement
Tree depth and complexity affect model performance.

#### Task
1. [🔍] What is the relationship between tree depth and the number of leaf nodes?
2. [🔍] How does tree depth relate to the number of decision rules?
3. [🔍] What is the maximum depth of a tree with 15 leaf nodes?
4. [🔍] Why might a very deep tree be problematic?

For a detailed explanation of this question, see [Question 11: Tree Complexity](L6_1_11_explanation.md).

## Question 12

### Problem Statement
Decision paths in trees represent specific rules.

#### Task
1. [📚] How many decision paths exist in a tree with 8 leaf nodes?
2. [📚] What is the length of the longest path in a tree of depth 4?
3. [📚] How do you represent a decision path as a logical expression?
4. [📚] What is the relationship between path length and prediction confidence?

For a detailed explanation of this question, see [Question 12: Decision Paths](L6_1_12_explanation.md).

## Question 13

### Problem Statement
Feature importance can be inferred from tree structure.

#### Task
1. [🔍] How can you determine which features are most important from a tree?
2. [🔍] What does it mean if a feature appears near the root of the tree?
3. [🔍] How does feature importance relate to information gain?
4. [🔍] Can a feature appear multiple times in different parts of the tree?

For a detailed explanation of this question, see [Question 13: Feature Importance](L6_1_13_explanation.md).

## Question 14

### Problem Statement
Decision trees can be represented in different formats.

#### Task
1. [📚] What are the three main ways to represent a decision tree?
2. [📚] How do you convert a tree structure to a set of rules?
3. [📚] What is the advantage of rule-based representation?
4. [📚] How do you represent a tree in a tabular format?

For a detailed explanation of this question, see [Question 14: Tree Representations](L6_1_14_explanation.md).

## Question 15

### Problem Statement
Tree construction follows specific principles.

#### Task
1. [🔍] What is the principle of greedy tree construction?
2. [🔍] Why don't we build all possible trees and choose the best one?
3. [🔍] What is the computational complexity of building a decision tree?
4. [🔍] How does the greedy approach affect the final tree quality?

For a detailed explanation of this question, see [Question 15: Construction Principles](L6_1_15_explanation.md).

## Question 16

### Problem Statement
Decision trees can be used for different prediction tasks.

#### Task
1. [📚] What is the difference between binary and multi-class classification trees?
2. [📚] How do you handle ordinal classification (e.g., rating scales)?
3. [📚] What is the difference between classification and regression tree outputs?
4. [📚] Can a single tree handle both classification and regression tasks?

For a detailed explanation of this question, see [Question 16: Prediction Tasks](L6_1_16_explanation.md).

## Question 17

### Problem Statement
Tree evaluation involves multiple metrics.

#### Task
1. [🔍] What are the main evaluation metrics for classification trees?
2. [🔍] What are the main evaluation metrics for regression trees?
3. [🔍] How do you measure tree interpretability?
4. [🔍] What is the trade-off between accuracy and interpretability?

For a detailed explanation of this question, see [Question 17: Evaluation Metrics](L6_1_17_explanation.md).

## Question 18

### Problem Statement
Decision trees have specific use cases and limitations.

#### Task
1. [📚] **Use Case 1**: When would you choose a decision tree over a neural network?
2. [📚] **Use Case 2**: When would you choose a decision tree over logistic regression?
3. [📚] **Limitation 1**: What happens when features are highly correlated?
4. [📚] **Limitation 2**: How do decision trees handle non-linear relationships?

For a detailed explanation of this question, see [Question 18: Use Cases and Limitations](L6_1_18_explanation.md).
