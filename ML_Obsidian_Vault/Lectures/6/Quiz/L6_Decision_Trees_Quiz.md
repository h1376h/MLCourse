# Lecture 6: Decision Trees Quiz

## Overview
This quiz contains 9 questions from different topics covered in section 6 of the lectures on Decision Trees, including ID3, C4.5, entropy and information gain in tree context, overfitting, underfitting, pruning, and Random Forest.

## Question 1

### Problem Statement
Consider a decision tree for classifying whether a person will buy a product based on three features:
- Age: {Young, Middle, Old}
- Income: {Low, Medium, High}
- Education: {High School, College, Graduate}

The training data shows the following distribution:
- Young + Low + High School: 80% buy, 20% don't buy
- Young + Low + College: 60% buy, 40% don't buy
- Young + Medium + High School: 70% buy, 30% don't buy
- Middle + Low + College: 40% buy, 60% don't buy
- Middle + Medium + Graduate: 90% buy, 10% don't buy
- Old + High + Graduate: 85% buy, 15% don't buy

#### Task
1. Calculate the entropy of the root node (overall dataset)
2. Calculate the information gain for splitting on Age
3. Calculate the information gain for splitting on Income
4. Which feature should be chosen as the root split according to ID3 algorithm?

For a detailed explanation of this question, see [Question 1: Entropy and Information Gain Calculations](L6_1_explanation.md).

## Question 2

### Problem Statement
Consider the following dataset for a binary classification problem:

| Feature A | Feature B | Feature C | Class |
|-----------|-----------|-----------|-------|
| 0         | 0         | 0         | 0     |
| 0         | 0         | 1         | 0     |
| 0         | 1         | 0         | 1     |
| 0         | 1         | 1         | 1     |
| 1         | 0         | 0         | 0     |
| 1         | 0         | 1         | 1     |
| 1         | 1         | 0         | 1     |
| 1         | 1         | 1         | 1     |

#### Task
1. Build the complete decision tree using the ID3 algorithm
2. Show the information gain calculation for each feature at each step
3. What is the depth of the resulting tree?
4. Is this tree optimal in terms of depth? Explain why or why not

For a detailed explanation of this question, see [Question 2: ID3 Algorithm Implementation](L6_2_explanation.md).

## Question 3

### Problem Statement
Consider a decision tree that has been trained on a dataset with 1000 samples. The tree has 15 leaf nodes and shows 95% accuracy on the training set but only 78% accuracy on a validation set.

#### Task
1. Identify whether this tree is overfitting, underfitting, or well-fitted
2. What are the signs of overfitting in this scenario?
3. Suggest three different pruning strategies that could help
4. If you were to use cost-complexity pruning, what parameter would you tune and why?

For a detailed explanation of this question, see [Question 3: Overfitting Detection and Pruning Strategies](L6_3_explanation.md).

## Question 4

### Problem Statement
You are implementing the C4.5 algorithm and need to handle a continuous feature "Temperature" with the following values: [15, 18, 20, 22, 25, 28, 30, 32, 35, 38].

#### Task
1. List all possible binary split points for the Temperature feature
2. For each split point, calculate the information gain
3. Which split point would C4.5 choose?
4. How does C4.5's handling of continuous features differ from ID3?

For a detailed explanation of this question, see [Question 4: C4.5 Continuous Feature Handling](L6_4_explanation.md).

## Question 5

### Problem Statement
Consider a Random Forest with 100 decision trees, each trained on bootstrap samples of size 1000 from a dataset with 2000 samples and 20 features.

#### Task
1. How many samples are expected to be "out-of-bag" for any given tree?
2. What is the probability that a specific sample is not selected in any bootstrap sample?
3. Explain how out-of-bag estimation works for validation
4. If you increase the number of trees to 200, how does this affect the out-of-bag estimation?

For a detailed explanation of this question, see [Question 5: Random Forest and Out-of-Bag Estimation](L6_5_explanation.md).

## Question 6

### Problem Statement
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

#### Task
1. A decision tree with more leaf nodes always has lower training error than a tree with fewer leaf nodes.
2. Information gain always favors features with more possible values.
3. Pre-pruning is more computationally expensive than post-pruning.
4. Random Forest reduces overfitting by increasing model bias.
5. The Gini index and entropy always give the same feature ranking for splits.

For a detailed explanation of these true/false questions, see [Question 6: Decision Tree True/False Statements](L6_6_explanation.md).

## Question 7

### Problem Statement
You have a decision tree with the following structure:
- Root: Feature A (3-way split)
  - A=0: Feature B (2-way split)
    - B=0: Class 0 (95% confidence)
    - B=1: Class 1 (85% confidence)
  - A=1: Feature C (2-way split)
    - C=0: Class 0 (90% confidence)
    - C=1: Class 1 (80% confidence)
  - A=2: Class 1 (98% confidence)

#### Task
1. Calculate the total number of decision rules in this tree
2. What is the maximum depth of this tree?
3. If you wanted to convert this tree to a set of if-then rules, how many rules would you have?
4. Suggest a pruning strategy that could simplify this tree while maintaining performance

For a detailed explanation of this question, see [Question 7: Tree Structure Analysis and Pruning](L6_7_explanation.md).

## Question 8

### Problem Statement
The graphs below illustrate various concepts related to decision trees and ensemble methods. Each visualization represents different aspects of tree learning and ensemble performance.

![Decision Tree Depth vs Performance](../Images/L6_Quiz_8/tree_depth_performance.png)
![Information Gain for Different Features](../Images/L6_Quiz_8/information_gain_features.png)
![Random Forest vs Single Tree Performance](../Images/L6_Quiz_8/random_forest_comparison.png)

#### Task
Using only the information provided in these graphs (i.e., without any extra computation), determine:

1. What is the optimal tree depth that balances training and validation performance?
2. Which feature provides the highest information gain for the first split?
3. How many trees in the Random Forest are needed to achieve stable performance?
4. Explain the relationship between tree depth and overfitting based on the first graph.

For a detailed explanation of this question, see [Question 8: Visual Decision Tree Analysis](L6_8_explanation.md).

## Question 9

### Problem Statement
The visualizations below illustrate various concepts from decision tree learning and ensemble methods applied to a real-world dataset. Each visualization represents different aspects of tree construction, pruning, and ensemble performance.

![Feature Importance in Decision Tree](../Images/L6_Quiz_9/feature_importance_tree.png)
![Pruning Effect on Tree Performance](../Images/L6_Quiz_9/pruning_performance.png)
![Ensemble Size vs Generalization](../Images/L6_Quiz_9/ensemble_size_generalization.png)

#### Task
Using only the information provided in these visualizations, answer the following questions:

1. Which feature is most important for the decision tree's classification decisions?
2. What is the optimal pruning parameter that maximizes validation performance?
3. How does increasing the ensemble size affect the gap between training and validation performance?
4. Explain why the feature importance ranking might differ between a single decision tree and Random Forest.
5. Identify the point of diminishing returns for ensemble size based on the third graph.

For a detailed explanation of this question, see [Question 9: Decision Tree and Ensemble Visualization Analysis](L6_9_explanation.md).
