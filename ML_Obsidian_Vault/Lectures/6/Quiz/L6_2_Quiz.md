# Lecture 6.2: Entropy and Information Gain in Trees Quiz

## Overview
This quiz contains 18 questions covering different topics from section 6.2 of the lectures on Entropy and Information Gain in Trees, including entropy calculation, information gain, gain ratio, impurity measures, and missing value handling.

## Question 1

### Problem Statement
Consider a dataset with 100 samples where:
- 60 samples belong to Class A
- 40 samples belong to Class B

#### Task
1. [🔍] Calculate the entropy $H(S)$ of this dataset
2. [🔍] What would be the entropy if the classes were perfectly balanced (50-50)?
3. [🔍] What would be the entropy if all samples belonged to the same class?
4. [🔍] Explain why entropy is maximum when classes are equally likely

For a detailed explanation of this question, see [Question 1: Basic Entropy Calculation](L6_2_1_explanation.md).

## Question 2

### Problem Statement
A feature "Color" splits the dataset as follows:

| Color | Class A | Class B | Total |
|-------|---------|---------|-------|
| Red   | 20      | 10      | 30    |
| Blue  | 25      | 15      | 40    |
| Green | 15      | 15      | 30    |

#### Task
1. Calculate the entropy for each color value
2. Calculate the weighted average entropy after splitting
3. Calculate the information gain from this split
4. Would this be a good split for the root node? Explain why

For a detailed explanation of this question, see [Question 2: Information Gain Calculation](L6_2_2_explanation.md).

## Question 3

### Problem Statement
Consider a feature "Size" with the following split:

| Size | Class A | Class B | Total |
|------|---------|---------|-------|
| Small | 10      | 30      | 40    |
| Medium| 30      | 10      | 40    |
| Large | 20      | 0       | 20    |

#### Task
1. [🔍] Calculate the entropy for each size category
2. [🔍] Calculate the weighted average entropy after splitting
3. [🔍] Calculate the information gain from this split
4. [🔍] Compare this split with the Color split from Question 2

For a detailed explanation of this question, see [Question 3: Comparing Information Gains](L6_2_3_explanation.md).

## Question 4

### Problem Statement
Information gain can favor features with many values. Consider a feature "ID" that uniquely identifies each sample.

#### Task
1. [📚] What would be the information gain if we split on the ID feature?
2. [📚] Why is this problematic for decision tree construction?
3. [📚] What is the gain ratio and how does it address this issue?
4. [📚] How do you calculate the split information for gain ratio?

For a detailed explanation of this question, see [Question 4: Gain Ratio and Split Information](L6_2_4_explanation.md).

## Question 5

### Problem Statement
Consider a dataset with three features and their information gains:

| Feature | Information Gain | Gain Ratio |
|---------|------------------|------------|
| Age     | 0.8              | 0.6        |
| Income  | 0.7              | 0.8        |
| Location| 0.9              | 0.4        |

#### Task
1. [📚] Which feature would you choose using only information gain?
2. [📚] Which feature would you choose using gain ratio?
3. [📚] Explain the difference between these two approaches
4. [📚] When would you prefer gain ratio over information gain?

For a detailed explanation of this question, see [Question 5: Information Gain vs Gain Ratio](L6_2_5_explanation.md).

## Question 6

### Problem Statement
Besides entropy, other impurity measures can be used in decision trees.

#### Task
1. [🔍] What is the Gini index formula for a binary classification problem?
2. [🔍] Calculate the Gini index for a dataset with 70% Class A and 30% Class B
3. [🔍] What is the classification error impurity measure?
4. [🔍] Compare the ranges of entropy, Gini index, and classification error

For a detailed explanation of this question, see [Question 6: Impurity Measures](L6_2_6_explanation.md).

## Question 7

### Problem Statement
The Gini index is another popular impurity measure for decision trees.

#### Task
1. [📚] Calculate the Gini index for each color value in the dataset from Question 2
2. [📚] Calculate the weighted average Gini index after splitting
3. [📚] Compare the Gini-based split with the entropy-based split
4. [📚] When might you prefer Gini index over entropy?

For a detailed explanation of this question, see [Question 7: Gini Index Comparison](L6_2_7_explanation.md).

## Question 8

### Problem Statement
Real-world datasets often contain missing values that need to be handled.

#### Task
1. [📚] What are three common strategies for handling missing values in decision trees?
2. [📚] How does the "surrogate split" method work?
3. [📚] What happens if you simply ignore samples with missing values?
4. [📚] When would you choose imputation over surrogate splits?

For a detailed explanation of this question, see [Question 8: Missing Value Handling](L6_2_8_explanation.md).

## Question 9

### Problem Statement
Entropy has specific mathematical properties that make it useful for decision trees.

#### Task
1. [🔍] What is the range of possible entropy values for a binary classification problem?
2. [🔍] At what class distribution is entropy maximized?
3. [🔍] What is the entropy when one class has probability 1?
4. [🔍] How does entropy change as the number of classes increases?

For a detailed explanation of this question, see [Question 9: Entropy Properties](L6_2_9_explanation.md).

## Question 10

### Problem Statement
Information gain measures the reduction in uncertainty after splitting.

#### Task
1. [📚] What is the mathematical formula for information gain?
2. [📚] Can information gain ever be negative? Why or why not?
3. [📚] What does an information gain of 0 mean?
4. [📚] How does information gain relate to the quality of a split?

For a detailed explanation of this question, see [Question 10: Information Gain Formula](L6_2_10_explanation.md).

## Question 11

### Problem Statement
Gain ratio normalizes information gain to handle bias toward features with many values.

#### Task
1. [🔍] What is the formula for split information?
2. [🔍] How does split information relate to the number of feature values?
3. [🔍] What is the range of possible gain ratio values?
4. [🔍] When might gain ratio fail to provide a good split?

For a detailed explanation of this question, see [Question 11: Gain Ratio Details](L6_2_11_explanation.md).

## Question 12

### Problem Statement
Different impurity measures have different characteristics.

#### Task
1. [📚] Compare the computational complexity of entropy vs Gini index
2. [📚] Which impurity measure is more sensitive to class distribution changes?
3. [📚] What are the advantages of using classification error as an impurity measure?
4. [📚] When might you choose one impurity measure over another?

For a detailed explanation of this question, see [Question 12: Impurity Measure Comparison](L6_2_12_explanation.md).

## Question 13

### Problem Statement
Continuous features require special handling in decision trees.

#### Task
1. [🔍] How do you find the optimal split point for a continuous feature?
2. [🔍] What is the computational complexity of finding the best split point?
3. [🔍] How many possible split points are there for n unique values?
4. [🔍] What happens if multiple features have the same information gain?

For a detailed explanation of this question, see [Question 13: Continuous Feature Splitting](L6_2_13_explanation.md).

## Question 14

### Problem Statement
Feature selection criteria affect tree quality.

#### Task
1. [📚] What are the main criteria for selecting the best feature to split on?
2. [📚] How do you handle ties in information gain?
3. [📚] What is the relationship between feature selection and tree depth?
4. [📚] How does feature selection affect the final tree structure?

For a detailed explanation of this question, see [Question 14: Feature Selection Criteria](L6_2_14_explanation.md).

## Question 15

### Problem Statement
Missing values pose challenges for decision tree construction.

#### Task
1. [🔍] What percentage of missing values can a decision tree typically handle?
2. [🔍] How does the surrogate split method work?
3. [🔍] What are the pros and cons of imputation vs surrogate splits?
4. [🔍] How do you handle missing values in the target variable?

For a detailed explanation of this question, see [Question 15: Missing Value Strategies](L6_2_15_explanation.md).

## Question 16

### Problem Statement
Information gain can be calculated for different types of splits.

#### Task
1. [📚] How do you calculate information gain for a binary split?
2. [📚] How do you calculate information gain for a multi-way split?
3. [📚] What is the relationship between split quality and information gain?
4. [📚] How does information gain change as you go deeper in the tree?

For a detailed explanation of this question, see [Question 16: Information Gain Calculation](L6_2_16_explanation.md).

## Question 17

### Problem Statement
Impurity measures can be extended to multi-class problems.

#### Task
1. [🔍] How do you calculate entropy for a 3-class problem?
2. [🔍] How do you calculate Gini index for a 4-class problem?
3. [🔍] What is the maximum entropy for a 5-class problem?
4. [🔍] How do impurity measures scale with the number of classes?

For a detailed explanation of this question, see [Question 17: Multi-Class Impurity](L6_2_17_explanation.md).

## Question 18

### Problem Statement
Advanced entropy concepts are important for decision trees.

#### Task
1. [📚] **Concept 1**: What is conditional entropy and how is it used?
2. [📚] **Concept 2**: What is mutual information and how does it relate to information gain?
3. [📚] **Concept 3**: What is cross-entropy and when is it used?
4. [📚] **Concept 4**: How do you handle class imbalance in entropy calculations?

For a detailed explanation of this question, see [Question 18: Advanced Entropy Concepts](L6_2_18_explanation.md).
