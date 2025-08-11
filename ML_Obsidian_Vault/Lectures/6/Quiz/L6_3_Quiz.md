# Lecture 6.3: ID3 Algorithm Quiz

## Overview
This quiz contains 18 questions covering different topics from section 6.3 of the lectures on ID3 Algorithm, including ID3 steps, recursive tree construction, stopping criteria, categorical feature handling, algorithm limitations, implementation details, and optimization.

## Question 1

### Problem Statement
The ID3 algorithm follows a recursive approach to build decision trees.

#### Task
1. [🔍] What are the main steps of the ID3 algorithm?
2. [🔍] How does ID3 choose the best feature for splitting at each node?
3. [🔍] What is the base case for stopping recursion?
4. [🔍] Why is ID3 considered a greedy algorithm?

For a detailed explanation of this question, see [Question 1: ID3 Algorithm Overview](L6_3_1_explanation.md).

## Question 2

### Problem Statement
Consider a dataset with the following class distribution:

| Class | Count |
|-------|-------|
| Yes   | 8     |
| No    | 4     |

#### Task
1. [📚] Calculate the entropy of this dataset
2. [📚] If a feature splits this into two branches with distributions [6,2] and [2,2], calculate the information gain
3. [📚] Would this be a good split according to ID3?
4. [📚] What is the next step in ID3 after finding the best split?

For a detailed explanation of this question, see [Question 2: ID3 Split Selection](L6_3_2_explanation.md).

## Question 3

### Problem Statement
ID3 uses stopping criteria to prevent infinite recursion.

#### Task
1. [🔍] What are the three main stopping criteria in ID3?
2. [🔍] Why is it important to have stopping criteria?
3. [🔍] What happens when all features have been used?
4. [🔍] How do you handle cases where no features remain but the node is not pure?

For a detailed explanation of this question, see [Question 3: ID3 Stopping Criteria](L6_3_3_explanation.md).

## Question 4

### Problem Statement
Consider building a decision tree for a weather dataset with features:

| Feature | Values |
|---------|--------|
| Outlook | Sunny, Rainy, Cloudy |
| Temperature | Hot, Mild, Cool |
| Humidity | High, Normal |
| Windy | True, False |

#### Task
1. [📚] How many possible leaf nodes could this tree have?
2. [📚] What is the maximum depth of the tree?
3. [📚] How would ID3 handle categorical features with many values?
4. [📚] What are the limitations of ID3 for this dataset?

For a detailed explanation of this question, see [Question 4: ID3 Tree Construction](L6_3_4_explanation.md).

## Question 5

### Problem Statement
ID3 has several advantages and limitations compared to other algorithms.

#### Task
1. [📚] **Advantage 1**: How does ID3 handle missing values?
2. [📚] **Advantage 2**: Why is ID3 computationally efficient?
3. [📚] **Limitation 1**: How does ID3 handle continuous features?
4. [📚] **Limitation 2**: What happens when ID3 encounters noise in the data?

For a detailed explanation of this question, see [Question 5: ID3 Advantages and Limitations](L6_3_5_explanation.md).

## Question 6

### Problem Statement
ID3 builds trees recursively by partitioning the dataset at each node.

#### Task
1. [🔍] Explain the recursive partitioning process in ID3 step by step
2. [🔍] What happens to the dataset size as you go deeper in the tree?
3. [🔍] How does ID3 ensure that each recursive call works on a subset of the original data?
4. [🔍] Why is the recursive approach computationally efficient?

For a detailed explanation of this question, see [Question 6: Recursive Partitioning](L6_3_6_explanation.md).

## Question 7

### Problem Statement
ID3 handles categorical features by creating multi-way splits.

#### Task
1. [📚] How does ID3 create splits for a categorical feature with 4 values?
2. [📚] What is the maximum number of branches a single categorical feature can create?
3. [📚] Why might multi-way splits be problematic for some datasets?
4. [📚] How would you handle a categorical feature with 20 different values?

For a detailed explanation of this question, see [Question 7: Handling Categorical Features](L6_3_7_explanation.md).

## Question 8

### Problem Statement
ID3 has several limitations that affect its practical use.

#### Task
1. [📚] **Limitation 1**: Why does ID3 struggle with continuous features?
2. [📚] **Limitation 2**: How does ID3 handle noise in the data?
3. [📚] **Limitation 3**: What happens when ID3 encounters missing values?
4. [📚] **Limitation 4**: Why might ID3 create overly complex trees?

For a detailed explanation of this question, see [Question 8: ID3 Limitations](L6_3_8_explanation.md).

## Question 9

### Problem Statement
ID3 implementation requires careful handling of edge cases.

#### Task
1. [🔍] What happens if all features have been used but the node is not pure?
2. [🔍] How do you handle a feature that creates empty branches?
3. [🔍] What is the "majority class" rule and when is it used?
4. [🔍] How do you handle cases where no features provide positive information gain?

For a detailed explanation of this question, see [Question 9: Edge Case Handling](L6_3_9_explanation.md).

## Question 10

### Problem Statement
ID3 can be optimized for better performance.

#### Task
1. [📚] What is the time complexity of building a decision tree with ID3?
2. [📚] How can you optimize feature selection in ID3?
3. [📚] What is the space complexity of storing a decision tree?
4. [📚] How does the dataset size affect ID3 performance?

For a detailed explanation of this question, see [Question 10: Performance Optimization](L6_3_10_explanation.md).

## Question 11

### Problem Statement
ID3 has specific requirements for data preprocessing.

#### Task
1. [🔍] What data types can ID3 handle without preprocessing?
2. [🔍] How do you handle numerical features in ID3?
3. [🔍] What happens if you have duplicate samples in your dataset?
4. [🔍] How do you handle features with constant values?

For a detailed explanation of this question, see [Question 11: Data Preprocessing](L6_3_11_explanation.md).

## Question 12

### Problem Statement
ID3 tree construction follows a specific order.

#### Task
1. [📚] Why does ID3 choose the feature with highest information gain first?
2. [📚] What is the relationship between feature selection order and tree depth?
3. [📚] How does the order of feature selection affect the final tree structure?
4. [📚] Can the order of feature selection affect tree performance?

For a detailed explanation of this question, see [Question 12: Feature Selection Order](L6_3_12_explanation.md).

## Question 13

### Problem Statement
ID3 can be extended with additional stopping criteria.

#### Task
1. [🔍] What is the minimum samples per leaf stopping criterion?
2. [🔍] What is the maximum depth stopping criterion?
3. [🔍] What is the minimum information gain threshold?
4. [🔍] How do these criteria affect tree size and performance?

For a detailed explanation of this question, see [Question 13: Extended Stopping Criteria](L6_3_13_explanation.md).

## Question 14

### Problem Statement
ID3 implementation can vary in different programming languages.

#### Task
1. [📚] What are the key data structures needed for ID3 implementation?
2. [📚] How do you represent a decision tree node in code?
3. [📚] What is the recursive function signature for ID3?
4. [📚] How do you handle the return value from recursive calls?

For a detailed explanation of this question, see [Question 14: Implementation Details](L6_3_14_explanation.md).

## Question 15

### Problem Statement
ID3 has specific memory and storage requirements.

#### Task
1. [🔍] How much memory does a decision tree node typically require?
2. [🔍] What is the relationship between dataset size and tree memory usage?
3. [🔍] How do you estimate the total memory needed for a complete tree?
4. [🔍] What happens if you run out of memory during tree construction?

For a detailed explanation of this question, see [Question 15: Memory Requirements](L6_3_15_explanation.md).

## Question 16

### Problem Statement
ID3 can be parallelized for better performance.

#### Task
1. [📚] Which parts of ID3 can be parallelized?
2. [📚] How do you parallelize feature evaluation?
3. [📚] What are the challenges of parallelizing recursive tree construction?
4. [📚] When is parallelization most beneficial for ID3?

For a detailed explanation of this question, see [Question 16: Parallelization](L6_3_16_explanation.md).

## Question 17

### Problem Statement
ID3 can be adapted for different types of problems.

#### Task
1. [🔍] How do you adapt ID3 for multi-class classification?
2. [🔍] How do you adapt ID3 for regression problems?
3. [🔍] How do you adapt ID3 for multi-output problems?
4. [🔍] What modifications are needed for each adaptation?

For a detailed explanation of this question, see [Question 17: Problem Adaptations](L6_3_17_explanation.md).

## Question 18

### Problem Statement
ID3 has evolved into more advanced algorithms.

#### Task
1. [📚] **Evolution 1**: What improvements does C4.5 make over ID3?
2. [📚] **Evolution 2**: What improvements does CART make over ID3?
3. [📚] **Evolution 3**: What improvements do modern tree algorithms make?
4. [📚] Why is ID3 still important to understand despite its limitations?

For a detailed explanation of this question, see [Question 18: Algorithm Evolution](L6_3_18_explanation.md).
