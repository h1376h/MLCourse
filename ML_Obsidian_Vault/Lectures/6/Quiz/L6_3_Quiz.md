# Lecture 6.3: ID3 Algorithm Quiz

## Overview
This quiz contains 5 questions covering different topics from section 6.3 of the lectures on ID3 Algorithm, including ID3 steps, recursive tree construction, stopping criteria, and algorithm implementation.

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
