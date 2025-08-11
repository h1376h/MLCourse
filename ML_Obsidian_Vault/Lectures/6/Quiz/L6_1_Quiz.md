# Lecture 6.1: Foundations of Decision Trees Quiz

## Overview
This quiz contains 5 questions covering different topics from section 6.1 of the lectures on Foundations of Decision Trees, including tree structure, decision-making process, feature space partitioning, and decision boundaries.

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
