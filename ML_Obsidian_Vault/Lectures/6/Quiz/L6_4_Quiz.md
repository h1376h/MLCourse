# Lecture 6.4: C4.5 Algorithm Quiz

## Overview
This quiz contains 5 questions covering different topics from section 6.4 of the lectures on C4.5 Algorithm, including C4.5 improvements over ID3, gain ratio, handling continuous features, and pruning techniques.

## Question 1

### Problem Statement
C4.5 is an improvement over the ID3 algorithm.

#### Task
1. [🔍] What are the main improvements that C4.5 makes over ID3?
2. [🔍] How does C4.5 address the bias toward features with many values?
3. [🔍] What is the main splitting criterion used in C4.5?
4. [🔍] Why is C4.5 considered more robust than ID3?

For a detailed explanation of this question, see [Question 1: C4.5 Improvements over ID3](L6_4_1_explanation.md).

## Question 2

### Problem Statement
C4.5 uses gain ratio instead of information gain for feature selection.

#### Task
1. [📚] What is the mathematical formula for gain ratio?
2. [📚] How does gain ratio normalize information gain?
3. [📚] What is the range of possible values for gain ratio?
4. [📚] When might gain ratio fail to provide a good split?

For a detailed explanation of this question, see [Question 2: Gain Ratio in C4.5](L6_4_2_explanation.md).

## Question 3

### Problem Statement
C4.5 can handle continuous features through discretization.

#### Task
1. [🔍] How does C4.5 find the best split point for continuous features?
2. [🔍] What is the computational complexity of finding optimal split points?
3. [🔍] How many possible split points are there for a continuous feature with $n$ unique values?
4. [🔍] What happens if multiple features have the same gain ratio?

For a detailed explanation of this question, see [Question 3: Continuous Feature Handling](L6_4_3_explanation.md).

## Question 4

### Problem Statement
C4.5 includes pruning techniques to prevent overfitting.

#### Task
1. [📚] What is the difference between pre-pruning and post-pruning?
2. [📚] How does C4.5 implement post-pruning?
3. [📚] What is the confidence factor in C4.5 pruning?
4. [📚] How do you choose the optimal confidence factor?

For a detailed explanation of this question, see [Question 4: C4.5 Pruning Methods](L6_4_4_explanation.md).

## Question 5

### Problem Statement
Consider a dataset with the following features and their metrics:

| Feature | Information Gain | Split Information | Gain Ratio |
|---------|------------------|-------------------|------------|
| Age     | 0.8              | 1.2               | 0.67       |
| Income  | 0.6              | 0.8               | 0.75       |
| Location| 0.9              | 2.1               | 0.43       |

#### Task
1. [📚] Which feature would ID3 choose based on information gain?
2. [📚] Which feature would C4.5 choose based on gain ratio?
3. [📚] Explain why these choices differ
4. [📚] Which choice is better for generalization and why?

For a detailed explanation of this question, see [Question 5: C4.5 vs ID3 Feature Selection](L6_4_5_explanation.md).
