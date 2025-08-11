# Lecture 6.2: Entropy and Information Gain in Trees Quiz

## Overview
This quiz contains 5 questions covering different topics from section 6.2 of the lectures on Entropy and Information Gain in Trees, including entropy calculation, information gain, gain ratio, and their application in decision tree construction.

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
1. [📚] Calculate the entropy for each color value
2. [📚] Calculate the weighted average entropy after splitting
3. [📚] Calculate the information gain from this split
4. [📚] Would this be a good split for the root node? Explain why

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
