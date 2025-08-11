# Lecture 7.2: Bagging (Bootstrap Aggregating) Quiz

## Overview
This quiz contains 5 questions covering different topics from section 7.2 of the lectures on Bagging, including bootstrap sampling, bagging process, diversity creation, and bagging advantages/limitations.

## Question 1

### Problem Statement
Bagging uses bootstrap sampling to create diverse training sets.

#### Task
1. [🔍] What is bootstrap sampling and how does it work?
2. [🔍] If you have a dataset with 1000 samples, how many samples will each bootstrap sample contain?
3. [🔍] What is the expected number of unique samples in each bootstrap sample?
4. [🔍] Why is bootstrap sampling important for bagging?

For a detailed explanation of this question, see [Question 1: Bootstrap Sampling in Bagging](L7_2_1_explanation.md).

## Question 2

### Problem Statement
The bagging process follows specific steps to create an ensemble.

#### Task
1. [📚] What are the main steps in the bagging algorithm?
2. [📚] How many base learners are typically used in bagging?
3. [📚] How do you combine predictions from different base learners?
4. [📚] What is the difference between bagging and simple model averaging?

For a detailed explanation of this question, see [Question 2: Bagging Algorithm Steps](L7_2_2_explanation.md).

## Question 3

### Problem Statement
Bagging creates diversity through data sampling.

#### Task
1. [🔍] How does bagging create diversity among base learners?
2. [🔍] What is the relationship between bootstrap sample size and diversity?
3. [🔍] Why is diversity important for bagging performance?
4. [🔍] How can you increase diversity in a bagging ensemble?

For a detailed explanation of this question, see [Question 3: Diversity in Bagging](L7_2_3_explanation.md).

## Question 4

### Problem Statement
Consider a bagging ensemble with the following characteristics:

| Parameter | Value |
|-----------|-------|
| Number of trees | 100 |
| Bootstrap sample size | 1000 |
| Original dataset size | 1000 |
| Base learner | Decision Tree |

#### Task
1. [📚] How many different training datasets will be created?
2. [📚] What is the expected number of unique samples per bootstrap sample?
3. [📚] How many samples will be out-of-bag for each tree on average?
4. [📚] What is the purpose of out-of-bag samples?

For a detailed explanation of this question, see [Question 4: Bagging Parameters and OOB](L7_2_4_explanation.md).

## Question 5

### Problem Statement
Bagging has both advantages and limitations compared to single models.

#### Task
1. [📚] **Advantage 1**: How does bagging reduce variance?
2. [📚] **Advantage 2**: Why is bagging robust to outliers?
3. [📚] **Limitation 1**: What are the computational costs of bagging?
4. [📚] **Limitation 2**: When might bagging not improve performance?

For a detailed explanation of this question, see [Question 5: Bagging Advantages and Limitations](L7_2_5_explanation.md).
