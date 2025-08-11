# Lecture 9.5: Filter Methods Quiz

## Overview
This quiz contains 5 questions covering different topics from section 9.5 of the lectures on Filter Methods, including filter algorithms, variance threshold, information gain, chi-square test, and filter method advantages/limitations.

## Question 1

### Problem Statement
Consider the following dataset with 4 features:

| Sample | F1 | F2 | F3 | F4 | Target |
|--------|----|----|----|----|--------|
| 1      | 1  | 0  | 5  | 1  | 0      |
| 2      | 1  | 0  | 5  | 1  | 0      |
| 3      | 1  | 0  | 5  | 1  | 0      |
| 4      | 0  | 1  | 5  | 0  | 1      |
| 5      | 0  | 1  | 5  | 0  | 1      |

#### Task
1. [🔍] Calculate the variance of each feature
2. [🔍] If you set a variance threshold of 0.1, which features would be removed?
3. [🔍] Why might F3 be problematic for this dataset?
4. [🔍] What is the advantage of using variance threshold?

For a detailed explanation of this question, see [Question 1: Variance Threshold Filter](L9_5_1_explanation.md).

## Question 2

### Problem Statement
Information gain measures the reduction in entropy when a feature is used for splitting.

#### Task
1. [📚] What is entropy and how is it calculated?
2. [📚] How is information gain calculated for a feature?
3. [📚] What does a high information gain indicate?
4. [📚] What are the limitations of information gain?

For a detailed explanation of this question, see [Question 2: Information Gain Calculation](L9_5_2_explanation.md).

## Question 3

### Problem Statement
The Chi-square test is used for categorical feature selection.

#### Task
1. [📚] What is the Chi-square test and when is it appropriate?
2. [📚] How do you calculate the Chi-square statistic?
3. [📚] How do you interpret the Chi-square p-value?
4. [📚] What are the assumptions of the Chi-square test?

For a detailed explanation of this question, see [Question 3: Chi-Square Test for Features](L9_5_3_explanation.md).

## Question 4

### Problem Statement
Filter methods have several advantages over other feature selection approaches.

#### Task
1. [📚] **Advantage 1**: How do filter methods handle computational efficiency?
2. [📚] **Advantage 2**: Why are filter methods independent of the learning algorithm?
3. [📚] **Advantage 3**: How do filter methods scale with dataset size?
4. [📚] **Advantage 4**: What makes filter methods interpretable?

For a detailed explanation of this question, see [Question 4: Filter Method Advantages](L9_5_4_explanation.md).

## Question 5

### Problem Statement
Filter methods also have limitations that should be considered.

#### Task
1. [📚] **Limitation 1**: How do filter methods handle feature interactions?
2. [📚] **Limitation 2**: Why might filter methods miss relevant features?
3. [📚] **Limitation 3**: How do filter methods perform with non-linear relationships?
4. [📚] **Limitation 4**: What is the "redundancy problem" in filter methods?

For a detailed explanation of this question, see [Question 5: Filter Method Limitations](L9_5_5_explanation.md).
