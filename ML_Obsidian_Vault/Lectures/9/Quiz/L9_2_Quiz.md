# Lecture 9.2: Univariate Feature Selection Quiz

## Overview
This quiz contains 5 questions covering different topics from section 9.2 of the lectures on Univariate Feature Selection, including statistical tests, correlation analysis, mutual information, and univariate methods.

## Question 1

### Problem Statement
Consider the following dataset with features and a binary target:

| Sample | Feature 1 | Feature 2 | Feature 3 | Target |
|--------|-----------|-----------|-----------|--------|
| 1      | 1.2       | 0.8       | 5.1       | 0      |
| 2      | 2.1       | 1.5       | 4.8       | 0      |
| 3      | 8.9       | 7.2       | 1.3       | 1      |
| 4      | 9.1       | 7.8       | 1.1       | 1      |
| 5      | 8.7       | 7.1       | 1.4       | 1      |

#### Task
1. [🔍] Calculate the correlation between Feature 1 and the Target
2. [🔍] Calculate the correlation between Feature 2 and the Target
3. [🔍] Which feature has the strongest linear relationship with the target?
4. [🔍] Would you expect Feature 3 to be selected by univariate methods? Why?

For a detailed explanation of this question, see [Question 1: Correlation Analysis for Feature Selection](L9_2_1_explanation.md).

## Question 2

### Problem Statement
Statistical tests can be used for univariate feature selection.

#### Task
1. [📚] What is the Chi-square test and when is it appropriate?
2. [📚] What is the F-test and when is it appropriate?
3. [📚] What is the t-test and when is it appropriate?
4. [📚] How do you interpret p-values in feature selection?

For a detailed explanation of this question, see [Question 2: Statistical Tests for Feature Selection](L9_2_2_explanation.md).

## Question 3

### Problem Statement
Mutual information measures the mutual dependence between variables.

#### Task
1. [📚] What is mutual information and how is it calculated?
2. [📚] What are the advantages of mutual information over correlation?
3. [📚] What range of values can mutual information take?
4. [📚] How does mutual information handle non-linear relationships?

For a detailed explanation of this question, see [Question 3: Mutual Information for Feature Selection](L9_2_3_explanation.md).

## Question 4

### Problem Statement
Univariate feature selection methods include SelectKBest and SelectPercentile.

#### Task
1. [📚] How does SelectKBest work?
2. [📚] How does SelectPercentile work?
3. [📚] What are the advantages of univariate selection?
4. [📚] What are the limitations of univariate selection?

For a detailed explanation of this question, see [Question 4: Univariate Selection Methods](L9_2_4_explanation.md).

## Question 5

### Problem Statement
Consider different scenarios for univariate feature selection.

#### Task
1. [📚] **Scenario A**: Binary classification with numerical features
2. [📚] **Scenario B**: Regression with mixed feature types
3. [📚] **Scenario C**: Multi-class classification with categorical features
4. [📚] **Scenario D**: High-dimensional dataset with many features

For each scenario, suggest the most appropriate univariate selection method and explain why.

For a detailed explanation of this question, see [Question 5: Univariate Selection Applications](L9_2_5_explanation.md).
