# Lecture 9.1: Foundations of Feature Engineering Quiz

## Overview
This quiz contains 5 questions covering different topics from section 9.1 of the lectures on Foundations of Feature Engineering, including feature types, quality, dimensionality curse, and the feature engineering process.

## Question 1

### Problem Statement
Consider the following dataset with different feature types:

| Customer ID | Age | Income | Education Level | Purchase Amount | Is Premium |
|-------------|-----|--------|-----------------|-----------------|------------|
| 001         | 25  | 45000  | Bachelor        | 150.50          | Yes        |
| 002         | 42  | 75000  | Master          | 89.99           | No         |
| 003         | 31  | 60000  | High School     | 200.00          | Yes        |

#### Task
1. [🔍] Identify the feature type for each column (numerical, categorical, ordinal, binary)
2. [🔍] Which features are continuous vs discrete?
3. [🔍] How would you encode the "Education Level" feature for machine learning?
4. [🔍] What preprocessing might be needed for the "Income" feature?

For a detailed explanation of this question, see [Question 1: Feature Types and Classification](L9_1_1_explanation.md).

## Question 2

### Problem Statement
Feature quality is crucial for machine learning performance.

#### Task
1. [📚] What makes a feature "relevant" to the target variable?
2. [📚] What is feature redundancy and why is it problematic?
3. [📚] How can you identify noisy features in a dataset?
4. [📚] What is the relationship between feature quality and model performance?

For a detailed explanation of this question, see [Question 2: Feature Quality Assessment](L9_1_2_explanation.md).

## Question 3

### Problem Statement
The "curse of dimensionality" affects machine learning in high-dimensional spaces.

#### Task
1. [📚] What is the curse of dimensionality?
2. [📚] How does it affect distance-based algorithms?
3. [📚] What is the "empty space phenomenon"?
4. [📚] How can feature engineering help mitigate dimensionality issues?

For a detailed explanation of this question, see [Question 3: Curse of Dimensionality](L9_1_3_explanation.md).

## Question 4

### Problem Statement
Feature engineering follows a systematic process.

#### Task
1. [📚] What are the main steps in the feature engineering process?
2. [📚] How do you identify which features to create?
3. [📚] What is the role of domain knowledge in feature engineering?
4. [📚] How do you validate that new features improve model performance?

For a detailed explanation of this question, see [Question 4: Feature Engineering Process](L9_1_4_explanation.md).

## Question 5

### Problem Statement
Consider different scenarios for feature engineering.

#### Task
1. [📚] **Scenario A**: Text classification with word frequency features
2. [📚] **Scenario B**: Time series prediction with lag features
3. [📚] **Scenario C**: Image classification with pixel intensity features
4. [📚] **Scenario D**: Customer churn prediction with behavioral features

For each scenario, suggest appropriate feature engineering techniques.

For a detailed explanation of this question, see [Question 5: Feature Engineering Applications](L9_1_5_explanation.md).
