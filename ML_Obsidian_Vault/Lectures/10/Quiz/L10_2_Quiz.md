# Lecture 10.2: Classification Evaluation Metrics Quiz

## Overview
This quiz contains 5 questions covering different topics from section 10.2 of the lectures on Classification Evaluation Metrics, including accuracy, precision, recall, F1-score, confusion matrix, and micro/macro/weighted metrics.

## Question 1

### Problem Statement
Consider the following confusion matrix for a binary classification problem:

|                 | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| Actual Positive | 80                 | 20                 |
| Actual Negative | 15                 | 85                 |

#### Task
1. [🔍] Calculate the accuracy of this classifier
2. [🔍] Calculate the precision for the positive class
3. [🔍] Calculate the recall (sensitivity) for the positive class
4. [🔍] Calculate the F1-score for the positive class

For a detailed explanation of this question, see [Question 1: Basic Classification Metrics](L10_2_1_explanation.md).

## Question 2

### Problem Statement
Precision and recall are important metrics for classification evaluation.

#### Task
1. [📚] What does precision measure and when is it important?
2. [📚] What does recall measure and when is it important?
3. [📚] What is the relationship between precision and recall?
4. [📚] When would you prefer high precision over high recall?

For a detailed explanation of this question, see [Question 2: Precision and Recall Understanding](L10_2_2_explanation.md).

## Question 3

### Problem Statement
The F1-score combines precision and recall into a single metric.

#### Task
1. [📚] What is the mathematical formula for F1-score?
2. [📚] Why use the harmonic mean instead of arithmetic mean?
3. [📚] What does an F1-score of 0, 0.5, and 1.0 mean?
4. [📚] When is F1-score more appropriate than accuracy?

For a detailed explanation of this question, see [Question 3: F1-Score Calculation and Interpretation](L10_2_3_explanation.md).

## Question 4

### Problem Statement
Multi-class classification requires different evaluation approaches.

#### Task
1. [📚] What is micro-averaging and how is it calculated?
2. [📚] What is macro-averaging and how is it calculated?
3. [📚] What is weighted averaging and when is it useful?
4. [📚] How do you choose between micro, macro, and weighted metrics?

For a detailed explanation of this question, see [Question 4: Multi-Class Evaluation Metrics](L10_2_4_explanation.md).

## Question 5

### Problem Statement
Consider different classification scenarios and appropriate metrics.

#### Task
1. [📚] **Scenario A**: Medical diagnosis where false negatives are costly
2. [📚] **Scenario B**: Spam detection where false positives are annoying
3. [📚] **Scenario C**: Balanced multi-class problem with equal class importance
4. [📚] **Scenario D**: Imbalanced dataset with rare positive cases

For each scenario, recommend the most appropriate evaluation metrics and explain your choice.

For a detailed explanation of this question, see [Question 5: Metric Selection for Different Scenarios](L10_2_5_explanation.md).
