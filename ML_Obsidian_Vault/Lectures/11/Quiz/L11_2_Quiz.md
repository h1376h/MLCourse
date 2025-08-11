# Lecture 11.2: Evaluation Metrics for Imbalanced Data Quiz

## Overview
This quiz contains 5 questions covering different topics from section 11.2 of the lectures on Evaluation Metrics for Imbalanced Data, including why accuracy fails, precision-recall focus, F1-score, ROC curves, and precision-recall curves.

## Question 1

### Problem Statement
Consider a classifier with the following performance on imbalanced data:

| Metric | Value |
|--------|-------|
| Accuracy | 95%   |
| Precision | 20%   |
| Recall | 60%   |
| F1-Score | 30%   |

#### Task
1. [🔍] Why is accuracy misleading in this case?
2. [🔍] What does the low precision indicate?
3. [🔍] What does the moderate recall indicate?
4. [🔍] Which metric best represents the actual performance?

For a detailed explanation of this question, see [Question 1: Why Accuracy Fails](L11_2_1_explanation.md).

## Question 2

### Problem Statement
Precision and recall are more appropriate for imbalanced data.

#### Task
1. [📚] What does precision measure in imbalanced classification?
2. [📚] What does recall measure in imbalanced classification?
3. [📚] Why are precision and recall more informative than accuracy?
4. [📚] How do you interpret precision and recall for minority classes?

For a detailed explanation of this question, see [Question 2: Precision and Recall for Imbalanced Data](L11_2_2_explanation.md).

## Question 3

### Problem Statement
F1-score combines precision and recall for balanced evaluation.

#### Task
1. [📚] What is the mathematical formula for F1-score?
2. [📚] Why use harmonic mean instead of arithmetic mean?
3. [📚] What does F1-score tell you about minority class performance?
4. [📚] When is F1-score most appropriate for imbalanced data?

For a detailed explanation of this question, see [Question 3: F1-Score for Imbalanced Data](L11_2_3_explanation.md).

## Question 4

### Problem Statement
ROC curves and AUC have limitations for imbalanced data.

#### Task
1. [📚] Why can ROC curves be misleading for imbalanced data?
2. [📚] What happens to ROC curves when the positive class is rare?
3. [📚] How does AUC behave with severe class imbalance?
4. [📚] When are ROC curves still useful for imbalanced problems?

For a detailed explanation of this question, see [Question 4: ROC Curves and Imbalanced Data](L11_2_4_explanation.md).

## Question 5

### Problem Statement
Precision-recall curves are often better for imbalanced data.

#### Task
1. [📚] What do precision-recall curves show?
2. [📚] Why are PR curves more informative for imbalanced data?
3. [📚] How do you interpret the baseline in PR curves?
4. [📚] What is the area under the PR curve (AUPRC)?

For a detailed explanation of this question, see [Question 5: Precision-Recall Curves](L11_2_5_explanation.md).
