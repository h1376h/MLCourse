# Lecture 10.4: ROC Curves and AUC Quiz

## Overview
This quiz contains 5 questions covering different topics from section 10.4 of the lectures on ROC Curves and AUC, including ROC construction, AUC calculation, ROC interpretation, and ROC vs precision-recall curves.

## Question 1

### Problem Statement
Consider the following predictions and true labels for a binary classifier:

| Sample | True Label | Predicted Probability |
|--------|------------|----------------------|
| 1      | 1          | 0.9                  |
| 2      | 1          | 0.8                  |
| 3      | 0          | 0.7                  |
| 4      | 1          | 0.6                  |
| 5      | 0          | 0.5                  |
| 6      | 0          | 0.4                  |

#### Task
1. [🔍] Calculate True Positive Rate (TPR) and False Positive Rate (FPR) at threshold 0.5
2. [🔍] Calculate TPR and FPR at threshold 0.7
3. [🔍] Calculate TPR and FPR at threshold 0.9
4. [🔍] Plot these three points on an ROC curve

For a detailed explanation of this question, see [Question 1: ROC Curve Construction](L10_4_1_explanation.md).

## Question 2

### Problem Statement
ROC curves plot True Positive Rate vs False Positive Rate across different thresholds.

#### Task
1. [📚] What does the x-axis represent in an ROC curve?
2. [📚] What does the y-axis represent in an ROC curve?
3. [📚] What does a point (0,0) on the ROC curve mean?
4. [📚] What does a point (1,1) on the ROC curve mean?

For a detailed explanation of this question, see [Question 2: ROC Curve Interpretation](L10_4_2_explanation.md).

## Question 3

### Problem Statement
Area Under the Curve (AUC) measures the overall performance of a classifier.

#### Task
1. [📚] What is the mathematical definition of AUC?
2. [📚] What range of values can AUC take?
3. [📚] What does an AUC of 0.5, 0.8, and 1.0 mean?
4. [📚] How do you interpret AUC in terms of classifier performance?

For a detailed explanation of this question, see [Question 3: AUC Calculation and Meaning](L10_4_3_explanation.md).

## Question 4

### Problem Statement
ROC curves and precision-recall curves serve different purposes.

#### Task
1. [📚] When should you use ROC curves vs precision-recall curves?
2. [📚] How do the curves differ for imbalanced datasets?
3. [📚] What does the baseline look like in each type of curve?
4. [📚] Which curve is more informative for rare positive cases?

For a detailed explanation of this question, see [Question 4: ROC vs Precision-Recall Curves](L10_4_4_explanation.md).

## Question 5

### Problem Statement
Consider different scenarios for using ROC curves and AUC.

#### Task
1. [📚] **Scenario A**: Balanced binary classification with equal class costs
2. [📚] **Scenario B**: Imbalanced dataset with rare positive class
3. [📚] **Scenario C**: Multi-class classification problem
4. [📚] **Scenario D**: Cost-sensitive classification with different error costs

For each scenario, explain whether ROC curves and AUC are appropriate and why.

For a detailed explanation of this question, see [Question 5: ROC Curves in Different Scenarios](L10_4_5_explanation.md).
