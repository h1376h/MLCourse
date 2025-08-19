# Lecture 9.4: ROC Curves and AUC Quiz

## Overview
This quiz contains 25 comprehensive questions covering ROC curves, AUC calculation, ROC interpretation, comparison with precision-recall curves, and practical applications. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Consider a binary classification model that predicts whether emails are spam or not spam.

**Model Predictions (sorted by probability):**
| Email | True Label | Predicted Probability |
|-------|------------|----------------------|
| 1     | Spam       | 0.95                 |
| 2     | Spam       | 0.88                 |
| 3     | Not Spam   | 0.82                 |
| 4     | Spam       | 0.75                 |
| 5     | Not Spam   | 0.68                 |
| 6     | Not Spam   | 0.45                 |
| 7     | Spam       | 0.32                 |
| 8     | Not Spam   | 0.15                 |

#### Task
1. Calculate the True Positive Rate (TPR) and False Positive Rate (FPR) for threshold 0.5
2. Calculate TPR and FPR for threshold 0.8
3. Calculate TPR and FPR for threshold 0.3
4. Plot these three points on a simple ROC curve
5. Estimate the AUC visually from your plot

For a detailed explanation of this question, see [Question 1: Basic ROC Construction](L9_4_1_explanation.md).

## Question 2

### Problem Statement
You have a medical diagnosis model with the following performance at different thresholds:

**Threshold Performance:**
| Threshold | TPR | FPR |
|-----------|-----|-----|
| 0.1       | 0.95| 0.80|
| 0.3       | 0.85| 0.45|
| 0.5       | 0.70| 0.25|
| 0.7       | 0.45| 0.10|
| 0.9       | 0.20| 0.02|

#### Task
1. Plot the ROC curve using these points
2. Calculate the AUC using the trapezoidal rule
3. What does an AUC of 0.5 represent?
4. What does an AUC of 1.0 represent?
5. Is this model better than random guessing? Explain

For a detailed explanation of this question, see [Question 2: ROC Curve Plotting](L9_4_2_explanation.md).

## Question 3

### Problem Statement
Consider a fraud detection system with the following confusion matrix at threshold 0.5:

|                | Predicted Fraud | Predicted Normal |
|----------------|-----------------|------------------|
| Actual Fraud   | 80              | 20               |
| Actual Normal  | 40              | 860              |

#### Task
1. Calculate the True Positive Rate (sensitivity)
2. Calculate the False Positive Rate (1 - specificity)
3. If you lower the threshold to 0.3, what happens to TPR and FPR?
4. If you raise the threshold to 0.7, what happens to TPR and FPR?
5. What threshold would you choose if false positives are more costly than false negatives?

For a detailed explanation of this question, see [Question 3: Fraud Detection ROC](L9_4_3_explanation.md).

## Question 4

### Problem Statement
You're comparing two models for a disease detection task.

**Model A Performance:**
- AUC: 0.85
- At threshold 0.5: TPR = 0.80, FPR = 0.15

**Model B Performance:**
- AUC: 0.78
- At threshold 0.5: TPR = 0.75, FPR = 0.10

#### Task
1. Which model has better overall performance? Justify
2. Which model has better performance at threshold 0.5?
3. If you need high sensitivity (TPR), which model would you choose?
4. If you need low false positive rate, which model would you choose?
5. Draw a simple sketch showing how the ROC curves might look

For a detailed explanation of this question, see [Question 4: Model Comparison](L9_4_4_explanation.md).

## Question 5

### Problem Statement
Consider a model that predicts whether a customer will churn.

**Threshold Analysis:**
| Threshold | TPR | FPR | Precision | Recall |
|-----------|-----|-----|-----------|--------|
| 0.1       | 0.95| 0.60| 0.45      | 0.95   |
| 0.3       | 0.85| 0.35| 0.58      | 0.85   |
| 0.5       | 0.70| 0.20| 0.70      | 0.70   |
| 0.7       | 0.50| 0.10| 0.80      | 0.50   |
| 0.9       | 0.25| 0.02| 0.90      | 0.25   |

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. Plot the Precision-Recall curve
4. Which curve would you use if false positives are more costly?
5. Which curve would you use if false negatives are more costly?

For a detailed explanation of this question, see [Question 5: ROC vs Precision-Recall](L9_4_5_explanation.md).

## Question 6

### Problem Statement
You have a model that predicts whether a student will pass a course.

**Performance at Different Thresholds:**
| Threshold | TPR | FPR |
|-----------|-----|-----|
| 0.2       | 0.90| 0.70|
| 0.4       | 0.75| 0.40|
| 0.6       | 0.60| 0.20|
| 0.8       | 0.35| 0.05|

#### Task
1. Plot the ROC curve
2. Calculate the AUC using the trapezoidal rule
3. What does the shape of this curve tell you about the model?
4. If you want 80% sensitivity, what threshold would you use?
5. What would be the corresponding false positive rate at that threshold?

For a detailed explanation of this question, see [Question 6: Student Performance ROC](L9_4_6_explanation.md).

## Question 7

### Problem Statement
Consider a model that predicts whether a house will sell within 30 days.

**Threshold Performance:**
| Threshold | TPR | FPR |
|-----------|-----|-----|
| 0.1       | 0.98| 0.85|
| 0.3       | 0.85| 0.55|
| 0.5       | 0.70| 0.30|
| 0.7       | 0.45| 0.12|
| 0.9       | 0.20| 0.03|

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. What does this AUC value indicate about the model?
4. If you need 90% sensitivity, what threshold would you use?
5. What would be the false positive rate at that threshold?

For a detailed explanation of this question, see [Question 7: House Sale Prediction ROC](L9_4_7_explanation.md).

## Question 8

### Problem Statement
You're evaluating a model that predicts whether a customer will make a purchase.

**Model Performance:**
- AUC: 0.72
- At threshold 0.5: TPR = 0.65, FPR = 0.25

#### Task
1. Is this model better than random guessing? Explain
2. What does an AUC of 0.72 mean in practical terms?
3. If you want to maximize precision, would you raise or lower the threshold?
4. If you want to maximize recall, would you raise or lower the threshold?
5. Draw a simple ROC curve that would correspond to AUC = 0.72

For a detailed explanation of this question, see [Question 8: Purchase Prediction ROC](L9_4_8_explanation.md).

## Question 9

### Problem Statement
Consider a model that predicts whether a patient will be readmitted to the hospital.

**Threshold Analysis:**
| Threshold | TPR | FPR | Precision |
|-----------|-----|-----|-----------|
| 0.2       | 0.90| 0.50| 0.45      |
| 0.4       | 0.75| 0.30| 0.60      |
| 0.6       | 0.55| 0.15| 0.75      |
| 0.8       | 0.30| 0.05| 0.85      |

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. Plot the Precision-Recall curve
4. Which threshold would you choose for clinical use?
5. Justify your threshold choice based on the business context

For a detailed explanation of this question, see [Question 9: Hospital Readmission ROC](L9_4_9_explanation.md).

## Question 10

### Problem Statement
You have a model that predicts whether a transaction is fraudulent.

**Performance Metrics:**
- AUC: 0.88
- At threshold 0.5: TPR = 0.82, FPR = 0.08

#### Task
1. What does an AUC of 0.88 indicate about the model?
2. If you lower the threshold to 0.3, what happens to TPR and FPR?
3. If you raise the threshold to 0.7, what happens to TPR and FPR?
4. What threshold would you choose if you want to minimize false positives?
5. What threshold would you choose if you want to maximize fraud detection?

For a detailed explanation of this question, see [Question 10: Fraud Detection Thresholds](L9_4_10_explanation.md).

## Question 11

### Problem Statement
Consider a model that predicts whether a customer will upgrade their subscription.

**Threshold Performance:**
| Threshold | TPR | FPR |
|-----------|-----|-----|
| 0.1       | 0.95| 0.75|
| 0.3       | 0.80| 0.45|
| 0.5       | 0.65| 0.25|
| 0.7       | 0.40| 0.10|
| 0.9       | 0.15| 0.02|

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. What does the shape of this curve tell you about the model?
4. If you want 85% sensitivity, what threshold would you use?
5. What would be the corresponding false positive rate?

For a detailed explanation of this question, see [Question 11: Subscription Upgrade ROC](L9_4_11_explanation.md).

## Question 12

### Problem Statement
You're comparing three models for a recommendation system.

**Model Performance:**
| Model | AUC | TPR at 0.5 | FPR at 0.5 |
|-------|-----|-------------|-------------|
| A     | 0.75| 0.70        | 0.20        |
| B     | 0.82| 0.75        | 0.25        |
| C     | 0.68| 0.65        | 0.15        |

#### Task
1. Which model has the best overall performance?
2. Which model has the best performance at threshold 0.5?
3. If you need high precision, which model would you choose?
4. If you need high recall, which model would you choose?
5. Draw a simple sketch showing the relative positions of the three ROC curves

For a detailed explanation of this question, see [Question 12: Recommendation System Comparison](L9_4_12_explanation.md).

## Question 13

### Problem Statement
Consider a model that predicts whether a student will complete an online course.

**Threshold Analysis:**
| Threshold | TPR | FPR | Precision | Recall |
|-----------|-----|-----|-----------|--------|
| 0.2       | 0.92| 0.65| 0.50      | 0.92   |
| 0.4       | 0.78| 0.40| 0.65      | 0.78   |
| 0.6       | 0.60| 0.22| 0.78      | 0.60   |
| 0.8       | 0.35| 0.08| 0.88      | 0.35   |

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. Plot the Precision-Recall curve
4. Which threshold would you choose for course planning?
5. Justify your threshold choice based on educational goals

For a detailed explanation of this question, see [Question 13: Course Completion ROC](L9_4_13_explanation.md).

## Question 14

### Problem Statement
You have a model that predicts whether a customer will contact support.

**Performance Metrics:**
- AUC: 0.65
- At threshold 0.5: TPR = 0.60, FPR = 0.30

#### Task
1. Is this model better than random guessing? Explain
2. What does an AUC of 0.65 indicate about the model?
3. If you want to minimize support calls, would you raise or lower the threshold?
4. If you want to maximize support call detection, would you raise or lower the threshold?
5. Draw a simple ROC curve that would correspond to AUC = 0.65

For a detailed explanation of this question, see [Question 14: Support Contact Prediction](L9_4_14_explanation.md).

## Question 15

### Problem Statement
Consider a model that predicts whether a product will be returned.

**Threshold Performance:**
| Threshold | TPR | FPR |
|-----------|-----|-----|
| 0.1       | 0.98| 0.80|
| 0.3       | 0.85| 0.50|
| 0.5       | 0.70| 0.25|
| 0.7       | 0.45| 0.10|
| 0.9       | 0.20| 0.02|

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. What does the shape of this curve tell you about the model?
4. If you want 90% sensitivity, what threshold would you use?
5. What would be the corresponding false positive rate?

For a detailed explanation of this question, see [Question 15: Product Return Prediction](L9_4_15_explanation.md).

## Question 16

### Problem Statement
You're evaluating a model that predicts whether a customer will leave a review.

**Threshold Analysis:**
| Threshold | TPR | FPR | Precision |
|-----------|-----|-----|-----------|
| 0.2       | 0.88| 0.55| 0.48      |
| 0.4       | 0.72| 0.32| 0.62      |
| 0.6       | 0.55| 0.18| 0.75      |
| 0.8       | 0.32| 0.06| 0.85      |

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. Plot the Precision-Recall curve
4. Which threshold would you choose for review collection?
5. Justify your threshold choice based on marketing goals

For a detailed explanation of this question, see [Question 16: Review Prediction ROC](L9_4_16_explanation.md).

## Question 17

### Problem Statement
Consider a model that predicts whether a job candidate will be successful.

**Performance Metrics:**
- AUC: 0.78
- At threshold 0.5: TPR = 0.72, FPR = 0.18

#### Task
1. What does an AUC of 0.78 indicate about the model?
2. If you lower the threshold to 0.3, what happens to TPR and FPR?
3. If you raise the threshold to 0.7, what happens to TPR and FPR?
4. What threshold would you choose if you want to minimize bad hires?
5. What threshold would you choose if you want to maximize good hire detection?

For a detailed explanation of this question, see [Question 17: Job Candidate Prediction](L9_4_17_explanation.md).

## Question 18

### Problem Statement
You have a model that predicts whether a customer will make a repeat purchase.

**Threshold Performance:**
| Threshold | TPR | FPR |
|-----------|-----|-----|
| 0.1       | 0.94| 0.70|
| 0.3       | 0.80| 0.40|
| 0.5       | 0.65| 0.20|
| 0.7       | 0.40| 0.08|
| 0.9       | 0.18| 0.02|

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. What does the shape of this curve tell you about the model?
4. If you want 85% sensitivity, what threshold would you use?
5. What would be the corresponding false positive rate?

For a detailed explanation of this question, see [Question 18: Repeat Purchase Prediction](L9_4_18_explanation.md).

## Question 19

### Problem Statement
Consider a model that predicts whether a patient will respond to a treatment.

**Threshold Analysis:**
| Threshold | TPR | FPR | Precision |
|-----------|-----|-----|-----------|
| 0.2       | 0.90| 0.60| 0.50      |
| 0.4       | 0.75| 0.35| 0.65      |
| 0.6       | 0.60| 0.18| 0.78      |
| 0.8       | 0.35| 0.06| 0.88      |

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. Plot the Precision-Recall curve
4. Which threshold would you choose for clinical trials?
5. Justify your threshold choice based on medical ethics

For a detailed explanation of this question, see [Question 19: Treatment Response ROC](L9_4_19_explanation.md).

## Question 20

### Problem Statement
You're evaluating a model that predicts whether a house will sell above asking price.

**Performance Metrics:**
- AUC: 0.71
- At threshold 0.5: TPR = 0.68, FPR = 0.22

#### Task
1. Is this model better than random guessing? Explain
2. What does an AUC of 0.71 indicate about the model?
3. If you want to maximize profit, would you raise or lower the threshold?
4. If you want to maximize sales volume, would you raise or lower the threshold?
5. Draw a simple ROC curve that would correspond to AUC = 0.71

For a detailed explanation of this question, see [Question 20: House Price Prediction](L9_4_20_explanation.md).

## Question 21

### Problem Statement
Consider a model that predicts whether a customer will upgrade to premium service.

**Threshold Performance:**
| Threshold | TPR | FPR |
|-----------|-----|-----|
| 0.1       | 0.96| 0.75|
| 0.3       | 0.82| 0.45|
| 0.5       | 0.68| 0.25|
| 0.7       | 0.45| 0.10|
| 0.9       | 0.22| 0.03|

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. What does the shape of this curve tell you about the model?
4. If you want 90% sensitivity, what threshold would you use?
5. What would be the corresponding false positive rate?

For a detailed explanation of this question, see [Question 21: Premium Upgrade Prediction](L9_4_21_explanation.md).

## Question 22

### Problem Statement
You're evaluating a model that predicts whether a student will graduate on time.

**Threshold Analysis:**
| Threshold | TPR | FPR | Precision |
|-----------|-----|-----|-----------|
| 0.2       | 0.89| 0.58| 0.52      |
| 0.4       | 0.74| 0.33| 0.66      |
| 0.6       | 0.58| 0.17| 0.79      |
| 0.8       | 0.33| 0.05| 0.89      |

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. Plot the Precision-Recall curve
4. Which threshold would you choose for academic planning?
5. Justify your threshold choice based on educational outcomes

For a detailed explanation of this question, see [Question 22: Graduation Prediction ROC](L9_4_22_explanation.md).

## Question 23

### Problem Statement
Consider a model that predicts whether a customer will recommend your service.

**Performance Metrics:**
- AUC: 0.76
- At threshold 0.5: TPR = 0.70, FPR = 0.20

#### Task
1. What does an AUC of 0.76 indicate about the model?
2. If you lower the threshold to 0.3, what happens to TPR and FPR?
3. If you raise the threshold to 0.7, what happens to TPR and FPR?
4. What threshold would you choose if you want to maximize positive recommendations?
5. What threshold would you choose if you want to minimize false positive recommendations?

For a detailed explanation of this question, see [Question 23: Recommendation Prediction](L9_4_23_explanation.md).

## Question 24

### Problem Statement
You have a model that predicts whether a customer will make a large purchase (>$500).

**Threshold Performance:**
| Threshold | TPR | FPR |
|-----------|-----|-----|
| 0.1       | 0.95| 0.80|
| 0.3       | 0.80| 0.45|
| 0.5       | 0.65| 0.22|
| 0.7       | 0.42| 0.08|
| 0.9       | 0.18| 0.02|

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. What does the shape of this curve tell you about the model?
4. If you want 85% sensitivity, what threshold would you use?
5. What would be the corresponding false positive rate?

For a detailed explanation of this question, see [Question 24: Large Purchase Prediction](L9_4_24_explanation.md).

## Question 25

### Problem Statement
Consider a model that predicts whether a customer will cancel their subscription.

**Threshold Analysis:**
| Threshold | TPR | FPR | Precision |
|-----------|-----|-----|-----------|
| 0.2       | 0.87| 0.52| 0.55      |
| 0.4       | 0.72| 0.28| 0.70      |
| 0.6       | 0.55| 0.15| 0.82      |
| 0.8       | 0.30| 0.05| 0.90      |

#### Task
1. Plot the ROC curve
2. Calculate the AUC
3. Plot the Precision-Recall curve
4. Which threshold would you choose for retention campaigns?
5. Justify your threshold choice based on customer lifetime value

For a detailed explanation of this question, see [Question 25: Subscription Cancellation ROC](L9_4_25_explanation.md).
