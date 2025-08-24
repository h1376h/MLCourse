# Lecture 9.2: Classification Evaluation Metrics Quiz

## Overview
This quiz contains 25 comprehensive questions covering classification evaluation metrics, including accuracy, precision, recall, F1 score, confusion matrix, multi-class evaluation, and practical applications. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Consider a binary classification model that predicts whether emails are spam or not spam.

**Confusion Matrix:**

|                | Predicted Spam | Predicted Not Spam |
|----------------|----------------|-------------------|
| Actual Spam    | $45$           | $5$               |
| Actual Not Spam| $10$           | $40$              |

#### Task
1. Calculate the accuracy of this model
2. Calculate the precision for the spam class
3. Calculate the recall for the spam class
4. Calculate the F1 score for the spam class
5. How many false positives does this model make?

For a detailed explanation of this question, see [Question 1: Basic Metrics Calculation](L9_2_1_explanation.md).

## Question 2

### Problem Statement
You have a medical diagnosis model with the following performance:

**Test Results:**
- Total patients: $1000$
- Actual positive cases: $100$
- Actual negative cases: $900$
- True positives: $80$
- False positives: $50$
- True negatives: $850$
- False negatives: $20$

#### Task
1. Calculate sensitivity (recall)
2. Calculate specificity
3. Calculate positive predictive value (precision)
4. Calculate negative predictive value
5. Is this model better at identifying positive or negative cases? Justify

For a detailed explanation of this question, see [Question 2: Medical Diagnosis Metrics](L9_2_2_explanation.md).

## Question 3

### Problem Statement
Consider a fraud detection system with the following confusion matrix:

|                | Predicted Fraud | Predicted Normal |
|----------------|-----------------|------------------|
| Actual Fraud   | $25$            | $75$             |
| Actual Normal  | $10$            | $890$            |

#### Task
1. Calculate the overall accuracy
2. Calculate precision for fraud detection
3. Calculate recall for fraud detection
4. Calculate F1 score for fraud detection
5. Why might accuracy be misleading for this problem?

For a detailed explanation of this question, see [Question 3: Fraud Detection Evaluation](L9_2_3_explanation.md).

## Question 4

### Problem Statement
You're comparing two models for a binary classification problem.

**Model A:**
- Precision: $0.85$
- Recall: $0.70$
- F1 Score: $0.77$

**Model B:**
- Precision: $0.75$
- Recall: $0.90$
- F1 Score: $0.82$

#### Task
1. Which model has higher precision?
2. Which model has higher recall?
3. Which model has higher F1 score?
4. If false positives are more costly than false negatives, which model would you choose?
5. Calculate the harmonic mean of precision and recall for Model A to verify the F1 score

For a detailed explanation of this question, see [Question 4: Model Comparison](L9_2_4_explanation.md).

## Question 5

### Problem Statement
Consider a multi-class classification problem with 3 classes: A, B, and C.

**Confusion Matrix:**

|     | Pred A | Pred B | Pred C |
|-----|--------|--------|--------|
| A   | $15$   | $2$    | $3$    |
| B   | $1$    | $18$   | $1$    |
| C   | $2$    | $1$    | $17$   |

#### Task
1. Calculate the overall accuracy
2. Calculate precision for class A
3. Calculate recall for class B
4. Calculate F1 score for class C
5. Which class is the model best at predicting?

For a detailed explanation of this question, see [Question 5: Multi-class Metrics](L9_2_5_explanation.md).

## Question 6

### Problem Statement
You're evaluating a model that predicts customer satisfaction levels.

**Results:**
- Very Satisfied: $150$ correct, $20$ incorrect
- Satisfied: $200$ correct, $30$ incorrect
- Neutral: $100$ correct, $25$ incorrect
- Dissatisfied: $80$ correct, $15$ incorrect
- Very Dissatisfied: $70$ correct, $10$ incorrect

#### Task
1. Calculate the overall accuracy
2. Calculate precision for "Very Satisfied" class
3. Calculate recall for "Dissatisfied" class
4. Calculate the macro-averaged F1 score
5. Which class has the highest precision?

For a detailed explanation of this question, see [Question 6: Customer Satisfaction Metrics](L9_2_6_explanation.md).

## Question 7

### Problem Statement
Consider a model that predicts whether a student will pass or fail a course.

**Results:**
- True Positives: $80$
- False Positives: $15$
- True Negatives: $70$
- False Negatives: $35$

#### Task
1. Calculate accuracy
2. Calculate precision
3. Calculate recall
4. Calculate F1 score
5. If failing to identify at-risk students is more costly than false alarms, which metric is most important?

For a detailed explanation of this question, see [Question 7: Student Performance Prediction](L9_2_7_explanation.md).

## Question 8

### Problem Statement
You have a model that predicts whether a customer will make a purchase.

**Performance by Customer Segment:**
- High-value customers: $90\%$ precision, $85\%$ recall
- Medium-value customers: $75\%$ precision, $80\%$ recall
- Low-value customers: $60\%$ precision, $70\%$ recall

#### Task
1. Calculate the weighted average precision (weighted by segment size)
2. Calculate the weighted average recall
3. Which segment has the best F1 score?
4. How would you evaluate the overall model performance?
5. Suggest one way to improve performance on low-value customers

For a detailed explanation of this question, see [Question 8: Customer Segment Analysis](L9_2_8_explanation.md).

## Question 9

### Problem Statement
Consider a model that predicts whether a loan application should be approved.

**Confusion Matrix:**

|                | Predicted Approve | Predicted Deny |
|----------------|-------------------|----------------|
| Actual Approve | $120$            | $30$           |
| Actual Deny    | $20$             | $130$          |

#### Task
1. Calculate the overall accuracy
2. Calculate precision for approval predictions
3. Calculate recall for approval predictions
4. Calculate the false positive rate
5. If denying a good loan is more costly than approving a bad loan, which metric should you prioritize?

For a detailed explanation of this question, see [Question 9: Loan Approval Metrics](L9_2_9_explanation.md).

## Question 10

### Problem Statement
You're comparing three models for a disease detection task.

**Model Performance:**

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| A     | $0.85$    | $0.70$ | $0.77$   |
| B     | $0.75$    | $0.90$ | $0.82$   |
| C     | $0.80$    | $0.80$ | $0.80$   |

#### Task
1. Which model has the highest precision?
2. Which model has the highest recall?
3. Which model has the highest F1 score?
4. If early detection is crucial, which model would you choose?
5. Calculate the arithmetic mean of precision and recall for each model

For a detailed explanation of this question, see [Question 10: Disease Detection Comparison](L9_2_10_explanation.md).

## Question 11

### Problem Statement
Consider a model that predicts whether a product will be returned.

**Results by Product Category:**
- Electronics: $85\%$ precision, $80\%$ recall
- Clothing: $70\%$ precision, $75\%$ recall
- Books: $90\%$ precision, $85\%$ recall
- Home & Garden: $75\%$ precision, $70\%$ recall

#### Task
1. Calculate the macro-averaged precision
2. Calculate the macro-averaged recall
3. Which category has the best F1 score?
4. How would you evaluate the overall model performance?
5. Suggest one way to improve performance on clothing category

For a detailed explanation of this question, see [Question 11: Product Return Prediction](L9_2_11_explanation.md).

## Question 12

### Problem Statement
You have a model that predicts whether a customer will churn.

**Performance by Subscription Length:**
- New customers (< 6 months): $75\%$ precision, $70\%$ recall
- Medium-term (6-24 months): $80\%$ precision, $85\%$ recall
- Long-term (> 24 months): $85\%$ precision, $90\%$ recall

#### Task
1. Calculate the weighted average precision (assuming equal segment sizes)
2. Calculate the weighted average recall
3. Which segment has the best F1 score?
4. What does this performance pattern suggest?
5. How would you improve the model for new customers?

For a detailed explanation of this question, see [Question 12: Customer Churn Analysis](L9_2_12_explanation.md).

## Question 13

### Problem Statement
Consider a model that predicts whether a job candidate will be successful.

**Confusion Matrix:**

|                | Predicted Success | Predicted Failure |
|----------------|-------------------|-------------------|
| Actual Success | $45$             | $15$             |
| Actual Failure | $10$             | $30$             |

#### Task
1. Calculate the overall accuracy
2. Calculate precision for success predictions
3. Calculate recall for success predictions
4. Calculate the false negative rate
5. If hiring a poor candidate is more costly than missing a good one, which metric is most important?

For a detailed explanation of this question, see [Question 13: Job Candidate Prediction](L9_2_13_explanation.md).

## Question 14

### Problem Statement
You're evaluating a model that predicts whether a transaction is fraudulent.

**Results by Transaction Amount:**
- Small (< $\$100$): $90\%$ precision, $85\%$ recall
- Medium ($\$100$-$\$1000$): $80\%$ precision, $90\%$ recall
- Large (> $\$1000$): $95\%$ precision, $75\%$ recall

#### Task
1. Calculate the macro-averaged precision
2. Calculate the macro-averaged recall
3. Which amount range has the best F1 score?
4. What does this performance pattern suggest?
5. How would you improve the model for large transactions?

For a detailed explanation of this question, see [Question 14: Fraud Detection by Amount](L9_2_14_explanation.md).

## Question 15

### Problem Statement
Consider a model that predicts whether a student will graduate on time.

**Performance by Major:**
- Engineering: $85\%$ precision, $80\%$ recall
- Business: $80\%$ precision, $85\%$ recall
- Arts: $75\%$ precision, $70\%$ recall
- Sciences: $90\%$ precision, $85\%$ recall

#### Task
1. Calculate the macro-averaged precision
2. Calculate the macro-averaged recall
3. Which major has the best F1 score?
4. What does this performance pattern suggest?
5. Suggest one way to improve performance on Arts majors

For a detailed explanation of this question, see [Question 15: Graduation Prediction by Major](L9_2_15_explanation.md).

## Question 16

### Problem Statement
You have a model that predicts whether a customer will upgrade their subscription.

**Results by Customer Type:**
- Free users: $70\%$ precision, $75\%$ recall
- Basic subscribers: $80\%$ precision, $85\%$ recall
- Premium subscribers: $85\%$ precision, $80\%$ recall

#### Task
1. Calculate the weighted average precision (assuming $50\%$ free, $30\%$ basic, $20\%$ premium)
2. Calculate the weighted average recall
3. Which customer type has the best F1 score?
4. What does this performance pattern suggest?
5. How would you improve the model for free users?

For a detailed explanation of this question, see [Question 16: Subscription Upgrade Prediction](L9_2_16_explanation.md).

## Question 17

### Problem Statement
Consider a model that predicts whether a patient will respond to a treatment.

**Confusion Matrix:**

|                | Predicted Respond | Predicted No Response |
|----------------|-------------------|----------------------|
| Actual Respond | $60$             | $20$                |
| Actual No Response | $15$         | $105$               |

#### Task
1. Calculate the overall accuracy
2. Calculate precision for response predictions
3. Calculate recall for response predictions
4. Calculate the false positive rate
5. If identifying non-responders is crucial, which metric should you prioritize?

For a detailed explanation of this question, see [Question 17: Treatment Response Prediction](L9_2_17_explanation.md).

## Question 18

### Problem Statement
You're evaluating a model that predicts whether a house will sell within 30 days.

**Performance by Price Range:**
- Low (< $\$200K$): $75\%$ precision, $80\%$ recall
- Medium ($\$200K$-$500K$): $80\%$ precision, $85\%$ recall
- High (> $\$500K$): $85\%$ precision, $70\%$ recall

#### Task
1. Calculate the macro-averaged precision
2. Calculate the macro-averaged recall
3. Which price range has the best F1 score?
4. What does this performance pattern suggest?
5. How would you improve the model for high-priced houses?

For a detailed explanation of this question, see [Question 18: House Sale Prediction](L9_2_18_explanation.md).

## Question 19

### Problem Statement
Consider a model that predicts whether a customer will leave a review.

**Results by Purchase Category:**
- Electronics: $80\%$ precision, $85\%$ recall
- Clothing: $75\%$ precision, $80\%$ recall
- Books: $90\%$ precision, $75\%$ recall
- Food: $70\%$ precision, $90\%$ recall

#### Task
1. Calculate the macro-averaged precision
2. Calculate the macro-averaged recall
3. Which category has the best F1 score?
4. What does this performance pattern suggest?
5. Suggest one way to improve performance on food category

For a detailed explanation of this question, see [Question 19: Review Prediction by Category](L9_2_19_explanation.md).

## Question 20

### Problem Statement
You have a model that predicts whether a student will pass a certification exam.

**Performance by Study Method:**
- Self-study: $75\%$ precision, $70\%$ recall
- Online course: $80\%$ precision, $85\%$ recall
- In-person training: $85\%$ precision, $80\%$ recall
- Hybrid approach: $90\%$ precision, $85\%$ recall

#### Task
1. Calculate the macro-averaged precision
2. Calculate the macro-averaged recall
3. Which study method has the best F1 score?
4. What does this performance pattern suggest?
5. How would you improve the model for self-study students?

For a detailed explanation of this question, see [Question 20: Certification Exam Prediction](L9_2_20_explanation.md).

## Question 21

### Problem Statement
Consider a model that predicts whether a customer will recommend your service.

**Confusion Matrix:**

|                | Predicted Recommend | Predicted Not Recommend |
|----------------|---------------------|-------------------------|
| Actual Recommend | $85$              | $15$                    |
| Actual Not Recommend | $20$        | $80$                    |

#### Task
1. Calculate the overall accuracy
2. Calculate precision for recommendation predictions
3. Calculate recall for recommendation predictions
4. Calculate the false negative rate
5. If identifying promoters is crucial, which metric should you prioritize?

For a detailed explanation of this question, see [Question 21: Customer Recommendation Prediction](L9_2_21_explanation.md).

## Question 22

### Problem Statement
You're evaluating a model that predicts whether a patient will be readmitted to the hospital.

**Results by Age Group:**
- Young (18-40): $80\%$ precision, $85\%$ recall
- Middle-aged (41-65): $85\%$ precision, $80\%$ recall
- Elderly (65+): $90\%$ precision, $75\%$ recall

#### Task
1. Calculate the macro-averaged precision
2. Calculate the macro-averaged recall
3. Which age group has the best F1 score?
4. What does this performance pattern suggest?
5. How would you improve the model for elderly patients?

For a detailed explanation of this question, see [Question 22: Hospital Readmission Prediction](L9_2_22_explanation.md).

## Question 23

### Problem Statement
Consider a model that predicts whether a customer will make a repeat purchase.

**Performance by Purchase Frequency:**
- First-time buyers: $70\%$ precision, $75\%$ recall
- Occasional buyers: $80\%$ precision, $85\%$ recall
- Regular buyers: $85\%$ precision, $90\%$ recall
- Loyal customers: $90\%$ precision, $85\%$ recall

#### Task
1. Calculate the macro-averaged precision
2. Calculate the macro-averaged recall
3. Which customer type has the best F1 score?
4. What does this performance pattern suggest?
5. Suggest one way to improve performance on first-time buyers

For a detailed explanation of this question, see [Question 23: Repeat Purchase Prediction](L9_2_23_explanation.md).

## Question 24

### Problem Statement
You have a model that predicts whether a student will complete an online course.

**Results by Course Duration:**
- Short course (< 4 weeks): $85\%$ precision, $80\%$ recall
- Medium course (4-8 weeks): $80\%$ precision, $85\%$ recall
- Long course (> 8 weeks): $75\%$ precision, $70\%$ recall

#### Task
1. Calculate the macro-averaged precision
2. Calculate the macro-averaged recall
3. Which course duration has the best F1 score?
4. What does this performance pattern suggest?
5. How would you improve the model for long courses?

For a detailed explanation of this question, see [Question 24: Course Completion Prediction](L9_2_24_explanation.md).

## Question 25

### Problem Statement
Consider a model that predicts whether a customer will contact customer support.

**Confusion Matrix:**

|                | Predicted Contact | Predicted No Contact |
|----------------|-------------------|----------------------|
| Actual Contact | $90$             | $10$                |
| Actual No Contact | $25$         | $175$               |

#### Task
1. Calculate the overall accuracy
2. Calculate precision for contact predictions
3. Calculate recall for contact predictions
4. Calculate the false positive rate
5. If reducing support calls is the goal, which metric is most important?

For a detailed explanation of this question, see [Question 25: Customer Support Contact Prediction](L9_2_25_explanation.md).
