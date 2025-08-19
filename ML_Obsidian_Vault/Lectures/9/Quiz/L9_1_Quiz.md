# Lecture 9.1: Foundations of Model Evaluation Quiz

## Overview
This quiz contains 20 comprehensive questions covering the foundations of model evaluation, including generalization, overfitting/underfitting, bias-variance tradeoff, evaluation processes, and practical applications. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
You're a data scientist evaluating a machine learning model that predicts student exam scores. The model achieves $95\%$ accuracy on training data but only $70\%$ accuracy on test data.

#### Task
1. What is this phenomenon called?
2. Calculate the training error rate
3. Calculate the generalization error rate
4. Is this model overfitting or underfitting? Explain in one sentence
5. What would you expect to happen if you increase the model complexity?

For a detailed explanation of this question, see [Question 1: Overfitting Detection](L9_1_1_explanation.md).

## Question 2

### Problem Statement
You're comparing three models for a customer churn prediction task:

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|-------|------------------|-------------------|---------------|
| A     | $98\%$              | $75\%$               | $72\%$           |
| B     | $85\%$              | $82\%$               | $80\%$           |
| C     | $70\%$              | $68\%$               | $65\%$           |

#### Task
1. Which model is overfitting? Justify your answer
2. Which model is underfitting? Justify your answer
3. Which model would you choose for deployment? Explain why
4. Calculate the generalization gap for each model
5. If you could only use one model, which would you select and why?

For a detailed explanation of this question, see [Question 2: Model Performance Analysis](L9_1_2_explanation.md).

## Question 3

### Problem Statement
Consider the bias-variance tradeoff in machine learning using a concrete example.

**Scenario:** You're building a model to predict house prices based on square footage.

#### Task
1. Define bias in the context of machine learning
2. Define variance in the context of machine learning
3. For a high-bias, low-variance model, what type of error would you expect to see?
4. For a low-bias, high-variance model, what type of error would you expect to see?
5. Draw a simple graph showing the relationship between model complexity and bias/variance

For a detailed explanation of this question, see [Question 3: Bias-Variance Tradeoff](L9_1_3_explanation.md).

## Question 4

### Problem Statement
You're evaluating a model that predicts student exam scores based on study hours.

**Training Data Results:**
- Model predicts: $85$, $78$, $92$, $88$, $76$
- Actual scores: $82$, $80$, $90$, $85$, $79$

**Test Data Results:**
- Model predicts: $87$, $81$, $89$, $83$
- Actual scores: $85$, $78$, $92$, $80$

#### Task
1. Calculate the training error (mean absolute error)
2. Calculate the test error (mean absolute error)
3. Is this model generalizing well? Explain
4. What does the difference between training and test error suggest?
5. Suggest one way to improve the model's generalization

For a detailed explanation of this question, see [Question 4: Generalization Analysis](L9_1_4_explanation.md).

## Question 5

### Problem Statement
Consider the following learning curves for three different models:

**Model X:** Training error decreases from $40\%$ to $5\%$, Validation error decreases from $45\%$ to $8\%$
**Model Y:** Training error decreases from $40\%$ to $2\%$, Validation error decreases from $45\%$ to $25\%$
**Model Z:** Training error decreases from $40\%$ to $35\%$, Validation error decreases from $45\%$ to $42\%$

#### Task
1. Which model shows signs of overfitting?
2. Which model shows signs of underfitting?
3. Which model has the best generalization?
4. For Model Y, what would happen if you collected more training data?
5. Sketch the learning curves for Model Y showing the expected behavior

For a detailed explanation of this question, see [Question 5: Learning Curve Analysis](L9_1_5_explanation.md).

## Question 6

### Problem Statement
You have a dataset with $1000$ samples and want to evaluate a model properly.

#### Task
1. What is the minimum recommended size for a test set?
2. If you use $80\%$ for training and $20\%$ for testing, how many samples are in each set?
3. Why is it important to have a separate test set?
4. What is the purpose of a validation set?
5. Draw a diagram showing the data split strategy you would recommend

For a detailed explanation of this question, see [Question 6: Data Splitting Strategy](L9_1_6_explanation.md).

## Question 7

### Problem Statement
Consider a model that predicts whether a customer will buy a product.

**Training Results:**
- $500$ positive examples, $500$ negative examples
- Model correctly predicts $450$ positive and $480$ negative cases

**Test Results:**
- $100$ positive examples, $100$ negative examples
- Model correctly predicts $85$ positive and $90$ negative cases

#### Task
1. Calculate training accuracy
2. Calculate test accuracy
3. Calculate the generalization gap
4. Is this model overfitting? Justify your answer
5. What would you expect to happen if you increase the model complexity?

For a detailed explanation of this question, see [Question 7: Binary Classification Evaluation](L9_1_7_explanation.md).

## Question 8

### Problem Statement
You're building a model to predict house prices and have limited data.

#### Task
1. What is the main challenge with limited data in model evaluation?
2. Name two techniques you could use to evaluate the model with limited data
3. Why might cross-validation be particularly useful in this scenario?
4. What is the risk of using the same data for both training and testing?
5. If you have $200$ houses, suggest a specific evaluation strategy

For a detailed explanation of this question, see [Question 8: Limited Data Evaluation](L9_1_8_explanation.md).

## Question 9

### Problem Statement
Consider the relationship between model complexity and performance.

#### Task
1. Draw a simple graph showing training error vs model complexity
2. On the same graph, show validation error vs model complexity
3. Mark the optimal model complexity point
4. What happens to training error as complexity increases?
5. What happens to validation error after the optimal point?

For a detailed explanation of this question, see [Question 9: Complexity-Performance Relationship](L9_1_9_explanation.md).

## Question 10

### Problem Statement
You're evaluating a model that predicts student grades based on attendance and study hours.

**Model Performance:**
- Training MSE: $15.2$
- Validation MSE: $18.7$
- Test MSE: $19.1$

#### Task
1. What do these numbers tell you about the model?
2. Calculate the generalization gap
3. Is this model overfitting or underfitting?
4. What would you expect to happen if you reduce model complexity?
5. Suggest one specific action to improve the model

For a detailed explanation of this question, see [Question 10: Regression Model Evaluation](L9_1_10_explanation.md).

## Question 11

### Problem Statement
Consider the concept of irreducible error in machine learning.

#### Task
1. What is irreducible error?
2. Can you eliminate irreducible error completely? Explain
3. Give an example of irreducible error in predicting house prices
4. How does irreducible error affect the bias-variance tradeoff?
5. If you have high irreducible error, what should your evaluation strategy be?

For a detailed explanation of this question, see [Question 11: Irreducible Error](L9_1_11_explanation.md).

## Question 12

### Problem Statement
You have a model that performs differently on different types of data.

**Performance by Data Type:**
- Numerical features: $85\%$ accuracy
- Categorical features: $72\%$ accuracy
- Mixed features: $78\%$ accuracy

#### Task
1. What does this performance variation suggest?
2. How would you evaluate the overall model performance?
3. What might cause this performance difference?
4. How would you report the model's performance to stakeholders?
5. Suggest one way to improve performance on categorical features

For a detailed explanation of this question, see [Question 12: Heterogeneous Performance](L9_1_12_explanation.md).

## Question 13

### Problem Statement
Consider the evaluation of a model in production vs development.

#### Task
1. What is the difference between offline and online evaluation?
2. Why might a model perform differently in production?
3. Name two challenges of online evaluation
4. What is concept drift and how does it affect evaluation?
5. How would you design an evaluation strategy for a production model?

For a detailed explanation of this question, see [Question 13: Production vs Development Evaluation](L9_1_13_explanation.md).

## Question 14

### Problem Statement
You're comparing two models and need to determine which is better.

**Model A:** Training accuracy $88\%$, Test accuracy $82\%$
**Model B:** Training accuracy $85\%$, Test accuracy $84\%$

#### Task
1. Which model has better generalization?
2. Calculate the generalization gap for each model
3. Which model would you choose for deployment? Justify
4. What additional information would you need to make a confident decision?
5. How would you test if the performance difference is statistically significant?

For a detailed explanation of this question, see [Question 14: Model Comparison](L9_1_14_explanation.md).

## Question 15

### Problem Statement
Design an evaluation strategy for a new machine learning project.

**Project Details:**
- Binary classification problem
- $5000$ training samples
- $1000$ test samples
- Need to compare $3$ different algorithms
- Must be completed within $2$ hours

#### Task
1. What evaluation metrics would you use?
2. How would you split your data?
3. What validation strategy would you recommend?
4. How would you ensure fair comparison between algorithms?
5. What would you do if one algorithm takes too long to train?

For a detailed explanation of this question, see [Question 15: Evaluation Strategy Design](L9_1_15_explanation.md).

## Question 16

### Problem Statement
You're analyzing a model that shows different performance across different demographic groups.

**Performance by Age Group:**
- $18$-$25$: $78\%$ accuracy
- $26$-$40$: $85\%$ accuracy
- $41$-$60$: $82\%$ accuracy
- $60+$: $75\%$ accuracy

#### Task
1. What does this performance variation suggest about the model?
2. How would you evaluate the overall fairness of this model?
3. What might cause the lower performance in the $18$-$25$ and $60+$ groups?
4. How would you report this to stakeholders?
5. Suggest one way to improve performance across all age groups

For a detailed explanation of this question, see [Question 16: Fairness in Model Evaluation](L9_1_16_explanation.md).

## Question 17

### Problem Statement
Consider a model that predicts stock prices with the following performance:

**Daily Performance:**
- Monday: $92\%$ accuracy
- Tuesday: $88\%$ accuracy
- Wednesday: $95\%$ accuracy
- Thursday: $87\%$ accuracy
- Friday: $90\%$ accuracy

#### Task
1. Calculate the average daily accuracy
2. Calculate the standard deviation of daily accuracy
3. What does this variation suggest about the model?
4. Is this level of variation acceptable for stock prediction? Explain
5. How would you evaluate the model's stability over time?

For a detailed explanation of this question, see [Question 17: Temporal Model Evaluation](L9_1_17_explanation.md).

## Question 18

### Problem Statement
You're evaluating a model that predicts customer satisfaction scores ($1$-$10$ scale).

**Training Results:**
- Mean predicted score: $7.2$
- Mean actual score: $7.1$
- Standard deviation of predictions: $1.8$
- Standard deviation of actual scores: $1.9$

**Test Results:**
- Mean predicted score: $6.8$
- Mean actual score: $7.0$
- Standard deviation of predictions: $1.6$
- Standard deviation of actual scores: $1.9$

#### Task
1. Is this model overfitting or underfitting? Explain
2. Calculate the bias in predictions for both training and test sets
3. What does the change in standard deviation suggest?
4. How would you improve this model's generalization?
5. What evaluation metrics would be most appropriate for this problem?

For a detailed explanation of this question, see [Question 18: Regression Model Bias Analysis](L9_1_18_explanation.md).

## Question 19

### Problem Statement
Consider a model that performs differently in different seasons.

**Seasonal Performance:**
- Spring: $85\%$ accuracy
- Summer: $78\%$ accuracy
- Fall: $82\%$ accuracy
- Winter: $75\%$ accuracy

#### Task
1. What does this seasonal variation suggest about the model?
2. How would you evaluate the model's robustness across seasons?
3. What might cause the lower performance in summer and winter?
4. How would you design a validation strategy to account for seasonality?
5. Suggest one way to make the model more seasonally robust

For a detailed explanation of this question, see [Question 19: Seasonal Model Evaluation](L9_1_19_explanation.md).

## Question 20

### Problem Statement
You're building a model to predict whether a student will pass a course based on various features.

**Feature Performance Analysis:**
- Study hours: $85\%$ accuracy when used alone
- Previous GPA: $78\%$ accuracy when used alone
- Attendance rate: $72\%$ accuracy when used alone
- Combined features: $89\%$ accuracy

#### Task
1. What does this performance pattern suggest about feature interactions?
2. How would you evaluate the contribution of each feature?
3. What evaluation strategy would you use to understand feature importance?
4. How would you test if the improvement from combining features is significant?
5. Suggest one way to further improve the model's performance

For a detailed explanation of this question, see [Question 20: Feature-Based Model Evaluation](L9_1_20_explanation.md).
