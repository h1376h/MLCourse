# Lecture 3.6: Model Evaluation and Validation Quiz

## Overview
This quiz contains 19 questions covering different topics from section 3.6 of the lectures on Model Evaluation and Validation in linear regression.

## Question 1

### Problem Statement
You have trained a linear regression model on a dataset with 200 samples. To evaluate its performance, you calculated the following metrics:

| Metric | Training Data | Test Data |
|--------|--------------|-----------|
| MSE    | 15.2         | 22.8      |
| MAE    | 3.1          | 3.9       |
| RMSE   | 3.9          | 4.8       |
| $R^2$  | 0.82         | 0.73      |

#### Task
1. Explain what each of these metrics measures and how they're calculated
2. Based on the table, is your model likely overfitting? Explain why or why not
3. If the test data has 50 samples, calculate the Sum of Squared Errors (SSE) on the test data
4. Which of these metrics is scale-dependent and which is scale-independent? Explain the practical implications of this difference

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Understanding Error Metrics](L3_6_1_explanation.md).

## Question 2

### Problem Statement
Consider the following problem of training vs. testing error. You've collected a dataset with 1000 samples and are deciding how to split it between training and testing sets.

#### Task
1. Explain the purpose of splitting data into training and testing sets
2. If you use 70% of your data for training, how many samples will be in each set?
3. Describe how training error and testing error typically behave as the complexity of a model increases
4. If you observe that your model has a training error of 0.05 and a testing error of 0.25, what can you conclude about your model's performance? What steps might you take to improve it?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Training vs Testing Split](L3_6_2_explanation.md).

## Question 3

### Problem Statement
You are evaluating a linear regression model using a hold-out validation approach. Your dataset contains 500 samples.

#### Task
1. Describe the simple hold-out method and its implementation steps
2. List at least three limitations of the simple hold-out method
3. If you use a 70-15-15 split for training-validation-test, how many samples will be in each set?
4. Explain how the choice of the random seed for splitting could affect your model evaluation results

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Hold-Out Validation](L3_6_3_explanation.md).

## Question 4

### Problem Statement
You're using K-fold cross-validation to evaluate a linear regression model on a dataset with 100 samples.

#### Task
1. Explain how K-fold cross-validation works and its advantages over the simple hold-out method
2. If you use 5-fold cross-validation, how many training and validation samples will be used in each fold?
3. Explain the difference between K-fold cross-validation and leave-one-out cross-validation (LOOCV)
4. Calculate the computational cost (in terms of model training) of performing 5-fold CV versus LOOCV on this dataset

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Cross-Validation Methods](L3_6_4_explanation.md).

## Question 5

### Problem Statement
You're analyzing learning curves for a linear regression model with varying dataset sizes. You've collected the following results:

| Training Samples | Training Error | Validation Error |
|------------------|----------------|------------------|
| 10               | 2.1            | 12.5             |
| 50               | 3.2            | 7.8              |
| 100              | 3.9            | 6.2              |
| 200              | 4.5            | 5.8              |
| 500              | 5.0            | 5.3              |

#### Task
1. Plot by hand (rough sketch) the learning curves showing training and validation error versus number of training samples
2. Based on these curves, does the model have high bias, high variance, or both? Explain your reasoning
3. Predict what the curves would look like if you continued increasing the dataset to 1000 samples
4. What actions would you recommend to improve the model's performance based on these curves?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Learning Curves Analysis](L3_6_5_explanation.md).

## Question 6

### Problem Statement
Consider a linear regression model that has been fit to a dataset. You've calculated the residuals (the differences between the predicted and actual values) and want to analyze them.

#### Task
1. Explain what residual analysis is and why it's important for validating model assumptions
2. List four common patterns in residual plots and what each pattern indicates about the model
3. If residuals exhibit a clear U-shaped pattern when plotted against the predicted values, what does this suggest about your model?
4. How would you check if the residuals follow a normal distribution? Describe a simple test and sketch what a normal vs. non-normal distribution of residuals might look like

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Residual Analysis](L3_6_6_explanation.md).

## Question 7

### Problem Statement
You've fitted three polynomial regression models to a dataset with 100 samples. The models have degrees 1 (linear), 3 (cubic), and 5 (quintic) respectively.

| Model  | Training Error | Test Error | AIC    | BIC    |
|--------|---------------|------------|--------|--------|
| Linear | 28.4          | 30.2       | 312.4  | 318.7  |
| Cubic  | 18.7          | 24.5       | 295.3  | 310.8  |
| Quintic| 16.2          | 29.8       | 290.1  | 315.2  |

#### Task
1. Based on the test error, which model would you select? Explain your choice
2. Define what AIC and BIC are and how they're calculated
3. Explain how AIC and BIC help with model selection and how they penalize model complexity
4. Based on the AIC and BIC values, which model would you select? Is this different from your answer in task 1? Explain why or why not

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Model Selection with Information Criteria](L3_6_7_explanation.md).

## Question 8

### Problem Statement
You're analyzing the expected behavior of training and true error curves for models with different complexities. 

#### Task
1. Sketch how training and test error curves typically behave as a function of model complexity for:
   a) A situation where you have limited data (e.g., 20 samples)
   b) A situation where you have abundant data (e.g., 10,000 samples)
2. Explain how these curves relate to the concepts of bias and variance
3. For a simple linear model and a complex 10th-degree polynomial model, describe how the training and test error curves would behave as you increase the number of training samples from 10 to 10,000
4. Using the error decomposition framework, explain how total error can be broken down into structural error and approximation error, and what each represents

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Expected Error Curves and Decomposition](L3_6_8_explanation.md).

## Question 9

### Problem Statement
You have the following predictions and actual values for 5 test samples:

| Sample | Actual Value | Predicted Value |
|--------|--------------|----------------|
| 1      | 10           | 12             |
| 2      | 15           | 13             |
| 3      | 8            | 7              |
| 4      | 12           | 14             |
| 5      | 20           | 19             |

#### Task
1. Calculate the Mean Absolute Error (MAE)
2. Calculate the Mean Squared Error (MSE)
3. Calculate the Root Mean Squared Error (RMSE)
4. If the variance of the actual values is 21.7, calculate the $R^2$ (coefficient of determination) value

For a detailed explanation of this problem, including step-by-step calculations, see [Question 9: Basic Error Calculations](L3_6_9_explanation.md).

## Question 10

### Problem Statement
You are performing 10-fold cross-validation on a dataset with 150 samples.

#### Task
1. How many samples will be in each fold?
2. For each fold, how many samples will be used for training and how many for validation?
3. If running the model on the training set takes approximately 2 seconds, estimate the total time required to complete the entire cross-validation process
4. If you decide to use LOOCV instead, how many model trainings will be required?

For a detailed explanation of this problem, including step-by-step solutions, see [Question 10: Cross-Validation Calculations](L3_6_10_explanation.md).

## Question 11

### Problem Statement
Consider a simple linear regression model that predicts house prices based on square footage. The model has been evaluated on a test set with the following metrics:
- Sum of Squared Errors (SSE): 2500
- Total Sum of Squares (TSS): 10000
- Number of test samples: 25

#### Task
1. Calculate the Mean Squared Error (MSE)
2. Calculate the $R^2$ (coefficient of determination)
3. Calculate the Root Mean Squared Error (RMSE)
4. Would you consider this model to be performing well? Explain your reasoning based on these metrics

For a detailed explanation of this problem, including step-by-step calculations, see [Question 11: Evaluating Model Performance](L3_6_11_explanation.md).

## Question 12

### Problem Statement
You're comparing different data splitting strategies for evaluating a model. Your dataset has 200 samples.

#### Task
1. Calculate how many samples would be in each set for:
   a) A 60-20-20 training-validation-test split
   b) A 70-30 training-test split with 5-fold cross-validation on the training set
2. How many models will you train in total for each approach?
3. Which approach would likely give you a more reliable estimate of model performance? Explain why
4. If your dataset is highly imbalanced (e.g., 90% of samples are from one class), what additional consideration should you make when splitting the data?

For a detailed explanation of this problem, including step-by-step solutions, see [Question 12: Data Splitting Strategies](L3_6_12_explanation.md).

## Question 13

### Problem Statement
You've collected information about two models' performance on the same dataset:

Model A:
- Training Error: 15.2
- Test Error: 18.7
- Number of parameters: 5

Model B:
- Training Error: 12.8
- Test Error: 20.5
- Number of parameters: 12

#### Task
1. Calculate the AIC for both models using the formula $\text{AIC} = n \cdot \ln(\text{MSE}) + 2p$, where $n$ is the number of samples (assume $n=100$) and $p$ is the number of parameters
2. Calculate the BIC for both models using the formula $\text{BIC} = n \cdot \ln(\text{MSE}) + p \cdot \ln(n)$
3. Based on the test error, AIC, and BIC, which model would you choose? Explain your reasoning
4. What does this example tell you about the relationship between model complexity and generalization?

For a detailed explanation of this problem, including step-by-step calculations, see [Question 13: Information Criteria Calculation](L3_6_13_explanation.md).

## Question 14

### Problem Statement
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

#### Task
1. K-fold cross-validation with $K=n$ (where $n$ is the number of samples) is equivalent to leave-one-out cross-validation.
2. When comparing models using information criteria, the model with the highest AIC value should be selected.
3. A model with high bias will typically show a large gap between training and test error.
4. The $R^2$ (coefficient of determination) metric can be negative for poorly performing models.
5. In residual analysis, randomly distributed residuals around zero suggest that the linear regression assumptions are violated.
6. Cross-validation is primarily used to prevent overfitting during model training.
7. The Mean Absolute Error (MAE) is more sensitive to outliers than the Mean Squared Error (MSE).
8. In learning curves, if the validation error continues to decrease as more training samples are added, adding more data is likely to improve model performance.

For a detailed explanation of these true/false questions, see [Question 14: TRUE/FALSE Evaluation Concepts](L3_6_14_explanation.md).

## Question 15

### Problem Statement
Fill in the blanks with the appropriate terms related to model evaluation and validation.

#### Task
Fill in each blank with the most appropriate term:

1. The process of dividing data into training and test sets is commonly known as $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$.
2. The metric that measures the proportion of variance in the dependent variable that is predictable from the independent variables is called $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$.
3. $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ measures the average absolute difference between predicted and actual values.
4. $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ cross-validation is a special case where each fold contains exactly one sample.
5. When test error decreases and then increases as model complexity increases, this phenomenon is called $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$.
6. The difference between the predicted values and the actual values are called $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$.
7. A model with high variance but low bias is likely to be $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$.
8. The $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ criterion penalizes model complexity more severely than AIC when the sample size is large.

For a detailed explanation of these fill-in-the-blank questions, see [Question 15: Key Concepts in Model Evaluation](L3_6_15_explanation.md).

## Question 16

### Problem Statement
Multiple choice questions on model evaluation and validation concepts.

#### Task
For each question, select the BEST answer.

1. Which of the following metrics is invariant to the scale of the target variable?
   a) Mean Squared Error (MSE)
   b) Mean Absolute Error (MAE)
   c) Root Mean Squared Error (RMSE)
   d) $R^2$ (coefficient of determination)

2. When using K-fold cross-validation, increasing the value of K typically:
   a) Decreases bias but increases variance in the estimation of model performance
   b) Increases bias but decreases variance in the estimation of model performance
   c) Increases both bias and variance
   d) Decreases both bias and variance

3. Learning curves show:
   a) How model parameters change during training
   b) How error metrics change as model complexity increases
   c) How error metrics change as training set size changes
   d) How error metrics change across different random initializations

4. If a model has a training error of 0.1 and a test error of 0.5, the most likely problem is:
   a) Underfitting
   b) Overfitting
   c) Insufficient feature engineering
   d) Non-linearity in the data

5. Which of the following is NOT an advantage of cross-validation over the simple hold-out method?
   a) More efficient use of available data
   b) More reliable estimate of model performance
   c) Less sensitive to how the data is split
   d) Computationally less expensive

For a detailed explanation of these multiple-choice questions, see [Question 16: Model Evaluation Multiple Choice](L3_6_16_explanation.md).

## Question 17

### Problem Statement
Match each concept in Column A with the most appropriate description in Column B.

#### Task
Match the items in Column A with the correct item in Column B.

**Column A:**
1. Residual Analysis
2. AIC (Akaike Information Criterion)
3. BIC (Bayesian Information Criterion)
4. Learning Curves
5. Cross-Validation

**Column B:**
a) A measure that balances model fit and complexity with a stronger penalty for complexity than AIC
b) A technique for visualizing model performance as training set size increases
c) A statistical approach for examining the differences between predicted and actual values
d) A method that splits the data into multiple subsets to validate model performance
e) A measure that balances model fit and complexity with a penalty term of $2p$, where $p$ is the number of parameters

For a detailed explanation of this matching exercise, see [Question 17: Matching Evaluation Concepts](L3_6_17_explanation.md).

## Question 18

### Problem Statement
Analyze the following scenario and answer the questions.

You are building a linear regression model to predict house prices. After training your model, you notice that:
- The training error is very low
- The test error is much higher than the training error
- The residuals exhibit a clear pattern when plotted against the predicted values
- The learning curve shows that validation error initially decreases with more training data but then plateaus

#### Task
1. [📚] What problem(s) is your model likely facing? Be specific.
2. [📚] Describe TWO specific strategies you could use to address the identified problems.
3. [📚] Which evaluation technique would be most appropriate to assess if your strategies have improved the model? Explain your choice.
4. [📚] If you had to choose between collecting more training data or adding more features to your model in this scenario, which would you recommend? Justify your answer.

For a detailed explanation of this scenario-based question, see [Question 18: Regression Model Diagnostics](L3_6_18_explanation.md).

## Question 19

### Problem Statement
Consider how expected training error and expected test error behave as a function of the number of training examples for models of different complexity.

#### Task
1. [📚] Sketch by hand the expected training error curve and expected test error curve as functions of the number of training examples $N$ for:
   a. A simple model with high bias
   b. A complex model with high variance
2. [📚] Explain why the training error curve and test error curve converge as $N$ increases, and identify what value they converge to
3. [📚] Why does the training error typically increase with more training samples while the test error decreases?
4. [📚] Which model (simple or complex) requires more training data to achieve good generalization, and why?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 19: Error Curves and Sample Size](L3_6_19_explanation.md).

## Question 20

### Problem Statement
In linear regression, we can decompose the error into different components to better understand model performance.

#### Task
1. [📚] Define the structural error and approximation error in linear regression, and explain what each represents
2. [📚] Write down the mathematical expressions for both types of errors in terms of the optimal parameters $\boldsymbol{w}^*$ (with infinite data) and the estimated parameters $\hat{\boldsymbol{w}}$ (with finite data)
3. [📚] Prove that the expected error can be decomposed as:
   $$E_{\boldsymbol{x},y}[(y - \hat{\boldsymbol{w}}^T \boldsymbol{x})^2] = E_{\boldsymbol{x},y}[(y - \boldsymbol{w}^{*T} \boldsymbol{x})^2] + E_{\boldsymbol{x}}[(\boldsymbol{w}^{*T} \boldsymbol{x} - \hat{\boldsymbol{w}}^T \boldsymbol{x})^2]$$
4. [📚] Explain how increasing the number of training examples affects each error component
5. [📚] Describe how model complexity (e.g., polynomial degree) affects each error component

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 20: Error Decomposition in Linear Regression](L3_6_20_explanation.md). 