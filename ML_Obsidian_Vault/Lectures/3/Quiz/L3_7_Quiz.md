# Lecture 3.7: Regularization in Linear Models Quiz

## Overview
This quiz contains 22 questions covering different topics from section 3.7 of the lectures on Regularization in Linear Models, including bias-variance tradeoff, ridge regression, lasso regression, elastic net, and methods for selecting regularization parameters.

## Question 1

### Problem Statement
Consider a 9th degree polynomial model fitting a dataset with just 10 data points. The model has extremely large coefficient values and fits the training data perfectly.

In this problem:
- The polynomial model is $$f(x) = w_0 + w_1x + w_2x^2 + \ldots + w_9x^9$$
- The coefficients have very large magnitudes (e.g., $w_5 = 640042.26$, $w_9 = 125201.43$)
- The training error is nearly zero

#### Task
1. Explain what phenomenon this model is likely experiencing
2. Describe how this problem would affect the model's performance on new, unseen data
3. Explain the relationship between the large coefficient values and the model's generalization ability
4. Suggest a technique to address this issue without changing the degree of the polynomial

For a detailed explanation of this problem, including step-by-step analysis, see [Question 1: Detecting Overfitting in Polynomial Models](L3_7_1_explanation.md).

## Question 2

### Problem Statement
The bias-variance tradeoff is a fundamental concept in machine learning. Consider a linear model fitting data generated from a sine function.

In this problem:
- A linear model without regularization has bias = 0.21, variance = 1.69
- A linear model with regularization has bias = 0.23, variance = 0.33

#### Task
1. Calculate the total expected error (bias$^2$ + variance) for both models
2. Explain why regularization increased the bias but decreased the variance
3. Describe the geometric interpretation of the regularization effect on the parameter space
4. Explain why, despite having higher bias, the regularized model has better overall performance

For a detailed explanation of this problem, including the mathematical analysis of bias-variance tradeoff, see [Question 2: Bias-Variance Tradeoff Analysis](L3_7_2_explanation.md).

## Question 3

### Problem Statement
In ridge regression (L2 regularization), we modify the cost function by adding a penalty term:

$$J_{\text{ridge}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_2^2$$

#### Task
1. Derive the closed-form solution for the ridge regression weights
2. Explain why ridge regression always has a unique solution even when $\boldsymbol{X}^T\boldsymbol{X}$ is not invertible
3. Describe how the regularization parameter $\lambda$ affects the solution as it approaches:
   a. $\lambda \rightarrow 0$
   b. $\lambda \rightarrow \infty$
4. Explain why the regularization effect is stronger for directions with smaller eigenvalues

For a detailed explanation of this problem, including the full mathematical derivation, see [Question 3: Ridge Regression Mathematics](L3_7_3_explanation.md).

## Question 4

### Problem Statement
Consider a polynomial regression model with degree 5 fitted to a dataset. The unregularized model has the following coefficients:
$w_0 = 1.2$, $w_1 = 15.7$, $w_2 = -45.3$, $w_3 = 86.9$, $w_4 = -67.2$, $w_5 = 21.5$

When L2 regularization is applied with parameter $\lambda = 10$, the coefficients change to:
$w_0 = 1.1$, $w_1 = 5.2$, $w_2 = -8.4$, $w_3 = 12.1$, $w_4 = -7.5$, $w_5 = 2.3$

#### Task
1. Calculate the L2 norm of the coefficient vector (excluding $w_0$) for both models
2. Explain why the intercept term $w_0$ is typically not regularized
3. Describe how the coefficients change as a result of regularization and why
4. Sketch how you would expect the fitted curves to differ between the two models

For a detailed explanation of this problem, including coefficient analysis, see [Question 4: Polynomial Coefficient Analysis](L3_7_4_explanation.md).

## Question 5

### Problem Statement
Lasso regression (L1 regularization) uses the following cost function:

$$J_{\text{lasso}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_1$$

#### Task
1. Explain why lasso regression tends to produce sparse solutions (many weights exactly zero)
2. Describe the geometric intuition behind why L1 regularization leads to sparsity
3. Compare the effect of lasso regularization to ridge regularization when features are correlated
4. Explain a practical scenario where lasso regression would be preferred over ridge regression

For a detailed explanation of this problem, including geometric interpretation, see [Question 5: Lasso Regression and Sparsity](L3_7_5_explanation.md).

## Question 6

### Problem Statement
Elastic Net combines both L1 and L2 penalties:

$$J_{\text{elastic}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda_1\|\boldsymbol{w}\|_1 + \lambda_2\|\boldsymbol{w}\|_2^2$$

#### Task
1. Describe the advantage of Elastic Net over Lasso when features are highly correlated
2. Explain how Elastic Net balances the properties of Ridge and Lasso regression
3. In what scenario would you prefer Elastic Net over both Ridge and Lasso?
4. Describe how you would approach selecting both regularization parameters $\lambda_1$ and $\lambda_2$

For a detailed explanation of this problem, including the benefits of Elastic Net, see [Question 6: Elastic Net Regularization](L3_7_6_explanation.md).

## Question 7

### Problem Statement
Consider the regularization path for a linear regression model with 5 features as the regularization parameter $\lambda$ varies from very large to very small values.

#### Task
1. Describe what a regularization path is and how it visualizes the effect of regularization
2. Explain the typical behavior of coefficient values in the regularization path for:
   a. Ridge regression
   b. Lasso regression
3. How would you use the regularization path to select an appropriate value of $\lambda$?
4. What does it mean when different features enter the model at different points along the lasso regularization path?

For a detailed explanation of this problem, including regularization path analysis, see [Question 7: Regularization Path Analysis](L3_7_7_explanation.md).

## Question 8

### Problem Statement
From a Bayesian perspective, regularization can be interpreted as imposing a prior distribution on the model parameters.

#### Task
1. Describe the prior distribution on weights that corresponds to L2 regularization
2. Describe the prior distribution on weights that corresponds to L1 regularization
3. Derive the connection between maximum a posteriori (MAP) estimation with a Gaussian prior and ridge regression
4. Explain how the regularization parameter $\lambda$ relates to the variance of the prior distribution

For a detailed explanation of this problem, including the Bayesian interpretation, see [Question 8: Bayesian Interpretation of Regularization](L3_7_8_explanation.md).

## Question 9

### Problem Statement
Cross-validation is commonly used to select the optimal regularization parameter $\lambda$.

#### Task
1. Describe the K-fold cross-validation procedure for selecting $\lambda$
2. Explain why using only training error to select $\lambda$ would lead to a poor choice
3. In a regularization context, describe what the validation curve typically looks like as $\lambda$ increases from very small to very large values
4. Describe a practical strategy for efficiently testing multiple values of $\lambda$

For a detailed explanation of this problem, including validation strategies, see [Question 9: Selecting Regularization Parameters](L3_7_9_explanation.md).

## Question 10

### Problem Statement
Early stopping in iterative optimization methods (like gradient descent) can be viewed as a form of implicit regularization.

#### Task
1. Explain how early stopping acts as a regularization technique
2. Compare early stopping to explicit regularization methods like Ridge and Lasso
3. Describe the relationship between the number of iterations and the effective model complexity
4. Explain how you would determine the optimal stopping point in practice

For a detailed explanation of this problem, including the regularization effect of early stopping, see [Question 10: Early Stopping as Regularization](L3_7_10_explanation.md).

## Question 11

### Problem Statement
Consider a simple linear regression model with two features. The unregularized solution is $w_1 = 5$ and $w_2 = 3$.

#### Task
1. Calculate the L1 norm and L2 norm of this weight vector
2. If we apply ridge regression with $\lambda = 0.5$, would the regularized weights have larger or smaller norms? Explain why.
3. If we apply lasso regression with a small $\lambda$, which coefficient might be reduced to zero first? Explain your reasoning.
4. Draw a simple diagram showing the L1 and L2 constraint regions in this 2D weight space

For a detailed explanation of this problem, including norm calculations, see [Question 11: Comparing L1 and L2 Norms](L3_7_11_explanation.md).

## Question 12

### Problem Statement
A colleague is selecting a regularization method for a machine learning task but is confused about which one to choose. They have a dataset with 100 features, but they suspect only about 10 features are truly relevant.

#### Task
1. Which regularization method would you recommend and why?
2. How would your recommendation change if instead all 100 features were somewhat relevant but highly correlated?
3. For your recommended method in the original scenario, explain how you would select the regularization parameter
4. Briefly explain the concept of "effective degrees of freedom" in regularized models

For a detailed explanation of this problem, including feature selection strategy, see [Question 12: Practical Regularization Selection](L3_7_12_explanation.md).

## Question 13

### Problem Statement
Consider the following regularized cost functions with parameter $\lambda = 2$:

$$J_A(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - \boldsymbol{w}^T \boldsymbol{x}^{(i)})^2 + 2\|\boldsymbol{w}\|_2^2$$

$$J_B(\boldsymbol{w}) = \sum_{i=1}^{n} (y^{(i)} - \boldsymbol{w}^T \boldsymbol{x}^{(i)})^2 + 2\|\boldsymbol{w}\|_1$$

#### Task
1. [📚] If we have a weight vector $\boldsymbol{w} = [0.5, -1.5, 2.0]$, calculate the penalty term for both Models A and B
2. [🔍] Which model would likely produce more zero coefficients and why?
3. [📚] Describe one advantage of Model A over Model B
4. [📚] If we double $\lambda$ to 4, how would the penalty terms change for both models?

For a detailed explanation of this problem, including penalty calculations, see [Question 13: Comparing Regularization Penalties](L3_7_13_explanation.md).

## Question 14

### Problem Statement
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

#### Task
1. Increasing the regularization parameter always increases the bias
2. An unregularized model will always have lower training error than a regularized version of the same model
3. If two models have the same training error, the one with smaller coefficient magnitudes will likely generalize better
4. Lasso regression typically produces more sparse models than Ridge regression with the same $\lambda$ value

For a detailed explanation of this problem, including analysis of each statement, see [Question 14: Regularization True/False](L3_7_14_explanation.md).

## Question 15

### Problem Statement
A data scientist has trained a linear regression model on housing data. When plotting the model's predictions vs. actual values, they notice that predictions vary wildly for similar houses, suggesting overfitting.

#### Task
1. Draw a simple sketch illustrating how an overfitted model might look compared to a well-regularized model on this housing data
2. List three specific symptoms of overfitting that might be observed in this housing price prediction model
3. If the model has 50 features, explain how you would apply regularization to address the overfitting
4. Describe a simple approach to determine if regularization has successfully addressed the overfitting problem

For a detailed explanation of this problem, including visualization concepts, see [Question 15: Visualizing Regularization Effects](L3_7_15_explanation.md).

## Question 16

### Problem Statement
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

#### Task
1. Ridge regression and Lasso regression will produce identical models when applied to the same dataset
2. As the regularization parameter $\lambda$ approaches infinity in ridge regression, all coefficient values will approach zero
3. The primary goal of regularization is to improve model performance on the training set
4. Both L1 and L2 regularization penalize large coefficient values, but in different ways
5. Early stopping in gradient descent prevents the model from reaching the global minimum of the cost function

For a detailed explanation of these statements, see [Question 16: Regularization Theory](L3_7_16_explanation.md).

## Question 17

### Problem Statement
Multiple Choice Questions on Regularization in Linear Models.

#### Task
Select the best answer for each question:

1. Which of the following is NOT a common method of regularization in linear regression?
   a) L1 norm penalty (Lasso)
   b) L2 norm penalty (Ridge)
   c) L0 norm penalty
   d) Elastic Net
   
2. As the regularization parameter in ridge regression increases, what happens to the model?
   a) Bias decreases, variance increases
   b) Bias increases, variance decreases
   c) Both bias and variance increase
   d) Both bias and variance decrease
   
3. Which regularization method is most likely to produce exactly zero coefficients?
   a) Ridge regression
   b) Lasso regression
   c) Both produce the same number of zero coefficients
   d) Neither produces exactly zero coefficients
   
4. From a Bayesian perspective, ridge regression can be interpreted as imposing what type of prior on the model parameters?
   a) Uniform prior
   b) Laplace prior
   c) Gaussian prior
   d) Cauchy prior

For detailed explanations of these questions, see [Question 17: Multiple Choice on Regularization](L3_7_17_explanation.md).

## Question 18

### Problem Statement
Match each concept in Column A with the most appropriate description in Column B.

#### Task
Match the items in Column A with the correct description in Column B:

**Column A:**
1. Elastic Net
2. Regularization Path
3. Early Stopping
4. Cross-Validation
5. Ridge Regression

**Column B:**
a) A visualization showing how coefficient values change as the regularization parameter varies
b) A technique that combines L1 and L2 penalties to get the best of both approaches
c) A method that adds the sum of squared weights to the cost function
d) A form of implicit regularization where iteration is halted before convergence
e) A technique to select the optimal regularization parameter by estimating model performance on unseen data

For the correct matches and explanations, see [Question 18: Matching Regularization Concepts](L3_7_18_explanation.md).

## Question 19

### Problem Statement
Fill in the blanks with the appropriate terms related to regularization in linear models.

#### Task
Fill in each blank with the most appropriate term:

1. In ridge regression, the penalty term is proportional to the $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ norm of the weight vector.
2. The $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ regularization method is more likely to produce sparse solutions with many coefficients exactly zero.
3. From a Bayesian perspective, L1 regularization corresponds to a $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ prior on the model weights.
4. As the regularization parameter increases, the $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ of the model typically increases while the $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ decreases.
5. The $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ shows how coefficient values change as the regularization parameter varies.
6. $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ combines the penalty terms from both ridge and lasso regression.

For the correct answers and explanations, see [Question 19: Fill in the Blanks](L3_7_19_explanation.md).

## Question 20

### Problem Statement
Short Answer Questions on Regularization Methods.

#### Task
Provide brief answers (1-3 sentences) to each of the following questions:

1. [📚] Why might you choose Elastic Net over pure Lasso or Ridge regression?
2. [📚] How does early stopping in gradient descent function as a form of regularization?
3. [📚] What is the relationship between the regularization parameter and the variance of the prior distribution in the Bayesian interpretation?
4. [📚] Why does L1 regularization (Lasso) tend to produce sparse coefficients while L2 regularization (Ridge) does not?
5. [📚] How would you use cross-validation to select the optimal regularization parameter?

For detailed answers to these questions, see [Question 20: Short Answer Questions](L3_7_20_explanation.md).

## Question 21

### Problem Statement
Consider a linear regression problem where you suspect many features might be irrelevant, and some of the relevant features are highly correlated with each other. You're deciding between three regularization approaches:

1. Ridge Regression: $J_{\text{ridge}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_2^2$
2. Lasso Regression: $J_{\text{lasso}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_1$
3. Elastic Net: $J_{\text{elastic}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda_1\|\boldsymbol{w}\|_1 + \lambda_2\|\boldsymbol{w}\|_2^2$

#### Task
1. [📚] Compare and contrast how these three regularization methods would handle irrelevant features
2. [📚] Explain which method would be most effective for dealing with the highly correlated features and why
3. [📚] For a dataset with 1000 features where only about 100 are relevant, which method would likely produce the most interpretable model?
4. [📚] Draw a simple 2D diagram showing the constraint regions imposed by L1 and L2 regularization, and explain geometrically why L1 regularization promotes sparsity
5. [📚] If computational efficiency is a concern, which method might present the most challenges and why?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 21: Comparing Regularization Methods](L3_7_21_explanation.md).

## Question 22

### Problem Statement
In ridge regression, we modify the standard linear regression cost function by adding a penalty term proportional to the squared norm of the weights. Consider the ridge regression cost function:

$$J_{\text{ridge}}(\boldsymbol{w}) = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\|\boldsymbol{w}\|_2^2$$

#### Task
1. [📚] Write down the closed-form solution for the ridge regression parameters $\boldsymbol{w}_{\text{ridge}}$
2. [📚] Explain why the matrix $(\boldsymbol{X}^T\boldsymbol{X} + \lambda\boldsymbol{I})$ is always invertible, even when $\boldsymbol{X}^T\boldsymbol{X}$ is not
3. [📚] For a dataset with highly correlated features, describe mathematically how ridge regression helps address the multicollinearity problem
4. [📚] As $\lambda \rightarrow \infty$, what happens to the ridge regression parameters $\boldsymbol{w}_{\text{ridge}}$? Provide a mathematical explanation for this behavior
5. [📚] In terms of eigenvalues and eigenvectors of $\boldsymbol{X}^T\boldsymbol{X}$, explain why ridge regression has a stronger effect on directions with smaller eigenvalues

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 22: Ridge Regression Mathematics](L3_7_22_explanation.md). 