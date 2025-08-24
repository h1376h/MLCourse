# Lecture 5.5: SVM Regression Quiz

## Overview
This quiz contains 20 questions covering different topics from section 5.5 of the lectures on Support Vector Regression (SVR), ε-insensitive Loss, Linear and Nonlinear SVR, ν-SVR, and SVR Implementation.

## Question 1

### Problem Statement
Consider the fundamental differences between SVM classification and Support Vector Regression (SVR).

#### Task
1. In classification, we maximize the margin between classes. What is the analogous concept in SVR?
2. Instead of discrete class labels, SVR predicts continuous values. How does this change the loss function?
3. Define support vectors in the context of regression
4. For the regression function $f(x) = \mathbf{w}^T\mathbf{x} + b$, what does the ε-tube represent geometrically?
5. Compare the decision boundary concept in classification vs the regression function in SVR
6. Design a temperature forecasting system using humidity and pressure readings with ±2°C accuracy tolerance. The system should ignore small prediction errors and be robust to sensor malfunctions. Design a prediction tube system, confidence scoring, and outlier detection. If using 3 sensors, how do you combine readings for better predictions?

For a detailed explanation of this problem, see [Question 1: SVR Fundamentals](L5_5_1_explanation.md).

## Question 2

### Problem Statement
Analyze the ε-insensitive loss function: $L_ε(y, f(x)) = \max(0, |y - f(x)| - ε)$.

Given the following predictions and true values with $ε = 0.3$:
- $(y_1, f(x_1)) = (2.5, 2.8)$
- $(y_2, f(x_2)) = (1.0, 1.5)$
- $(y_3, f(x_3)) = (3.2, 2.7)$
- $(y_4, f(x_4)) = (0.8, 0.9)$
- $(y_5, f(x_5)) = (4.1, 3.5)$

#### Task
1. Calculate the ε-insensitive loss for each prediction
2. Which points lie within the ε-tube?
3. Compare these losses to squared loss: $(y - f(x))^2$
4. Sketch the ε-insensitive loss function for $ε = 0.3$ (you can draw this by hand)
5. Show that the ε-insensitive loss is more robust to outliers than squared loss
6. Design a prediction system for daily closing prices using a $3.00 tolerance for acceptable errors. Actual prices: $25.00, $10.00, $32.00, $8.00, $41.00; Predicted prices: $28.00, $15.00, $27.00, $9.00, $35.00. Design a profit scoring system, calculate prediction accuracy, and create a risk assessment system. Determine optimal tolerance for maximizing profits while minimizing risk.

For a detailed explanation of this problem, see [Question 2: ε-insensitive Loss Analysis](L5_5_2_explanation.md).

## Question 3

### Problem Statement
Derive and analyze the SVR optimization problem:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^n (\xi_i + \xi_i^*)$$
$$\text{subject to: } y_i - \mathbf{w}^T\mathbf{x}_i - b \leq ε + \xi_i$$
$$\mathbf{w}^T\mathbf{x}_i + b - y_i \leq ε + \xi_i^*$$
$$\xi_i, \xi_i^* \geq 0$$

#### Task
1. Explain why we need both $\xi_i$ and $\xi_i^*$ (upper and lower slack variables)
2. What do the constraints represent geometrically in terms of the ε-tube?
3. Prove that at most one of $\xi_i$ or $\xi_i^*$ can be non-zero for any training point
4. Write the Lagrangian for this optimization problem
5. Derive the KKT conditions for optimality

For a detailed explanation of this problem, see [Question 3: SVR Optimization Formulation](L5_5_3_explanation.md).

## Question 4

### Problem Statement
Solve a small SVR problem analytically.

Consider the 1D dataset:
- $(x_1, y_1) = (1, 2)$
- $(x_2, y_2) = (2, 3)$
- $(x_3, y_3) = (3, 5)$ (potential outlier)

With $ε = 0.5$ and $C = 1$.

#### Task
1. Set up the primal optimization problem
2. Identify which points will be support vectors
3. For a linear model $f(x) = wx + b$, solve for the optimal $w$ and $b$
4. Calculate the slack variables $\xi_i$ and $\xi_i^*$ for each point
5. Verify that your solution satisfies all KKT conditions

For a detailed explanation of this problem, see [Question 4: SVR Analytical Solution](L5_5_4_explanation.md).

## Question 5

### Problem Statement
Categorize points in SVR based on their relationship to the ε-tube.

#### Task
1. For points inside the ε-tube: what are the values of $\xi_i$, $\xi_i^*$, $\alpha_i$, and $\alpha_i^*$?
2. For points exactly on the ε-tube boundary: what are the constraint conditions?
3. For points outside the ε-tube: which Lagrange multipliers are non-zero?
4. Create a decision tree for classifying points based on their $(α_i, α_i^*, ξ_i, ξ_i^*)$ values
5. How does the number of support vectors relate to the choice of $ε$?

For a detailed explanation of this problem, see [Question 5: Support Vector Classification in SVR](L5_5_5_explanation.md).

## Question 6

### Problem Statement
Analyze the effect of the ε parameter on SVR behavior.

#### Task
1. Derive the limit behavior as $ε \to 0$: what happens to the loss function and solution?
2. Derive the limit behavior as $ε \to \infty$: what happens to the optimization problem?
3. For a noisy dataset, predict how $ε$ affects bias vs variance tradeoff
4. Design an experiment to determine the optimal $ε$ value using cross-validation
5. Show that smaller $ε$ generally leads to more support vectors

For a detailed explanation of this problem, see [Question 6: ε Parameter Effects](L5_5_6_explanation.md).

## Question 7

### Problem Statement
Compare ε-SVR with ν-SVR formulation.

The ν-SVR optimization problem is:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*, ε} \frac{1}{2}||\mathbf{w}||^2 + C\left(ν ε + \frac{1}{n}\sum_{i=1}^n (\xi_i + \xi_i^*)\right)$$

#### Task
1. What is the key difference in parameterization between ε-SVR and ν-SVR?
2. What does the parameter $ν \in (0,1)$ control in ν-SVR?
3. How is $ε$ automatically determined in ν-SVR?
4. Prove that $ν$ provides an upper bound on the fraction of training errors
5. When would you prefer ν-SVR over ε-SVR in practice?

For a detailed explanation of this problem, see [Question 7: ν-SVR vs ε-SVR](L5_5_7_explanation.md).

## Question 8

### Problem Statement
Extend SVR to nonlinear regression using kernels.

#### Task
1. Show how the kernel trick applies to SVR: rewrite the decision function in terms of support vectors
2. For the RBF kernel $K(x, z) = \exp(-\gamma(x-z)^2)$, what happens to the regression function's flexibility as $\gamma$ increases?
3. Design a kernel appropriate for time series regression with seasonal patterns
4. How does kernel choice affect the interpretability of SVR models?
5. Compare the computational complexity of linear vs nonlinear SVR

For a detailed explanation of this problem, see [Question 8: Nonlinear SVR with Kernels](L5_5_8_explanation.md).

## Question 9

### Problem Statement
Compare SVR with other regression methods systematically.

#### Task
1. **vs Linear Regression**: Compare robustness to outliers, assumptions, and computational complexity
2. **vs Ridge Regression**: How do the regularization mechanisms differ?
3. **vs LASSO**: Compare feature selection capabilities and sparsity
4. **vs Neural Networks**: Compare expressiveness, interpretability, and training complexity
5. Design experiments to empirically compare these methods on synthetic and real datasets

For a detailed explanation of this problem, see [Question 9: SVR vs Other Regression Methods](L5_5_9_explanation.md).

## Question 10

### Problem Statement
Design a comprehensive hyperparameter tuning strategy for SVR.

#### Task
1. For RBF kernel SVR, what hyperparameters need to be tuned simultaneously?
2. Design a nested cross-validation scheme for unbiased hyperparameter selection
3. What search ranges would you use for $C$, $ε$, and $\gamma$ parameters?
4. Design early stopping criteria for expensive parameter searches
5. How would you detect overfitting vs underfitting during the tuning process?

For a detailed explanation of this problem, see [Question 10: SVR Hyperparameter Tuning](L5_5_10_explanation.md).

## Question 11

### Problem Statement
Implement robust SVR for datasets with different types of noise and outliers.

#### Task
1. How does SVR handle Gaussian noise vs heavy-tailed noise distributions?
2. Design a preprocessing pipeline to detect and handle outliers before SVR training
3. Compare the robustness of different ε values for datasets with varying noise levels
4. Design a robust scaling strategy that's less sensitive to outliers
5. Design an adaptive ε selection method based on data characteristics

For a detailed explanation of this problem, see [Question 11: Robust SVR Implementation](L5_5_11_explanation.md).

## Question 12

### Problem Statement
Develop SVR for multi-output regression problems.

#### Task
1. Extend the SVR formulation to predict multiple correlated outputs simultaneously
2. How would you share information between different output dimensions?
3. Design appropriate kernel functions for multi-output scenarios
4. Compare the computational complexity of independent SVRs vs joint multi-output SVR
5. Design a practical algorithm for multi-output SVR with RBF kernels

For a detailed explanation of this problem, see [Question 12: Multi-output SVR](L5_5_12_explanation.md).

## Question 13

### Problem Statement
Apply SVR to time series forecasting and sequential data.

#### Task
1. How would you transform a time series forecasting problem into a supervised regression problem for SVR?
2. Design appropriate feature engineering techniques for temporal data
3. How would you handle non-stationarity in time series using SVR?
4. Design a sliding window approach for online SVR learning
5. Compare SVR with ARIMA and LSTM for time series prediction

For a detailed explanation of this problem, see [Question 13: SVR for Time Series](L5_5_13_explanation.md).

## Question 14

### Problem Statement
Develop practical implementation guidelines and best practices for SVR.

#### Task
1. Create a comprehensive preprocessing checklist for SVR (scaling, outlier detection, feature selection)
2. Design model validation strategies specific to regression problems
3. How would you explain SVR predictions to non-technical stakeholders?
4. Design uncertainty quantification for SVR predictions
5. Design a production pipeline for real-time SVR inference with performance monitoring

For a detailed explanation of this problem, see [Question 14: SVR Best Practices](L5_5_14_explanation.md).

## Question 15

### Problem Statement
For the regression function $f(x) = 0.5x + 1$ and ε = 0.3, calculate losses for data points:
- $(1, 2.1)$, $(2, 1.8)$, $(3, 2.2)$, $(4, 3.5)$, $(5, 2.9)$

#### Task
1. Calculate $f(x_i)$ for each input $x_i$
2. Compute $L_ε(y_i, f(x_i)) = \max(0, |y_i - f(x_i)| - ε)$ for each point
3. Calculate total ε-insensitive loss $\sum_i L_ε(y_i, f(x_i))$
4. Compare with squared loss $\sum_i (y_i - f(x_i))^2$
5. Identify which points would be support vectors (contribute to loss)

For a detailed explanation of this problem, see [Question 15: ε-Loss Calculations](L5_5_15_explanation.md).

## Question 16

### Problem Statement
Consider the SVR problem with data points:
- $(1, 2.5)$, $(2, 3.2)$, $(3, 4.1)$, $(4, 4.8)$

Using linear SVR with $ε = 0.2$ and $C = 1$.

#### Task
1. Write the complete primal optimization problem with slack variables
2. List all constraints for the 4 data points
3. For solution $f(x) = 0.8x + 1.5$, calculate all slack variables $\xi_i$ and $\xi_i^*$
4. Compute the total objective function value
5. Identify which points are support vectors based on slack values

For a detailed explanation of this problem, see [Question 16: SVR Problem Setup](L5_5_16_explanation.md).

## Question 17

### Problem Statement
Compare ε-SVR and ν-SVR on the same dataset.

Given ν-SVR with $ν = 0.4$ and $C = 2$ vs ε-SVR with $ε = 0.3$ and $C = 2$.

#### Task
1. What does $ν = 0.4$ mean in terms of support vectors and errors?
2. What can you say about the fraction of support vectors?
3. How is ε automatically determined in ν-SVR?
4. Derive the relationship between ν and the effective ε
5. When would you prefer ν-SVR over ε-SVR?

For a detailed explanation of this problem, see [Question 17: ν-SVR Analysis](L5_5_17_explanation.md).

## Question 18

### Problem Statement
Apply RBF kernel SVR to nonlinear regression data.

Data points: $(0, 1)$, $(1, 1.5)$, $(2, 4)$, $(3, 4.2)$, $(4, 2)$

#### Task
1. Compute the $5 \times 5$ kernel matrix with $\gamma = 0.5$
2. Explain why this kernel can model nonlinear relationships
3. Write the SVR prediction function in terms of support vectors
4. Given $\alpha_1 = 0.3, \alpha_2 = 0, \alpha_3 = 0.7, \alpha_4 = 0.2, \alpha_5 = 0$, calculate prediction for $x = 2.5$
5. Predict how increasing $\gamma$ would affect the prediction function

For a detailed explanation of this problem, see [Question 18: Kernel SVR Application](L5_5_18_explanation.md).

## Question 19

### Problem Statement
Compare performance on a dataset with outliers.

Dataset: $(1, 2)$, $(2, 4)$, $(3, 6)$, $(4, 8)$, $(5, 100)$ ← outlier

#### Task
1. Fit $y = ax + b$ using least squares and calculate MSE
2. Apply SVR with $ε = 1$ and predict the fitted line
3. Calculate how much the outlier affects each method's solution
4. Quantify robustness by comparing solutions with and without the outlier
5. Discuss the bias-variance trade-off for each method

For a detailed explanation of this problem, see [Question 19: Robustness Comparison](L5_5_19_explanation.md).

## Question 20

### Problem Statement
Study how SVR parameters affect performance.

#### Task
1. For fixed $ε = 0.1$, predict performance changes as $C: 0.1 \to 1 \to 10$
2. For fixed $C = 1$, predict changes as $ε: 0.01 \to 0.1 \to 1$
3. Sketch expected validation error vs. $C$ and vs. $ε$ (you can draw this by hand)
4. Design a 2D grid search strategy for $(C, ε)$ optimization
5. Describe how to detect overfitting in each parameter dimension

For a detailed explanation of this problem, see [Question 20: Parameter Sensitivity](L5_5_20_explanation.md).