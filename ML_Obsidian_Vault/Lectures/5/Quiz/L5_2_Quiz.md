# Lecture 5.2: Hard Margin and Soft Margin SVMs Quiz

## Overview
This quiz contains 14 questions covering different topics from section 5.2 of the lectures on Hard Margin SVM, Soft Margin SVM, Slack Variables, Regularization Trade-offs, Hinge Loss Function, and KKT Conditions.

## Question 1

### Problem Statement
Consider a 2D dataset with outliers:
- Class $+1$: $(3, 2)$, $(4, 3)$, $(5, 2)$, $(1, 4)$ (potential outlier)
- Class $-1$: $(0, 0)$, $(1, 1)$, $(0, 2)$

#### Task
1. Plot the data and determine if it's linearly separable
2. Explain why hard margin SVM would fail on this dataset
3. Calculate the minimum number of constraint violations needed to make the data separable
4. Design a soft margin SVM formulation that handles the outlier appropriately
5. What would be the effect of removing the outlier $(1, 4)$ on the hard margin solution?

For a detailed explanation of this problem, see [Question 1: Hard Margin Limitations](L5_2_1_explanation.md).

## Question 2

### Problem Statement
Analyze the soft margin SVM optimization problem:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^n \xi_i$$
$$\text{subject to: } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

#### Task
1. Derive this formulation from the hard margin case by introducing slack variables
2. What is the geometric interpretation of each slack variable $\xi_i$?
3. Prove that the constraint $\xi_i \geq 0$ is necessary for the formulation to make sense
4. Show that in the optimal solution, $\xi_i = \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))$
5. What happens to the problem when $C \to \infty$?

For a detailed explanation of this problem, see [Question 2: Soft Margin Formulation](L5_2_2_explanation.md).

## Question 3

### Problem Statement
Given a dataset with the following training points and their optimal slack variables:
- $(\mathbf{x}_1, y_1) = ((2, 1), +1)$, $\xi_1 = 0$
- $(\mathbf{x}_2, y_2) = ((1, 2), +1)$, $\xi_2 = 0.3$
- $(\mathbf{x}_3, y_3) = ((0, 0), -1)$, $\xi_3 = 0$
- $(\mathbf{x}_4, y_4) = ((1, 0), -1)$, $\xi_4 = 1.2$

#### Task
1. Interpret what each slack variable value means geometrically
2. Which points are correctly classified and which are misclassified?
3. Which points lie within the margin, on the margin, or outside the margin?
4. If the hyperplane is $x_1 + x_2 - 1.5 = 0$, verify the slack variable values
5. Calculate the total penalty $\sum_{i=1}^4 \xi_i$ contributed to the objective function

For a detailed explanation of this problem, see [Question 3: Slack Variable Analysis](L5_2_3_explanation.md).

## Question 4

### Problem Statement
Analyze the effect of the regularization parameter $C$ on soft margin SVM behavior.

#### Task
1. For $C = 0.1, 1, 10, 100$, predict the qualitative behavior of the classifier
2. Derive the relationship between $C$ and the bias-variance tradeoff
3. As $C$ increases, how does the number of support vectors typically change?
4. Design an experiment to find the optimal $C$ using validation curves
5. Prove that the soft margin SVM solution approaches the hard margin solution as $C \to \infty$

For a detailed explanation of this problem, see [Question 4: Regularization Parameter Analysis](L5_2_4_explanation.md).

## Question 5

### Problem Statement
Consider the hinge loss function: $L_h(y, f(x)) = \max(0, 1 - y \cdot f(x))$ where $f(x) = \mathbf{w}^T\mathbf{x} + b$.

#### Task
1. Calculate the hinge loss for the following predictions:
   - $y = +1$, $f(x) = 2.5$
   - $y = +1$, $f(x) = 0.8$
   - $y = +1$, $f(x) = -0.3$
   - $y = -1$, $f(x) = -1.7$
   - $y = -1$, $f(x) = 0.4$
2. Plot the hinge loss as a function of $y \cdot f(x)$
3. Show that $\xi_i = L_h(y_i, f(\mathbf{x}_i))$ in the soft margin formulation
4. Compare the derivative properties of hinge loss vs squared loss
5. Prove that hinge loss upper bounds the 0-1 loss

For a detailed explanation of this problem, see [Question 5: Hinge Loss Analysis](L5_2_5_explanation.md).

## Question 6

### Problem Statement
Compare different loss functions for classification:
- 0-1 Loss: $L_{01}(y, f(x)) = \mathbb{I}[y \cdot f(x) \leq 0]$
- Hinge Loss: $L_h(y, f(x)) = \max(0, 1 - y \cdot f(x))$
- Logistic Loss: $L_{\ell}(y, f(x)) = \log(1 + e^{-y \cdot f(x)})$
- Squared Loss: $L_s(y, f(x)) = (y - f(x))^2$

#### Task
1. Plot all four loss functions on the same graph for $y = +1$ and $f(x) \in [-3, 3]$
2. Calculate the loss values for each function when $y = +1$ and $f(x) = 0.5$
3. Which losses are convex and which are not? Prove your answers
4. Which loss is most robust to outliers and why?
5. Derive the gradients of each loss function with respect to $f(x)$

For a detailed explanation of this problem, see [Question 6: Loss Function Comparison](L5_2_6_explanation.md).

## Question 7

### Problem Statement
Derive and analyze the KKT conditions for soft margin SVM.

The Lagrangian is:
$$L = \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^n \xi_i - \sum_{i=1}^n \alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 + \xi_i] - \sum_{i=1}^n \mu_i \xi_i$$

#### Task
1. Write out all KKT stationarity conditions
2. Derive the constraint $\sum_{i=1}^n \alpha_i y_i = 0$
3. Show that $\alpha_i + \mu_i = C$ for all $i$
4. Prove that $0 \leq \alpha_i \leq C$ for all $i$
5. Classify training points based on their $\alpha_i$ and $\xi_i$ values

For a detailed explanation of this problem, see [Question 7: KKT Conditions Derivation](L5_2_7_explanation.md).

## Question 8

### Problem Statement
Categorize support vectors in soft margin SVM based on KKT conditions.

#### Task
1. For $\alpha_i = 0$, what can you conclude about the point $\mathbf{x}_i$?
2. For $0 < \alpha_i < C$, derive the conditions on $\xi_i$ and the point's position
3. For $\alpha_i = C$, what are the possible scenarios for $\xi_i$?
4. Create a decision tree for classifying points based on $(\alpha_i, \xi_i)$ values
5. Given the following points, classify them:
   - Point A: $\alpha_A = 0$, $\xi_A = 0$
   - Point B: $\alpha_B = 0.5C$, $\xi_B = 0$
   - Point C: $\alpha_C = C$, $\xi_C = 0.8$
   - Point D: $\alpha_D = C$, $\xi_D = 1.5$

For a detailed explanation of this problem, see [Question 8: Support Vector Classification](L5_2_8_explanation.md).

## Question 9

### Problem Statement
Solve a small soft margin SVM problem analytically.

Consider the dataset:
- $\mathbf{x}_1 = (1, 0)$, $y_1 = +1$
- $\mathbf{x}_2 = (0, 1)$, $y_2 = +1$
- $\mathbf{x}_3 = (-1, 0)$, $y_3 = -1$
- $\mathbf{x}_4 = (0, -1)$, $y_4 = -1$
- $\mathbf{x}_5 = (0.1, 0.1)$, $y_5 = -1$ (outlier)

#### Task
1. Set up the dual optimization problem for soft margin SVM with $C = 1$
2. Solve for the optimal Lagrange multipliers $\alpha_i$
3. Calculate the optimal weight vector $\mathbf{w}^*$ and bias $b^*$
4. Determine the slack variable $\xi_5$ for the outlier point
5. Verify that your solution satisfies all KKT conditions

For a detailed explanation of this problem, see [Question 9: Analytical Solution Example](L5_2_9_explanation.md).

## Question 10

### Problem Statement
Design experiments to understand the bias-variance tradeoff in soft margin SVMs.

#### Task
1. Describe how you would generate synthetic datasets to study the effect of $C$
2. Design a cross-validation scheme to select the optimal $C$ value
3. Predict how training error and validation error curves will look as functions of $C$
4. What metrics would you use to evaluate the bias and variance components?
5. How would noise level in the data affect the optimal choice of $C$?

For a detailed explanation of this problem, see [Question 10: Experimental Design](L5_2_10_explanation.md).

## Question 11

### Problem Statement
Analyze the computational complexity differences between hard and soft margin SVMs.

#### Task
1. Compare the number of optimization variables in hard vs soft margin formulations
2. How does the addition of slack variables affect the QP problem structure?
3. What is the worst-case time complexity for solving soft margin SVM?
4. How does the choice of $C$ affect convergence properties of optimization algorithms?
5. Estimate the memory requirements for storing the extended problem formulation

For a detailed explanation of this problem, see [Question 11: Computational Complexity](L5_2_11_explanation.md).

## Question 12

### Problem Statement
Investigate the relationship between soft margin SVM and regularized empirical risk minimization.

#### Task
1. Show that soft margin SVM can be written as: $\min_{\mathbf{w}, b} \frac{\lambda}{2}||\mathbf{w}||^2 + \frac{1}{n}\sum_{i=1}^n L_h(y_i, \mathbf{w}^T\mathbf{x}_i + b)$
2. What is the relationship between $\lambda$ and $C$?
3. Compare this to Ridge regression formulation
4. How does this connect SVM to the general framework of regularized learning?
5. What other loss functions could be substituted while maintaining convexity?

For a detailed explanation of this problem, see [Question 12: Regularized Risk Minimization](L5_2_12_explanation.md).

## Question 13

### Problem Statement
Study the geometric interpretation of the soft margin in 2D.

#### Task
1. For the hyperplane $x_1 + 2x_2 - 3 = 0$, draw the margin boundaries
2. Sketch points with different slack variable values: $\xi = 0, 0.5, 1.0, 1.5$
3. Show how the margin changes as $C$ varies from $0.1$ to $10$
4. Illustrate the effect of adding an outlier point on the decision boundary
5. Compare the margins achieved with $C = 1$ vs $C = 100$ for the same dataset

For a detailed explanation of this problem, see [Question 13: Geometric Interpretation](L5_2_13_explanation.md).

## Question 14

### Problem Statement
Practical implementation considerations for soft margin SVMs.

#### Task
1. How would you handle the case where all points are outliers (all $\xi_i > 0$)?
2. What preprocessing steps are essential for soft margin SVM?
3. Design a grid search strategy for finding optimal $C$ values
4. How would you detect when $C$ is too small or too large from the solution characteristics?
5. What stopping criteria would you use for iterative optimization algorithms?

For a detailed explanation of this problem, see [Question 14: Implementation Considerations](L5_2_14_explanation.md).