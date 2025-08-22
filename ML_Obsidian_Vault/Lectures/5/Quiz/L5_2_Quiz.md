# Lecture 5.2: Hard Margin and Soft Margin SVMs Quiz

## Overview
This quiz contains 10 questions covering different topics from section 5.2 of the lectures on Hard Margin SVM, Soft Margin SVM, Slack Variables, Regularization Trade-offs, and Hinge Loss Function.

## Question 1

### Problem Statement
Consider a simple 2D dataset with potential outliers:
- Class +1: $(2, 3)$, $(3, 2)$, $(4, 3)$, and one potential outlier at $(1, 4)$
- Class -1: $(0, 0)$, $(1, 1)$, $(0, 2)$

#### Task
1. [📚] Sketch these points and try to draw a separating line for the hard margin case
2. [🔍] Explain what would happen to the hard margin SVM if the outlier $(1, 4)$ makes the data non-linearly separable
3. [🔍] Why might the hard margin approach be problematic in this scenario?
4. [📚] How would soft margin SVM handle this situation differently?

For a detailed explanation of this problem, see [Question 1: Hard vs Soft Margin with Outliers](L5_2_1_explanation.md).

## Question 2

### Problem Statement
Consider the hard margin SVM optimization problem:
$$\min_{\mathbf{w}, b} \frac{1}{2}||\mathbf{w}||^2$$
$$\text{subject to: } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n$$

#### Task
1. [🔍] What assumption does this formulation make about the data?
2. [📚] What happens if even one training point violates the constraint $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$?
3. [📚] Why can't this formulation handle noisy or overlapping data?
4. [🔍] In what practical scenarios might hard margin SVM still be useful?

For a detailed explanation of this problem, see [Question 2: Hard Margin Limitations](L5_2_2_explanation.md).

## Question 3

### Problem Statement
Consider the soft margin SVM optimization problem:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^n \xi_i$$
$$\text{subject to: } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

#### Task
1. [📚] What do the slack variables $\xi_i$ represent geometrically?
2. [📚] Why do we need the constraint $\xi_i \geq 0$?
3. [🔍] How does the term $C\sum_{i=1}^n \xi_i$ in the objective function affect the optimization?
4. [📚] What is the modified constraint compared to the hard margin case?

For a detailed explanation of this problem, see [Question 3: Soft Margin Formulation](L5_2_3_explanation.md).

## Question 4

### Problem Statement
Consider the interpretation of slack variable values for different training points.

#### Task
1. [📚] For a point with $\xi_i = 0$, what can you say about its position relative to the margin?
2. [📚] For a point with $0 < \xi_i < 1$, where is this point located?
3. [📚] For a point with $\xi_i = 1$, what is special about this point?
4. [📚] For a point with $\xi_i > 1$, what does this indicate about the classification?
5. [🔍] Sketch a 2D example showing points with different slack variable values

For a detailed explanation of this problem, see [Question 4: Slack Variable Interpretation](L5_2_4_explanation.md).

## Question 5

### Problem Statement
Consider the regularization parameter $C$ and its effect on the SVM behavior.

#### Task
1. [📚] When $C \to \infty$, what happens to the soft margin SVM? What does it become equivalent to?
2. [📚] When $C \to 0$, what happens to the decision boundary and slack variables?
3. [🔍] How does increasing $C$ affect the bias-variance tradeoff?
4. [🔍] How does increasing $C$ typically affect the number of support vectors?
5. [📚] Give a practical example of when you might choose a small vs. large value of $C$

For a detailed explanation of this problem, see [Question 5: Regularization Parameter C Effects](L5_2_5_explanation.md).

## Question 6

### Problem Statement
Consider the hinge loss function: $L(y, f(x)) = \max(0, 1 - y \cdot f(x))$ where $f(x) = \mathbf{w}^T\mathbf{x} + b$.

#### Task
1. [📚] For a correctly classified point with $y \cdot f(x) = 2$, what is the hinge loss?
2. [📚] For a point on the margin boundary with $y \cdot f(x) = 1$, what is the hinge loss?
3. [📚] For a misclassified point with $y \cdot f(x) = -0.5$, what is the hinge loss?
4. [🔍] Why is the hinge loss called "hinge" loss? Describe its shape.
5. [📚] How does the hinge loss relate to the slack variables in soft margin SVM?

For a detailed explanation of this problem, see [Question 6: Hinge Loss Calculation](L5_2_6_explanation.md).

## Question 7

### Problem Statement
Compare the hinge loss with other loss functions commonly used in machine learning.

#### Task
1. [🔍] How does the hinge loss compare to the 0-1 loss (simple classification error)?
2. [🔍] How does the hinge loss compare to the logistic loss used in logistic regression?
3. [🔍] How does the hinge loss compare to the squared loss $(y - f(x))^2$?
4. [📚] Why is the hinge loss considered a good approximation to the 0-1 loss?
5. [🔍] What are the computational advantages of hinge loss over 0-1 loss?

For a detailed explanation of this problem, see [Question 7: Loss Function Comparison](L5_2_7_explanation.md).

## Question 8

### Problem Statement
Consider the KKT (Karush-Kuhn-Tucker) conditions for the soft margin SVM.

#### Task
1. [📚] For the soft margin SVM, what are the types of points based on their KKT conditions?
2. [📚] For a point with $\alpha_i = 0$, what can you say about its slack variable and position?
3. [📚] For a point with $0 < \alpha_i < C$, what constraints must be satisfied?
4. [📚] For a point with $\alpha_i = C$, what does this indicate?
5. [🔍] How do the KKT conditions help identify support vectors in soft margin SVM?

For a detailed explanation of this problem, see [Question 8: KKT Conditions](L5_2_8_explanation.md).

## Question 9

### Problem Statement
Consider the relationship between different types of support vectors in soft margin SVM.

#### Task
1. [🔍] What is the difference between "margin support vectors" and "error support vectors"?
2. [📚] For margin support vectors, what are the values of $\alpha_i$ and $\xi_i$?
3. [📚] For error support vectors, what are the possible values of $\alpha_i$ and $\xi_i$?
4. [🔍] Which type of support vectors lie exactly on the margin boundary?
5. [📚] How does the presence of error support vectors affect the decision boundary?

For a detailed explanation of this problem, see [Question 9: Types of Support Vectors](L5_2_9_explanation.md).

## Question 10

### Problem Statement
Consider practical strategies for hyperparameter selection in soft margin SVMs.

#### Task
1. [🔍] Describe a systematic approach for selecting the optimal value of $C$ using cross-validation
2. [📚] What metrics should you monitor when tuning $C$ (training error, validation error, number of support vectors)?
3. [🔍] How would you detect if your chosen $C$ value leads to overfitting or underfitting?
4. [📚] What role does the validation curve (performance vs. $C$) play in hyperparameter selection?
5. [🔍] In a scenario with severe class imbalance, how might you adjust your approach to selecting $C$?

For a detailed explanation of this problem, see [Question 10: Hyperparameter Selection](L5_2_10_explanation.md).
