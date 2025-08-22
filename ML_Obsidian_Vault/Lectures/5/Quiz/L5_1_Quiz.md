# Lecture 5.1: Maximum Margin Classifiers Quiz

## Overview
This quiz contains 12 questions covering different topics from section 5.1 of the lectures on Maximum Margin Theory, Geometric Interpretation, Linear Separability, Support Vectors, Dual Formulation, and Decision Functions.

## Question 1

### Problem Statement
Consider a simple 2D binary classification problem with four training points:
- Class +1: $(3, 2)$ and $(4, 3)$
- Class -1: $(1, 1)$ and $(2, 0)$

#### Task
1. [📚] Sketch these points in a 2D coordinate system
2. [📚] Draw a possible separating hyperplane (line) that separates the two classes
3. [🔍] Explain what "margin" means in geometric terms for this example
4. [🔍] Why might there be multiple valid separating lines, and which one should we prefer?

For a detailed explanation of this problem, see [Question 1: Basic Margin Concept](L5_1_1_explanation.md).

## Question 2

### Problem Statement
Consider the mathematical definition of margin for a linear classifier with decision boundary $\mathbf{w}^T\mathbf{x} + b = 0$.

#### Task
1. [📚] If a point $\mathbf{x}_i$ with label $y_i$ satisfies $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$, what can you say about this point?
2. [📚] What is the geometric margin of a point $\mathbf{x}_i$ from the hyperplane?
3. [📚] For the hyperplane $2x_1 + 3x_2 - 6 = 0$, calculate the distance from point $(1, 2)$ to this hyperplane
4. [🔍] Explain why we normalize the decision boundary equation when computing geometric margin

For a detailed explanation of this problem, see [Question 2: Geometric Margin Calculation](L5_1_2_explanation.md).

## Question 3

### Problem Statement
Consider the primal optimization problem for maximum margin classification:
$$\min_{\mathbf{w}, b} \frac{1}{2}||\mathbf{w}||^2$$
$$\text{subject to: } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n$$

#### Task
1. [📚] Why do we minimize $\frac{1}{2}||\mathbf{w}||^2$ instead of maximizing the margin directly?
2. [📚] What does the constraint $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$ ensure?
3. [🔍] How is this optimization problem different from minimizing classification errors?
4. [🔍] What type of optimization problem is this (linear, quadratic, etc.)?

For a detailed explanation of this problem, see [Question 3: Primal Optimization Formulation](L5_1_3_explanation.md).

## Question 4

### Problem Statement
Consider the concept of support vectors in maximum margin classification.

#### Task
1. [🔍] Define what a support vector is in mathematical terms
2. [📚] For a linearly separable dataset, what property do all support vectors share?
3. [📚] If you remove a non-support vector from the training set, how does this affect the optimal hyperplane?
4. [📚] If you remove a support vector from the training set, how does this affect the optimal hyperplane?
5. [🔍] Why are support vectors called "support" vectors?

For a detailed explanation of this problem, see [Question 4: Support Vector Properties](L5_1_4_explanation.md).

## Question 5

### Problem Statement
Consider the Lagrangian dual formulation of the maximum margin problem:
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j$$
$$\text{subject to: } \sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \geq 0$$

#### Task
1. [📚] What do the Lagrange multipliers $\alpha_i$ represent?
2. [📚] For a support vector, what can you say about its corresponding $\alpha_i$?
3. [📚] For a non-support vector, what is the value of its corresponding $\alpha_i$?
4. [🔍] Why is the dual formulation often preferred over the primal for SVMs?

For a detailed explanation of this problem, see [Question 5: Dual Formulation and Lagrange Multipliers](L5_1_5_explanation.md).

## Question 6

### Problem Statement
Consider a small dataset with three points:
- $\mathbf{x}_1 = (1, 0)$, $y_1 = +1$
- $\mathbf{x}_2 = (0, 1)$, $y_2 = +1$  
- $\mathbf{x}_3 = (-1, -1)$, $y_3 = -1$

#### Task
1. [📚] Sketch these points and identify which ones are likely to be support vectors
2. [📚] Write the decision function in terms of support vectors: $f(\mathbf{x}) = \sum_{i \in SV} \alpha_i y_i \mathbf{x}_i^T\mathbf{x} + b$
3. [📚] If $\alpha_1 = 0.5$, $\alpha_2 = 0.5$, and $\alpha_3 = 1.0$, verify that the dual constraint $\sum_i \alpha_i y_i = 0$ is satisfied
4. [📚] Calculate the decision value for the point $\mathbf{x} = (0, 0)$ using these parameters (assume $b = 0$)

For a detailed explanation of this problem, see [Question 6: Support Vector Decision Function](L5_1_6_explanation.md).

## Question 7

### Problem Statement
Compare the maximum margin classifier with the perceptron algorithm for linear classification.

#### Task
1. [🔍] What is the key difference in the objective functions of these two algorithms?
2. [📚] For a linearly separable dataset, will both algorithms find a separating hyperplane?
3. [🔍] Which algorithm is more sensitive to outliers and why?
4. [📚] Give an example of a simple 2D dataset where both algorithms would find the same solution
5. [🔍] Explain why the maximum margin solution often generalizes better

For a detailed explanation of this problem, see [Question 7: Comparison with Perceptron](L5_1_7_explanation.md).

## Question 8

### Problem Statement
Consider the computational complexity and optimization aspects of maximum margin classification.

#### Task
1. [🔍] What is the time complexity of training a maximum margin classifier using quadratic programming?
2. [📚] How many optimization variables are in the primal formulation vs. the dual formulation?
3. [🔍] For which scenarios (high-dimensional features vs. many samples) is the dual formulation preferred?
4. [📚] What happens to the optimization problem when the data is not linearly separable?

For a detailed explanation of this problem, see [Question 8: Computational Complexity](L5_1_8_explanation.md).

## Question 9

### Problem Statement
Consider the geometric interpretation of the maximum margin hyperplane.

#### Task
1. [📚] In 2D, the separating hyperplane is a line. What is the equation of the line equidistant from two parallel lines $\mathbf{w}^T\mathbf{x} + b = 1$ and $\mathbf{w}^T\mathbf{x} + b = -1$?
2. [📚] What is the distance between these two parallel lines in terms of $||\mathbf{w}||$?
3. [🔍] Why do we say that support vectors "touch" the margin boundary?
4. [📚] Sketch a 2D example showing the hyperplane, margin boundaries, and support vectors

For a detailed explanation of this problem, see [Question 9: Geometric Interpretation](L5_1_9_explanation.md).

## Question 10

### Problem Statement
Consider the conditions for linear separability and the uniqueness of the maximum margin solution.

#### Task
1. [🔍] What does it mean for a dataset to be linearly separable?
2. [📚] For a linearly separable dataset, is the maximum margin hyperplane unique? Why or why not?
3. [📚] If you have a dataset with only two points (one from each class), how many support vectors will there be?
4. [🔍] What is the minimum number of support vectors needed to define a hyperplane in $d$-dimensional space?

For a detailed explanation of this problem, see [Question 10: Linear Separability and Uniqueness](L5_1_10_explanation.md).

## Question 11

### Problem Statement
Consider the relationship between the margin width and the norm of the weight vector.

#### Task
1. [📚] If the margin width is $\frac{2}{||\mathbf{w}||}$, and we want to maximize the margin, what should we do to $||\mathbf{w}||$?
2. [📚] Given the constraint $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$, explain why we can't simply set $\mathbf{w} = \mathbf{0}$ to minimize $||\mathbf{w}||^2$
3. [🔍] What happens to the decision boundary if we scale both $\mathbf{w}$ and $b$ by the same positive constant?
4. [📚] Why is the normalization constraint (setting the functional margin to 1) necessary in the formulation?

For a detailed explanation of this problem, see [Question 11: Margin Width and Weight Vector](L5_1_11_explanation.md).

## Question 12

### Problem Statement
Consider the decision function and prediction confidence in maximum margin classification.

#### Task
1. [📚] Write the decision function for classifying a new point $\mathbf{x}$ using the maximum margin classifier
2. [📚] How can the magnitude of $f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b$ be interpreted in terms of confidence?
3. [🔍] For two points with decision values $f(\mathbf{x}_1) = +2.5$ and $f(\mathbf{x}_2) = +0.1$, which prediction is more confident and why?
4. [📚] What does it mean when $f(\mathbf{x}) = 0$ for a test point?
5. [🔍] How does the distance from the separating hyperplane relate to prediction confidence?

For a detailed explanation of this problem, see [Question 12: Decision Function and Confidence](L5_1_12_explanation.md).
