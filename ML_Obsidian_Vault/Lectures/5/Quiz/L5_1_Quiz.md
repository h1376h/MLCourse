# Lecture 5.1: Maximum Margin Classifiers Quiz

## Overview
This quiz contains 4 questions covering different topics from section 5.1 of the lectures on Maximum Margin Theory, Geometric Interpretation, Linear Separability, and Support Vectors.

## Question 1

### Problem Statement
Consider a binary classification problem with linearly separable data. You have a training dataset with two classes represented by points in 2D space.

#### Task
1. Explain what the "margin" means in the context of maximum margin classification
2. Why is maximizing the margin beneficial for generalization?
3. Draw a simple 2D example showing the optimal separating hyperplane and the margin
4. What happens to the decision boundary if we remove non-support vector points from the training set?

For a detailed explanation of this problem, see [Question 1: Maximum Margin Concepts](L5_1_1_explanation.md).

## Question 2

### Problem Statement
Consider the mathematical formulation of the maximum margin classifier. For a linearly separable dataset, we want to find the hyperplane that maximizes the margin.

#### Task
1. Write the mathematical formulation of the primal optimization problem for maximum margin classification
2. Explain what the constraint $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$ represents geometrically
3. Why do we minimize $\frac{1}{2}|\mathbf{w}|^2$ instead of directly maximizing the margin?
4. What is the relationship between the weight vector $\mathbf{w}$ and the optimal separating hyperplane?

For a detailed explanation of this problem, see [Question 2: Mathematical Formulation](L5_1_2_explanation.md).

## Question 3

### Problem Statement
Consider the dual formulation of the maximum margin classifier and the concept of support vectors.

#### Task
1. Write the dual optimization problem for the maximum margin classifier
2. Explain the significance of the Lagrange multipliers $\alpha_i$ in the dual formulation
3. Define what support vectors are and explain their geometric significance
4. Why can the optimal hyperplane be expressed solely in terms of support vectors?
5. What happens to the solution if we add more training points that are not support vectors?

For a detailed explanation of this problem, see [Question 3: Dual Formulation and Support Vectors](L5_1_3_explanation.md).

## Question 4

### Problem Statement
Consider the decision function of a maximum margin classifier and compare it with the perceptron algorithm.

#### Task
1. Write the decision function for classifying a new point $\mathbf{x}$ using the trained maximum margin classifier
2. How does the maximum margin classifier differ from the perceptron algorithm in terms of:
   - The objective function being optimized
   - The final decision boundary
   - Sensitivity to outliers
3. Explain why the maximum margin classifier typically generalizes better than the perceptron
4. Give an example scenario where both algorithms would produce the same result

For a detailed explanation of this problem, see [Question 4: Decision Function and Comparison with Perceptron](L5_1_4_explanation.md).
