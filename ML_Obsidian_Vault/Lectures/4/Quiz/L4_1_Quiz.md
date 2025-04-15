# Lecture 4.1: Foundations of Linear Classification Quiz

## Overview
This quiz contains 4 questions from different topics covered in section 4.1 of the lectures on Foundations of Linear Classification.

## Question 1

### Problem Statement
Consider a binary classification problem with two features x₁ and x₂. We have the following data points:
- Class 0: (1, 1), (2, 1), (1, 2)
- Class 1: (4, 4), (5, 4), (4, 5)

#### Task
1. Plot these points in a 2D feature space and identify whether they appear to be linearly separable
2. Find a linear decision boundary in the form of w₁x₁ + w₂x₂ + b = 0 that separates the two classes
3. For a new point (3, 3), which class would your model predict?
4. What is the geometric interpretation of the weights w₁ and w₂ in relation to the decision boundary?

## Question 2

### Problem Statement
Consider a binary classification task with features x ∈ ℝᵈ and labels y ∈ {0, 1}.

#### Task
1. Write the equation for the linear decision boundary in this d-dimensional space
2. Define the margin of a linear classifier and explain its significance
3. Explain how the sign of w^T x + b determines the predicted class
4. Describe the relationship between linear classification and logistic regression

## Question 3

### Problem Statement
Consider a feature space visualization where positive and negative examples are clearly separated, but not by a linear boundary.

#### Task
1. Explain why a linear classifier would fail in this scenario
2. Describe the concept of feature transformation and how it can help
3. Provide an example of a specific feature transformation that might make the data linearly separable
4. Discuss the tradeoffs involved when using more complex feature transformations

## Question 4

### Problem Statement
Consider the following loss functions for linear classification:
- 0-1 loss
- Hinge loss
- Logistic loss
- Exponential loss

#### Task
1. Define each loss function mathematically
2. Explain the key differences between these loss functions
3. Describe the advantages and disadvantages of each
4. Which loss functions are convex, and why is convexity important for optimization? 