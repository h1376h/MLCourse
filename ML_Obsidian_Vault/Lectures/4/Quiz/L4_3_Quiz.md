# Lecture 4.3: Linear Separability and Optimization Quiz

## Overview
This quiz contains 4 questions from different topics covered in section 4.3 of the lectures on Linear Separability and Optimization.

## Question 1

### Problem Statement
Consider a binary classification problem in ℝ². 

#### Task
1. Define what it means for a dataset to be linearly separable
2. Prove that if two sets of points in ℝ² are linearly separable, then there exists an infinite number of linear decision boundaries that separate them
3. For n points in a general position in ℝᵈ (no d+1 points lie on the same hyperplane), what is the maximum number of different ways to label these points such that they remain linearly separable?
4. Provide an example of a simple dataset in ℝ² that is not linearly separable, and explain why

## Question 2

### Problem Statement
Consider the following loss functions for linear classifiers:

#### Task
1. Define the hinge loss function used in Support Vector Machines
2. Define the logistic loss function used in logistic regression
3. Compare and contrast these two loss functions in terms of:
   a. Mathematical properties
   b. Sensitivity to outliers
   c. Probabilistic interpretation
   d. Decision boundary characteristics
4. For which scenarios would you prefer one loss function over the other? Explain your reasoning

## Question 3

### Problem Statement
Consider the gradient descent algorithm for optimizing a linear classifier.

#### Task
1. Write down the general form of the gradient descent update rule
2. Derive the gradient of the logistic loss function for binary classification
3. Compare batch gradient descent, mini-batch gradient descent, and stochastic gradient descent in terms of:
   a. Computational efficiency
   b. Convergence properties
   c. Implementation complexity
4. Explain how learning rate scheduling can improve gradient descent convergence

## Question 4

### Problem Statement
Consider the following optimization problems for linear classification:

#### Task
1. Formulate the optimization problem for finding the maximum margin separator (hard-margin SVM)
2. Explain what the margin represents geometrically and why maximizing it is desirable
3. How does the soft-margin SVM modification address the issue of non-linearly separable data?
4. Compare the computational complexity of solving the optimization problems for:
   a. Perceptron algorithm
   b. Logistic regression using gradient descent
   c. Support Vector Machines using quadratic programming 