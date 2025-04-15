# Lecture 4.2: The Perceptron Algorithm Quiz

## Overview
This quiz contains 4 questions from different topics covered in section 4.2 of the lectures on The Perceptron Algorithm.

## Question 1

### Problem Statement
Consider the perceptron algorithm for binary classification. Given a linearly separable dataset with features x ∈ ℝᵈ and labels y ∈ {-1, 1}.

#### Task
1. Write down the perceptron update rule for misclassified points
2. Explain the geometric interpretation of the update rule
3. Under what conditions will the perceptron algorithm converge?
4. What does the perceptron convergence theorem state about the number of mistakes made during training?

## Question 2

### Problem Statement
Consider the following dataset for binary classification:
- Class +1: (3, 3), (4, 1), (1, 4)
- Class -1: (1, 1), (2, 2), (1, 2)

#### Task
1. Initialize a perceptron with weights w = [0, 0] and bias b = 0
2. Trace through the first 3 iterations of the perceptron algorithm, assuming you visit points in the order listed
3. For each iteration, show:
   a. The decision boundary equation
   b. Whether each point is correctly classified
   c. The updated weights and bias after processing misclassified points
4. Draw the initial decision boundary and the decision boundary after these 3 iterations

## Question 3

### Problem Statement
Consider the limitations of the basic perceptron algorithm.

#### Task
1. What happens when the data is not linearly separable? Explain with an example
2. How can the perceptron algorithm be modified to handle non-linearly separable data?
3. Compare the perceptron algorithm with logistic regression for binary classification
4. Explain how the voted perceptron or averaged perceptron algorithms improve upon the basic perceptron

## Question 4

### Problem Statement
Consider implementing the perceptron algorithm for a practical application.

#### Task
1. Write pseudocode for the perceptron learning algorithm
2. How would you determine an appropriate learning rate?
3. Describe how feature scaling affects the perceptron algorithm performance
4. Explain how you would adapt the perceptron algorithm for multiclass classification using the one-vs-all approach 