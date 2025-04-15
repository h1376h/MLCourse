# Lecture 4.4: Multi-class Classification Quiz

## Overview
This quiz contains 4 questions from different topics covered in section 4.4 of the lectures on Multi-class Classification.

## Question 1

### Problem Statement
Consider extending binary classification techniques to the multi-class setting.

#### Task
1. Describe the one-vs-all (also called one-vs-rest) approach for multi-class classification
2. Describe the one-vs-one approach for multi-class classification
3. For a problem with K classes, how many binary classifiers are needed for:
   a. One-vs-all approach
   b. One-vs-one approach
4. Compare the computational complexity and potential advantages/disadvantages of these two approaches

## Question 2

### Problem Statement
Consider a multi-class classification problem with K=4 classes and n=1000 data points in a d=10 dimensional feature space.

#### Task
1. Explain how the one-vs-all approach would be implemented for this problem using logistic regression as the base classifier
2. If your binary classifiers output probabilities, how would you combine these probabilities to make a final prediction?
3. How would you evaluate the performance of your multi-class classifier? Describe appropriate metrics
4. If the classes are imbalanced (e.g., class 1: 600 points, class 2: 200 points, class 3: 100 points, class 4: 100 points), what challenges might arise and how would you address them?

## Question 3

### Problem Statement
Consider implementing a multi-class classification system for handwritten digit recognition (10 classes).

#### Task
1. Compare the following approaches for this problem:
   a. One-vs-all with linear classifiers
   b. One-vs-one with linear classifiers
   c. Direct multi-class approaches (e.g., multinomial logistic regression)
2. What are the computational and memory requirements for training and prediction in each approach?
3. How would feature engineering impact the performance of these approaches?
4. If computational resources are limited, which approach would you recommend and why?

## Question 4

### Problem Statement
Consider the following confusion matrix for a multi-class classifier with 3 classes:

```
             Predicted
             C1  C2  C3
Actual  C1   85  10   5
        C2   15  75  10
        C3    5  20  75
```

#### Task
1. Calculate the following metrics:
   a. Overall accuracy
   b. Per-class precision
   c. Per-class recall
   d. Per-class F1-score
2. Based on this confusion matrix, which classes are most confused with each other?
3. Suggest possible reasons for the observed confusion and how you might modify your classifier to improve performance
4. If class C3 represents a rare but critical condition in a medical diagnosis application, which metric would be most important to optimize, and how might this affect your approach? 