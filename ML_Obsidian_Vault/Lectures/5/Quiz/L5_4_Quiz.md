# Lecture 5.4: Multi-class SVM Approaches Quiz

## Overview
This quiz contains 4 questions covering different topics from section 5.4 of the lectures on One-vs-Rest, One-vs-One, Decision Strategies, and Multi-class Classification Performance.

## Question 1

### Problem Statement
Consider the challenge of extending binary SVMs to handle multi-class classification problems.

#### Task
1. Explain why SVMs are inherently binary classifiers and what challenges arise when dealing with multi-class problems
2. Compare and contrast the One-vs-Rest (OvR) and One-vs-One (OvO) approaches for multi-class SVM
3. For a dataset with K classes:
   - How many binary classifiers are trained in OvR?
   - How many binary classifiers are trained in OvO?
   - What are the computational implications of each approach?
4. Give an example scenario where OvR might be preferred over OvO, and vice versa

For a detailed explanation of this problem, see [Question 1: Multi-class SVM Strategies](L5_4_1_explanation.md).

## Question 2

### Problem Statement
Consider the One-vs-Rest (OvR) approach for multi-class SVM classification.

#### Task
1. Describe the training procedure for OvR multi-class SVM with K classes
2. Explain the decision strategy used during prediction with OvR
3. What potential issues can arise with OvR classification?
   - Class imbalance problems
   - Ambiguous regions where no classifier is confident
   - Overlapping decision regions
4. How would you handle a situation where multiple classifiers claim a test point or no classifier claims it?
5. Discuss the scalability of OvR as the number of classes increases

For a detailed explanation of this problem, see [Question 2: One-vs-Rest Implementation](L5_4_2_explanation.md).

## Question 3

### Problem Statement
Consider the One-vs-One (OvO) approach for multi-class SVM classification.

#### Task
1. Describe the training procedure for OvO multi-class SVM with K classes
2. Explain the voting mechanism typically used in OvO for final classification
3. What are the advantages of OvO over OvR in terms of:
   - Training data balance
   - Individual classifier complexity
   - Robustness to outliers
4. Calculate the number of pairwise classifiers needed for:
   - 3 classes
   - 5 classes
   - 10 classes
   - 100 classes
5. At what point does the computational overhead of OvO become prohibitive?

For a detailed explanation of this problem, see [Question 3: One-vs-One Implementation](L5_4_3_explanation.md).

## Question 4

### Problem Statement
Consider alternative approaches to multi-class SVM and performance evaluation metrics.

#### Task
1. Describe the direct multi-class SVM approach that simultaneously optimizes for all classes
2. What are Error-Correcting Output Codes (ECOC) and how can they be applied to multi-class SVM?
3. Compare the following multi-class approaches in terms of training time, prediction time, and accuracy:
   - One-vs-Rest
   - One-vs-One  
   - Direct multi-class optimization
4. What evaluation metrics are appropriate for multi-class SVM performance assessment?
5. How would you handle severe class imbalance in a multi-class SVM problem (e.g., 95% of samples belong to one class)?

For a detailed explanation of this problem, see [Question 4: Advanced Multi-class Methods and Evaluation](L5_4_4_explanation.md).
