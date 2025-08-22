# Lecture 5.4: Multi-class SVM Approaches Quiz

## Overview
This quiz contains 13 questions covering different topics from section 5.4 of the lectures on One-vs-Rest, One-vs-One, Decision Strategies, Multi-class Classification Performance, Error-Correcting Output Codes, and Direct Multi-class Methods.

## Question 1

### Problem Statement
Consider a 4-class classification problem with classes $\{A, B, C, D\}$ and $n = 1000$ training samples distributed as: $n_A = 400$, $n_B = 300$, $n_C = 200$, $n_D = 100$.

#### Task
1. For One-vs-Rest (OvR), specify the 4 binary classification problems and their class distributions
2. For One-vs-One (OvO), list all $\binom{4}{2} = 6$ pairwise problems and their sample sizes
3. Calculate the class imbalance ratio for each OvR classifier
4. Which approach (OvR or OvO) suffers more from class imbalance in this scenario?
5. Design a cost-sensitive modification to handle the imbalance

For a detailed explanation of this problem, see [Question 1: Multi-class Problem Setup](L5_4_1_explanation.md).

## Question 2

### Problem Statement
Analyze the One-vs-Rest approach mathematically.

For a $K$-class problem, the OvR approach trains $K$ binary classifiers $f_k(\mathbf{x}) = \mathbf{w}_k^T\mathbf{x} + b_k$.

#### Task
1. Write the optimization problem for the $k$-th binary classifier
2. During prediction, show that the OvR decision rule is: $\hat{y} = \arg\max_k f_k(\mathbf{x})$
3. What problems arise when multiple classifiers output positive values?
4. What problems arise when all classifiers output negative values?
5. Derive a confidence measure for OvR predictions based on the margin between the top two scores

For a detailed explanation of this problem, see [Question 2: OvR Mathematical Analysis](L5_4_2_explanation.md).

## Question 3

### Problem Statement
Work through a concrete OvR example.

Consider a 3-class problem with the following OvR classifier outputs for a test point $\mathbf{x}$:
- $f_1(\mathbf{x}) = +1.2$ (Class 1 vs Rest)
- $f_2(\mathbf{x}) = -0.3$ (Class 2 vs Rest)  
- $f_3(\mathbf{x}) = +0.8$ (Class 3 vs Rest)

#### Task
1. What is the predicted class using the standard OvR rule?
2. Calculate a confidence score for this prediction
3. How would you handle the ambiguous case where $f_1(\mathbf{x}) = +0.9$ and $f_3(\mathbf{x}) = +0.9$?
4. Convert the decision values into class probabilities using softmax
5. What additional information would help resolve prediction ambiguities?

For a detailed explanation of this problem, see [Question 3: OvR Concrete Example](L5_4_3_explanation.md).

## Question 4

### Problem Statement
Analyze the One-vs-One approach and its voting mechanism.

For $K$ classes, OvO trains $\binom{K}{2}$ binary classifiers $f_{ij}(\mathbf{x})$ for each pair $(i,j)$.

#### Task
1. Write the decision rule for classifier $f_{ij}(\mathbf{x})$: when does it vote for class $i$ vs class $j$?
2. Implement the majority voting scheme: $\hat{y} = \arg\max_k \sum_{j \neq k} \mathbb{I}[f_{kj}(\mathbf{x}) > 0]$
3. For $K = 5$ classes, what's the maximum and minimum number of votes a class can receive?
4. Design a tie-breaking mechanism for cases where multiple classes receive the same number of votes
5. Derive a confidence measure based on the vote distribution

For a detailed explanation of this problem, see [Question 4: OvO Voting Analysis](L5_4_4_explanation.md).

## Question 5

### Problem Statement
Solve a complete OvO example.

Consider a 4-class problem with classes $\{1, 2, 3, 4\}$. For a test point, the 6 pairwise classifiers output:
- $f_{12}(\mathbf{x}) = +0.5$ (class 1 beats class 2)
- $f_{13}(\mathbf{x}) = -0.2$ (class 3 beats class 1)
- $f_{14}(\mathbf{x}) = +0.8$ (class 1 beats class 4)
- $f_{23}(\mathbf{x}) = +1.1$ (class 2 beats class 3)
- $f_{24}(\mathbf{x}) = +0.6$ (class 2 beats class 4)
- $f_{34}(\mathbf{x}) = -0.4$ (class 4 beats class 3)

#### Task
1. Create the vote tally for each class
2. Determine the winning class using majority voting
3. Calculate the vote confidence ratio
4. What would happen if we used the sum of decision values instead of vote counting?
5. Identify potential inconsistencies in the pairwise decisions (voting cycles)

For a detailed explanation of this problem, see [Question 5: OvO Complete Example](L5_4_5_explanation.md).

## Question 6

### Problem Statement
Compare computational complexity of OvR vs OvO approaches.

#### Task
1. For $K$ classes and $n$ training samples, derive the total number of training samples used by OvR vs OvO
2. Calculate the number of binary classifiers for $K = 3, 5, 10, 50, 100$
3. At what value of $K$ does OvO require more classifiers than OvR?
4. Compare the training time complexity assuming each binary SVM takes $O(n^2)$ time
5. Compare the prediction time for both approaches
6. Design a parallel training strategy for both OvR and OvO

For a detailed explanation of this problem, see [Question 6: Complexity Comparison](L5_4_6_explanation.md).

## Question 7

### Problem Statement
Explore Error-Correcting Output Codes (ECOC) for multi-class classification.

#### Task
1. Design a 4-bit ECOC matrix for 4 classes with error-correcting capability
2. Show how to decode predictions using Hamming distance
3. Calculate the error-correcting capability of your code
4. Compare the number of binary classifiers needed vs OvR and OvO for 8 classes
5. Design an optimal ECOC matrix that maximizes the minimum Hamming distance between codewords

For a detailed explanation of this problem, see [Question 7: Error-Correcting Codes](L5_4_7_explanation.md).

## Question 8

### Problem Statement
Design and implement multi-class evaluation metrics.

Consider a 3-class confusion matrix:
$$C = \begin{pmatrix} 85 & 5 & 10 \\ 8 & 78 & 14 \\ 12 & 7 & 81 \end{pmatrix}$$

#### Task
1. Calculate the overall accuracy
2. Compute precision, recall, and F1-score for each class
3. Calculate macro-averaged and micro-averaged metrics
4. Compute the balanced accuracy
5. Design a cost-sensitive evaluation metric where misclassifying class 1 as class 3 costs twice as much as other errors

For a detailed explanation of this problem, see [Question 8: Multi-class Evaluation](L5_4_8_explanation.md).

## Question 9

### Problem Statement
Investigate direct multi-class SVM formulations.

The direct approach optimizes:
$$\min_{\mathbf{W}, \boldsymbol{\xi}} \frac{1}{2}||\mathbf{W}||_F^2 + C\sum_{i=1}^n \sum_{k \neq y_i} \xi_{ik}$$
$$\text{subject to: } \mathbf{w}_{y_i}^T\mathbf{x}_i - \mathbf{w}_k^T\mathbf{x}_i \geq 1 - \xi_{ik}, \quad \xi_{ik} \geq 0$$

#### Task
1. Interpret the constraint: what does it ensure for the correct class vs incorrect classes?
2. How many slack variables are needed for $n$ samples and $K$ classes?
3. Compare this to the total complexity of OvR and OvO approaches
4. Derive the dual formulation for the direct multi-class SVM
5. What are the advantages and disadvantages compared to decomposition methods?

For a detailed explanation of this problem, see [Question 9: Direct Multi-class Formulation](L5_4_9_explanation.md).

## Question 10

### Problem Statement
Handle severe class imbalance in multi-class scenarios.

Consider a dataset with 5 classes where class 1 has 80% of the data and classes 2-5 have 5% each.

#### Task
1. How would class imbalance affect OvR classifiers differently?
2. Design class-weighted loss functions for both OvR and OvO
3. Implement SMOTE (Synthetic Minority Oversampling) strategy for multi-class problems
4. How would you modify the decision thresholds to account for class imbalance?
5. Design an evaluation protocol that fairly assesses performance across all classes

For a detailed explanation of this problem, see [Question 10: Class Imbalance Handling](L5_4_10_explanation.md).

## Question 11

### Problem Statement
Implement hierarchical classification strategies.

#### Task
1. Design a binary tree classifier for 8 classes that minimizes expected classification cost
2. How would you handle the case where intermediate nodes make errors?
3. Compare the number of classifiers needed vs flat multi-class approaches
4. Design a strategy for learning the optimal hierarchy from data
5. How would you handle new classes that don't fit well into the existing hierarchy?

For a detailed explanation of this problem, see [Question 11: Hierarchical Classification](L5_4_11_explanation.md).

## Question 12

### Problem Statement
Optimize multi-class SVM hyperparameters systematically.

#### Task
1. For OvR with RBF kernels, design a cross-validation strategy to tune $C$ and $\gamma$
2. Should you use the same hyperparameters for all binary classifiers or tune them separately?
3. How would you handle the computational cost of tuning $K$ separate classifiers?
4. Design an early stopping criterion for expensive hyperparameter searches
5. Implement a warm-start strategy that leverages solutions from similar binary problems

For a detailed explanation of this problem, see [Question 12: Hyperparameter Optimization](L5_4_12_explanation.md).

## Question 13

### Problem Statement
Design a comprehensive practical implementation framework.

#### Task
1. Create a decision flowchart for choosing between OvR, OvO, and direct methods based on dataset characteristics
2. Implement preprocessing pipelines specific to multi-class problems
3. Design efficient prediction pipelines that minimize computational cost
4. How would you handle online learning scenarios where new classes are added incrementally?
5. Implement model selection that jointly optimizes kernel choice, multi-class strategy, and hyperparameters

For a detailed explanation of this problem, see [Question 13: Practical Implementation Framework](L5_4_13_explanation.md).