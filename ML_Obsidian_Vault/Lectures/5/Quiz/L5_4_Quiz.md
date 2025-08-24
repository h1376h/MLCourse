# Lecture 5.4: Multi-class SVM Approaches Quiz

## Overview
This quiz contains 20 questions covering different topics from section 5.4 of the lectures on One-vs-Rest, One-vs-One, Decision Strategies, Multi-class Classification Performance, Error-Correcting Output Codes, and Direct Multi-class Methods.

## Question 1

### Problem Statement
Consider a 4-class classification problem with classes $\{A, B, C, D\}$ and $n = 1000$ training samples distributed as: $n_A = 400$, $n_B = 300$, $n_C = 200$, $n_D = 100$.

#### Task
1. For One-vs-Rest (OvR), specify the 4 binary classification problems and their class distributions
2. For One-vs-One (OvO), list all $\binom{4}{2} = 6$ pairwise problems and their sample sizes
3. Calculate the class imbalance ratio for each OvR classifier
4. Which approach (OvR or OvO) suffers more from class imbalance in this scenario?
5. Design a cost-sensitive modification to handle the imbalance
6. Organize a tournament with 4 teams: Alpha (400 fans), Beta (300 fans), Gamma (200 fans), Delta (100 fans). Compare Format A (One-vs-Rest: each team vs combined "All Others") and Format B (One-vs-One: round-robin). Design a fairness scoring system accounting for team size differences and calculate competition balance for each format. Determine which format gives smaller teams better chances to win.

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
2. Design the majority voting scheme: $\hat{y} = \arg\max_k \sum_{j \neq k} \mathbb{I}[f_{kj}(\mathbf{x}) > 0]$
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
6. Design a voting system for 4 candidates (Alice, Bob, Carol, David) using pairwise comparisons. Results: Alice beats Bob (0.5), Carol beats Alice (0.2), Alice beats David (0.8), Bob beats Carol (1.1), Bob beats David (0.6), David beats Carol (0.4). Design a vote counting system, calculate election confidence, and create a preference ranking. Handle potential voting paradoxes.

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
$$C = \begin{bmatrix} 85 & 5 & 10 \\ 8 & 78 & 14 \\ 12 & 7 & 81 \end{bmatrix}$$

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
3. Design SMOTE (Synthetic Minority Oversampling) strategy for multi-class problems
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
5. Design a warm-start strategy that leverages solutions from similar binary problems

For a detailed explanation of this problem, see [Question 12: Hyperparameter Optimization](L5_4_12_explanation.md).

## Question 13

### Problem Statement
Design a comprehensive practical implementation framework.

#### Task
1. Create a decision flowchart for choosing between OvR, OvO, and direct methods based on dataset characteristics
2. Design preprocessing pipelines specific to multi-class problems
3. Design efficient prediction pipelines that minimize computational cost
4. How would you handle online learning scenarios where new classes are added incrementally?
5. Design model selection that jointly optimizes kernel choice, multi-class strategy, and hyperparameters

For a detailed explanation of this problem, see [Question 13: Practical Implementation Framework](L5_4_13_explanation.md).

## Question 14

### Problem Statement
For a 3-class dataset with classes {A, B, C}, design One-vs-Rest classification.

Training data:
- Class A: $(1, 2)$, $(2, 1)$
- Class B: $(3, 3)$, $(4, 2)$
- Class C: $(1, 4)$, $(2, 5)$

#### Task
1. Set up the three binary classification problems (A vs not-A, B vs not-B, C vs not-C)
2. For each binary problem, list the positive and negative class labels
3. Given decision functions:
   - $f_A(\mathbf{x}) = 0.3x_1 + 0.4x_2 - 1.2$
   - $f_B(\mathbf{x}) = -0.2x_1 + 0.6x_2 - 0.8$
   - $f_C(\mathbf{x}) = 0.1x_1 - 0.3x_2 + 0.5$
4. Classify test points $(2, 3)$ and $(1, 1)$ using maximum decision value
5. Rank the predictions by confidence level

For a detailed explanation of this problem, see [Question 14: OvR Implementation](L5_4_14_explanation.md).

## Question 15

### Problem Statement
For a 4-class problem {A, B, C, D}, design One-vs-One classification.

#### Task
1. List all $\binom{4}{2} = 6$ binary classifiers needed
2. For test point $\mathbf{x}$, given pairwise decisions:
   - A vs B: A wins
   - A vs C: C wins  
   - A vs D: A wins
   - B vs C: B wins
   - B vs D: B wins
   - C vs D: D wins
3. Count votes for each class
4. If there were a tie, describe two tie-breaking strategies
5. Design a confidence measure based on vote margins

For a detailed explanation of this problem, see [Question 15: OvO Vote Counting](L5_4_15_explanation.md).

## Question 16

### Problem Statement
Design Error-Correcting Output Codes for a 5-class problem.

#### Task
1. Design a $5 \times 7$ ECOC matrix where each row represents a class and each column a binary classifier
2. Calculate minimum Hamming distance between any two codewords
3. Show that your code can correct at least 1 bit error
4. For output vector $(+1, -1, +1, -1, +1, +1, -1)$, find the closest codeword
5. Explain why 7 classifiers are needed instead of the minimum $\lceil \log_2(5) \rceil = 3$

For a detailed explanation of this problem, see [Question 16: ECOC Design](L5_4_16_explanation.md).

## Question 17

### Problem Statement
Compare training and prediction costs for different multi-class strategies.

For dataset with $n$ samples, $d$ features, $K$ classes:

#### Task
1. Calculate training time complexity for:
   - OvR: $O(?)$
   - OvO: $O(?)$
   - Direct multi-class: $O(?)$
2. Compare memory usage for storing $K$ vs $\binom{K}{2}$ vs 1 classifier
3. For $n_{SV}$ support vectors per classifier, compare prediction costs
4. At what number of classes $K$ does OvO become more expensive than OvR?
5. Calculate actual numbers for $K = 10, n = 1000, d = 100$

For a detailed explanation of this problem, see [Question 17: Complexity Analysis](L5_4_17_explanation.md).

## Question 18

### Problem Statement
Calculate comprehensive evaluation metrics for a 4-class confusion matrix.

Confusion matrix:
```
     A   B   C   D
A   85   3   2   0
B    4  76   5   5
C    1   8  82   4
D    0   3   1  86
```

#### Task
1. Calculate overall accuracy and per-class accuracy
2. Compute precision, recall, and F1-score for each class
3. Calculate macro-averaged and micro-averaged precision, recall, F1
4. Compute balanced accuracy accounting for class sizes
5. Identify the most confusing class pairs and suggest improvements

For a detailed explanation of this problem, see [Question 18: Multi-class Metrics](L5_4_18_explanation.md).

## Question 19

### Problem Statement
Design strategies for severely imbalanced multi-class data.

Dataset distribution:
- Class 1: 8000 samples (80%)
- Class 2: 800 samples (8%)
- Class 3: 600 samples (6%)
- Class 4: 400 samples (4%)
- Class 5: 200 samples (2%)

#### Task
1. Calculate the class distribution for each OvR binary problem
2. Design class weights for the cost-sensitive loss function
3. Adjust decision thresholds to achieve balanced recall across classes
4. Design a combined over/under-sampling approach
5. Choose appropriate evaluation metrics for this imbalanced scenario

For a detailed explanation of this problem, see [Question 19: Imbalance Strategies](L5_4_19_explanation.md).

## Question 20

### Problem Statement
Design efficient hyperparameter tuning for multi-class RBF-SVM.

#### Task
1. For OvR with RBF kernels, define the hyperparameter search space for $C$ and $\gamma$
2. Compare grid search vs random search vs Bayesian optimization
3. Design stratified k-fold CV that maintains class balance in each fold
4. With limited computational budget, design an efficient search strategy
5. Use hyperparameters from one binary classifier to warm-start others

For a detailed explanation of this problem, see [Question 20: Hyperparameter Tuning](L5_4_20_explanation.md).