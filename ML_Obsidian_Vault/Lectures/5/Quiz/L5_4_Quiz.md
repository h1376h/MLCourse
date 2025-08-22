# Lecture 5.4: Multi-class SVM Approaches Quiz

## Overview
This quiz contains 10 questions covering different topics from section 5.4 of the lectures on One-vs-Rest, One-vs-One, Decision Strategies, Multi-class Classification Performance, and Error-Correcting Output Codes.

## Question 1

### Problem Statement
Consider a 4-class classification problem with classes A, B, C, and D.

#### Task
1. [🔍] Explain why standard binary SVMs cannot directly handle this multi-class problem
2. [📚] How many binary classifiers would you need to train using the One-vs-Rest (OvR) approach?
3. [📚] How many binary classifiers would you need to train using the One-vs-One (OvO) approach?
4. [📚] List all the binary classification problems for the OvO approach

For a detailed explanation of this problem, see [Question 1: Multi-class Problem Setup](L5_4_1_explanation.md).

## Question 2

### Problem Statement
Consider the One-vs-Rest (OvR) training procedure for a 3-class problem with classes {Red, Blue, Green}.

#### Task
1. [📚] Describe what the three binary classifiers would be trained to distinguish
2. [📚] For the "Red vs. Not-Red" classifier, what would be the positive and negative class labels for the training data?
3. [🔍] What is a potential class imbalance issue in OvR, and why does it occur?
4. [📚] During prediction, how do you combine the outputs of the three binary classifiers to make a final decision?

For a detailed explanation of this problem, see [Question 2: One-vs-Rest Training](L5_4_2_explanation.md).

## Question 3

### Problem Statement
Consider the decision-making process in One-vs-Rest classification.

#### Task
1. [📚] If the three OvR classifiers output scores: Red: +2.1, Blue: -0.5, Green: +0.8, what would be the predicted class?
2. [🔍] What happens if all classifiers output negative scores (no class claims the point)?
3. [🔍] What happens if multiple classifiers output positive scores?
4. [📚] How can you convert the raw SVM decision values into confidence scores or probabilities?

For a detailed explanation of this problem, see [Question 3: OvR Decision Making](L5_4_3_explanation.md).

## Question 4

### Problem Statement
Consider the One-vs-One (OvO) approach for a 4-class problem with classes {A, B, C, D}.

#### Task
1. [📚] List all six pairwise binary classifiers that need to be trained
2. [📚] For each test point, how many classifiers need to make predictions?
3. [📚] If the pairwise results are: A beats B, A beats C, A beats D, B beats C, B beats D, C beats D, what is the final prediction using majority voting?
4. [🔍] What happens if there's a tie in the voting (e.g., each class gets the same number of votes)?

For a detailed explanation of this problem, see [Question 4: One-vs-One Voting](L5_4_4_explanation.md).

## Question 5

### Problem Statement
Compare the computational complexity of OvR vs. OvO approaches.

#### Task
1. [📚] For $K$ classes, write the formulas for the number of binary classifiers in OvR and OvO
2. [📚] Calculate the number of classifiers for $K = 5, 10, 20, 100$ classes for both approaches
3. [🔍] At what number of classes does OvO require more classifiers than OvR?
4. [🔍] Compare the training time complexity: which approach scales better with the number of classes?
5. [📚] Compare the prediction time: which approach is faster during inference?

For a detailed explanation of this problem, see [Question 5: Computational Complexity Comparison](L5_4_5_explanation.md).

## Question 6

### Problem Statement
Consider the advantages and disadvantages of OvR vs. OvO approaches.

#### Task
1. [🔍] **OvR Advantage**: Why might OvR be preferred when you have many classes?
2. [🔍] **OvR Disadvantage**: What is the main issue with class imbalance in OvR?
3. [🔍] **OvO Advantage**: Why might OvO produce more balanced datasets for each binary classifier?
4. [🔍] **OvO Disadvantage**: What computational overhead does OvO introduce?
5. [📚] In what scenarios would you choose OvR over OvO, and vice versa?

For a detailed explanation of this problem, see [Question 6: OvR vs OvO Trade-offs](L5_4_6_explanation.md).

## Question 7

### Problem Statement
Consider Error-Correcting Output Codes (ECOC) for multi-class classification.

#### Task
1. [🔍] What is the basic idea behind ECOC for multi-class classification?
2. [📚] For a 4-class problem, design a simple ECOC matrix with 3 binary classifiers
3. [📚] How do you decode the predictions from multiple binary classifiers using your ECOC matrix?
4. [🔍] What advantage does ECOC provide over simple OvR or OvO approaches in terms of error correction?

For a detailed explanation of this problem, see [Question 7: Error-Correcting Output Codes](L5_4_7_explanation.md).

## Question 8

### Problem Statement
Consider multi-class evaluation metrics and their interpretation.

#### Task
1. [📚] For a 3-class confusion matrix, what metrics can you calculate?
2. [📚] How do you compute macro-averaged and micro-averaged precision and recall?
3. [🔍] When might macro-averaging be preferred over micro-averaging, and vice versa?
4. [📚] What is the difference between multi-class accuracy and balanced accuracy?
5. [🔍] How do you handle severe class imbalance in multi-class evaluation?

For a detailed explanation of this problem, see [Question 8: Multi-class Evaluation Metrics](L5_4_8_explanation.md).

## Question 9

### Problem Statement
Consider direct multi-class SVM formulations that optimize all classes simultaneously.

#### Task
1. [🔍] What is the main advantage of direct multi-class optimization over decomposition methods (OvR/OvO)?
2. [📚] How does the optimization problem change when we try to optimize all classes simultaneously?
3. [🔍] What are the computational challenges of direct multi-class SVMs?
4. [📚] Why are decomposition methods (OvR/OvO) still commonly used despite the availability of direct methods?

For a detailed explanation of this problem, see [Question 9: Direct Multi-class SVMs](L5_4_9_explanation.md).

## Question 10

### Problem Statement
Consider practical considerations for implementing multi-class SVMs.

#### Task
1. [📚] How would you handle class imbalance in a dataset where one class has 80% of the samples?
2. [🔍] What preprocessing steps are important for multi-class SVM (feature scaling, sampling, etc.)?
3. [📚] How would you approach hyperparameter tuning for a multi-class SVM?
4. [🔍] What strategies can you use to speed up training and prediction for large-scale multi-class problems?
5. [📚] How do you decide between OvR and OvO for a specific real-world application?

For a detailed explanation of this problem, see [Question 10: Practical Multi-class Implementation](L5_4_10_explanation.md).
