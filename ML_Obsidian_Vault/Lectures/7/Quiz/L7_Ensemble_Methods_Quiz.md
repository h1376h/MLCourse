# Lecture 7: Ensemble Methods Quiz

## Overview
This quiz contains 9 questions from different topics covered in section 7 of the lectures on Ensemble Methods, including AdaBoost, Bagging, Boosting, Random Forest, Stacking, and other ensemble learning techniques.

## Question 1

### Problem Statement
Consider a binary classification problem with 1000 training samples. You want to build an ensemble using bagging with 10 decision trees as base learners.

#### Task
1. How many samples will each tree be trained on using bootstrap sampling?
2. What is the expected number of unique samples in each bootstrap sample?
3. Calculate the probability that a specific sample is not selected in any bootstrap sample
4. Explain why bagging helps reduce variance in predictions

For a detailed explanation of this question, see [Question 1: Bagging and Bootstrap Sampling](L7_1_explanation.md).

## Question 2

### Problem Statement
You are implementing AdaBoost with decision stumps (depth-1 decision trees) as weak learners. After 5 iterations, you have the following weights for training samples: [0.1, 0.15, 0.2, 0.25, 0.3] and corresponding weak learner errors: [0.4, 0.35, 0.3, 0.25, 0.2].

#### Task
1. Calculate the weight update factor α for each iteration
2. Show how the sample weights change after the 5th iteration
3. What is the final ensemble prediction for a sample that gets predictions [1, -1, 1, -1, 1] from the weak learners?
4. Explain why AdaBoost focuses more on misclassified samples

For a detailed explanation of this question, see [Question 2: AdaBoost Weight Updates and Predictions](L7_2_explanation.md).

## Question 3

### Problem Statement
Consider a Random Forest with 100 trees trained on a dataset with 20 features and 1000 samples. Each tree uses bootstrap sampling and considers a random subset of 5 features at each split.

#### Task
1. What is the probability that a specific feature is considered at a given split?
2. How does feature subsampling help with tree diversity?
3. Calculate the expected number of trees that will use a specific feature
4. Explain the relationship between feature subsampling and overfitting

For a detailed explanation of this question, see [Question 3: Random Forest Feature Subsampling](L7_3_explanation.md).

## Question 4

### Problem Statement
You have three base models with the following performance on a validation set:
- Model A: 85% accuracy
- Model B: 82% accuracy  
- Model C: 78% accuracy

You want to create an ensemble using different combination strategies.

#### Task
1. Calculate the ensemble accuracy using simple majority voting
2. If you use weighted voting based on individual accuracies, what weights would you assign?
3. What is the minimum accuracy improvement needed for the ensemble to be worthwhile?
4. Explain why ensemble performance might be worse than the best individual model

For a detailed explanation of this question, see [Question 4: Ensemble Combination Strategies](L7_4_explanation.md).

## Question 5

### Problem Statement
Consider a stacking ensemble with 5 base models and a meta-learner. You have 1000 training samples and want to use 5-fold cross-validation for generating meta-features.

#### Task
1. How many predictions will be generated for each base model?
2. What is the shape of the meta-feature matrix used to train the meta-learner?
3. Explain why using the same data for training base models and meta-learner leads to overfitting
4. Suggest an alternative approach to avoid this overfitting issue

For a detailed explanation of this question, see [Question 5: Stacking and Cross-Validation](L7_5_explanation.md).

## Question 6

### Problem Statement
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

#### Task
1. Boosting always reduces bias compared to bagging.
2. Random Forest can handle missing values without imputation.
3. Stacking with linear regression as meta-learner always improves performance.
4. Ensemble diversity is more important than individual model accuracy.
5. AdaBoost is more prone to overfitting than Random Forest.

For a detailed explanation of these true/false questions, see [Question 6: Ensemble Methods True/False Statements](L7_6_explanation.md).

## Question 7

### Problem Statement
You have a dataset with 1000 samples and want to build an ensemble. You can choose between:
- Option A: 50 shallow trees (max_depth=3)
- Option B: 10 deep trees (max_depth=10)
- Option C: 100 very shallow trees (max_depth=1)

#### Task
1. Which option would likely have the highest training accuracy?
2. Which option would likely generalize best to unseen data?
3. Rank the options by computational complexity (training time)
4. Suggest a hybrid approach that might give better results

For a detailed explanation of this question, see [Question 7: Ensemble Size and Complexity Tradeoffs](L7_7_explanation.md).

## Question 8

### Problem Statement
The graphs below illustrate various concepts related to ensemble methods and their performance characteristics. Each visualization represents different aspects of ensemble learning.

![Bagging vs Boosting Performance](../Images/L7_Quiz_8/bagging_vs_boosting.png)
![Ensemble Size vs Error Rate](../Images/L7_Quiz_8/ensemble_size_error.png)
![Feature Importance in Ensemble](../Images/L7_Quiz_8/ensemble_feature_importance.png)

#### Task
Using only the information provided in these graphs (i.e., without any extra computation), determine:

1. Which ensemble method (Bagging or Boosting) shows better generalization performance?
2. What is the optimal ensemble size that minimizes error rate?
3. How does feature importance change as ensemble size increases?
4. Explain the relationship between ensemble size and overfitting based on the second graph.

For a detailed explanation of this question, see [Question 8: Visual Ensemble Analysis](L7_8_explanation.md).

## Question 9

### Problem Statement
The visualizations below illustrate various concepts from ensemble learning applied to a real-world classification problem. Each visualization represents different aspects of ensemble performance, diversity, and optimization.

![Model Diversity vs Ensemble Performance](../Images/L7_Quiz_9/diversity_performance.png)
![Learning Curves for Different Ensemble Methods](../Images/L7_Quiz_9/learning_curves_ensembles.png)
![Hyperparameter Tuning Impact](../Images/L7_Quiz_9/hyperparameter_tuning.png)

#### Task
Using only the information provided in these visualizations, answer the following questions:

1. What is the optimal level of diversity that maximizes ensemble performance?
2. Which ensemble method shows the fastest convergence during training?
3. How does hyperparameter tuning affect the gap between training and validation performance?
4. Explain why increasing model diversity beyond a certain point hurts performance.
5. Identify the ensemble method that would be most suitable for online learning based on the learning curves.

For a detailed explanation of this question, see [Question 9: Ensemble Methods Visualization Analysis](L7_9_explanation.md).
