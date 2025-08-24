# Lecture 7.1: Foundations of Ensemble Learning Quiz

## Overview
This quiz contains 6 questions covering different topics from section 7.1 of the lectures on Foundations of Ensemble Learning, including ensemble concepts, bias-variance trade-off, types of ensembles, diversity importance, and combination strategies.

## Question 1

### Problem Statement
Consider three individual models that make predictions on the same test set:

| Model | Accuracy |
|-------|----------|
| A     | 75%      |
| B     | 78%      |
| C     | 72%      |

#### Task
1. What is the average accuracy of the individual models?
2. If you use simple majority voting, what is the minimum accuracy the ensemble could achieve?
3. What is the maximum accuracy the ensemble could achieve?
4. Explain why an ensemble might perform better than the average of individual models.

For a detailed explanation of this question, see [Question 1: Ensemble Performance Analysis](L7_1_1_explanation.md).

## Question 2

### Problem Statement
Ensemble learning combines multiple base learners to improve performance.

#### Task
1. Name the three main types of ensemble methods.
2. What is the key principle behind ensemble learning?
3. Why is diversity among base learners important?
4. Give an example of when ensemble learning might not help.

For a detailed explanation of this question, see [Question 2: Ensemble Learning Principles](L7_1_2_explanation.md).

## Question 3

### Problem Statement
Consider an ensemble with 5 base models that use different combination strategies.

#### Task
1. If using simple averaging, what weight does each model have?
2. If using weighted averaging with weights $[0.3, 0.2, 0.2, 0.15, 0.15]$, what is the sum of weights?
3. What is the advantage of weighted averaging over simple averaging?
4. When might simple averaging be preferred over weighted averaging?

For a detailed explanation of this question, see [Question 3: Ensemble Combination Strategies](L7_1_3_explanation.md).

## Question 4

### Problem Statement
Ensemble diversity is crucial for good performance.

#### Task
1. What are three ways to create diversity among base learners?
2. Why is too much diversity a problem?
3. What is the relationship between diversity and ensemble size?
4. How can you measure diversity in an ensemble?

For a detailed explanation of this question, see [Question 4: Ensemble Diversity](L7_1_4_explanation.md).

## Question 5

### Problem Statement
Ensemble learning has both advantages and challenges.

#### Task
1. **Advantage 1**: How does ensemble learning improve generalization?
2. **Advantage 2**: How does ensemble learning reduce overfitting?
3. **Challenge 1**: What are the computational costs of ensembles?
4. **Challenge 2**: How do ensembles affect model interpretability?

For a detailed explanation of this question, see [Question 5: Ensemble Advantages and Challenges](L7_1_5_explanation.md).

## Question 6

### Problem Statement
Ensembles are often used to manage the bias-variance trade-off.

#### Task
1. [📚] According to the bias-variance trade-off, what are the characteristics of a typical "weak learner"?
2. [📚] Which ensemble technique is primarily used to decrease the variance of unstable models?
3. [📚] Which ensemble technique is primarily used to decrease the bias of weak learners?
4. [📚] Explain why averaging the outputs of multiple high-variance models can lead to an overall reduction in the ensemble's variance.

For a detailed explanation of this question, see [Question 6: Bias-Variance Trade-off](L7_1_6_explanation.md).