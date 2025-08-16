# Lecture 7.3: AdaBoost Algorithm Quiz

## Overview
This quiz tests your understanding of the AdaBoost algorithm, including weak learners, weight updates, algorithm steps, and theoretical foundations.

## Question 1

### Problem Statement
AdaBoost uses weak learners and adaptively updates sample weights.

#### Task
1. What is a weak learner in AdaBoost?
2. Why does AdaBoost focus on misclassified samples?
3. How does AdaBoost combine weak learners?
4. What is the main difference between AdaBoost and bagging?

**Answer**:
1. A weak learner is a model that performs slightly better than random guessing (>50% accuracy for binary classification)
2. AdaBoost focuses on misclassified samples to learn from mistakes and improve overall performance
3. AdaBoost combines weak learners using weighted voting based on their performance
4. Main difference: AdaBoost adaptively updates sample weights and uses weighted combination, while bagging uses equal weights and simple averaging

## Question 2

### Problem Statement
Consider an AdaBoost ensemble with 3 weak learners that have the following errors:
- Weak Learner 1: ε₁ = 0.3
- Weak Learner 2: ε₂ = 0.25
- Weak Learner 3: ε₃ = 0.2

#### Task
1. Calculate the weight α for each weak learner using the formula α = 0.5 × ln((1-ε)/ε)
2. Which weak learner has the highest weight?
3. What does a higher weight indicate about a weak learner?
4. How would the final prediction be calculated?

**Answer**:
1. α₁ = 0.5 × ln(0.7/0.3) ≈ 0.423
   α₂ = 0.5 × ln(0.75/0.25) ≈ 0.549
   α₃ = 0.5 × ln(0.8/0.2) ≈ 0.693
2. Weak Learner 3 has the highest weight (α₃ ≈ 0.693)
3. Higher weight indicates better performance (lower error rate)
4. Final prediction = sign(α₁×h₁ + α₂×h₂ + α₃×h₃) where h_i are the weak learner predictions

## Question 3

### Problem Statement
AdaBoost updates sample weights after each iteration.

#### Task
1. What happens to the weights of correctly classified samples?
2. What happens to the weights of misclassified samples?
3. Why does AdaBoost normalize weights after updating?
4. How does weight updating help AdaBoost learn?

**Answer**:
1. Correctly classified samples get their weights decreased
2. Misclassified samples get their weights increased
3. Weights are normalized to maintain a probability distribution and prevent numerical issues
4. Weight updating helps AdaBoost focus on difficult samples, forcing subsequent weak learners to learn from previous mistakes

## Question 4

### Problem Statement
AdaBoost has theoretical guarantees and convergence properties.

#### Task
1. What is the theoretical bound on AdaBoost's training error?
2. Why does AdaBoost typically not overfit even with many iterations?
3. What is the relationship between weak learner performance and ensemble performance?
4. When might AdaBoost fail to converge?

**Answer**:
1. Training error ≤ exp(-2 × Σ α²) where α are the weak learner weights
2. AdaBoost doesn't overfit because it focuses on reducing training error and has theoretical guarantees
3. Better weak learners (lower error rates) lead to better ensemble performance
4. AdaBoost might fail when weak learners have error rates ≥ 0.5 (worse than random guessing)

## Question 5

### Problem Statement
AdaBoost can be sensitive to noisy data and outliers.

#### Task
1. Why is AdaBoost sensitive to noisy data?
2. What happens to sample weights for outliers?
3. How can you make AdaBoost more robust to noise?
4. When would you choose AdaBoost over other ensemble methods?

**Answer**:
1. AdaBoost is sensitive to noise because it gives high weights to misclassified samples, including noisy ones
2. Outliers get high weights, making the algorithm focus too much on them
3. Make AdaBoost more robust by: using regularization, limiting iterations, or preprocessing data to remove noise
4. Choose AdaBoost when: you have clean data, want fast convergence, or need interpretable weak learners
