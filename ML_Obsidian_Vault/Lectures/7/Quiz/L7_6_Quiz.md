# Topic 7.6: Stacking and Blending Quiz

## Overview
This quiz tests your understanding of stacking and blending ensemble methods, including meta-learning, cross-validation, and combination strategies.

## Question 1

### Problem Statement
Stacking (stacked generalization) uses a meta-learner to combine base models.

#### Task
1. What is the difference between base learners and meta-learner in stacking?
2. Why is cross-validation important in stacking?
3. What is the purpose of the meta-feature matrix?
4. How does stacking differ from simple averaging?

**Answer**:
1. Base learners make predictions on the data, meta-learner learns how to combine these predictions optimally
2. Cross-validation prevents data leakage between base model training and meta-learner training
3. Meta-feature matrix contains predictions from base models, used to train the meta-learner
4. Stacking learns optimal combination weights, while simple averaging uses equal weights

## Question 2

### Problem Statement
Consider a stacking ensemble with 3 base models and 1000 training samples using 5-fold cross-validation.

#### Task
1. How many predictions will be generated for each base model?
2. What is the shape of the meta-feature matrix?
3. Why can't you use the same data for training base models and meta-learner?
4. What would happen if you ignored cross-validation in stacking?

**Answer**:
1. Each base model generates 1000 predictions (one per sample)
2. Meta-feature matrix shape: 1000 × 3 (samples × base models)
3. Using same data would cause data leakage and overfitting, as meta-learner would see training data
4. Without cross-validation, the meta-learner would overfit to the training data, leading to poor generalization

## Question 3

### Problem Statement
Blending is an alternative to stacking for combining models.

#### Task
1. How does blending differ from stacking?
2. What is the advantage of blending over stacking?
3. What is the disadvantage of blending?
4. When would you choose blending over stacking?

**Answer**:
1. Blending uses a holdout validation set instead of cross-validation to generate meta-features
2. Advantage: simpler implementation, faster training, less prone to overfitting
3. Disadvantage: uses less data for training base models, may be less robust
4. Choose blending when: you have limited computational resources, want simpler implementation, or have large datasets

## Question 4

### Problem Statement
Meta-learning in stacking requires careful design.

#### Task
1. What types of algorithms can be used as meta-learners?
2. Why might a simple meta-learner (like linear regression) work well?
3. What is the risk of using a complex meta-learner?
4. How do you evaluate the performance of a stacking ensemble?

**Answer**:
1. Meta-learners can be: linear models, decision trees, neural networks, or any supervised learning algorithm
2. Simple meta-learner works well because it's less prone to overfitting and can generalize better
3. Complex meta-learner may overfit to the meta-features, especially with limited data
4. Evaluate using cross-validation on the meta-features or a separate test set

## Question 5

### Problem Statement
Stacking and blending have different tradeoffs.

#### Task
1. Compare the computational complexity of stacking vs blending
2. Which method is more prone to overfitting?
3. How do you choose between stacking and blending?
4. What are the key considerations for successful stacking/blending?

**Answer**:
1. Stacking is more computationally expensive due to cross-validation, blending is faster
2. Stacking is more prone to overfitting due to complex meta-learning
3. Choose stacking for better performance, blending for simplicity and speed
4. Key considerations: sufficient data, diverse base models, proper validation, avoiding data leakage
