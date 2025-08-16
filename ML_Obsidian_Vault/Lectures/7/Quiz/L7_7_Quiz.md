# Lecture 7.7: Ensemble Model Selection and Tuning Quiz

## Overview
This quiz tests your understanding of ensemble model selection, hyperparameter tuning, cross-validation strategies, and optimization techniques.

## Question 1

### Problem Statement
Choosing base learners for an ensemble is crucial for performance.

#### Task
1. What are three criteria for selecting base learners?
2. Why is diversity among base learners important?
3. What happens if all base learners are too similar?
4. Give an example of a good combination of base learners

**Answer**:
1. Three criteria: individual performance, diversity, computational efficiency
2. Diversity ensures models can correct each other's errors and provide complementary predictions
3. If all learners are too similar, the ensemble won't benefit from combination and may overfit
4. Good combination: decision tree + linear model + neural network (different algorithms, different biases)

## Question 2

### Problem Statement
Hyperparameter tuning is essential for ensemble performance.

#### Task
1. What are the main hyperparameters to tune in Random Forest?
2. What are the main hyperparameters to tune in AdaBoost?
3. How does cross-validation help with hyperparameter selection?
4. What is the tradeoff between tuning effort and performance gain?

**Answer**:
1. Random Forest: number of trees, max depth, min samples per leaf, features per split
2. AdaBoost: number of estimators, learning rate, base estimator type
3. Cross-validation provides unbiased estimates of hyperparameter performance
4. Tradeoff: more tuning may improve performance but increases computational cost and risk of overfitting

## Question 3

### Problem Statement
Ensemble size affects both performance and computational cost.

#### Task
1. How does increasing ensemble size typically affect performance?
2. What is the point of diminishing returns for ensemble size?
3. How do you balance performance and computational cost?
4. What factors influence the optimal ensemble size?

**Answer**:
1. Increasing ensemble size typically improves performance but with diminishing returns
2. Point of diminishing returns: when adding more models provides minimal performance improvement
3. Balance by finding the sweet spot where performance gain justifies computational cost
4. Factors: dataset size, complexity, base learner performance, available computational resources

## Question 4

### Problem Statement
Cross-validation strategies for ensembles require careful design.

#### Task
1. Why is nested cross-validation important for ensembles?
2. What is the difference between outer and inner cross-validation?
3. How do you avoid data leakage in ensemble cross-validation?
4. What are the computational tradeoffs of different CV strategies?

**Answer**:
1. Nested CV prevents overfitting by using separate folds for hyperparameter tuning and performance estimation
2. Outer CV estimates final performance, inner CV tunes hyperparameters
3. Avoid leakage by ensuring validation data for hyperparameter tuning is separate from training data
4. Tradeoffs: more folds = better estimates but higher computational cost, fewer folds = faster but less reliable

## Question 5

### Problem Statement
Ensemble interpretability is important for many applications.

#### Task
1. What makes ensembles harder to interpret than single models?
2. How can you make Random Forest more interpretable?
3. What is the tradeoff between interpretability and performance?
4. When is interpretability crucial for ensemble methods?

**Answer**:
1. Ensembles are harder to interpret because they combine multiple models with complex interactions
2. Make Random Forest interpretable by: limiting tree depth, using feature importance, visualizing individual trees
3. Tradeoff: simpler ensembles are more interpretable but may have lower performance
4. Interpretability is crucial in: medical diagnosis, financial decisions, legal applications, and when explaining decisions to stakeholders
