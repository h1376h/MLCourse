# Lecture 7.4: Advanced Boosting Algorithms Quiz

## Overview
This quiz tests your understanding of advanced boosting algorithms, including gradient boosting, XGBoost, LightGBM, and CatBoost.

## Question 1

### Problem Statement
Gradient boosting is different from AdaBoost in how it constructs weak learners.

#### Task
1. How does gradient boosting differ from AdaBoost?
2. What is the role of gradients in gradient boosting?
3. Why is gradient boosting called "gradient" boosting?
4. What types of loss functions can gradient boosting handle?

**Answer**:
1. Gradient boosting builds weak learners sequentially to minimize a loss function, while AdaBoost focuses on sample weights
2. Gradients indicate the direction of steepest increase in the loss function
3. It's called "gradient" boosting because it uses gradients of the loss function to guide the learning process
4. Gradient boosting can handle various loss functions: regression (MSE), classification (log loss), and custom loss functions

## Question 2

### Problem Statement
XGBoost (Extreme Gradient Boosting) is an optimized version of gradient boosting.

#### Task
1. What does the "X" in XGBoost stand for?
2. What are the main optimizations in XGBoost?
3. How does XGBoost handle regularization?
4. What is the advantage of XGBoost over standard gradient boosting?

**Answer**:
1. "X" stands for "Extreme" - indicating enhanced performance and optimization
2. Main optimizations: parallel processing, cache-aware access, out-of-core computation, tree pruning
3. XGBoost uses L1 (Lasso) and L2 (Ridge) regularization on both leaf weights and tree structure
4. Advantages: faster training, better regularization, built-in cross-validation, handles missing values

## Question 3

### Problem Statement
LightGBM is designed for efficiency and speed.

#### Task
1. What does "Light" in LightGBM refer to?
2. How does LightGBM differ from XGBoost in tree construction?
3. What is the main advantage of LightGBM for large datasets?
4. When would you choose LightGBM over XGBoost?

**Answer**:
1. "Light" refers to the lightweight and fast nature of the algorithm
2. LightGBM uses leaf-wise tree growth instead of level-wise, creating more unbalanced but efficient trees
3. Main advantage: much faster training and lower memory usage, especially for large datasets
4. Choose LightGBM when: speed is crucial, memory is limited, or working with very large datasets

## Question 4

### Problem Statement
CatBoost is designed to handle categorical features efficiently.

#### Task
1. What does "Cat" in CatBoost refer to?
2. How does CatBoost handle categorical variables differently?
3. What is the main innovation in CatBoost?
4. When would you prefer CatBoost over other boosting algorithms?

**Answer**:
1. "Cat" refers to "Categorical" - the algorithm's strength in handling categorical features
2. CatBoost uses ordered boosting and target encoding to handle categorical variables without preprocessing
3. Main innovation: ordered boosting that prevents target leakage and overfitting
4. Prefer CatBoost when: you have many categorical features, want automatic feature handling, or need to avoid preprocessing

## Question 5

### Problem Statement
Regularization is important in advanced boosting algorithms to prevent overfitting.

#### Task
1. What are the main regularization techniques in advanced boosting?
2. How does early stopping work in boosting?
3. What is the relationship between learning rate and regularization?
4. How do you choose the optimal number of boosting iterations?

**Answer**:
1. Main regularization: L1/L2 regularization, tree depth limits, minimum samples per leaf, learning rate reduction
2. Early stopping monitors validation performance and stops training when it starts degrading
3. Lower learning rate provides implicit regularization by making smaller updates, reducing overfitting
4. Optimal iterations: use cross-validation to find the point where validation performance plateaus or starts decreasing
