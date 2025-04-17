# Machine Learning Course - Questions and Quizzes

This document contains only the questions, quizzes, and practice exercises for the Machine Learning course, organized by lecture. Use this for focused study and exam preparation.

## Course Information

- **Instructor**: Dr. Zahra Rahimi
- **For comprehensive course materials**: See [[index|main index]]

## Lecture 1: Machine Learning Fundamentals

### Types of Learning (Section 1.1)
- **Quiz**: [[Lectures/1/Quiz/L1_1_Quiz|Quiz on Types of Learning]]
  - Identifying appropriate learning approaches
  - Understanding supervised, unsupervised, reinforcement, and semi-supervised learning
  - Application scenarios for different learning types
- **Examples**: 
  - Email spam detection (supervised learning)
  - Customer segmentation (unsupervised learning)
  - Game-playing agents (reinforcement learning)
  - Semi-supervised image recognition
- **Slides**: 
  - Slide 5: Distinguish between regression and classification tasks
  - Slide 8: Identify learning types from real-world scenarios
  - Slide 12: Match algorithms to appropriate learning paradigms

### Identifying Learning Problems (Section 1.2)
- **Quiz**: [[Lectures/1/Quiz/L1_2_Quiz|Quiz on Identifying Learning Problems]]
  - Pattern recognition techniques
  - Learning problem formulation
  - Real-world applications
- **Examples**:
  - Face recognition system design
  - Natural language processing problem formulation
  - Anomaly detection in network traffic
  - Time series prediction for stock prices
- **Slides**:
  - Slide 15: Formalize a given problem as a learning task
  - Slide 18: Identify input and output variables in learning scenarios
  - Slide 22: Determine appropriate feature representations

### Generalization Concepts (Section 1.3)
- **Quiz**: [[Lectures/1/Quiz/L1_3_Quiz|Quiz on Generalization Concepts]]
  - Underfitting and overfitting
  - Bias-variance tradeoff
  - Regularization techniques
- **Examples**:
  - Polynomial regression with varying degrees
  - Cross-validation strategies implementation
  - Learning curve interpretation
  - Feature selection to reduce overfitting
- **Slides**:
  - Slide 27: Analyze model performance graphs for under/overfitting
  - Slide 31: Calculate bias and variance components
  - Slide 35: Apply regularization to given problems

### Well-posed Learning Problems (Section 1.4)
- **Quiz**: [[Lectures/1/Quiz/L1_4_Quiz|Quiz on Well-posed Learning Problems]]
  - Core theoretical foundations
  - Examples of properly formulated learning problems
  - Problem identification for ML suitability
- **Examples**:
  - Credit risk assessment formulation
  - Medical diagnosis modeling
  - Converting ill-posed problems to well-posed ones
  - Problem identification and boundary definition
- **Slides**:
  - Slide 41: Evaluate learning problem formulations
  - Slide 45: Apply Hadamard's criteria to learning problems
  - Slide 48: Transform problems for ML suitability

## Lecture 2: Probability and Statistical Foundations

### Probability Fundamentals
- **Quiz**: [[Lectures/2/Quiz/L2_1_Quiz|Quiz on Probability Fundamentals]]
  - Basic probability concepts
  - Important distributions for ML
  - Probability calculations
- **Examples**:
  - Bayesian updating in medical tests
  - Normal distribution applications in data normalization
  - Bernoulli trials in A/B testing
  - Joint probability calculations for feature relationships
- **Slides**:
  - Slide 5: Calculate conditional probabilities
  - Slide 9: Work with probability distributions
  - Slide 14: Apply Bayes' theorem to machine learning problems

### Information Theory and Entropy
- **Quiz**: [[Lectures/2/Quiz/L2_2_Quiz|Quiz on Information Theory]]
  - Entropy and information gain
  - Cross-entropy and KL divergence
  - Applications to ML
- **Examples**:
  - Decision tree information gain calculations
  - Feature selection using mutual information
  - Model comparison with KL divergence
  - Entropy-based text analysis
- **Slides**:
  - Slide 20: Calculate entropy for given probability distributions
  - Slide 25: Compute information gain for feature evaluation
  - Slide 28: Apply cross-entropy in classification problems

### Statistical Estimation
- **Quiz**: [[Lectures/2/Quiz/L2_3_Quiz|Quiz on Statistical Estimation]]
  - Maximum Likelihood Estimation (MLE)
  - Bias and consistency
  - Parameter estimation
- **Examples**:
  - MLE for Gaussian distribution parameters
  - Sample mean estimation with confidence intervals
  - Consistency proofs for estimators
  - Estimation in linear models
- **Slides**:
  - Slide 32: Derive maximum likelihood estimators
  - Slide 38: Analyze estimator properties
  - Slide 42: Apply MLE to machine learning models

### Bayesian Inference
- **Quiz**: [[Lectures/2/Quiz/L2_4_Quiz|Quiz on Bayesian Inference]]
  - Prior and posterior distributions
  - Maximum A Posteriori (MAP) estimation
  - Bayesian vs. frequentist approaches
- **Examples**:
  - Bayesian linear regression implementation
  - Prior selection strategies
  - Posterior computation for classification
  - Bayesian model comparison
- **Slides**:
  - Slide 46: Compare MAP and MLE approaches
  - Slide 50: Apply Bayesian inference to parameter estimation
  - Slide 54: Select appropriate priors for learning problems

## Lecture 3: Linear Regression

### Linear Regression Fundamentals
- **Quiz**: [[Lectures/3/Quiz/L3_1_Quiz|Quiz on Linear Regression Fundamentals]]
  - Simple linear regression
  - Loss functions
  - Analytical solutions
- **Examples**:
  - House price prediction implementation
  - Different loss function comparisons
  - Derivation of normal equations
  - Coefficient interpretation in real datasets
- **Slides**:
  - Slide 4: Formulate regression problems mathematically
  - Slide 8: Derive the normal equation solution
  - Slide 12: Interpret regression coefficients

### Optimization Techniques
- **Quiz**: [[Lectures/3/Quiz/L3_2_Quiz|Quiz on Optimization Techniques]]
  - Gradient Descent
  - Stochastic Gradient Descent
  - Learning rate selection
- **Examples**:
  - Implementing batch gradient descent
  - Learning rate scheduling strategies
  - SGD with momentum implementation
  - Convergence visualization
- **Slides**:
  - Slide 16: Calculate gradients for optimization
  - Slide 22: Compare optimization algorithm behaviors
  - Slide 25: Select appropriate learning rates

### Regularization Techniques
- **Quiz**: [[Lectures/3/Quiz/L3_3_Quiz|Quiz on Regularization]]
  - Ridge regression (L2)
  - Lasso regression (L1)
  - Elastic Net
- **Examples**:
  - Feature selection with Lasso
  - Ridge regression for multicollinearity
  - Regularization path analysis
  - Elastic Net for sparse high-dimensional data
- **Slides**:
  - Slide 30: Apply regularization to ill-conditioned problems
  - Slide 34: Select regularization parameters
  - Slide 37: Compare regularization techniques

### Model Evaluation
- **Quiz**: [[Lectures/3/Quiz/L3_4_Quiz|Quiz on Model Evaluation]]
  - Train/test splits
  - Cross-validation techniques
  - Performance metrics
- **Examples**:
  - K-fold cross-validation implementation
  - Learning curve interpretation
  - RMSE vs. MAE comparison
  - R-squared analysis
- **Slides**:
  - Slide 42: Design validation strategies
  - Slide 46: Calculate and interpret performance metrics
  - Slide 50: Apply model selection techniques

## Lecture 4: Linear Classifiers

### Perceptron Algorithm
- **Quiz**: [[Lectures/4/Quiz/L4_1_Quiz|Quiz on Perceptron Algorithm]]
  - Binary classification
  - Decision boundaries
  - Limitations of perceptron
- **Examples**:
  - Perceptron implementation for binary classification
  - Decision boundary visualization
  - Convergence analysis
  - Linearly non-separable data scenarios
- **Slides**:
  - Slide 5: Trace perceptron learning algorithm steps
  - Slide 10: Analyze perceptron convergence conditions
  - Slide 15: Identify perceptron limitations

### Linear Classification
- **Quiz**: [[Lectures/4/Quiz/L4_2_Quiz|Quiz on Linear Classification]]
  - Multi-class classification
  - One-vs-All and One-vs-One strategies
  - Error measures
- **Examples**:
  - Handwritten digit classification
  - One-vs-All strategy implementation
  - Error measure comparison
  - Decision boundary visualization for multiple classes
- **Slides**:
  - Slide 20: Compare multi-class classification strategies
  - Slide 25: Calculate error measures for classification
  - Slide 30: Design linear classification systems

## Lecture 5: Logistic Regression

### Binary Classification
- **Quiz**: [[Lectures/5/Quiz/L5_1_Quiz|Quiz on Binary Classification]]
  - Logistic function
  - Decision boundaries
  - Probability interpretation
- **Examples**:
  - Credit approval system implementation
  - Probability threshold selection
  - ROC curve analysis
  - Feature importance interpretation
- **Slides**:
  - Slide 5: Derive logistic regression model
  - Slide 10: Interpret logistic regression coefficients
  - Slide 15: Analyze decision boundary properties

### Multi-class Logistic Regression
- **Quiz**: [[Lectures/5/Quiz/L5_2_Quiz|Quiz on Multi-class Logistic Regression]]
  - Softmax function
  - Cross-entropy loss
  - Implementation challenges
- **Examples**:
  - MNIST digit classification
  - Softmax implementation details
  - Loss function comparison
  - Handling class imbalance
- **Slides**:
  - Slide 20: Derive softmax regression equations
  - Slide 25: Calculate cross-entropy loss
  - Slide 30: Compare multi-class strategies

### Optimization Methods
- **Quiz**: [[Lectures/5/Quiz/L5_3_Quiz|Quiz on Optimization Methods]]
  - Newton's method
  - Gradient-based methods
  - Convergence properties
- **Examples**:
  - Newton's method implementation
  - Conjugate gradient approach
  - Optimization landscape visualization
  - Second-order methods comparison
- **Slides**:
  - Slide 35: Derive Newton's update equations
  - Slide 40: Analyze convergence rates
  - Slide 45: Select optimization strategies for specific problems

## Lecture 6: Support Vector Machines

### Maximum Margin Classifiers
- **Quiz**: [[Lectures/6/Quiz/L6_1_Quiz|Quiz on Maximum Margin Classifiers]]
  - Optimal separating hyperplanes
  - Support vectors
  - Margin maximization
- **Examples**:
  - Hard margin SVM implementation
  - Geometric interpretation of margin
  - Support vector identification
  - Dual form derivation
- **Slides**:
  - Slide 5: Formulate the margin maximization problem
  - Slide 10: Identify support vectors in examples
  - Slide 15: Convert primal to dual form

### Kernel Methods
- **Quiz**: [[Lectures/6/Quiz/L6_2_Quiz|Quiz on Kernel Methods]]
  - Kernel trick
  - Common kernel functions
  - Feature space transformations
- **Examples**:
  - RBF kernel implementation
  - Kernel selection strategies
  - Feature mapping visualization
  - Mercer's theorem application
- **Slides**:
  - Slide 20: Apply the kernel trick to non-linear problems
  - Slide 25: Calculate kernel matrices for given data
  - Slide 30: Select appropriate kernels for specific problems

### SVM Variants
- **Quiz**: [[Lectures/6/Quiz/L6_3_Quiz|Quiz on SVM Variants]]
  - Soft margin SVM
  - Multi-class SVMs
  - SVM regression
- **Examples**:
  - C parameter tuning for soft margin SVM
  - One-vs-One implementation for multi-class SVM
  - SVR for housing price prediction
  - Comparison with other regression methods
- **Slides**:
  - Slide 35: Formulate the soft margin optimization problem
  - Slide 40: Apply SVM to regression tasks
  - Slide 45: Design multi-class SVM systems

## Practice Exams
- [[Practice_Exam_1]]: Covers Lectures 1-3
- [[Practice_Exam_2]]: Covers Lectures 4-6
- [[Final_Exam_Practice]]: Comprehensive practice covering all lectures

## How to Use This Document

1. Complete the quiz for each section before moving to the next
2. Review incorrect answers and revisit the related lecture material
3. Use practice exams to assess overall understanding
4. For comprehensive content and explanations, refer to the [[index|main index]] 