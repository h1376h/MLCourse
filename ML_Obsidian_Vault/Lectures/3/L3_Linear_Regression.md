# Lecture 3: Linear Regression

## Overview
This module covers fundamental concepts in linear regression, one of the most widely used statistical methods in machine learning. You'll learn everything from basic theory to advanced applications and optimization techniques for linear models, providing a foundation for many other ML algorithms. The module emphasizes the bias-variance tradeoff, different hypothesis spaces, cost function optimization, and practical methods for model evaluation and regularization.

### Lecture 3.1: Mathematical Foundations of Linear Models
- [[L3_1_Linear_Model_Theory|Linear Model Theory]]: Vector spaces and linear algebra fundamentals
- Matrix Properties in Linear Models: Hat matrix, projection, and eigendecomposition
- Best Unrestricted Regression Function: Mathematical derivation and properties
- Gauss-Markov Theorem: BLUE properties and optimality
- Statistical Properties: Bias, variance, and efficiency of estimators
- Error Decomposition: Breaking down prediction error into components
- Mathematical Foundations Examples: Theoretical examples and proofs
- Required Reading: Sections 3.1-3.1.3 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L3_1_Quiz]]: Test your understanding of linear model mathematics

### Lecture 3.2: Simple Linear Regression
- [[L3_2_Linear_Regression_Formulation|Linear Regression Formulation]]: Problem setup and notation
- [[L3_2_Simple_Linear_Regression|Simple Linear Regression]]: Modeling with a single independent variable 
- Univariate vs. Multivariate Formulations: Mathematical representations
- [[L3_2_Cost_Function|Cost Function]]: MSE optimization objectives and visualization
- [[L3_2_Least_Squares|Least Squares Method]]: Derivation and geometric interpretation
- [[L3_2_Analytical_Solution|Analytical Solution]]: Closed-form solution with normal equations
- [[L3_2_Error_Models|Error Models]]: Gaussian and other error distributions in regression
- Linear Regression Assumptions: Testing and implications
- [[L3_2_Examples|Simple Linear Regression Examples]]: Practical applications with code
- Required Reading: Section 3.1.4-3.1.5 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L3_2_Quiz]]: Test your understanding of simple linear regression

### Lecture 3.3: Probabilistic View of Linear Regression
- Probabilistic Framework: Linear regression as probabilistic inference
- MLE for Linear Regression: Maximum likelihood formulation
- Likelihood Function: Constructing the likelihood for linear models
- Gaussian Assumptions: Noise models and implications
- Maximum Likelihood Derivation: Step-by-step derivation
- Equivalence to Least Squares: Connection between MLE and OLS
- Parameter Distribution: Uncertainty in parameter estimates
- Predictive Distribution: Making probabilistic predictions
- Structural and Approximation Error: Error decomposition analysis
- Log-Likelihood Optimization: Numerical considerations
- Probabilistic Linear Regression Examples: Implementation and demonstrations
- Required Reading: Sections 3.1.5-3.2.2 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: L3_3_Quiz: Test your understanding of probabilistic regression

### Lecture 3.4: Multiple Linear Regression
- Multiple Linear Regression: Extending to multiple variables
- Matrix Formulation: Vectorized representation for efficiency
- Multicollinearity: Causes, detection, and solutions
- Interaction Terms: Modeling feature interactions
- Feature Engineering: Creating effective features
- Dummy Variables: Handling categorical predictors
- Polynomial Regression: Modeling nonlinear relationships with examples
- Basis Function Expansion: Radial basis functions and splines
- Curse of Dimensionality: Challenges with high-dimensional data
- Multiple Regression Examples: Practical applications with code
- Required Reading: Section 3.2.3-3.2.5 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: L3_4_Quiz: Test your understanding of multiple linear regression

### Lecture 3.5: Optimization Techniques for Linear Regression
- Gradient Descent: Iterative optimization for linear regression
- Batch Gradient Descent: Full dataset approach with math derivation
- Stochastic Gradient Descent: Scaling to large datasets
- Mini-batch Gradient Descent: Balancing efficiency and stability
- Normal Equations vs Gradient Descent: Tradeoffs and considerations
- Learning Rate Selection: Strategies for convergence
- Least Mean Squares (LMS) Algorithm: Implementation details
- Feature Scaling: Preprocessing for optimization
- Convergence Analysis: Theoretical guarantees
- Advanced Optimizers: Momentum, RMSprop, Adam
- Online Learning: Sequential updates for streaming data
- Optimization Examples: Implementation of various methods
- Required Reading: Chapter 4.3 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: L3_5_Quiz: Test your understanding of optimization techniques

### Lecture 3.6: Model Evaluation and Validation
- Error Metrics: MSE, MAE, RMSE, R-squared
- Training vs Testing Error: Generalization assessment
- Train-Test Split: Data partitioning strategies
- Simple Hold-Out Method: Implementation and limitations
- Cross Validation: K-fold and leave-one-out methods
- Learning Curves: Diagnosing bias and variance
- Expected Training and True Error Curves: Analysis of model behavior
- Residual Analysis: Validating model assumptions
- Hypothesis Testing: Statistical significance of coefficients
- Model Complexity Selection: Balancing fit and generalization
- Information Criteria: AIC, BIC for model selection
- Evaluation Examples: Practical implementation of evaluation methods
- Required Reading: Chapter 1.3 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: L3_6_Quiz: Test your understanding of model evaluation

### Lecture 3.7: Regularization in Linear Models
- Overfitting in Linear Models: Detection and prevention
- Bias-Variance Tradeoff: Fundamental ML concept with visual examples
- Regularization Theory: Mathematical foundations
- Ridge Regression (L2): Implementation, effects, and mathematical derivation
- Regularization Path: Effect of regularization parameter lambda
- Lasso Regression (L1): Feature selection capabilities
- Elastic Net: Combining L1 and L2 regularization
- Polynomial Coefficient Analysis: Effect of regularization on coefficient values
- Bayesian Interpretation: Priors and regularization
- Selecting Regularization Parameter: Cross-validation approaches
- Early Stopping: Implicit regularization in iterative methods
- Regularization Examples: Practical code demonstrations
- Required Reading: Chapter 3.3 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: L3_7_Quiz: Test your understanding of regularization in linear models

### Lecture 3.8: Advanced Linear Regression Topics
- Feature Selection Techniques: Methods for optimal feature sets
- Robust Regression: Handling outliers and non-Gaussian noise
- Weighted Least Squares: Accommodating heteroscedasticity
- Generalized Linear Models: Beyond the Gaussian case
- Radial Basis Functions: Prototypes and similarity-based prediction
- Gaussian Basis Functions: Implementation and applications
- Regression Diagnostics: Advanced validation techniques
- Variable Selection Methods: Stepwise, LARS, elastic net
- Bayesian Linear Regression: Full posterior approach
- Time Series Regression: Handling temporal data
- Instrumental Variables: Addressing causality
- Quantile Regression: Beyond mean prediction
- Implementation Considerations: Scaling and stability
- Advanced Examples: Complex applications and case studies
- Required Reading: Chapters 3.4-3.5 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: L3_8_Quiz: Test your understanding of advanced regression topics

## Related Videos
- [Mathematical Foundations of Linear Models](https://www.youtube.com/watch?v=zPG4NjIkCjc)
- [Simple Linear Regression](https://www.youtube.com/watch?v=Dn6b9fCIUpM)
- [Probabilistic View of Linear Regression](https://www.youtube.com/watch?v=dQw_w4aGqYc)
- [Multiple Linear Regression Explained](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- [Gradient Descent for Linear Regression](https://www.youtube.com/watch?v=fSytzGwwBVw)
- [Model Evaluation Techniques](https://www.youtube.com/watch?v=Q81RR3yKn30)
- [Regularization in Linear Models](https://www.youtube.com/watch?v=yR2Paq9Zbk8)
- [Advanced Linear Regression Topics](https://www.youtube.com/watch?v=yR2Paq9Zbk8)

## All Quizzes
Test your understanding with these quizzes:
- [[L3_1_Quiz]]: Mathematical Foundations of Linear Models
- [[L3_2_Quiz]]: Simple Linear Regression
- L3_3_Quiz: Probabilistic View of Linear Regression
- L3_4_Quiz: Multiple Linear Regression
- L3_5_Quiz: Optimization for Linear Regression
- L3_6_Quiz: Model Evaluation and Validation
- L3_7_Quiz: Regularization in Linear Models
- L3_8_Quiz: Advanced Linear Regression Topics