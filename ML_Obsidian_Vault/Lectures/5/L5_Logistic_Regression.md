# Lecture 5: Logistic Regression

## Overview
This module explores logistic regression, a fundamental classification algorithm that extends linear models to probability-based classification. You'll learn about the theory, implementation, and optimization of logistic regression for both binary and multi-class problems, as well as regularization techniques to prevent overfitting.

### Lecture 5.1: Binary Logistic Regression Foundations
- [[L5_1_Classification_Probability|Classification as Probability]]: Probabilistic approach to classification
- [[L5_1_Logistic_Regression_Model|Logistic Regression Model]]: Core concepts and formulation
- [[L5_1_Linear_vs_Logistic|Linear vs. Logistic Regression]]: Key differences and use cases
- [[L5_1_Binary_Classification|Binary Classification]]: Two-class problem formulation
- [[L5_1_Sigmoid_Function|Sigmoid Function]]: Mathematical properties and role in logistic regression
- [[L5_1_Decision_Boundary|Decision Boundary]]: Geometric interpretation in feature space
- [[L5_1_Probability_Interpretation|Probability Interpretation]]: Understanding predicted probabilities
- [[L5_1_Threshold_Selection|Threshold Selection]]: Setting classification boundaries
- [[L5_1_Examples|Binary Logistic Regression Examples]]: Implementation with visualizations
- Required Reading: Chapter 4.3.2 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L5_1_Quiz]]: Test your understanding of binary logistic regression

### Lecture 5.2: Maximum Likelihood for Logistic Regression
- [[L5_2_Likelihood_Function|Likelihood Function]]: Construction for logistic regression
- [[L5_2_Log_Likelihood|Log-Likelihood]]: Converting to log space for numerical stability
- [[L5_2_MLE_Derivation|MLE Derivation]]: Mathematical derivation of parameter estimation
- [[L5_2_Cross_Entropy_Loss|Cross Entropy Loss]]: Connection to maximum likelihood
- [[L5_2_Cost_Function|Cost Function]]: Optimization objective for logistic regression
- [[L5_2_Gradient_Computation|Gradient Computation]]: Deriving gradients for optimization
- [[L5_2_Hessian_Matrix|Hessian Matrix]]: Second-order derivatives
- [[L5_2_Convexity|Convexity Properties]]: Guarantee of global optimum
- [[L5_2_MLE_Examples|MLE Examples]]: Step-by-step implementation of MLE
- Required Reading: Chapter 4.3.3 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L5_2_Quiz]]: Test your understanding of maximum likelihood estimation

### Lecture 5.3: Optimization Methods for Logistic Regression
- [[L5_3_Optimization_Overview|Optimization Overview]]: Techniques for logistic regression
- [[L5_3_Gradient_Descent|Gradient Descent]]: First-order optimization
- [[L5_3_Batch_vs_Stochastic|Batch vs. Stochastic Gradient Descent]]: Tradeoffs and implementation
- [[L5_3_Learning_Rate|Learning Rate Selection]]: Strategies for convergence
- [[L5_3_Newtons_Method|Newton's Method]]: Second-order optimization technique
- [[L5_3_Quasi_Newton|Quasi-Newton Methods]]: BFGS and L-BFGS algorithms
- [[L5_3_Conjugate_Gradient|Conjugate Gradient]]: Alternative optimization approach
- [[L5_3_Convergence_Criteria|Convergence Criteria]]: Determining when to stop
- [[L5_3_Optimization_Examples|Optimization Examples]]: Implementation of various methods
- Required Reading: Chapter 4.3.3 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L5_3_Quiz]]: Test your understanding of optimization methods

### Lecture 5.4: Multi-class Logistic Regression
- [[L5_4_Multi_Class_Extension|Multi-Class Extension]]: Beyond binary classification
- [[L5_4_One_vs_Rest|One-vs-Rest Approach]]: Binary decomposition strategy
- [[L5_4_Multinomial_Logistic|Multinomial Logistic Regression]]: Direct multi-class approach
- [[L5_4_Softmax_Function|Softmax Function]]: Generalizing sigmoid to multiple classes
- [[L5_4_Softmax_Properties|Softmax Properties]]: Mathematical characteristics and behavior
- [[L5_4_Multi_Class_MLE|Multi-Class MLE]]: Maximum likelihood for multiple classes
- [[L5_4_Multi_Class_Cross_Entropy|Multi-Class Cross Entropy]]: Loss function formulation
- [[L5_4_Gradient_Multi_Class|Gradient Computation for Multi-Class]]: Optimization
- [[L5_4_Multi_Class_Examples|Multi-Class Examples]]: Implementation with visualizations
- Required Reading: Chapter 4.3.4 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L5_4_Quiz]]: Test your understanding of multi-class logistic regression

### Lecture 5.5: Regularized Logistic Regression
- [[L5_5_Overfitting_Classification|Overfitting in Classification]]: Recognizing and preventing
- [[L5_5_Regularization_Theory|Regularization Theory]]: Controlling model complexity
- [[L5_5_L1_Regularization|L1 Regularization]]: Lasso penalty for feature selection
- [[L5_5_L2_Regularization|L2 Regularization]]: Ridge penalty for coefficient shrinkage
- [[L5_5_Elastic_Net|Elastic Net]]: Combining L1 and L2 regularization
- [[L5_5_Regularization_Paths|Regularization Paths]]: Effect of regularization parameter
- [[L5_5_Bayesian_Perspective|Bayesian Perspective]]: Priors and regularization
- [[L5_5_Regularization_Parameter|Regularization Parameter Selection]]: Cross-validation
- [[L5_5_Regularized_Examples|Regularized Examples]]: Implementation and effects
- Required Reading: Chapter 4.3.2 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L5_5_Quiz]]: Test your understanding of regularized logistic regression

### Lecture 5.6: Evaluation Metrics for Logistic Regression
- [[L5_6_Classification_Metrics|Classification Metrics]]: Accuracy, precision, recall, F1-score
- [[L5_6_Confusion_Matrix|Confusion Matrix]]: Comprehensive evaluation framework
- [[L5_6_ROC_Curves|ROC Curves]]: Receiver Operating Characteristic analysis
- [[L5_6_AUC|Area Under Curve (AUC)]]: Aggregate performance measure
- [[L5_6_Precision_Recall|Precision-Recall Curves]]: Alternative to ROC for imbalanced data
- [[L5_6_Calibration|Probability Calibration]]: Reliability of predicted probabilities
- [[L5_6_Log_Loss|Log Loss]]: Evaluating probabilistic predictions
- [[L5_6_Brier_Score|Brier Score]]: Squared error for probabilistic predictions
- [[L5_6_Evaluation_Examples|Evaluation Examples]]: Practical metrics implementation
- Required Reading: Chapter 4.5 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L5_6_Quiz]]: Test your understanding of evaluation metrics

### Lecture 5.7: Comparison with Other Classifiers
- [[L5_7_Linear_Classifiers_Comparison|Linear Classifiers Comparison]]: Logistic vs. others
- [[L5_7_Logistic_vs_LDA|Logistic Regression vs. LDA]]: Probabilistic approaches compared
- [[L5_7_Logistic_vs_SVM|Logistic Regression vs. SVM]]: Margin-based comparison
- [[L5_7_Logistic_vs_Decision_Trees|Logistic Regression vs. Decision Trees]]: Interpretability
- [[L5_7_Logistic_vs_Neural_Networks|Logistic Regression vs. Neural Networks]]: Connections
- [[L5_7_Model_Selection|Model Selection]]: Choosing the right classifier
- [[L5_7_Ensembling|Ensembling Logistic Models]]: Boosting and bagging
- [[L5_7_Computational_Complexity|Computational Complexity]]: Training and inference costs
- [[L5_7_Comparison_Examples|Classifier Comparison Examples]]: Empirical evaluation
- Required Reading: Chapter 4.5 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L5_7_Quiz]]: Test your understanding of classifier comparisons

### Lecture 5.8: Advanced Logistic Regression Topics
- [[L5_8_Feature_Engineering|Feature Engineering]]: Creating effective features
- [[L5_8_Feature_Selection|Feature Selection]]: Identifying relevant features
- [[L5_8_Handling_Missing_Data|Handling Missing Data]]: Strategies for incomplete datasets
- [[L5_8_Imbalanced_Classes|Imbalanced Classes]]: Techniques for uneven class distributions
- [[L5_8_Sparsity|Sparsity in Logistic Regression]]: Efficient representation
- [[L5_8_Interpretability|Model Interpretability]]: Understanding coefficients
- [[L5_8_Online_Learning|Online Learning]]: Incremental training
- [[L5_8_Large_Scale_Logistic|Large-Scale Logistic Regression]]: Scaling to big data
- [[L5_8_Advanced_Examples|Advanced Examples]]: Complex real-world applications
- Required Reading: Chapter 4.5 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L5_8_Quiz]]: Test your understanding of advanced logistic regression topics

## Programming Resources
- [[L5_Logistic_Regression_Python_Guide|Python Implementation Guide]]: Step-by-step implementation
- [[L5_Binary_Logistic_Regression_Implementation|Building Binary Logistic Regression]]: Code tutorial
- [[L5_Multi_Class_Implementation|Multi-class Logistic Regression Implementation]]: Softmax approach
- [[L5_Scikit_Learn_Logistic|Using Scikit-learn for Logistic Regression]]: Library tutorial
- [[L5_Visualizing_Decision_Boundaries|Visualizing Decision Boundaries]]: Plotting techniques
- [[L5_Regularization_Implementation|Implementing Regularization]]: Code examples
- [[L5_Advanced_Logistic_Code|Advanced Logistic Regression Techniques]]: Implementation

## Related Slides
*(not included in the repo)*
- Binary_Logistic_Regression.pdf
- Maximum_Likelihood_Estimation.pdf
- Optimization_Methods.pdf
- Multi_Class_Logistic_Regression.pdf
- Regularized_Logistic_Regression.pdf
- Evaluation_Metrics.pdf
- Classifier_Comparison.pdf
- Advanced_Logistic_Regression.pdf

## Related Videos
- [Introduction to Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [The Sigmoid Function Explained](https://www.youtube.com/watch?v=TPqr8t919YM)
- [Maximum Likelihood for Logistic Regression](https://www.youtube.com/watch?v=BfKanl1aSG0)
- [Newton's Method for Optimization](https://www.youtube.com/watch?v=p1PV-r6lpvg)
- [Multi-class Classification & Softmax](https://www.youtube.com/watch?v=hMDQAM2Nh8k)
- [Regularization in Logistic Regression](https://www.youtube.com/watch?v=qbakZL5eaJM)
- [Evaluating Classification Models](https://www.youtube.com/watch?v=ZlpXLMUcTF0)
- [Comparing Classification Algorithms](https://www.youtube.com/watch?v=DCgPRYR5weI)

## All Quizzes
Test your understanding with these quizzes:
- [[L5_1_Quiz]]: Binary Logistic Regression Foundations
- [[L5_2_Quiz]]: Maximum Likelihood for Logistic Regression
- [[L5_3_Quiz]]: Optimization Methods for Logistic Regression
- [[L5_4_Quiz]]: Multi-class Logistic Regression
- [[L5_5_Quiz]]: Regularized Logistic Regression
- [[L5_6_Quiz]]: Evaluation Metrics for Logistic Regression
- [[L5_7_Quiz]]: Comparison with Other Classifiers
- [[L5_8_Quiz]]: Advanced Logistic Regression Topics 