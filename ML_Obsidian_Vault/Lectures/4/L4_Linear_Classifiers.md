# Lecture 4: Linear Classifiers

## Overview
This module introduces fundamental concepts of linear classification methods, from basic principles to advanced algorithms. You'll learn how linear decision boundaries can separate classes in feature space and explore various linear classification techniques for binary and multi-class problems, including probabilistic approaches.

### Lecture 4.1: Foundations of Linear Classification
- [[L4_1_Classification_vs_Regression|Classification vs Regression]]: Key differences and problem formulation
- [[L4_1_Linear_Classification|Linear Classification]]: Core concepts of linear decision boundaries
- [[L4_1_Decision_Boundaries|Decision Boundaries]]: Geometric interpretation in feature space
- [[L4_1_Feature_Space|Feature Space]]: Understanding data representation for classification
- [[L4_1_Discriminant_Functions|Discriminant Functions]]: Linear functions for classification
- [[L4_1_Binary_Classification|Binary Classification]]: Two-class problem formulation
- [[L4_1_Classification_Metrics|Classification Metrics]]: Accuracy, precision, recall, F1-score
- [[L4_1_Confusion_Matrix|Confusion Matrix]]: Comprehensive evaluation framework
- [[L4_1_ROC_Curves|ROC Curves]]: Performance visualization across thresholds
- [[L4_1_Examples|Classification Examples]]: Practical applications with code
- Required Reading: Chapter 4.1 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L4_1_Quiz]]: Test your understanding of foundations of linear classification

### Lecture 4.2: The Perceptron Algorithm
- [[L4_2_Perceptron|Perceptron]]: Historical context and model structure
- [[L4_2_Binary_Perceptron|Binary Perceptron]]: Detailed algorithm for two classes
- [[L4_2_Geometric_Interpretation|Geometric Interpretation]]: Visual understanding of perceptron learning
- [[L4_2_Perceptron_Algorithm|Perceptron Algorithm]]: Step-by-step explanation and pseudocode
- [[L4_2_Convergence_Theorem|Convergence Theorem]]: Mathematical proof and conditions
- [[L4_2_Perceptron_Learning_Rule|Perceptron Learning Rule]]: Weight update mechanism
- [[L4_2_Kernelized_Perceptron|Kernelized Perceptron]]: Handling non-linear data
- [[L4_2_Initialization|Initialization Strategies]]: Starting points for perceptron weights
- [[L4_2_Limitations|Perceptron Limitations]]: XOR problem and other challenges
- [[L4_2_Perceptron_Examples|Perceptron Examples]]: Implementation with visualizations
- Required Reading: Chapter 4.1.7 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L4_2_Quiz]]: Test your understanding of the perceptron algorithm

### Lecture 4.3: Probabilistic Linear Classifiers
- [[L4_3_Discriminative_vs_Generative|Discriminative vs Generative]]: Approaches to classification
- [[L4_3_Logistic_Regression|Logistic Regression]]: Probabilistic linear classification
- [[L4_3_Sigmoid_Function|Sigmoid Function]]: Converting linear outputs to probabilities
- [[L4_3_MLE_Logistic_Regression|MLE for Logistic Regression]]: Maximum likelihood approach
- [[L4_3_Cross_Entropy_Loss|Cross Entropy Loss]]: Proper scoring rule for classification
- [[L4_3_MAP_Logistic_Regression|MAP for Logistic Regression]]: Bayesian perspective
- [[L4_3_Regularized_Logistic|Regularized Logistic Regression]]: Preventing overfitting
- [[L4_3_Gradient_Optimization|Gradient-Based Optimization]]: Finding optimal parameters
- [[L4_3_Newton_Method|Newton's Method]]: Second-order optimization
- [[L4_3_Implementation|Logistic Regression Implementation]]: Practical coding example
- [[L4_3_Probabilistic_Examples|Probabilistic Examples]]: Applications with code
- Required Reading: Chapter 4.3.2 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L4_3_Quiz]]: Test your understanding of probabilistic linear classifiers

### Lecture 4.4: Linear Separability and Loss Functions
- [[L4_4_Linear_Separability|Linear Separability]]: When classes can be perfectly separated
- [[L4_4_Linearly_Separable_Data|Linearly Separable Data]]: Properties and examples
- [[L4_4_Non_Separable_Cases|Non-Separable Cases]]: Handling overlapping classes
- [[L4_4_Pocket_Algorithm|Pocket Algorithm]]: Improving perceptron for non-separable data
- [[L4_4_Loss_Functions|Loss Functions]]: Hinge loss, logistic loss, exponential loss
- [[L4_4_Loss_Function_Properties|Loss Function Properties]]: Convexity, differentiability
- [[L4_4_Margin_Classifiers|Margin-Based Classification]]: Maximizing the decision boundary margin
- [[L4_4_SVM_Introduction|Support Vector Machine Introduction]]: Maximum margin classifiers
- [[L4_4_Linear_Discriminant_Analysis|Linear Discriminant Analysis]]: Statistical approach
- [[L4_4_Comparing_Methods|Comparing Classification Methods]]: Perceptron, LDA, Logistic, SVM
- [[L4_4_Examples|Linear Separability Examples]]: Visual examples and implementation
- Required Reading: Chapter 4.1.7 and 7.1 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L4_4_Quiz]]: Test your understanding of linear separability and loss functions

### Lecture 4.5: Optimization for Linear Classifiers
- [[L4_5_Classification_Optimization|Classification Optimization]]: Objective functions
- [[L4_5_Batch_Learning|Batch Learning]]: Processing all data at once
- [[L4_5_Online_Learning|Online Learning]]: Incremental updates with individual samples
- [[L4_5_Stochastic_vs_Batch|Stochastic vs Batch Learning]]: Tradeoffs and differences
- [[L4_5_Perceptron_Optimization|Perceptron Optimization]]: Improving convergence
- [[L4_5_Logistic_Regression_Optimization|Logistic Regression Optimization]]: Gradient methods
- [[L4_5_Voted_Perceptron|Voted Perceptron]]: Ensemble approach to perceptron learning
- [[L4_5_Averaged_Perceptron|Averaged Perceptron]]: Improving generalization
- [[L4_5_Passive_Aggressive|Passive-Aggressive Algorithms]]: Modern online learning
- [[L4_5_Coordinate_Descent|Coordinate Descent]]: Feature-wise optimization
- [[L4_5_Implementation_Examples|Optimization Examples]]: Code demonstrations
- Required Reading: Chapter 4.1.3-4.1.4 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L4_5_Quiz]]: Test your understanding of optimization for linear classifiers

### Lecture 4.6: Multi-class Classification Strategies
- [[L4_6_Multi_Class_Classification|Multi-Class Classification]]: Extending binary methods
- [[L4_6_One_vs_All|One-vs-All (OVA)]]: Training N binary classifiers
- [[L4_6_One_vs_One|One-vs-One (OVO)]]: Training N(N-1)/2 binary classifiers
- [[L4_6_Error_Correcting_Codes|Error-Correcting Output Codes]]: Robust multi-class approach
- [[L4_6_Multiclass_Perceptron|Multiclass Perceptron]]: Direct extension to multiple classes
- [[L4_6_Softmax_Regression|Softmax Regression]]: Multinomial logistic regression
- [[L4_6_Cross_Entropy_Multiclass|Multiclass Cross Entropy]]: Loss function for multiple classes
- [[L4_6_MLE_Softmax|MLE for Softmax Regression]]: Maximum likelihood estimation
- [[L4_6_Comparison_Strategies|Comparison of Strategies]]: Accuracy, complexity, scalability
- [[L4_6_Multiclass_LDA|Multiclass LDA]]: Statistical discrimination with multiple classes
- [[L4_6_Multiclass_Examples|Multi-class Examples]]: Implementation and visualization
- Required Reading: Chapter 4.1.2 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L4_6_Quiz]]: Test your understanding of multi-class classification

### Lecture 4.7: Advanced Linear Classifiers and Applications
- [[L4_7_Maximum_Margin_Classifiers|Maximum Margin Classifiers]]: Support Vector Machines
- [[L4_7_Kernel_Methods|Kernel Methods]]: Handling non-linearity in linear classifiers
- [[L4_7_Kernelized_Linear_Classifiers|Kernelized Linear Classifiers]]: Theory and implementation
- [[L4_7_Regularization_Techniques|Regularization for Classifiers]]: Preventing overfitting
- [[L4_7_Feature_Selection|Feature Selection]]: Identifying relevant features
- [[L4_7_Calibration|Probability Calibration]]: Getting reliable probabilities
- [[L4_7_Imbalanced_Data|Imbalanced Data]]: Handling class imbalance
- [[L4_7_Cost_Sensitive_Learning|Cost-Sensitive Learning]]: Accounting for different error costs
- [[L4_7_Confidence_Estimation|Confidence Estimation]]: Uncertainty in classification
- [[L4_7_Real_World_Applications|Real-World Applications]]: Case studies and examples
- [[L4_7_Advanced_Examples|Advanced Examples]]: Complex implementation cases
- Required Reading: Chapter 4.3 and 7.1 of "Pattern Recognition and Machine Learning" by Bishop
- Quiz: [[L4_7_Quiz]]: Test your understanding of advanced linear classifiers

## Programming Resources
- [[L4_Linear_Classifiers_Python_Guide|Python Implementation Guide]]: Step-by-step implementation
- [[L4_Perceptron_From_Scratch|Building a Perceptron from Scratch]]: Code tutorial
- [[L4_Logistic_Regression_Implementation|Logistic Regression Implementation]]: MLE approach
- [[L4_Scikit_Learn_Classifiers|Using Scikit-learn for Linear Classification]]: Library tutorial
- [[L4_Visualizing_Decision_Boundaries|Visualizing Decision Boundaries]]: Plotting techniques
- [[L4_Multi_Class_Implementation|Multi-class Classification in Python]]: Code examples
- [[L4_Advanced_Classification_Code|Advanced Classification Techniques]]: Implementation

## Related Slides
*(not included in the repo)*
- Linear_Classifiers_Foundations.pdf
- Perceptron_Algorithm_Deep_Dive.pdf
- Probabilistic_Linear_Classifiers.pdf
- Linear_Separability_and_Loss_Functions.pdf
- Optimization_for_Classifiers.pdf
- Multi_Class_Classification_Strategies.pdf
- Advanced_Linear_Classifiers.pdf

## Related Videos
- [Introduction to Linear Classification](https://www.youtube.com/watch?v=hCOIMkcsm_g)
- [The Perceptron Algorithm Explained](https://www.youtube.com/watch?v=4Gac5I64LM4)
- [Logistic Regression and MLE](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [Linear Separability and Loss Functions](https://www.youtube.com/watch?v=X4GdQhQ7Jbg)
- [Optimization for Linear Classifiers](https://www.youtube.com/watch?v=5RZzlKidqRc)
- [Multi-class Classification Strategies](https://www.youtube.com/watch?v=ZvaELFv5IpM)
- [Advanced Linear Classifiers](https://www.youtube.com/watch?v=XUj5JbQihlU)

## All Quizzes
Test your understanding with these quizzes:
- [[L4_1_Quiz]]: Foundations of Linear Classification
- [[L4_2_Quiz]]: The Perceptron Algorithm
- [[L4_3_Quiz]]: Probabilistic Linear Classifiers
- [[L4_4_Quiz]]: Linear Separability and Loss Functions
- [[L4_5_Quiz]]: Optimization for Linear Classifiers
- [[L4_6_Quiz]]: Multi-class Classification Strategies
- [[L4_7_Quiz]]: Advanced Linear Classifiers and Applications 