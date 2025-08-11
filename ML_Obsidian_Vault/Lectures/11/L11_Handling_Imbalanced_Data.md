# Lecture 11: Handling Imbalanced Data

## Overview
This module addresses the critical challenge of class imbalance in machine learning, covering techniques for handling skewed datasets, synthetic data generation methods like SMOTE, and various sampling strategies to improve model performance on minority classes.

### Lecture 11.1: Understanding Class Imbalance
- [[L11_1_Class_Imbalance_Concept|Class Imbalance Concept]]: What is class imbalance and why it matters
- [[L11_1_Imbalance_Problems|Problems with Imbalanced Data]]: Biased models and poor performance
- [[L11_1_Imbalance_Ratios|Imbalance Ratios]]: Mild, moderate, and severe imbalance
- [[L11_1_Real_World_Examples|Real-World Examples]]: Fraud detection, medical diagnosis
- [[L11_1_Imbalance_Detection|Detecting Imbalance]]: How to identify imbalanced datasets
- [[L11_1_Examples|Basic Examples]]: Simple imbalance demonstrations
- Required Reading: Chapter 8 of "Imbalanced Learning" by He and Garcia
- Quiz: [[L11_1_Quiz]]: Test your understanding of class imbalance concepts

### Lecture 11.2: Evaluation Metrics for Imbalanced Data
- [[L11_2_Imbalanced_Metrics|Metrics for Imbalanced Data]]: Why accuracy fails
- [[L11_2_Precision_Recall_Imbalanced|Precision and Recall]]: Focus on minority class
- [[L11_2_F1_Score_Imbalanced|F1 Score]]: Harmonic mean for imbalanced data
- [[L11_2_ROC_AUC_Imbalanced|ROC and AUC]]: ROC curves for imbalanced problems
- [[L11_2_Precision_Recall_Curves|Precision-Recall Curves]]: Better than ROC for imbalance
- [[L11_2_Examples|Metric Examples]]: Implementation and interpretation
- Required Reading: Chapter 8.1 of "Imbalanced Learning" by He and Garcia
- Quiz: [[L11_2_Quiz]]: Test your understanding of evaluation metrics for imbalanced data

### Lecture 11.3: Random Oversampling
- [[L11_3_Random_Oversampling|Random Oversampling]]: Simple duplication approach
- [[L11_3_Oversampling_Process|Oversampling Process]]: How to implement random oversampling
- [[L11_3_Oversampling_Advantages|Oversampling Advantages]]: When random oversampling helps
- [[L11_3_Oversampling_Limitations|Oversampling Limitations]]: Overfitting and memorization
- [[L11_3_Implementation_Considerations|Implementation Considerations]]: Practical aspects
- [[L11_3_Examples|Oversampling Examples]]: Implementation and case studies
- Required Reading: Chapter 9.1 of "Imbalanced Learning" by He and Garcia
- Quiz: [[L11_3_Quiz]]: Test your understanding of random oversampling

### Lecture 11.4: Random Undersampling
- [[L11_4_Random_Undersampling|Random Undersampling]]: Reducing majority class samples
- [[L11_4_Undersampling_Process|Undersampling Process]]: Implementation and techniques
- [[L11_4_Undersampling_Advantages|Undersampling Advantages]]: When to use undersampling
- [[L11_4_Undersampling_Limitations|Undersampling Limitations]]: Information loss concerns
- [[L11_4_Undersampling_Strategies|Undersampling Strategies]]: Different approaches
- [[L11_4_Examples|Undersampling Examples]]: Implementation and applications
- Required Reading: Chapter 9.2 of "Imbalanced Learning" by He and Garcia
- Quiz: [[L11_4_Quiz]]: Test your understanding of random undersampling

### Lecture 11.5: SMOTE and Synthetic Data Generation
- [[L11_5_SMOTE_Concept|SMOTE Concept]]: Synthetic Minority Over-sampling Technique
- [[L11_5_SMOTE_Algorithm|SMOTE Algorithm]]: Step-by-step SMOTE process
- [[L11_5_SMOTE_Implementation|SMOTE Implementation]]: How to implement SMOTE
- [[L11_5_SMOTE_Variants|SMOTE Variants]]: Borderline SMOTE, ADASYN
- [[L11_5_SMOTE_Advantages|SMOTE Advantages]]: Why SMOTE is effective
- [[L11_5_SMOTE_Limitations|SMOTE Limitations]]: When SMOTE fails
- [[L11_5_Examples|SMOTE Examples]]: Implementation and case studies
- Required Reading: "SMOTE: Synthetic Minority Over-sampling Technique" by Chawla et al.
- Quiz: [[L11_5_Quiz]]: Test your understanding of SMOTE and synthetic data generation

### Lecture 11.6: Advanced Synthetic Data Methods
- [[L11_6_Advanced_Synthetic|Advanced Synthetic Methods]]: Beyond SMOTE
- [[L11_6_ADASYN|ADASYN]]: Adaptive Synthetic Sampling
- [[L11_6_Borderline_SMOTE|Borderline SMOTE]]: Focusing on decision boundaries
- [[L11_6_Safe_Level_SMOTE|Safe Level SMOTE]]: Safe synthetic sample generation
- [[L11_6_Data_Augmentation|Data Augmentation]]: Creating variations of existing samples
- [[L11_6_Examples|Advanced Examples]]: Implementation and applications
- Required Reading: Chapter 9.3 of "Imbalanced Learning" by He and Garcia
- Quiz: [[L11_6_Quiz]]: Test your understanding of advanced synthetic data methods

### Lecture 11.7: Hybrid and Ensemble Methods
- [[L11_7_Hybrid_Methods|Hybrid Methods]]: Combining oversampling and undersampling
- [[L11_7_SMOTEENN|SMOTEENN]]: SMOTE with Edited Nearest Neighbors
- [[L11_7_SMOTETomek|SMOTETomek]]: SMOTE with Tomek links
- [[L11_7_Ensemble_Imbalanced|Ensemble Methods]]: Using multiple models
- [[L11_7_Balanced_Ensembles|Balanced Ensembles]]: Creating balanced model combinations
- [[L11_7_Examples|Hybrid Examples]]: Implementation and case studies
- Required Reading: Chapter 9.4 of "Imbalanced Learning" by He and Garcia
- Quiz: [[L11_7_Quiz]]: Test your understanding of hybrid and ensemble methods

### Lecture 11.8: Cost-Sensitive Learning and Best Practices
- [[L11_8_Cost_Sensitive_Learning|Cost-Sensitive Learning]]: Penalizing misclassification differently
- [[L11_8_Cost_Matrix|Cost Matrix]]: Defining misclassification costs
- [[L11_8_Algorithm_Modifications|Algorithm Modifications]]: Adapting algorithms for imbalance
- [[L11_8_Best_Practices|Best Practices]]: Guidelines for handling imbalanced data
- [[L11_8_Method_Selection|Method Selection]]: Choosing the right approach
- [[L11_8_Real_World_Applications|Real-World Applications]]: Case studies and examples
- Required Reading: Chapter 10 of "Imbalanced Learning" by He and Garcia
- Quiz: [[L11_8_Quiz]]: Test your understanding of cost-sensitive learning and best practices

## Programming Resources
- [[L11_Imbalanced_Data_Python_Guide|Python Implementation Guide]]: Step-by-step implementation
- [[L11_SMOTE_Implementation|SMOTE Algorithm Implementation]]: Code tutorial
- [[L11_Oversampling_Undersampling_Code|Sampling Methods Implementation]]: Implementation guide
- [[L11_Imbalanced_Metrics_Code|Evaluation Metrics for Imbalanced Data]]: Metric implementation
- [[L11_Scikit_Learn_Imbalanced|Using Scikit-learn for Imbalanced Data]]: Library tutorial
- [[L11_Advanced_Sampling_Code|Advanced Sampling Methods]]: Implementation guide
- [[L11_Cost_Sensitive_Learning_Code|Cost-Sensitive Learning Implementation]]: Cost matrix tutorial

## Related Slides
*(not included in the repo)*
- Class_Imbalance_Overview.pdf
- Evaluation_Metrics_Imbalanced.pdf
- Random_Sampling_Methods.pdf
- SMOTE_Algorithm_Deep_Dive.pdf
- Advanced_Synthetic_Methods.pdf
- Hybrid_Ensemble_Methods.pdf
- Cost_Sensitive_Learning.pdf
- Best_Practices_Imbalanced.pdf

## Related Videos
- [Introduction to Class Imbalance](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Evaluation Metrics for Imbalanced Data](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [SMOTE Algorithm Explained](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Handling Imbalanced Data](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Advanced Sampling Methods](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Cost-Sensitive Learning](https://www.youtube.com/watch?v=YaKMeAlHgqQ)
- [Best Practices for Imbalanced Data](https://www.youtube.com/watch?v=YaKMeAlHgqQ)

## All Quizzes
Test your understanding with these quizzes:
- [[L11_1_Quiz]]: Understanding Class Imbalance
- [[L11_2_Quiz]]: Evaluation Metrics for Imbalanced Data
- [[L11_3_Quiz]]: Random Oversampling
- [[L11_4_Quiz]]: Random Undersampling
- [[L11_5_Quiz]]: SMOTE and Synthetic Data Generation
- [[L11_6_Quiz]]: Advanced Synthetic Data Methods
- [[L11_7_Quiz]]: Hybrid and Ensemble Methods
- [[L11_8_Quiz]]: Cost-Sensitive Learning and Best Practices
