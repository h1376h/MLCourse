# Lecture 8.8: Feature Extraction vs. Selection Quiz

## Overview
This quiz contains 30 questions covering feature extraction vs selection, including dimensionality reduction, PCA, LDA, feature construction, and when to use each approach. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Dimensionality reduction is a key concept in feature engineering and machine learning.

#### Task
1. What is dimensionality reduction and why is it important?
2. What are the two main approaches to dimensionality reduction?
3. How do you choose between feature selection and extraction?
4. What are the trade-offs of each approach?
5. Compare the interpretability of both approaches

For a detailed explanation of this question, see [Question 1: Dimensionality Reduction Overview](L8_8_1_explanation.md).

## Question 2

### Problem Statement
Feature selection chooses a subset of the original features.

#### Task
1. What is feature selection and how does it work?
2. What are the advantages of feature selection?
3. What are the disadvantages of feature selection?
4. When is feature selection most appropriate?
5. Compare feature selection with other approaches

For a detailed explanation of this question, see [Question 2: Feature Selection Review](L8_8_2_explanation.md).

## Question 3

### Problem Statement
Feature extraction transforms features into a new space.

#### Task
1. What is feature extraction and how does it work?
2. What are the advantages of feature extraction?
3. What are the disadvantages of feature extraction?
4. When is feature extraction most appropriate?
5. Compare feature extraction with feature selection

For a detailed explanation of this question, see [Question 3: Feature Extraction](L8_8_3_explanation.md).

## Question 4

### Problem Statement
Principal Component Analysis (PCA) is a common feature extraction method.

#### Task
1. How does PCA work?
2. What are the main steps of PCA?
3. What are the advantages of PCA?
4. What are the disadvantages of PCA?
5. When would you use PCA?

For a detailed explanation of this question, see [Question 4: Principal Component Analysis](L8_8_4_explanation.md).

## Question 5

### Problem Statement
Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction method.

#### Task
1. How does LDA work?
2. What are the main steps of LDA?
3. What are the advantages of LDA?
4. What are the disadvantages of LDA?
5. When would you use LDA vs PCA?

For a detailed explanation of this question, see [Question 5: Linear Discriminant Analysis](L8_8_5_explanation.md).

## Question 6

### Problem Statement
Feature construction creates new features from existing ones.

#### Task
1. What is feature construction and how does it work?
2. What are examples of feature construction?
3. What are the advantages of feature construction?
4. What are the disadvantages of feature construction?
5. When would you use feature construction?

For a detailed explanation of this question, see [Question 6: Feature Construction and Engineering](L8_8_6_explanation.md).

## Question 7

### Problem Statement
Feature extraction and selection have different characteristics and trade-offs.

#### Task
1. What are the main differences between extraction and selection?
2. How do they differ in interpretability?
3. How do they differ in computational cost?
4. When would you choose extraction over selection?
5. Compare the generality of both approaches

For a detailed explanation of this question, see [Question 7: Extraction vs Selection](L8_8_7_explanation.md).

## Question 8

### Problem Statement
Consider a dataset with 4 original features and the following correlation matrix:

| Feature | F1   | F2   | F3   | F4   |
|---------|------|------|------|------|
| F1      | 1.0  | 0.8  | 0.2  | 0.1  |
| F2      | 0.8  | 1.0  | 0.3  | 0.2  |
| F3      | 0.2  | 0.3  | 1.0  | 0.7  |
| F4      | 0.1  | 0.2  | 0.7  | 1.0  |

#### Task
1. If you use feature selection, which features would you keep?
2. If you use PCA, how many principal components would you need?
3. What does the correlation structure suggest about feature redundancy?
4. Which approach would be better for this dataset?
5. Calculate the effective number of independent features

For a detailed explanation of this question, see [Question 8: Feature Analysis](L8_8_8_explanation.md).

## Question 9

### Problem Statement
PCA transforms data to maximize variance in the projected space.

#### Task
1. How do you calculate the principal components?
2. What does the eigenvalue tell you about each component?
3. How do you choose the number of components?
4. What happens to the data variance after PCA?
5. Compare PCA with other transformation methods

For a detailed explanation of this question, see [Question 9: PCA Details](L8_8_9_explanation.md).

## Question 10

### Problem Statement
LDA maximizes the ratio of between-class to within-class scatter.

#### Task
1. How do you calculate the between-class scatter matrix?
2. How do you calculate the within-class scatter matrix?
3. What does the discriminant function tell you?
4. How do you choose the number of discriminant functions?
5. Compare LDA with PCA for classification

For a detailed explanation of this question, see [Question 10: LDA Details](L8_8_10_explanation.md).

## Question 11

### Problem Statement
Feature construction can create polynomial, interaction, and ratio features.

#### Task
1. What are polynomial features and when are they useful?
2. What are interaction features and when are they useful?
3. What are ratio features and when are they useful?
4. How do you handle the explosion of constructed features?
5. Design a feature construction strategy

For a detailed explanation of this question, see [Question 11: Feature Construction Types](L8_8_11_explanation.md).

## Question 12

### Problem Statement
The curse of dimensionality affects both feature selection and extraction.

#### Task
1. How does high dimensionality affect feature selection?
2. How does it affect feature extraction?
3. Which approach is more robust to high dimensionality?
4. What strategies can you use to handle high dimensionality?
5. Compare the scalability of both approaches

For a detailed explanation of this question, see [Question 12: Dimensionality Impact](L8_8_12_explanation.md).

## Question 13

### Problem Statement
Different domains have different requirements for dimensionality reduction.

#### Task
1. What are the key considerations for medical diagnosis features?
2. How do financial applications differ from image recognition?
3. What's important for real-time sensor data?
4. Compare needs for text vs numerical data
5. Which domain would benefit most from feature extraction?

For a detailed explanation of this question, see [Question 13: Domain-Specific Requirements](L8_8_13_explanation.md).

## Question 14

### Problem Statement
Feature extraction and selection affect different types of machine learning algorithms.

#### Task
1. How do they affect linear models?
2. How do they affect tree-based models?
3. How do they affect neural networks?
4. Which algorithm type benefits most from each approach?
5. Compare the impact on different algorithm families

For a detailed explanation of this question, see [Question 14: Algorithm-Specific Effects](L8_8_14_explanation.md).

## Question 15

### Problem Statement
Consider a scenario where you have limited computational resources.

#### Task
1. What dimensionality reduction strategies would you use with limited time?
2. How do you prioritize different approaches?
3. What's the trade-off between speed and quality?
4. If you have 1 hour for dimensionality reduction, what would you do?
5. Design a resource-constrained strategy

For a detailed explanation of this question, see [Question 15: Resource Constraints](L8_8_15_explanation.md).

## Question 16

### Problem Statement
Feature extraction and selection can be applied at different stages of the pipeline.

#### Task
1. When is the best time to perform dimensionality reduction?
2. How does timing affect results?
3. What happens if you reduce dimensions before preprocessing?
4. How do you handle dimensionality reduction in online learning?
5. Compare different timing strategies

For a detailed explanation of this question, see [Question 16: Timing Considerations](L8_8_16_explanation.md).

## Question 17

### Problem Statement
The success of dimensionality reduction depends on the quality of the evaluation criteria.

#### Task
1. What makes a good evaluation criterion for feature selection?
2. What makes a good evaluation criterion for feature extraction?
3. How do you avoid overfitting in dimensionality reduction?
4. What's the role of cross-validation?
5. Compare different evaluation strategies

For a detailed explanation of this question, see [Question 17: Evaluation Criteria](L8_8_17_explanation.md).

## Question 18

### Problem Statement
Feature extraction and selection affect model robustness and stability.

#### Task
1. How do they improve model stability?
2. What happens to model performance with noisy features?
3. How do they affect cross-validation results?
4. If a model is unstable with 100 features, what would happen with 20 features?
5. Compare stability before and after dimensionality reduction

For a detailed explanation of this question, see [Question 18: Model Stability](L8_8_18_explanation.md).

## Question 19

### Problem Statement
Feature extraction and selection can reveal domain knowledge and insights.

#### Task
1. How can they help understand data relationships?
2. What insights can you gain from reduced features?
3. How do they help with feature engineering?
4. If certain patterns emerge, what does this suggest?
5. Compare the insights from different approaches

For a detailed explanation of this question, see [Question 19: Domain Insights](L8_8_19_explanation.md).

## Question 20

### Problem Statement
The relationship between features and target variables determines approach effectiveness.

#### Task
1. How do you measure feature-target relationships for selection?
2. How do you measure them for extraction?
3. What types of relationships are hard to detect?
4. How do you handle non-linear relationships?
5. Compare different relationship measures

For a detailed explanation of this question, see [Question 20: Feature-Target Relationships](L8_8_20_explanation.md).

## Question 21

### Problem Statement
Feature extraction and selection affect the entire machine learning workflow.

#### Task
1. How do they impact data preprocessing?
2. How do they affect model validation?
3. How do they impact deployment?
4. What changes in the workflow after dimensionality reduction?
5. Compare workflows with and without dimensionality reduction

For a detailed explanation of this question, see [Question 21: Workflow Impact](L8_8_21_explanation.md).

## Question 22

### Problem Statement
Consider a dataset with 1000 samples and 100 features where only 20 features are truly relevant.

#### Task
1. What percentage of features are irrelevant in this dataset?
2. If each irrelevant feature adds 1% noise, what's the total noise level?
3. How would this affect feature selection vs extraction?
4. What's the signal-to-noise ratio with all features vs relevant features only?
5. Which approach would be better for this dataset?

For a detailed explanation of this question, see [Question 22: Irrelevant Features Impact](L8_8_22_explanation.md).

## Question 23

### Problem Statement
Feature extraction and selection can be viewed as optimization problems.

#### Task
1. What is the objective function for feature selection?
2. What is the objective function for feature extraction?
3. What are the constraints in each problem?
4. How do you balance multiple objectives?
5. Compare different optimization approaches

For a detailed explanation of this question, see [Question 23: Optimization Formulation](L8_8_23_explanation.md).

## Question 24

### Problem Statement
The curse of dimensionality affects different approaches differently.

#### Task
1. How does high dimensionality affect feature selection?
2. How does it affect feature extraction?
3. Which approach is most robust to high dimensionality?
4. What strategies can you use to handle high dimensionality?
5. Compare the robustness of different approaches

For a detailed explanation of this question, see [Question 24: High Dimensionality](L8_8_24_explanation.md).

## Question 25

### Problem Statement
Feature extraction and selection affect different types of machine learning algorithms.

#### Task
1. How do they affect linear models?
2. How do they affect tree-based models?
3. How do they affect neural networks?
4. Which algorithm type benefits most from each approach?
5. Compare the impact on different algorithm families

For a detailed explanation of this question, see [Question 25: Algorithm-Specific Effects](L8_8_25_explanation.md).

## Question 26

### Problem Statement
Feature extraction and selection can be applied at different stages of the pipeline.

#### Task
1. When is the best time to perform dimensionality reduction?
2. How does timing affect results?
3. What happens if you reduce dimensions before preprocessing?
4. How do you handle dimensionality reduction in online learning?
5. Compare different timing strategies

For a detailed explanation of this question, see [Question 26: Timing Considerations](L8_8_26_explanation.md).

## Question 27

### Problem Statement
The success of dimensionality reduction depends on the quality of the evaluation criteria.

#### Task
1. What makes a good evaluation criterion for feature selection?
2. What makes a good evaluation criterion for feature extraction?
3. How do you avoid overfitting in dimensionality reduction?
4. What's the role of cross-validation?
5. Compare different evaluation strategies

For a detailed explanation of this question, see [Question 27: Evaluation Criteria](L8_8_27_explanation.md).

## Question 28

### Problem Statement
Feature extraction and selection affect model robustness and stability.

#### Task
1. How do they improve model stability?
2. What happens to model performance with noisy features?
3. How do they affect cross-validation results?
4. If a model is unstable with 100 features, what would happen with 20 features?
5. Compare stability before and after dimensionality reduction

For a detailed explanation of this question, see [Question 28: Model Stability](L8_8_28_explanation.md).

## Question 29

### Problem Statement
Feature extraction and selection can reveal domain knowledge and insights.

#### Task
1. How can they help understand data relationships?
2. What insights can you gain from reduced features?
3. How do they help with feature engineering?
4. If certain patterns emerge, what does this suggest?
5. Compare the insights from different approaches

For a detailed explanation of this question, see [Question 29: Domain Insights](L8_8_29_explanation.md).

## Question 30

### Problem Statement
Feature extraction and selection are part of a broader feature engineering strategy.

#### Task
1. How do they complement feature creation?
2. What's the relationship between different approaches?
3. How do you coordinate different feature engineering steps?
4. If you create 100 new features, how do you reduce dimensions?
5. Design a comprehensive feature engineering pipeline

For a detailed explanation of this question, see [Question 30: Feature Engineering Integration](L8_8_30_explanation.md).
