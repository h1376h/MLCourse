# Lecture 8.5: Filter Methods In-Depth Quiz

## Overview
This quiz contains 25 questions covering filter methods in-depth, including filter concepts, univariate vs multivariate filters, the Relief algorithm, and the advantages and disadvantages of filter methods. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Filter methods are preprocessing techniques that select features independently of the learning algorithm.

#### Task
1. What is the main characteristic of filter methods?
2. How do filter methods differ from wrapper methods?
3. When are filter methods most appropriate?
4. What types of evaluation criteria do filter methods use?
5. Compare filter vs wrapper methods in terms of speed

For a detailed explanation of this question, see [Question 1: Filter Method Overview](L8_5_1_explanation.md).

## Question 2

### Problem Statement
Univariate filters evaluate features individually, while multivariate filters consider feature subsets.

#### Task
1. What is the main difference between univariate and multivariate filters?
2. What are the advantages of univariate filters?
3. What are the advantages of multivariate filters?
4. When would you choose univariate over multivariate filters?
5. Compare the computational complexity of both approaches

For a detailed explanation of this question, see [Question 2: Univariate vs Multivariate Filters](L8_5_2_explanation.md).

## Question 3

### Problem Statement
The Relief algorithm is an instance-based feature weighting method.

#### Task
1. How does the Relief algorithm work?
2. What is the main idea behind Relief?
3. How does Relief handle different types of features?
4. What are the advantages of Relief?
5. Compare Relief with other filter methods

For a detailed explanation of this question, see [Question 3: Relief Algorithm](L8_5_3_explanation.md).

## Question 4

### Problem Statement
Filter methods have both advantages and disadvantages compared to other selection approaches.

#### Task
1. What are the main advantages of filter methods?
2. What are the main disadvantages?
3. How do filter methods handle feature interactions?
4. When would you choose filters over wrappers?
5. Compare the generality of different approaches

For a detailed explanation of this question, see [Question 4: Advantages and Disadvantages](L8_5_4_explanation.md).

## Question 5

### Problem Statement
Consider a dataset with 5 features and their correlation scores with the target:

| Feature | Correlation |
|---------|-------------|
| A       | 0.85        |
| B       | 0.72        |
| C       | 0.31        |
| D       | 0.68        |
| E       | 0.45        |

#### Task
1. If you use a correlation threshold of 0.5, which features would be selected?
2. If you want exactly 3 features, what threshold would you use?
3. What percentage of the maximum possible correlation do you capture with top 3?
4. If feature A costs $100 and improves accuracy by 5%, while feature B costs $50 and improves by 3%, which is more cost-effective?
5. Calculate the average correlation of selected vs unselected features

For a detailed explanation of this question, see [Question 5: Correlation-Based Filtering](L8_5_5_explanation.md).

## Question 6

### Problem Statement
Mutual information can be used as a filter criterion for feature selection.

#### Task
1. How does mutual information differ from correlation as a filter?
2. What types of relationships can mutual information detect?
3. How do you set thresholds for mutual information?
4. If feature A has mutual information 0.8 and feature B has 0.6, which is more relevant?
5. Compare mutual information vs correlation for non-linear relationships

For a detailed explanation of this question, see [Question 6: Mutual Information Filtering](L8_5_6_explanation.md).

## Question 7

### Problem Statement
Chi-square test is commonly used as a filter for categorical features.

#### Task
1. When is the chi-square test appropriate as a filter?
2. How do you interpret chi-square values for feature selection?
3. What are the limitations of chi-square filtering?
4. If the chi-square statistic is 15.2 with 4 degrees of freedom, what does this mean?
5. Compare chi-square vs other categorical feature filters

For a detailed explanation of this question, see [Question 7: Chi-Square Filtering](L8_5_7_explanation.md).

## Question 8

### Problem Statement
Filter methods can be combined to create more robust feature selection.

#### Task
1. How do you combine multiple filter criteria?
2. What are the advantages of combining filters?
3. How do you handle conflicting filter rankings?
4. If feature A ranks 1st by correlation but 3rd by mutual information, what does this suggest?
5. Design a multi-filter selection strategy

For a detailed explanation of this question, see [Question 8: Multi-Filter Approaches](L8_5_8_explanation.md).

## Question 9

### Problem Statement
The Relief algorithm updates feature weights based on nearest neighbor analysis.

#### Task
1. How does Relief update feature weights?
2. What is the role of nearest neighbors in Relief?
3. How do you choose the number of neighbors in Relief?
4. What happens to feature weights during Relief iterations?
5. Compare Relief with other instance-based methods

For a detailed explanation of this question, see [Question 9: Relief Algorithm Details](L8_5_9_explanation.md).

## Question 10

### Problem Statement
Filter methods can handle different types of data and features.

#### Task
1. How do filter methods handle continuous features?
2. How do they handle categorical features?
3. How do they handle mixed data types?
4. What preprocessing is needed for different feature types?
5. Compare filter approaches for different data types

For a detailed explanation of this question, see [Question 10: Data Type Handling](L8_5_10_explanation.md).

## Question 11

### Problem Statement
The curse of dimensionality affects filter methods differently than other approaches.

#### Task
1. How does high dimensionality affect filter methods?
2. What happens to feature relevance as dimensions increase?
3. How do you handle the sparsity problem in filters?
4. If you have 10,000 features, what filter strategy would you use?
5. Compare the robustness of different filter methods

For a detailed explanation of this question, see [Question 11: Dimensionality Impact](L8_5_11_explanation.md).

## Question 12

### Problem Statement
Filter methods can be viewed as optimization problems with different objectives.

#### Task
1. What is the objective function for filter methods?
2. What are the constraints in filter-based selection?
3. How do you balance multiple filter criteria?
4. If you want to maximize relevance while minimizing redundancy, how do you formulate this?
5. Compare different optimization approaches for filters

For a detailed explanation of this question, see [Question 12: Optimization Formulation](L8_5_12_explanation.md).

## Question 13

### Problem Statement
Different search strategies can be used with filter methods.

#### Task
1. What are the main types of search strategies for filters?
2. What's the trade-off between exhaustive and heuristic search?
3. When would you use random search with filters?
4. If you have limited time, what filter search strategy would you choose?
5. Compare different search approaches for filters

For a detailed explanation of this question, see [Question 13: Search Strategies](L8_5_13_explanation.md).

## Question 14

### Problem Statement
Filter methods affect different types of machine learning algorithms differently.

#### Task
1. How do filter methods affect linear models?
2. How do they affect tree-based models?
3. How do they affect neural networks?
4. Which algorithm type benefits most from filter methods?
5. Compare the impact on different algorithm families

For a detailed explanation of this question, see [Question 14: Algorithm-Specific Effects](L8_5_14_explanation.md).

## Question 15

### Problem Statement
Consider a scenario where you have limited computational resources for filter-based selection.

#### Task
1. What filter strategies would you use with limited time?
2. How do you prioritize feature evaluation?
3. What's the trade-off between speed and quality?
4. If you have 1 hour to evaluate 1000 features, how many can you assess?
5. Design an efficient filter-based selection strategy

For a detailed explanation of this question, see [Question 15: Resource Constraints](L8_5_15_explanation.md).

## Question 16

### Problem Statement
Filter methods can be applied at different stages of the machine learning pipeline.

#### Task
1. When is the best time to perform filter-based selection?
2. How does selection timing affect results?
3. What happens if you apply filters before preprocessing?
4. How do you handle filter selection in online learning?
5. Compare different timing strategies for filters

For a detailed explanation of this question, see [Question 16: Selection Timing](L8_5_16_explanation.md).

## Question 17

### Problem Statement
The success of filter methods depends on the quality of the evaluation criteria.

#### Task
1. What makes a good filter criterion?
2. How do you avoid overfitting in filter selection?
3. What's the role of cross-validation in filter methods?
4. If filter selection improves training accuracy but hurts validation accuracy, what does this suggest?
5. Compare different evaluation strategies for filters

For a detailed explanation of this question, see [Question 17: Evaluation Criteria](L8_5_17_explanation.md).

## Question 18

### Problem Statement
Filter methods affect model robustness and stability.

#### Task
1. How do filter methods improve model stability?
2. What happens to model performance with noisy features?
3. How do filters affect cross-validation results?
4. If a model is unstable with 100 features, what would happen with 20 features?
5. Compare stability before and after filter selection

For a detailed explanation of this question, see [Question 18: Model Stability](L8_5_18_explanation.md).

## Question 19

### Problem Statement
Different domains have different requirements for filter-based selection.

#### Task
1. What are the key considerations for medical diagnosis features?
2. How do financial applications differ from image recognition?
3. What's important for real-time sensor data?
4. Compare filter needs for text vs numerical data
5. Which domain would benefit most from filter methods?

For a detailed explanation of this question, see [Question 19: Domain-Specific Requirements](L8_5_19_explanation.md).

## Question 20

### Problem Statement
Filter methods can reveal domain knowledge and insights.

#### Task
1. How can filter methods help understand data relationships?
2. What insights can you gain from filtered features?
3. How do filters help with feature engineering?
4. If certain features are consistently filtered out, what does this suggest?
5. Compare the insights from filters vs other methods

For a detailed explanation of this question, see [Question 20: Domain Insights](L8_5_20_explanation.md).

## Question 21

### Problem Statement
The relationship between features and target variables determines filter effectiveness.

#### Task
1. How do you measure feature-target relationships for filters?
2. What types of relationships are hard to detect with filters?
3. How do you handle non-linear relationships?
4. If a feature has no linear correlation but high mutual information, what does this suggest?
5. Compare different relationship measures for filters

For a detailed explanation of this question, see [Question 21: Feature-Target Relationships](L8_5_21_explanation.md).

## Question 22

### Problem Statement
Filter methods affect the entire machine learning workflow.

#### Task
1. How do filter methods impact data preprocessing?
2. How do they affect model validation?
3. How do they impact deployment?
4. What changes in the workflow after filter selection?
5. Compare workflows with and without filter selection

For a detailed explanation of this question, see [Question 22: Workflow Impact](L8_5_22_explanation.md).

## Question 23

### Problem Statement
Consider a dataset with 1000 samples and 100 features where only 20 features are truly relevant.

#### Task
1. What percentage of features are irrelevant in this dataset?
2. If each irrelevant feature adds 1% noise, what's the total noise level?
3. How would this affect filter method performance?
4. What's the signal-to-noise ratio with all features vs relevant features only?
5. Calculate the probability of selecting only relevant features by random chance

For a detailed explanation of this question, see [Question 23: Irrelevant Features Impact](L8_5_23_explanation.md).

## Question 24

### Problem Statement
Filter methods can be viewed as a search and optimization problem.

#### Task
1. What is the objective function for filter-based selection?
2. What are the constraints in this optimization problem?
3. How do you balance multiple objectives?
4. If you want to maximize relevance while minimizing features, how do you formulate this?
5. Compare different optimization approaches for filters

For a detailed explanation of this question, see [Question 24: Optimization Formulation](L8_5_24_explanation.md).

## Question 25

### Problem Statement
Filter methods are part of a broader feature engineering strategy.

#### Task
1. How do filter methods complement feature creation?
2. What's the relationship between filtering and transformation?
3. How do you coordinate different feature engineering steps?
4. If you create 100 new features, how do you filter the best ones?
5. Design a comprehensive feature engineering pipeline

For a detailed explanation of this question, see [Question 25: Feature Engineering Integration](L8_5_25_explanation.md).
