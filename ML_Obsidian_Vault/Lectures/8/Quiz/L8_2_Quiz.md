# Lecture 8.2: Univariate Feature Selection Methods Quiz

## Overview
This quiz contains 30 questions covering univariate feature selection methods, including filter scoring, irrelevance criteria, correlation measures, mutual information, chi-square tests, and determining the optimal number of features. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Univariate feature selection considers one feature at a time independently.

#### Task
1. What is the main advantage of univariate methods?
2. What is the main limitation of univariate methods?
3. When are univariate methods most appropriate?
4. How do univariate methods handle feature interactions?
5. If you have 100 features, how many individual evaluations does univariate selection require?
6. Given a dataset with features $X_1, X_2, ..., X_n$ and target $Y$, calculate the computational complexity of univariate selection. If each feature evaluation takes 2 seconds and you have 500 features, how long will the complete selection process take? Express your answer in minutes and seconds.

For a detailed explanation of this question, see [Question 1: Univariate Approach](L8_2_1_explanation.md).

## Question 2

### Problem Statement
Univariate filter scoring ranks individual features based on their relevance to the target.

#### Task
1. What is the purpose of filter scoring?
2. How does filter scoring differ from wrapper methods?
3. What are the advantages of filter methods?
4. If a feature has a score of 0.8, what does this indicate?
5. Compare filter vs wrapper methods in terms of speed
6. A wrapper method takes 5 minutes to evaluate a single feature subset, while a filter method takes 30 seconds per feature. If you have 100 features and want to evaluate all possible subsets of size 1, 2, and 3, calculate the total time difference between wrapper and filter approaches. Which method is faster and by how much?

For a detailed explanation of this question, see [Question 2: Univariate Filter Scoring](L8_2_2_explanation.md).

## Question 3

### Problem Statement
Feature irrelevance can be defined using conditional probabilities and KL divergence.

#### Task
1. What does it mean for a feature to be irrelevant?
2. How do conditional probabilities help identify irrelevance?
3. What is KL divergence and how is it used?
4. If $P(y|x) = P(y)$ for all values of $x$, what does this suggest about feature $x$?
5. Calculate the KL divergence between two probability distributions
6. Given two probability distributions $P = [0.3, 0.4, 0.3]$ and $Q = [0.2, 0.5, 0.3]$, calculate the KL divergence $D_{KL}(P||Q)$. If $D_{KL}(P||Q) = 0.05$, what does this tell you about the relationship between features $P$ and $Q$?

For a detailed explanation of this question, see [Question 3: Criteria: Defining Feature Irrelevance](L8_2_3_explanation.md).

## Question 4

### Problem Statement
Pearson correlation measures linear relationships between features and targets.

#### Task
1. What is the formula for Pearson correlation?
2. What values can Pearson correlation take?
3. What does a correlation of 0 indicate?
4. If feature $X$ has correlation 0.7 with target $Y$, what does this mean?
5. Calculate the correlation between two simple datasets
6. Given the following data points: $X = [1, 2, 3, 4, 5]$ and $Y = [2, 4, 5, 4, 6]$, calculate the Pearson correlation coefficient step by step. Show your work including the calculation of means, deviations, and the final correlation value.

For a detailed explanation of this question, see [Question 4: Criteria: Pearson Correlation](L8_2_4_explanation.md).

## Question 5

### Problem Statement
Mutual information measures the dependence between features and targets.

#### Task
1. What is mutual information and how is it calculated?
2. How does mutual information differ from correlation?
3. What types of relationships can mutual information detect?
4. If mutual information is 0, what does this indicate?
5. Compare mutual information vs correlation for non-linear relationships
6. Given a joint probability distribution $P(X,Y)$ where $P(X=0,Y=0) = 0.3$, $P(X=0,Y=1) = 0.2$, $P(X=1,Y=0) = 0.1$, and $P(X=1,Y=1) = 0.4$, calculate the mutual information $I(X;Y)$. Show your calculations for marginal probabilities, entropies, and the final mutual information value.

For a detailed explanation of this question, see [Question 5: Criteria: Mutual Information](L8_2_5_explanation.md).

## Question 6

### Problem Statement
Chi-square test measures independence between categorical features and targets.

#### Task
1. What is the chi-square test statistic formula?
2. When is the chi-square test appropriate?
3. What does a high chi-square value indicate?
4. If the chi-square statistic is 15.2 with 4 degrees of freedom, what does this mean?
5. Calculate the chi-square statistic for a simple contingency table
6. Given the following contingency table:

| Feature\Target | Class 0 | Class 1 |
|----------------|---------|---------|
| Category A     | 25      | 15      |
| Category B     | 20      | 30      |

Calculate the chi-square statistic step by step. Show your expected frequencies, chi-square contributions, and the final statistic. With $\alpha = 0.05$ and 1 degree of freedom, is the feature independent of the target?

For a detailed explanation of this question, see [Question 6: Criteria: Chi-Square Test](L8_2_6_explanation.md).

## Question 7

### Problem Statement
Determining the optimal number of features to select is crucial for model performance.

#### Task
1. How do you use cross-validation to find the optimal 'k'?
2. What is the trade-off between too few and too many features?
3. How do you avoid overfitting in feature selection?
4. If accuracy peaks at 15 features, what's the optimal number?
5. Design a strategy to find the optimal feature count
6. Given the following cross-validation results for different numbers of features:

| Features | CV Accuracy | Std Dev |
|----------|-------------|---------|
| 5        | 0.82        | 0.03    |
| 10       | 0.87        | 0.02    |
| 15       | 0.89        | 0.02    |
| 20       | 0.88        | 0.03    |
| 25       | 0.86        | 0.04    |

Calculate the 95% confidence interval for each feature count and determine the optimal number of features. Use the rule: select the smallest number of features where the upper confidence bound of a larger feature set doesn't exceed the lower confidence bound of the current set.

For a detailed explanation of this question, see [Question 7: Determining the Number of Features to Select](L8_2_7_explanation.md).

## Question 8

### Problem Statement
Univariate methods have both advantages and disadvantages compared to multivariate approaches.

#### Task
1. What are the main advantages of univariate methods?
2. What are the main disadvantages?
3. How do univariate methods handle feature redundancy?
4. When would you choose univariate over multivariate methods?
5. Compare scalability of univariate vs multivariate approaches
6. If a multivariate method requires $O(n^2)$ operations per feature subset evaluation and a univariate method requires $O(n)$ operations per feature, calculate the computational advantage of univariate methods for datasets with 100, 1000, and 10000 features. Express the speedup as a ratio and percentage.

For a detailed explanation of this question, see [Question 8: Advantages and Disadvantages](L8_2_8_explanation.md).

## Question 9

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
1. Rank the features by relevance
2. If you select the top 3 features, which ones would you choose?
3. What percentage of the maximum possible correlation do you capture with top 3?
4. If feature A costs $100 and improves accuracy by 5%, while feature B costs $50 and improves by 3%, which is more cost-effective?
5. Calculate the average correlation of selected vs unselected features
6. Calculate the coefficient of variation (CV = standard deviation/mean) for the correlation scores. If you want to select features such that their combined correlation variance is minimized while maintaining an average correlation above 0.6, which features would you select? Show your calculations.

For a detailed explanation of this question, see [Question 9: Feature Ranking Analysis](L8_2_9_explanation.md).

## Question 10

### Problem Statement
Mutual information can detect non-linear relationships that correlation misses.

#### Task
1. Give an example where correlation is 0 but mutual information is high
2. How does mutual information handle categorical variables?
3. What's the relationship between mutual information and entropy?
4. If $I(X;Y) = H(X)$, what does this indicate?
5. Compare mutual information for different types of relationships
6. Consider a dataset where $X$ takes values $\{1, 2, 3, 4\}$ with equal probability and $Y = X^2 \mod 4$. Calculate the correlation between $X$ and $Y$, then calculate the mutual information. Show that correlation is 0 but mutual information reveals the relationship. What does this demonstrate about the limitations of correlation?

For a detailed explanation of this question, see [Question 10: Mutual Information Properties](L8_2_10_explanation.md).

## Question 11

### Problem Statement
Chi-square test requires understanding of degrees of freedom and significance.

#### Task
1. How do you calculate degrees of freedom for a contingency table?
2. What does the p-value tell you about feature relevance?
3. How do you interpret chi-square critical values?
4. If you have a 3×2 contingency table, what are the degrees of freedom?
5. Calculate the expected frequencies for a simple table
6. For a 4×3 contingency table, calculate the degrees of freedom. If the observed chi-square statistic is 18.5, find the critical value at $\alpha = 0.01$ and determine if the null hypothesis of independence should be rejected. Show your work using chi-square distribution tables.

For a detailed explanation of this question, see [Question 11: Chi-Square Test Details](L8_2_11_explanation.md).

## Question 12

### Problem Statement
Feature selection thresholds affect the number of selected features.

#### Task
1. How do you set thresholds for different selection criteria?
2. What happens if you set the threshold too high?
3. What happens if you set the threshold too low?
4. If you want exactly 20% of features, how do you set the threshold?
5. Design a threshold selection strategy
6. Given feature scores: [0.95, 0.87, 0.76, 0.65, 0.54, 0.43, 0.32, 0.21, 0.15, 0.08], calculate the threshold that would select exactly 30% of features. If you want to ensure that selected features have scores at least 2 standard deviations above the mean, what threshold would you use? Show your calculations.

For a detailed explanation of this question, see [Question 12: Threshold Selection](L8_2_12_explanation.md).

## Question 13

### Problem Statement
Cross-validation helps prevent overfitting in feature selection.

#### Task
1. How do you use cross-validation for feature selection?
2. What's the risk of not using cross-validation?
3. How many folds would you recommend for feature selection?
4. If you have 1000 samples, how many would be in each fold for 5-fold CV?
5. Design a cross-validation strategy for feature selection
6. For a dataset with 1200 samples, calculate the sample sizes for 3-fold, 5-fold, and 10-fold cross-validation. If each fold takes 2 minutes to process and you have 1 hour total, which CV strategy would you choose? Calculate the total processing time for each approach.

For a detailed explanation of this question, see [Question 13: Cross-Validation in Selection](L8_2_13_explanation.md).

## Question 14

### Problem Statement
Different selection criteria may give different feature rankings.

#### Task
1. Why might correlation and mutual information rank features differently?
2. How do you handle conflicting rankings?
3. What's the advantage of using multiple criteria?
4. If feature A ranks 1st by correlation but 3rd by mutual information, what does this suggest?
5. Design a multi-criteria selection approach
6. Given three features with rankings by correlation [1, 2, 3] and mutual information [2, 1, 3], calculate the Spearman rank correlation between these two ranking methods. If you use a weighted average approach with weights 0.6 for correlation and 0.4 for mutual information, what would be the final ranking? Show your calculations.

For a detailed explanation of this question, see [Question 14: Multiple Criteria](L8_2_14_explanation.md).

## Question 15

### Problem Statement
Feature selection affects model interpretability and performance.

#### Task
1. How does feature selection improve interpretability?
2. What's the relationship between features and model complexity?
3. How do you balance interpretability vs performance?
4. If you need to explain predictions to stakeholders, how many features would you select?
5. Compare interpretability of different feature counts
6. If model complexity increases exponentially with the number of features (complexity = $2^n$ where $n$ is the number of features), calculate the complexity for 5, 10, and 15 features. If stakeholders can understand models with complexity ≤ 1000, what's the maximum number of features you should select? Show your work.

For a detailed explanation of this question, see [Question 15: Interpretability vs Performance](L8_2_15_explanation.md).

## Question 16

### Problem Statement
Consider a binary classification problem with the following feature-target relationships:

| Feature | Correlation | Mutual Info | Chi-Square |
|---------|-------------|-------------|------------|
| Age     | 0.45        | 0.38        | 12.5       |
| Income  | 0.72        | 0.65        | 28.3       |
| Gender  | 0.15        | 0.42        | 18.7       |
| Education| 0.38       | 0.35        | 15.2       |

#### Task
1. Rank features by each criterion
2. Which features would you select if you want exactly 2?
3. Why do the rankings differ between criteria?
4. If you can only afford to collect 2 features, which would you choose?
5. Calculate the average ranking for each feature across all criteria
6. Normalize all three metrics to a 0-1 scale and calculate a composite score using weights: 40% correlation, 35% mutual information, and 25% chi-square. Which features would you select based on this composite score? Show your normalization and weighted average calculations.

For a detailed explanation of this question, see [Question 16: Multi-Criteria Ranking](L8_2_16_explanation.md).

## Question 17

### Problem Statement
Feature selection can be viewed as an optimization problem.

#### Task
1. What is the objective function for feature selection?
2. What are the constraints?
3. How do you handle multiple objectives?
4. If you want to maximize accuracy while minimizing features, how do you formulate this?
5. Design an optimization approach for feature selection
6. Formulate the feature selection problem as a multi-objective optimization: maximize accuracy (0-1 scale) and minimize the number of features. If accuracy = $0.8 + 0.02n - 0.001n^2$ where $n$ is the number of features, find the optimal number of features that maximizes the objective function $f(n) = \text{accuracy} - 0.1n$. Show your derivative calculations and optimization steps.

For a detailed explanation of this question, see [Question 17: Optimization Formulation](L8_2_17_explanation.md).

## Question 18

### Problem Statement
The curse of dimensionality affects feature selection strategies.

#### Task
1. How does high dimensionality affect univariate methods?
2. What happens to feature relevance as dimensions increase?
3. How do you handle the sparsity problem?
4. If you have 10,000 features, what selection strategy would you use?
5. Compare selection strategies for low vs high dimensional data
6. If the probability of a feature being relevant decreases exponentially with dimensionality as $P(\text{relevant}) = 0.1 \times 0.95^n$ where $n$ is the number of features, calculate the expected number of relevant features for datasets with 100, 1000, and 10000 features. If you need at least 5 relevant features for your model, what's the maximum dimensionality you should consider?

For a detailed explanation of this question, see [Question 18: High Dimensionality](L8_2_18_explanation.md).

## Question 19

### Problem Statement
Feature selection affects different types of machine learning algorithms.

#### Task
1. How does feature selection affect linear models?
2. How does it affect tree-based models?
3. How does it affect neural networks?
4. Which algorithm type benefits most from univariate selection?
5. Compare the impact on different algorithm families
6. If training time for a linear model is $T = 0.1n^2$ seconds and for a tree model is $T = 0.05n \log n$ seconds, where $n$ is the number of features, calculate the training time savings when reducing features from 100 to 20 for both models. Which model benefits more from feature selection in terms of training time reduction?

For a detailed explanation of this question, see [Question 19: Algorithm-Specific Effects](L8_2_19_explanation.md).

## Question 20

### Problem Statement
Feature selection can be applied at different stages of the pipeline.

#### Task
1. When is the best time to perform feature selection?
2. How does selection timing affect results?
3. What happens if you select features before preprocessing?
4. How do you handle selection in online learning?
5. Compare different timing strategies
6. If preprocessing takes 2 minutes per feature and feature selection takes 1 minute per feature, calculate the total pipeline time for three strategies: (1) preprocess all 100 features then select 20, (2) select 20 features then preprocess them, (3) preprocess 50 features then select 20. Which strategy is fastest and by how much?

For a detailed explanation of this question, see [Question 20: Selection Timing](L8_2_20_explanation.md).

## Question 21

### Problem Statement
Consider a scenario where you have limited computational resources.

#### Task
1. What selection strategies would you use with limited time?
2. How do you prioritize feature evaluation?
3. What's the trade-off between speed and quality?
4. If you have 1 hour to evaluate 1000 features, how many can you assess?
5. Design an efficient selection strategy
6. If feature evaluation time follows a power law distribution where the $i$-th feature takes $t_i = 0.1 \times i^{0.8}$ seconds, calculate the total time to evaluate the first 100, 500, and 1000 features. If you have exactly 1 hour, how many features can you evaluate? Show your calculations using the sum of the power series.

For a detailed explanation of this question, see [Question 21: Resource Constraints](L8_2_21_explanation.md).

## Question 22

### Problem Statement
Feature selection affects model robustness and stability.

#### Task
1. How does feature selection improve model stability?
2. What happens to model performance with noisy features?
3. How does selection affect cross-validation results?
4. If a model is unstable with 100 features, what would happen with 20 features?
5. Compare stability before and after selection
6. If model stability is measured as $S = 1 / (1 + 0.05n)$ where $n$ is the number of features, calculate the stability improvement when reducing from 100 to 20 features. If you want stability ≥ 0.8, what's the maximum number of features you should use? Show your calculations and interpret the results.

For a detailed explanation of this question, see [Question 22: Model Stability](L8_2_22_explanation.md).

## Question 23

### Problem Statement
Different domains have different feature selection requirements.

#### Task
1. What are the key considerations for medical diagnosis features?
2. How do financial applications differ from image recognition?
3. What's important for real-time sensor data?
4. Compare feature selection needs for text vs numerical data
5. Which domain would benefit most from univariate selection?
6. If medical diagnosis requires 99.9% confidence in feature relevance and financial applications require 95% confidence, calculate the minimum sample sizes needed for each domain assuming a binomial distribution. If you have 1000 samples, what confidence level can you achieve for feature relevance testing?

For a detailed explanation of this question, see [Question 23: Domain-Specific Requirements](L8_2_23_explanation.md).

## Question 24

### Problem Statement
Feature selection can reveal domain knowledge and insights.

#### Task
1. How can feature selection help understand data relationships?
2. What insights can you gain from selected features?
3. How does selection help with feature engineering?
4. If certain features are consistently selected, what does this suggest?
5. Compare the insights from selection vs extraction
6. If feature selection reveals that 80% of selected features are from the same domain category, calculate the probability that this occurred by chance if features were randomly distributed across 5 categories. Use the binomial distribution to determine if this clustering is statistically significant at $\alpha = 0.05$.

For a detailed explanation of this question, see [Question 24: Domain Insights](L8_2_24_explanation.md).

## Question 25

### Problem Statement
The relationship between features and target variables determines selection effectiveness.

#### Task
1. How do you measure feature-target relationships?
2. What types of relationships are hard to detect?
3. How do you handle non-linear relationships?
4. If a feature has no linear correlation but high mutual information, what does this suggest?
5. Compare different relationship measures
6. Given a feature $X$ with values $\{1, 2, 3, 4\}$ and target $Y$ with values $\{2, 4, 6, 8\}$, calculate both the Pearson correlation and the coefficient of determination $R^2$. If $Y = 2X$ exactly, verify that $R^2 = 1$. Show that correlation = 1 and explain why this represents a perfect linear relationship.

For a detailed explanation of this question, see [Question 25: Feature-Target Relationships](L8_2_25_explanation.md).

## Question 26

### Problem Statement
Feature selection affects the entire machine learning workflow.

#### Task
1. How does selection impact data preprocessing?
2. How does it affect model validation?
3. How does it impact deployment?
4. What changes in the workflow after selection?
5. Compare workflows with and without selection
6. If preprocessing time scales as $T_p = 0.5n$ minutes and model training time scales as $T_t = 0.1n^2$ minutes, where $n$ is the number of features, calculate the total workflow time for 50, 100, and 200 features. If you have 2 hours total, what's the maximum number of features you can process? Show your calculations.

For a detailed explanation of this question, see [Question 26: Workflow Impact](L8_2_26_explanation.md).

## Question 27

### Problem Statement
Consider a dataset with 1000 samples and 100 features where only 20 features are truly relevant.

#### Task
1. What percentage of features are irrelevant in this dataset?
2. If each irrelevant feature adds 1% noise, what's the total noise level?
3. How would this affect model performance?
4. What's the signal-to-noise ratio with all features vs relevant features only?
5. Calculate the probability of selecting only relevant features by random chance
6. If the probability of randomly selecting a relevant feature is $p = 20/100 = 0.2$, use the binomial distribution to calculate the probability of selecting exactly 15 relevant features when randomly choosing 20 features. What's the probability of selecting at least 15 relevant features? Show your calculations using the binomial formula.

For a detailed explanation of this question, see [Question 27: Irrelevant Features Impact](L8_2_27_explanation.md).

## Question 28

### Problem Statement
Feature selection can be viewed as a search and optimization problem.

#### Task
1. What is the objective function for feature selection?
2. What are the constraints in this optimization problem?
3. How do you balance multiple objectives?
4. If you want to maximize accuracy while minimizing features, how do you formulate this?
5. Compare different optimization approaches
6. Formulate the feature selection problem as a constrained optimization: maximize accuracy subject to the constraint that the number of features ≤ 25. If the accuracy function is $A(n) = 0.7 + 0.015n - 0.0002n^2$ where $n$ is the number of features, find the optimal number of features that maximizes accuracy within the constraint. Use calculus to find the maximum and verify it satisfies the constraint.

For a detailed explanation of this question, see [Question 28: Optimization Formulation](L8_2_28_explanation.md).

## Question 29

### Problem Statement
The success of feature selection depends on the quality of the evaluation criteria.

#### Task
1. What makes a good evaluation criterion?
2. How do you avoid overfitting in selection?
3. What's the role of cross-validation in selection?
4. If selection improves training accuracy but hurts validation accuracy, what does this suggest?
5. Compare different evaluation strategies
6. If training accuracy follows $A_{train} = 0.8 + 0.02n$ and validation accuracy follows $A_{val} = 0.8 + 0.01n - 0.001n^2$ where $n$ is the number of features, find the point where the gap between training and validation accuracy is maximized. This represents the point of maximum overfitting. Show your calculations and interpret the result.

For a detailed explanation of this question, see [Question 29: Evaluation Criteria](L8_2_29_explanation.md).

## Question 30

### Problem Statement
Feature selection is part of a broader feature engineering strategy.

#### Task
1. How does selection complement feature creation?
2. What's the relationship between selection and transformation?
3. How do you coordinate different feature engineering steps?
4. If you create 100 new features, how do you select the best ones?
5. Design a comprehensive feature engineering pipeline
6. If feature creation costs $C_c = 10$ minutes per feature and feature selection costs $C_s = 2$ minutes per feature, calculate the total cost for creating 50 features and then selecting the best 20. If you have a budget of 8 hours, what's the maximum number of features you can create and select? Show your calculations and design an optimal strategy.

For a detailed explanation of this question, see [Question 30: Feature Engineering Integration](L8_2_30_explanation.md).
