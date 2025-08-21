# Lecture 8.3: Multivariate Feature Selection Methods Quiz

## Overview
This quiz contains 15 questions covering multivariate feature selection methods, including when univariate methods fail, handling feature redundancy, search space problems, and feature clustering approaches. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Consider a dataset with 4 features: A, B, C, and D. Features A and B are individually weak predictors but when combined, they perfectly predict the target. Features C and D are individually strong predictors but redundant.

#### Task
1. Calculate the number of possible feature subsets (including empty set)
2. If univariate selection picks the top 2 features, which would it select and why?
3. If multivariate selection considers all subsets, which subset would be optimal?
4. What is the main advantage of multivariate methods in this scenario?
5. Calculate the search space size for 4 features vs 10 features
6. If you have a budget constraint where each feature costs $\$5$ and you can spend at most $\$15$, how many valid feature subsets exist? Calculate the total cost of all possible subsets.
7. If features A and B have individual correlations with target of $0.2$ and $0.3$ respectively, but their combined correlation is $0.95$, calculate the interaction strength using the formula: $\text{Interaction} = \text{Combined\_Correlation} - \max(\text{Individual\_Correlations}) - 0.1 \times \min(\text{Individual\_Correlations})$. What does this value indicate about feature synergy?

For a detailed explanation of this question, see [Question 1: Multivariate Approach](L8_3_1_explanation.md).

## Question 2

### Problem Statement
In a loan default prediction dataset, features include: income, debt_ratio, credit_score, and age. Income and debt_ratio together determine debt-to-income ratio, which is a strong predictor. Credit_score and age are moderately correlated ($r = 0.6$).

#### Task
1. Design a scenario where univariate selection would miss the income-debt_ratio interaction
2. If income and debt_ratio have correlation $r = 0.3$, calculate the redundancy score
3. How many independent feature combinations exist if you group correlated features?
4. Calculate the reduction in search space if you treat correlated features as groups
5. Design a multivariate selection strategy for this dataset
6. If the debt-to-income ratio is calculated as $\frac{\text{debt\_ratio} \times 100}{\text{income}}$, and you have sample values: income = $\$50,000$, debt_ratio = $0.4$, credit_score = $720$, age = $35$, calculate the actual debt-to-income ratio. If the target is $1$ for default and $0$ for no default, and the threshold for default is debt-to-income > $0.43$, what would be the prediction?
7. Calculate the VIF (Variance Inflation Factor) for income if the $R^2$ from regressing income on other features is $0.25$. If the rule of thumb is to remove features with VIF > $5$, would income be removed? Then calculate the condition number of the correlation matrix for all four features.

For a detailed explanation of this question, see [Question 2: Feature Interactions](L8_3_2_explanation.md).

## Question 3

### Problem Statement
You have a dataset with 20 features and want to find the optimal subset of 5-10 features.

#### Task
1. Calculate the total number of possible feature subsets
2. Calculate the number of subsets with exactly 7 features
3. If each subset evaluation takes $0.1$ seconds, how long would exhaustive search take?
4. If you use forward selection starting with 1 feature, how many evaluations are needed?
5. Design a heuristic search strategy that evaluates at most 1000 subsets
6. If you implement a "smart" search that skips subsets where the first 3 features are all from the same correlation cluster (assume 4 clusters of 5 features each), calculate how many subsets you can skip. What percentage of the search space does this represent?
7. If you use a genetic algorithm with population size 50, mutation rate $0.1$, and crossover rate $0.8$, calculate the expected number of unique feature subsets generated in 10 generations. Assume each generation produces 50 offspring and 10% of offspring are duplicates.

For a detailed explanation of this question, see [Question 3: Search Space Problem](L8_3_3_explanation.md).

## Question 4

### Problem Statement
A dataset has 100 features that can be grouped into 5 clusters based on correlation. Each cluster contains 20 features with average within-cluster correlation of $0.8$.

#### Task
1. Calculate the effective number of independent features after clustering
2. If you select one feature from each cluster, how does this reduce the search space?
3. Calculate the reduction in search space size (as a percentage)
4. If features within clusters have correlation > $0.7$, how many clusters would you expect?
5. Design a clustering strategy that maximizes feature independence
6. If you use hierarchical clustering with a correlation threshold of $0.7$, and the correlation matrix shows that features 1-20 have correlations ranging from $0.75$ to $0.95$, features 21-40 have correlations $0.65$ to $0.85$, and features 41-60 have correlations $0.55$ to $0.75$, how many clusters would you actually get? Calculate the average correlation within each resulting cluster.
7. Calculate the silhouette score for a feature if its average distance to features in its own cluster is $0.3$ and its average distance to features in the nearest other cluster is $0.7$. If the silhouette score ranges from $-1$ to $1$, interpret this value. Then calculate the Davies-Bouldin index if you have 3 clusters with average within-cluster distances of $0.4$, $0.5$, and $0.6$.

For a detailed explanation of this question, see [Question 4: Feature Clustering](L8_3_4_explanation.md).

## Question 5

### Problem Statement
You want to maximize model accuracy while minimizing the number of features. Your current model has 50 features with 85% accuracy.

#### Task
1. Formulate this as a multi-objective optimization problem
2. If accuracy increases by $0.5\%$ for each additional feature up to 20, then decreases by $0.1\%$ per feature, find the optimal number
3. Design a penalty function that balances accuracy and feature count
4. If you have a budget of 30 features maximum, how do you modify the objective?
5. Compare greedy vs exhaustive search for this optimization problem
6. If the penalty function is $P(\text{features}) = \lambda \times \text{features}^2$, where $\lambda$ is a tuning parameter, and you want the penalty to equal the accuracy gain when features = 25, calculate the value of $\lambda$. Then find the optimal number of features when $\lambda = 0.01$.
7. Calculate the Pareto frontier efficiency if you have three solutions: (20 features, 87% accuracy), (30 features, 88% accuracy), and (40 features, 86% accuracy). If the ideal point is (10 features, 90% accuracy), calculate the normalized distance to ideal for each solution using the formula: $\text{distance} = \sqrt{\frac{(\text{features}-10)^2}{100} + \frac{(\text{accuracy}-0.9)^2}{0.01}}$.

For a detailed explanation of this question, see [Question 5: Optimization Formulation](L8_3_5_explanation.md).

## Question 6

### Problem Statement
You have 25 features and limited computational time. You can evaluate at most 500 feature subsets.

#### Task
1. Compare the number of evaluations needed for exhaustive search vs forward selection
2. If forward selection adds one feature at a time, how many evaluations are needed?
3. Design a random search strategy that samples 500 subsets efficiently
4. Calculate the probability of finding the optimal subset with random sampling
5. Design a hybrid strategy combining forward selection and random sampling
6. If you use simulated annealing with temperature $T = 100$ and cooling rate $\alpha = 0.95$, calculate the temperature after 20 iterations. If the acceptance probability is $P = \exp(-\Delta E/T)$ where $\Delta E$ is the change in objective value, calculate the probability of accepting a worse solution when $\Delta E = 5$ and $T = 50$.

For a detailed explanation of this question, see [Question 6: Search Strategies](L8_3_6_explanation.md).

## Question 7

### Problem Statement
Consider three algorithms: linear regression, decision tree, and neural network. You have 100 features where 20 are truly relevant and 80 are noise.

#### Task
1. How would multivariate selection affect linear regression performance?
2. How would it affect decision tree performance?
3. How would it affect neural network training time?
4. Which algorithm benefits most from multivariate selection and why?
5. Calculate the expected performance improvement for each algorithm
6. If the noise features have correlation $0.1$ with the target and relevant features have correlation $0.7$, calculate the expected correlation of a random 10-feature subset. What's the probability that a random subset contains at least 8 relevant features?
7. Calculate the expected number of false positives if you use a correlation threshold of $0.3$ for feature selection. If the null hypothesis is that a feature has zero correlation with the target, and you have 100 features with $\alpha = 0.05$, how many false positives would you expect by chance alone?

For a detailed explanation of this question, see [Question 7: Algorithm Effects](L8_3_7_explanation.md).

## Question 8

### Problem Statement
You have 1 hour to perform multivariate feature selection on a dataset with 50 features. Each subset evaluation takes 2 seconds.

#### Task
1. Calculate how many feature subsets you can evaluate in 1 hour
2. If you use forward selection, how many features can you select in the time limit?
3. Design a time-efficient selection strategy for this constraint
4. What's the trade-off between evaluation time and selection quality?
5. If you can parallelize evaluations, how does this change your strategy?
6. If you can run 4 parallel evaluations and each evaluation time increases by $0.1$ seconds for each additional feature in the subset, calculate the maximum subset size you can evaluate within 1 hour. What's the optimal number of parallel processes to minimize total time?

For a detailed explanation of this question, see [Question 8: Resource Constraints](L8_3_8_explanation.md).

## Question 9

### Problem Statement
You're building a machine learning pipeline with preprocessing, feature selection, and model training.

#### Task
1. When should you perform multivariate selection relative to preprocessing?
2. If you select features before scaling, what problems might occur?
3. How does selection timing affect cross-validation results?
4. Design a pipeline that integrates selection at the optimal stage
5. Compare the workflow with early vs late feature selection
6. If preprocessing takes 5 minutes, feature selection takes 10 minutes, and model training takes 15 minutes, calculate the total pipeline time for different selection timings. If you can parallelize preprocessing and selection, what's the minimum total time? What's the time savings compared to sequential execution?
7. Calculate the pipeline efficiency if you can overlap preprocessing and feature selection by 3 minutes, and feature selection and model training by 2 minutes. If the critical path determines total time, what's the new critical path duration? Calculate the percentage improvement in pipeline efficiency.

For a detailed explanation of this question, see [Question 9: Selection Timing](L8_3_9_explanation.md).

## Question 10

### Problem Statement
You're using cross-validation to evaluate feature subsets. Training accuracy improves with more features, but validation accuracy peaks at 15 features.

#### Task
1. What does this pattern suggest about overfitting?
2. How do you modify your evaluation criterion to prevent overfitting?
3. If you use nested cross-validation, how many folds would you recommend?
4. Calculate the optimal number of features based on validation performance
5. Design an evaluation strategy that balances bias and variance
6. If you use 5-fold cross-validation and the standard deviation of accuracy across folds is $2\%$ for 10 features, $3\%$ for 15 features, and $5\%$ for 20 features, calculate the 95% confidence interval for each feature count. Which feature count has the most stable performance?

For a detailed explanation of this question, see [Question 10: Evaluation Criteria](L8_3_10_explanation.md).

## Question 11

### Problem Statement
A model with 100 features shows high variance in cross-validation results. You want to improve stability through feature selection.

#### Task
1. How does reducing features from 100 to 20 affect model stability?
2. If cross-validation variance decreases by 30% with feature selection, what does this suggest?
3. Design a stability-based feature selection criterion
4. How do you measure feature subset stability across different data splits?
5. Compare stability metrics before and after feature selection
6. If the original CV variance is $0.04$ and decreases to $0.028$ after feature selection, calculate the percentage improvement. If you want to achieve a target variance of $0.02$, how many more features should you remove? Assume a linear relationship between feature count and variance.
7. Calculate the Jaccard similarity between two feature subsets $S_1 = \{1,2,3,4,5\}$ and $S_2 = \{2,3,4,6,7\}$. If you have 5 different data splits and the average Jaccard similarity across all pairs is $0.6$, what's the stability index? Then calculate the Kuncheva index if the total number of features is 20.

For a detailed explanation of this question, see [Question 11: Model Stability](L8_3_11_explanation.md).

## Question 12

### Problem Statement
You're working on medical diagnosis with 200 features including lab results, symptoms, and patient demographics.

#### Task
1. What are the key considerations for medical feature selection?
2. If you need interpretability, how does this affect your selection strategy?
3. How do you handle missing data in multivariate selection?
4. Design a selection strategy that prioritizes medical interpretability
5. Compare selection approaches for medical vs financial applications
6. If 20% of features have missing data and you can only use features with < 10% missing values, how many features remain? If the missing data follows a pattern where lab results have 5% missing, symptoms have 15% missing, and demographics have 2% missing, calculate the expected number of features in each category after filtering.
7. Calculate the information gain for a lab result feature if the entropy of the target (disease/no disease) is $0.8$, and after splitting on the feature, the weighted average entropy becomes $0.5$. If you have 1000 patients and the feature has 3 possible values with frequencies 400, 300, and 300, calculate the information gain ratio.

For a detailed explanation of this question, see [Question 12: Domain Requirements](L8_3_12_explanation.md).

## Question 13

### Problem Statement
You have a target variable and want to measure the relationship with feature subsets of size 3-8.

#### Task
1. How do you measure the relationship between a feature subset and target?
2. If features have non-linear relationships with target, what measures would you use?
3. Calculate the number of possible 5-feature subsets from 20 total features
4. Design a test to detect non-linear feature subset-target relationships
5. Compare correlation, mutual information, and other relationship measures
6. If you want to test all subset sizes from 3 to 8 features, calculate the total number of evaluations needed. If you can only evaluate 1000 subsets, what's the maximum subset size you can test completely? What percentage of the total search space does this represent?
7. Calculate the canonical correlation between two feature subsets $X = \{x_1, x_2, x_3\}$ and $Y = \{y_1, y_2\}$ if their correlation matrix is $R = \begin{bmatrix} 0.8 & 0.6 \\ 0.6 & 0.9 \end{bmatrix}$. If the eigenvalues of $R$ are $\lambda_1 = 1.4$ and $\lambda_2 = 0.3$, what's the canonical correlation coefficient? Then calculate the redundancy index if the target has correlation $0.7$ with the first canonical variate.

For a detailed explanation of this question, see [Question 13: Feature-Target Relationships](L8_3_13_explanation.md).

## Question 14

### Problem Statement
Feature selection affects the entire machine learning workflow from data preparation to deployment.

#### Task
1. How does multivariate selection impact data preprocessing steps?
2. How does it affect model validation and testing?
3. What changes are needed in the deployment pipeline?
4. Design a workflow that integrates feature selection seamlessly
5. Compare the workflow complexity with and without multivariate selection
6. If the original workflow takes 2 hours and feature selection adds 30 minutes, but reduces model training time by 45 minutes due to fewer features, calculate the net time impact. If you deploy 10 models per month, what's the total time savings over 6 months?
7. Calculate the workflow efficiency improvement if you can reduce the feature selection time from 30 minutes to 20 minutes by using early stopping. If the efficiency is defined as $\frac{\text{useful\_output\_time}}{\text{total\_time}} \times 100\%$, and useful output time is model training + deployment, calculate the efficiency before and after optimization.

For a detailed explanation of this question, see [Question 14: Workflow Impact](L8_3_14_explanation.md).

## Question 15

### Problem Statement
You've created 50 new features through feature engineering and now need to select the best subset.

#### Task
1. How do you coordinate feature creation and selection?
2. If new features are combinations of original features, how does this affect selection?
3. Design a pipeline that creates and selects features iteratively
4. Calculate the total search space including original and engineered features
5. Design a comprehensive feature engineering and selection strategy
6. If the original 20 features can be combined in pairs to create 190 new features, and you can only keep the top 50 engineered features, calculate the total search space. If evaluating each subset takes 1 second, how long would it take to find the optimal subset? Design a strategy to reduce this to under 1 hour.

For a detailed explanation of this question, see [Question 15: Feature Engineering Integration](L8_3_15_explanation.md).
