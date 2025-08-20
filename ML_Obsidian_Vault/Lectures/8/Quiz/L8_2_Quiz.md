# Lecture 8.2: Univariate Feature Selection Methods Quiz

## Overview
This quiz contains 19 focused questions covering univariate feature selection methods, including filter scoring, irrelevance criteria, correlation measures, mutual information, chi-square tests, determining the optimal number of features, selection timing, resource constraints, domain-specific requirements, and domain insights. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Univariate feature selection considers one feature at a time independently.

#### Task
1. What is the main advantage of univariate methods?
2. What is the main limitation of univariate methods?
3. If you have $100$ features, how many individual evaluations does univariate selection require?
4. Given a dataset with features $X_1, X_2, ..., X_n$ and target $Y$, calculate the computational complexity of univariate selection. If each feature evaluation takes $2$ seconds and you have $500$ features, how long will the complete selection process take? Express your answer in minutes and seconds.

For a detailed explanation of this question, see [Question 1: Univariate Approach](L8_2_1_explanation.md).

## Question 2

### Problem Statement
Univariate filter scoring ranks individual features based on their relevance to the target.

#### Task
1. What is the purpose of filter scoring?
2. How does filter scoring differ from wrapper methods?
3. A wrapper method takes $5$ minutes to evaluate a single feature subset, while a filter method takes $30$ seconds per feature. If you have $100$ features and want to evaluate all possible subsets of size $1$, $2$, and $3$, calculate the total time difference between wrapper and filter approaches. Which method is faster and by how much?
4. If filter methods have $80\%$ accuracy in identifying relevant features, how many false positives would you expect with $100$ features where $20$ are truly relevant?

For a detailed explanation of this question, see [Question 2: Univariate Filter Scoring](L8_2_2_explanation.md).

## Question 3

### Problem Statement
Feature irrelevance can be defined using conditional probabilities and KL divergence.

#### Task
1. What does it mean for a feature to be irrelevant?
2. If $P(y|x) = P(y)$ for all values of $x$, what does this suggest about feature $x$?
3. Given two probability distributions $P = [0.3, 0.4, 0.3]$ and $Q = [0.2, 0.5, 0.3]$, calculate the KL divergence $D_{KL}(P||Q)$. If $D_{KL}(P||Q) = 0.05$, what does this tell you about the relationship between features $P$ and $Q$?
4. Calculate the KL divergence for uniform distributions $P = [0.25, 0.25, 0.25, 0.25]$ and $Q = [0.5, 0.5, 0, 0]$

For a detailed explanation of this question, see [Question 3: Criteria: Defining Feature Irrelevance](L8_2_3_explanation.md).

## Question 4

### Problem Statement
Pearson correlation measures linear relationships between features and targets.

#### Task
1. What is the formula for Pearson correlation?
2. What values can Pearson correlation take?
3. Given the following data points: $X = [1, 2, 3, 4, 5]$ and $Y = [2, 4, 5, 4, 6]$, calculate the Pearson correlation coefficient step by step. Show your work including the calculation of means, deviations, and the final correlation value.
4. If $Y = aX + b$, what is the correlation between $X$ and $Y$?
5. Calculate the correlation for $X = [1, 2, 3, 4, 5]$ and $Y = [1, 4, 9, 16, 25]$

For a detailed explanation of this question, see [Question 4: Criteria: Pearson Correlation](L8_2_4_explanation.md).

## Question 5

### Problem Statement
Mutual information measures the dependence between features and targets and can detect non-linear relationships.

#### Task
1. What is mutual information and how is it calculated?
2. How does mutual information differ from correlation?
3. Given a joint probability distribution $P(X,Y)$ where $P(X=0,Y=0) = 0.3$, $P(X=0,Y=1) = 0.2$, $P(X=1,Y=0) = 0.1$, and $P(X=1,Y=1) = 0.4$, calculate the mutual information $I(X;Y)$. Show your calculations for marginal probabilities, entropies, and the final mutual information value.
4. Consider a dataset where $X$ takes values $\{1, 2, 3, 4\}$ with equal probability and $Y = X^2 \bmod 4$. Calculate the correlation between $X$ and $Y$, then explain why correlation is $0$ but mutual information reveals the relationship.

For a detailed explanation of this question, see [Question 5: Criteria: Mutual Information](L8_2_5_explanation.md).

## Question 6

### Problem Statement
Chi-square test measures independence between categorical features and targets.

#### Task
1. What is the chi-square test statistic formula?
2. When is the chi-square test appropriate?
3. Given the following contingency table:

| Feature\Target | Class 0 | Class 1 |
|----------------|---------|---------|
| Category A     | $25$      | $15$      |
| Category B     | $20$      | $30$      |

Calculate the chi-square statistic step by step. Show your expected frequencies, chi-square contributions, and the final statistic. With $\alpha = 0.05$ and $1$ degree of freedom, is the feature independent of the target?
4. For a $4 \times 3$ contingency table, calculate the degrees of freedom. If the observed chi-square statistic is $18.5$, find the critical value at $\alpha = 0.01$ and determine if the null hypothesis of independence should be rejected.

For a detailed explanation of this question, see [Question 6: Criteria: Chi-Square Test](L8_2_6_explanation.md).

## Question 7

### Problem Statement
Determining the optimal number of features to select is crucial for model performance.

#### Task
1. How do you use cross-validation to find the optimal '$k$'?
2. What is the trade-off between too few and too many features?
3. Given the following cross-validation results for different numbers of features:

| Features | CV Accuracy | Std Dev |
|----------|-------------|---------|
| $5$        | $0.82$        | $0.03$    |
| $10$       | $0.87$        | $0.02$    |
| $15$       | $0.89$        | $0.02$    |
| $20$       | $0.88$        | $0.03$    |
| $25$       | $0.86$        | $0.04$    |

Calculate the $95\%$ confidence interval for each feature count and determine the optimal number of features. Use the rule: select the smallest number of features where the upper confidence bound of a larger feature set doesn't exceed the lower confidence bound of the current set.
4. For a dataset with $1200$ samples, calculate the sample sizes for $3$-fold, $5$-fold, and $10$-fold cross-validation.

For a detailed explanation of this question, see [Question 7: Determining the Number of Features to Select](L8_2_7_explanation.md).

## Question 8

### Problem Statement
Consider a dataset with $5$ features and their correlation scores with the target:

| Feature | Correlation |
|---------|-------------|
| A       | $0.85$        |
| B       | $0.72$        |
| C       | $0.31$        |
| D       | $0.68$        |
| E       | $0.45$        |

#### Task
1. Rank the features by relevance
2. If you select the top $3$ features, which ones would you choose?
3. Calculate the coefficient of variation ($CV = \frac{\text{standard deviation}}{\text{mean}}$) for the correlation scores
4. If you want to select features such that their combined correlation variance is minimized while maintaining an average correlation above $0.6$, which features would you select? Show your calculations.

For a detailed explanation of this question, see [Question 8: Feature Ranking Analysis](L8_2_8_explanation.md).

## Question 9

### Problem Statement
Consider a binary classification problem with the following feature-target relationships:

| Feature | Correlation | Mutual Info | Chi-Square |
|---------|-------------|-------------|------------|
| Age     | $0.45$        | $0.38$        | $12.5$       |
| Income  | $0.72$        | $0.65$        | $28.3$       |
| Gender  | $0.15$        | $0.42$        | $18.7$       |
| Education| $0.38$       | $0.35$        | $15.2$       |

#### Task
1. Rank features by each criterion
2. Which features would you select if you want exactly $2$?
3. Normalize all three metrics to a $0$-$1$ scale and calculate a composite score using weights: $40\%$ correlation, $35\%$ mutual information, and $25\%$ chi-square. Which features would you select based on this composite score? Show your normalization and weighted average calculations.
4. If you want to maximize the minimum score across all criteria, which features would you select?

For a detailed explanation of this question, see [Question 9: Multi-Criteria Ranking](L8_2_9_explanation.md).

## Question 10

### Problem Statement
Feature selection thresholds affect the number of selected features.

#### Task
1. How do you set thresholds for different selection criteria?
2. What happens if you set the threshold too high or too low?
3. Given feature scores: $[0.95, 0.87, 0.76, 0.65, 0.54, 0.43, 0.32, 0.21, 0.15, 0.08]$, calculate the threshold that would select exactly $30\%$ of features
4. If you want to ensure that selected features have scores at least $2$ standard deviations above the mean, what threshold would you use? Show your calculations.

For a detailed explanation of this question, see [Question 10: Threshold Selection](L8_2_10_explanation.md).

## Question 11

### Problem Statement
Feature selection can be viewed as an optimization problem where we aim to find the optimal subset of features that maximizes model performance while minimizing computational cost and complexity. The problem involves balancing multiple objectives: accuracy, interpretability, and computational efficiency.

#### Task
1. What is the objective function for feature selection? Explain the components and their trade-offs.
2. What are the constraints in this optimization problem? Consider practical limitations like computational resources, interpretability requirements, and domain-specific constraints. If you have a maximum budget of $1000$ computational units and each feature costs $50$ units to evaluate, what's the maximum number of features you can consider?
3. Formulate the feature selection problem as a multi-objective optimization: maximize accuracy ($0$-$1$ scale) and minimize the number of features. If accuracy $= 0.8 + 0.02n - 0.001n^2$ where $n$ is the number of features, find the optimal number of features that maximizes the objective function $f(n) = \text{accuracy} - 0.1n$. Show your derivative calculations and optimization steps. What is the maximum value of the objective function?
4. Formulate the feature selection problem as a constrained optimization: maximize accuracy subject to the constraint that the number of features $\leq 25$. If the accuracy function is $A(n) = 0.7 + 0.015n - 0.0002n^2$ where $n$ is the number of features, find the optimal number of features that maximizes accuracy within the constraint. What is the maximum achievable accuracy under this constraint?

For a detailed explanation of this question, see [Question 11: Optimization Formulation](L8_2_11_explanation.md).

## Question 12

### Problem Statement
The curse of dimensionality affects feature selection strategies.

#### Task
1. How does high dimensionality affect univariate methods?
2. What happens to feature relevance as dimensions increase?
3. If the probability of a feature being relevant decreases exponentially with dimensionality as $P(\text{relevant}) = 0.1 \times 0.95^n$ where $n$ is the number of features, calculate the expected number of relevant features for datasets with $100$, $1000$, and $10000$ features
4. If you need at least $5$ relevant features for your model, what's the maximum dimensionality you should consider?

For a detailed explanation of this question, see [Question 12: High Dimensionality](L8_2_12_explanation.md).

## Question 13

### Problem Statement
Feature selection affects different types of machine learning algorithms.

#### Task
1. How does feature selection affect linear models vs tree-based models?
2. Which algorithm type benefits most from univariate selection?
3. If training time for a linear model is $T = 0.1n^2$ seconds and for a tree model is $T = 0.05n \log n$ seconds, where $n$ is the number of features, calculate the training time savings when reducing features from $100$ to $20$ for both models
4. Which model benefits more from feature selection in terms of training time reduction?

For a detailed explanation of this question, see [Question 13: Algorithm-Specific Effects](L8_2_13_explanation.md).

## Question 14

### Problem Statement
Feature selection affects model robustness, stability, and interpretability.

#### Task
1. How does feature selection improve model stability?
2. How does feature selection improve interpretability?
3. If model stability is measured as $S = \frac{1}{1 + 0.05n}$ where $n$ is the number of features, calculate the stability improvement when reducing from $100$ to $20$ features
4. If model complexity increases exponentially with the number of features (complexity $= 2^n$ where $n$ is the number of features), calculate the complexity for $5$, $10$, and $15$ features. If stakeholders can understand models with complexity $\leq 1000$, what's the maximum number of features you should select?

For a detailed explanation of this question, see [Question 14: Interpretability vs Performance](L8_2_14_explanation.md).

## Question 15

### Problem Statement
Consider a dataset with $1000$ samples and $100$ features where only $20$ features are truly relevant.

#### Task
1. What percentage of features are irrelevant in this dataset?
2. If each irrelevant feature adds $1\%$ noise, what's the total noise level?
3. What's the signal-to-noise ratio with all features vs relevant features only?
4. If the probability of randomly selecting a relevant feature is $p = \frac{20}{100} = 0.2$, use the binomial distribution to calculate the probability of selecting exactly $15$ relevant features when randomly choosing $20$ features. What's the probability of selecting at least $15$ relevant features? Show your calculations using the binomial formula.

For a detailed explanation of this question, see [Question 15: Irrelevant Features Impact](L8_2_15_explanation.md).

## Question 16

### Problem Statement
Feature selection can be applied at different stages of the machine learning pipeline.

#### Task
1. When is the best time to perform feature selection?
2. What happens if you select features before preprocessing?
3. If preprocessing takes $2$ minutes per feature and feature selection takes $1$ minute per feature, calculate the total pipeline time for three strategies: $(1)$ preprocess all $100$ features then select $20$, $(2)$ select $20$ features then preprocess them, $(3)$ preprocess $50$ features then select $20$. Which strategy is fastest and by how much?
4. How do you handle feature selection in online learning scenarios?

For a detailed explanation of this question, see [Question 16: Selection Timing](L8_2_16_explanation.md).

## Question 17

### Problem Statement
Consider a scenario where you have limited computational resources for feature selection.

#### Task
1. What selection strategies would you use with limited time?
2. If feature evaluation time follows a power law distribution where the $i$-th feature takes $t_i = 0.1 \times i^{0.8}$ seconds, calculate the total time to evaluate the first $100$, $500$, and $1000$ features. If you have exactly $1$ hour, how many features can you evaluate? Show your calculations using the sum of the power series.
3. If you can only evaluate $10\%$ of features, which ones would you prioritize?
4. Design an efficient selection strategy for resource-constrained environments.

For a detailed explanation of this question, see [Question 17: Resource Constraints](L8_2_17_explanation.md).

## Question 18

### Problem Statement
Different domains have different feature selection requirements and constraints.

#### Task
1. What are the key considerations for medical diagnosis features?
2. How do financial applications differ from image recognition in feature selection needs?
3. If medical diagnosis requires $99.9\%$ confidence in feature relevance and financial applications require $95\%$ confidence, calculate the minimum sample sizes needed for each domain assuming a binomial distribution. If you have $1000$ samples, what confidence level can you achieve for feature relevance testing?
4. How do regulatory compliance requirements affect feature selection strategies?

For a detailed explanation of this question, see [Question 18: Domain-Specific Requirements](L8_2_18_explanation.md).

## Question 19

### Problem Statement
Feature selection can reveal domain knowledge and insights beyond just improving model performance.

#### Task
1. How can feature selection help understand data relationships?
2. If feature selection reveals that $80\%$ of selected features are from the same domain category, calculate the probability that this occurred by chance if features were randomly distributed across $5$ categories. Use the binomial distribution to determine if this clustering is statistically significant at $\alpha = 0.05$.
3. What insights can you gain from consistently selected features?
4. How does feature selection help with feature engineering decisions?

For a detailed explanation of this question, see [Question 19: Domain Insights](L8_2_19_explanation.md).
