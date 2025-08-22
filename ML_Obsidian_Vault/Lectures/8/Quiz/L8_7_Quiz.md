# Lecture 8.7: Search Strategies Quiz

## Overview
This quiz contains 31 questions covering search strategies for feature selection, including general procedures, stopping criteria, different search approaches (complete, heuristic, random), comparisons between methods, and cross-validation analysis. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Feature selection can be viewed as a search problem through the space of possible feature subsets.

#### Task
1. What are the four main components of the general search procedure?
2. How do you generate feature subsets?
3. How do you evaluate feature subsets?
4. What determines when to stop searching?
5. How do you validate the final selection?

For a detailed explanation of this question, see [Question 1: General Search Procedure](L8_7_1_explanation.md).

## Question 2

### Problem Statement
Stopping criteria determine when to terminate the feature selection search process.

#### Task
1. What are the main types of stopping criteria?
2. How do you set thresholds for stopping criteria?
3. What happens if you stop too early?
4. What happens if you stop too late?
5. Design a stopping strategy for feature selection

For a detailed explanation of this question, see [Question 2: Stopping Criteria](L8_7_2_explanation.md).

## Question 3

### Problem Statement
Different search strategies can be classified into three main categories.

#### Task
1. What are the three main types of search strategies?
2. What's the trade-off between completeness and efficiency?
3. When would you use each type of strategy?
4. How do you choose the right search strategy?
5. Compare the characteristics of each approach

For a detailed explanation of this question, see [Question 3: Search Strategies Overview](L8_7_3_explanation.md).

## Question 4

### Problem Statement
Exhaustive search evaluates all possible feature subsets.

#### Task
1. How does exhaustive search work?
2. What are the advantages of exhaustive search?
3. What are the disadvantages of exhaustive search?
4. When is exhaustive search feasible?
5. Calculate the number of subsets for different feature counts

For a detailed explanation of this question, see [Question 4: Exhaustive (Complete) Search](L8_7_4_explanation.md).

## Question 5

### Problem Statement
Heuristic search uses rules of thumb to guide the search process.

#### Task
1. What is heuristic search and how does it work?
2. What are the advantages of heuristic search?
3. What are the disadvantages of heuristic search?
4. When would you choose heuristic over exhaustive search?
5. Compare heuristic vs exhaustive search

For a detailed explanation of this question, see [Question 5: Heuristic Search](L8_7_5_explanation.md).

## Question 6

### Problem Statement
Sequential Forward Selection (SFS) is a greedy heuristic approach.

#### Task
1. How does SFS work?
2. What are the advantages of SFS?
3. What are the disadvantages of SFS?
4. When would you choose SFS?
5. Compare SFS with other search strategies

For a detailed explanation of this question, see [Question 6: Sequential Forward Selection](L8_7_6_explanation.md).

## Question 7

### Problem Statement
Sequential Backward Elimination (SBE) starts with all features and removes them one by one.

#### Task
1. How does SBE work?
2. What are the advantages of SBE?
3. What are the disadvantages of SBE?
4. When would you choose SBE over SFS?
5. Compare SBE with SFS

For a detailed explanation of this question, see [Question 7: Sequential Backward Elimination](L8_7_7_explanation.md).

## Question 8

### Problem Statement
Floating selection methods allow features to be added and removed during the search.

#### Task
1. How do floating selection methods work?
2. What are the advantages of floating selection?
3. What are the disadvantages of floating selection?
4. When would you choose floating selection?
5. Compare floating selection with SFS and SBE

For a detailed explanation of this question, see [Question 8: Floating Selection](L8_7_8_explanation.md).

## Question 9

### Problem Statement
Greedy algorithms make locally optimal choices at each step.

#### Task
1. What is a greedy algorithm and how does it work?
2. What are the advantages of greedy approaches?
3. What are the disadvantages of greedy approaches?
4. When do greedy algorithms work well?
5. Compare greedy vs optimal approaches

For a detailed explanation of this question, see [Question 9: Greedy Algorithms](L8_7_9_explanation.md).

## Question 10

### Problem Statement
Hill climbing is a greedy search strategy that moves toward better solutions.

#### Task
1. How does hill climbing work?
2. What are the advantages of hill climbing?
3. What are the disadvantages of hill climbing?
4. How do you handle local optima in hill climbing?
5. Compare hill climbing with other search strategies

For a detailed explanation of this question, see [Question 10: Hill Climbing](L8_7_10_explanation.md).

## Question 11

### Problem Statement
Best-first search maintains a priority queue of candidate solutions.

#### Task
1. How does best-first search work?
2. What are the advantages of best-first search?
3. What are the disadvantages of best-first search?
4. How do you implement the priority queue?
5. Compare best-first search with other strategies

For a detailed explanation of this question, see [Question 11: Best-First Search](L8_7_11_explanation.md).

## Question 12

### Problem Statement
Branch and bound search uses pruning to avoid exploring unpromising paths.

#### Task
1. How does branch and bound work?
2. What are the advantages of branch and bound?
3. What are the disadvantages of branch and bound?
4. How do you implement pruning in branch and bound?
5. Compare branch and bound with other approaches

For a detailed explanation of this question, see [Question 12: Branch and Bound Search](L8_7_12_explanation.md).

## Question 13

### Problem Statement
Random search strategies use stochastic methods to explore the feature space.

#### Task
1. How do random search strategies work?
2. What are the advantages of random search?
3. What are the disadvantages of random search?
4. When would you use random search?
5. Compare random search with deterministic approaches

For a detailed explanation of this question, see [Question 13: Random Search](L8_7_13_explanation.md).

## Question 14

### Problem Statement
Genetic algorithms use evolutionary principles to search for optimal feature subsets.

#### Task
1. How do genetic algorithms work for feature selection?
2. What are the main components of genetic algorithms?
3. What are the advantages of genetic algorithms?
4. What are the disadvantages of genetic algorithms?
5. Compare genetic algorithms with other search strategies

For a detailed explanation of this question, see [Question 14: Genetic Algorithms](L8_7_14_explanation.md).

## Question 15

### Problem Statement
Simulated annealing uses temperature-based probability to escape local optima.

#### Task
1. How does simulated annealing work?
2. What is the role of temperature in simulated annealing?
3. How do you set the cooling schedule?
4. What are the advantages of simulated annealing?
5. Compare simulated annealing with other approaches

For a detailed explanation of this question, see [Question 15: Simulated Annealing](L8_7_15_explanation.md).

## Question 16

### Problem Statement
Consider a dataset with 6 features and the following search space:

| Feature Subset | Score |
|----------------|-------|
| Feature 1      | 0.75  |
| Feature 2      | 0.68  |
| Feature 3      | 0.72  |
| Features 1+2   | 0.82  |
| Features 1+3   | 0.79  |
| Features 2+3   | 0.76  |
| All features   | 0.85  |

#### Task
1. If you use hill climbing starting with Feature 1, what path would you follow?
2. What's the final solution using hill climbing?
3. Is this the global optimum? Explain
4. If you use best-first search, what would be the search order?
5. Calculate the improvement at each step

For a detailed explanation of this question, see [Question 16: Search Strategy Analysis](L8_7_16_explanation.md).

## Question 17

### Problem Statement
Different search strategies have different computational complexities.

#### Task
1. What's the time complexity of exhaustive search?
2. What's the time complexity of greedy search?
3. What's the time complexity of genetic algorithms?
4. How do you choose based on computational constraints?
5. Compare the scalability of different approaches

For a detailed explanation of this question, see [Question 17: Computational Complexity](L8_7_17_explanation.md).

## Question 18

### Problem Statement
The choice of search strategy affects the quality of the final solution.

#### Task
1. How do different strategies affect solution quality?
2. What's the trade-off between speed and quality?
3. How do you measure solution quality?
4. When is a suboptimal solution acceptable?
5. Design a strategy selection approach

For a detailed explanation of this question, see [Question 18: Solution Quality](L8_7_18_explanation.md).

## Question 19

### Problem Statement
Hybrid approaches combine multiple search strategies.

#### Task
1. What are hybrid search approaches?
2. How do you combine different strategies?
3. What are the advantages of hybrid approaches?
4. What are the disadvantages of hybrid approaches?
5. Design a hybrid search strategy

For a detailed explanation of this question, see [Question 19: Hybrid Approaches](L8_7_19_explanation.md).

## Question 20

### Problem Statement
The curse of dimensionality affects different search strategies differently.

#### Task
1. How does high dimensionality affect exhaustive search?
2. How does it affect greedy search?
3. How does it affect random search?
4. Which strategy is most robust to high dimensionality?
5. Compare the robustness of different approaches

For a detailed explanation of this question, see [Question 20: Dimensionality Impact](L8_7_20_explanation.md).

## Question 21

### Problem Statement
Search strategies can be adapted for different types of feature selection problems.

#### Task
1. How do you adapt strategies for classification vs regression?
2. How do you adapt for different evaluation criteria?
3. How do you adapt for different data types?
4. What considerations are important for adaptation?
5. Design an adaptive search strategy

For a detailed explanation of this question, see [Question 21: Strategy Adaptation](L8_7_21_explanation.md).

## Question 22

### Problem Statement
Consider a scenario where you have limited computational resources for search.

#### Task
1. What search strategies would you use with limited time?
2. How do you prioritize search exploration?
3. What's the trade-off between exploration and exploitation?
4. If you have 1 hour to search, what strategy would you choose?
5. Design a resource-constrained search strategy

For a detailed explanation of this question, see [Question 22: Resource Constraints](L8_7_22_explanation.md).

## Question 23

### Problem Statement
Search strategies can be applied at different stages of the machine learning pipeline.

#### Task
1. When is the best time to perform search-based selection?
2. How does search timing affect results?
3. What happens if you search before preprocessing?
4. How do you handle search in online learning?
5. Compare different timing strategies for search

For a detailed explanation of this question, see [Question 23: Search Timing](L8_7_23_explanation.md).

## Question 24

### Problem Statement
The success of search strategies depends on the quality of the evaluation criteria.

#### Task
1. What makes a good evaluation criterion for search?
2. How do you avoid overfitting in search-based selection?
3. What's the role of cross-validation in search?
4. If search improves training accuracy but hurts validation accuracy, what does this suggest?
5. Compare different evaluation strategies for search

For a detailed explanation of this question, see [Question 24: Evaluation Criteria](L8_7_24_explanation.md).

## Question 25

### Problem Statement
Search strategies affect model robustness and stability.

#### Task
1. How do search strategies improve model stability?
2. What happens to model performance with noisy features?
3. How do search strategies affect cross-validation results?
4. If a model is unstable with 100 features, what would happen with 20 features?
5. Compare stability before and after search-based selection

For a detailed explanation of this question, see [Question 25: Model Stability](L8_7_25_explanation.md).

## Question 26

### Problem Statement
Different domains have different requirements for search-based selection.

#### Task
1. What are the key considerations for medical diagnosis features?
2. How do financial applications differ from image recognition?
3. What's important for real-time sensor data?
4. Compare search needs for text vs numerical data
5. Which domain would benefit most from search-based selection?

For a detailed explanation of this question, see [Question 26: Domain-Specific Requirements](L8_7_26_explanation.md).

## Question 27

### Problem Statement
Search strategies can reveal domain knowledge and insights.

#### Task
1. How can search strategies help understand data relationships?
2. What insights can you gain from search results?
3. How do search strategies help with feature engineering?
4. If certain feature combinations are consistently found, what does this suggest?
5. Compare the insights from different search strategies

For a detailed explanation of this question, see [Question 27: Domain Insights](L8_7_27_explanation.md).

## Question 28

### Problem Statement
The relationship between features and target variables determines search effectiveness.

#### Task
1. How do you measure feature subset-target relationships for search?
2. What types of relationships are hard to detect with search?
3. How do you handle non-linear relationships?
4. If a feature subset has no linear correlation but high mutual information, what does this suggest?
5. Compare different relationship measures for search

For a detailed explanation of this question, see [Question 28: Feature-Target Relationships](L8_7_28_explanation.md).

## Question 29

### Problem Statement
Search strategies affect the entire machine learning workflow.

#### Task
1. How do search strategies impact data preprocessing?
2. How do they affect model validation?
3. How do they impact deployment?
4. What changes in the workflow after search-based selection?
5. Compare workflows with and without search-based selection

For a detailed explanation of this question, see [Question 29: Workflow Impact](L8_7_29_explanation.md).

## Question 30

### Problem Statement
Search strategies are part of a broader feature engineering strategy.

#### Task
1. How do search strategies complement feature creation?
2. What's the relationship between search and transformation?
3. How do you coordinate different feature engineering steps?
4. If you create 100 new features, how do you search for the best subsets?
5. Design a comprehensive feature engineering pipeline

For a detailed explanation of this question, see [Question 30: Feature Engineering Integration](L8_7_30_explanation.md).

## Question 31

### Problem Statement
Feature selection using cross-validation requires careful analysis of performance across different feature subsets. Consider a feature selection scenario where features are ranked by a univariate score, and parameter $k$ (representing the number of features to select) is chosen using cross-validation.

#### Task
1. A 3-fold cross-validation was performed for $k \in \{2, 3, 4\}$ with the following results:
   - For $k = 2$: $\{0.78, 0.80, 0.76\}$
   - For $k = 3$: $\{0.81, 0.82, 0.79\}$
   - For $k = 4$: $\{0.82, 0.78, 0.80\}$
   
   Calculate the mean accuracy and standard deviation for each $k$ value.

2. For each $k$ value, calculate the 95% confidence interval using the t-distribution. Assume the critical t-value for 2 degrees of freedom at $\alpha = 0.05$ is $t_{0.025,2} = 4.303$.

3. Which $k$ would you select and why? Consider both mean performance and stability (standard deviation).

4. If selecting $k = 3$ means you might miss the potential performance of $k = 4$, what's the expected regret? Calculate the expected regret as $E[\text{Regret}] = \max(\text{mean}_{k=4}) - \text{mean}_{k=3}$.

5. Calculate the coefficient of variation ($CV = \frac{\text{std}}{\text{mean}}$) for each $k$ value. Which $k$ provides the most stable performance?

6. If you require a minimum mean accuracy of 0.80, which $k$ values meet this criterion? If you also require a maximum standard deviation of 0.02, which $k$ values satisfy both constraints?

7. Assume each additional feature adds computational cost $C(k) = 0.1k^2$ and provides utility $U(k) = \text{mean accuracy} - 0.05k$. Calculate the net benefit $B(k) = U(k) - C(k)$ for each $k$ value. Which $k$ maximizes net benefit?

8. If you were to perform 5-fold cross-validation instead of 3-fold, how would this affect your confidence in the results? Calculate the standard error of the mean for 5-fold CV assuming the same standard deviations.

**Note:** All calculations should be done by hand. Show all intermediate steps and round final answers to 4 decimal places where appropriate.

For a detailed explanation of this question, see [Question 31: Cross-Validation Feature Selection Analysis](L8_7_31_explanation.md).
