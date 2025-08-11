# Lecture 6.3: Decision Tree Algorithms (ID3, C4.5, CART) Quiz

## Overview
This quiz contains 33 comprehensive questions covering decision tree algorithms ID3, C4.5, and CART. Topics include algorithm foundations, splitting criteria, feature handling, missing values, pruning, complexity analysis, practical implementations, edge cases, cost functions, overfitting analysis, and modern extensions with detailed numerical examples.

## Question 1

### Problem Statement
The ID3 (Iterative Dichotomiser 3) algorithm forms the foundation of decision tree learning through its recursive partitioning approach.

#### Task
1. List the main steps of the ID3 algorithm in order
2. Explain how ID3 chooses the best feature for splitting at each node
3. What are the base cases for stopping recursion in ID3?
4. Why is ID3 considered a greedy algorithm? Provide a concise explanation

For a detailed explanation of this question, see [Question 1: ID3 Algorithm Foundations](L6_3_1_explanation.md).

## Question 2

### Problem Statement
Consider a binary classification dataset with the following class distribution:

| Class | Count |
|-------|-------|
| Yes   | 8     |
| No    | 4     |

#### Task
1. Calculate the entropy of this dataset (show your work)
2. If a feature splits this into two branches with distributions $[6,2]$ and $[2,2]$, calculate the information gain
3. Would this be considered a good split according to ID3? Justify your answer
4. What is the next step in the ID3 algorithm after finding the best split?

For a detailed explanation of this question, see [Question 2: ID3 Information Gain Calculation](L6_3_2_explanation.md).

## Question 3

### Problem Statement
ID3 employs specific stopping criteria to prevent infinite recursion and ensure proper tree termination.

#### Task
1. List the three main stopping criteria used by ID3
2. Explain why stopping criteria are essential for the algorithm
3. What happens when all available features have been used in the path from root to current node?
4. How should ID3 handle cases where no features remain but the node contains mixed classes?

For a detailed explanation of this question, see [Question 3: ID3 Stopping Criteria](L6_3_3_explanation.md).

## Question 4

### Problem Statement
Consider building a decision tree using ID3 for a weather prediction dataset with the following feature specifications:

| Feature | Possible Values |
|---------|----------------|
| Outlook | Sunny, Rainy, Cloudy |
| Temperature | Hot, Mild, Cool |
| Humidity | High, Normal |
| Windy | True, False |

#### Task
1. Calculate the maximum number of possible leaf nodes this tree could have
2. Determine the maximum depth the tree could reach
3. Explain how ID3 handles categorical features with different numbers of values
4. Identify the main limitations of ID3 when applied to this type of dataset

For a detailed explanation of this question, see [Question 4: ID3 Tree Structure Analysis](L6_3_4_explanation.md).

## Question 5

### Problem Statement
Evaluate whether each of the following statements about decision tree algorithms is TRUE or FALSE. Provide a brief justification for each answer.

#### Task
1. ID3 can handle continuous features directly without any preprocessing
2. C4.5's gain ratio always produces the same feature ranking as ID3's information gain
3. CART uses only binary splits regardless of the number of values in a categorical feature
4. The entropy of a pure node (all samples belong to one class) is always zero
5. C4.5's split information penalizes features with many values to reduce bias
6. CART can handle both classification and regression problems using the same tree structure
7. ID3 includes built-in pruning mechanisms to prevent overfitting
8. C4.5's handling of missing values is more sophisticated than ID3's approach
9. Information gain and Gini impurity always select the same feature for splitting
10. CART's binary splits always result in more interpretable trees than multi-way splits

For a detailed explanation of this question, see [Question 5: Decision Tree Algorithm Properties](L6_3_5_explanation.md).

## Question 6

### Problem Statement
Consider C4.5's improvement over ID3 in handling feature selection bias.

#### Task
1. What is the main problem with ID3's information gain regarding features with many values?
2. For a feature with values $\{A, B, C\}$ splitting a dataset of 12 samples into subsets of size $\{3, 5, 4\}$, calculate the split information
3. If the information gain for this split is 0.8, calculate the gain ratio
4. Explain in one sentence why split information corrects the bias

For a detailed explanation of this question, see [Question 6: C4.5 Gain Ratio Analysis](L6_3_6_explanation.md).

## Question 7

### Problem Statement
Design a simple "Algorithm Selection Game" where you must choose the most appropriate decision tree algorithm for different scenarios.

#### Task
For each scenario below, select the most suitable algorithm (ID3, C4.5, or CART) and explain your reasoning in 1-2 sentences:

1. **Small educational dataset**: 50 samples, 4 categorical features (2-3 values each), no missing data, interpretability is crucial
2. **Mixed-type dataset**: 1000 samples, 6 categorical features, 4 continuous features, 15% missing values
3. **High-cardinality problem**: 500 samples, features include customer ID, zip code, and product category with 50+ unique values
4. **Regression task**: Predicting house prices using both categorical (neighborhood, style) and continuous (size, age) features
5. **Noisy environment**: Dataset with many irrelevant features and some measurement errors

For a detailed explanation of this question, see [Question 7: Algorithm Selection Strategy](L6_3_7_explanation.md).

## Question 8

### Problem Statement
Create a "Decision Tree Construction Race" where you manually trace through the first split decision for all three algorithms on the same tiny dataset.

**Dataset: Restaurant Recommendation**
| Cuisine | Price | Rating | Busy | Recommend |
|---------|-------|--------|------|-----------|
| Italian | Low   | Good   | No   | Yes       |
| Chinese | High  | Poor   | Yes  | No        |
| Italian | High  | Good   | No   | Yes       |
| Mexican | Low   | Poor   | Yes  | No        |
| Chinese | Low   | Good   | Yes  | Yes       |
| Mexican | High  | Good   | No   | Yes       |

#### Task
1. **ID3 approach**: Calculate information gain for each feature and identify the best split
2. **C4.5 approach**: Calculate gain ratio for each feature and compare with ID3's choice
3. **CART approach**: For the Cuisine feature, evaluate all possible binary splits using Gini impurity
4. **Final comparison**: Which feature would each algorithm choose as the root? Explain any differences

For a detailed explanation of this question, see [Question 8: Multi-Algorithm Construction Trace](L6_3_8_explanation.md).

## Question 9

### Problem Statement
CART's binary splitting strategy differs fundamentally from ID3 and C4.5.

#### Task
1. For a categorical feature "Grade" with values $\{A, B, C, D\}$, list all possible binary splits CART would consider
2. Calculate the number of binary splits for a categorical feature with $k$ values
3. What does CART stand for and why can it handle regression problems?
4. Given class distributions: A(3,1), B(2,2), C(1,3), D(4,0), find the optimal binary split using Gini impurity

For a detailed explanation of this question, see [Question 9: CART Binary Splitting Strategy](L6_3_9_explanation.md).

## Question 10

### Problem Statement
Compare splitting criteria used by different decision tree algorithms.

#### Task
1. For class distribution $[6, 2]$, calculate both entropy and Gini impurity
2. For class distribution $[4, 4]$, calculate both measures
3. Which measure (entropy or Gini) reaches its maximum value for balanced distributions?
4. In practice, do entropy and Gini impurity usually lead to significantly different trees?

For a detailed explanation of this question, see [Question 10: Splitting Criteria Comparison](L6_3_10_explanation.md).

## Question 11

### Problem Statement
Match each algorithm feature on the left with its correct algorithm on the right:

#### Task
1. Uses only binary splits                    A) ID3 only
2. Handles continuous features directly       B) C4.5 only  
3. Uses information gain as splitting criterion C) CART only
4. Can perform regression                     D) Both ID3 and C4.5
5. Uses gain ratio to reduce bias             E) All three algorithms
6. Requires feature discretization for continuous data F) None of them

For a detailed explanation of this question, see [Question 11: Algorithm Feature Matching](L6_3_11_explanation.md).

## Question 12

### Problem Statement
Consider how C4.5 handles continuous features through optimal threshold selection.

#### Task
1. Why can't ID3 handle continuous features directly? Answer in one sentence
2. For ages $\{22, 25, 30, 35, 40\}$ with classes $\{No, No, Yes, Yes, No\}$, list all candidate threshold values
3. Calculate information gain for the threshold Age ≤ 27.5
4. How does C4.5's approach to continuous features differ from manual discretization?

For a detailed explanation of this question, see [Question 12: Continuous Feature Handling](L6_3_12_explanation.md).

## Question 13

### Problem Statement
Consider missing value handling strategies across different algorithms.

#### Task
1. How does ID3 typically handle missing values in practice?
2. Describe C4.5's "fractional instance" method in one sentence
3. What are CART's surrogate splits and why are they useful?
4. Given a dataset where 30% of samples have missing values for Feature A, which algorithm would be most robust?

For a detailed explanation of this question, see [Question 13: Missing Value Strategies](L6_3_13_explanation.md).

## Question 14

### Problem Statement
Apply the ID3 algorithm to this small dataset:

| Outlook | Temperature | Humidity | Wind | Play Tennis |
|---------|-------------|----------|------|-------------|
| Sunny   | Hot         | High     | Weak | No          |
| Sunny   | Hot         | High     | Strong | No        |
| Overcast| Hot         | High     | Weak | Yes         |
| Rain    | Mild        | High     | Weak | Yes         |
| Rain    | Cool        | Normal   | Weak | Yes         |
| Rain    | Cool        | Normal   | Strong | No        |

#### Task
1. Calculate the entropy of the entire dataset
2. Calculate information gain for the Outlook feature
3. Calculate information gain for the Wind feature
4. Which feature should ID3 choose as the root node?

For a detailed explanation of this question, see [Question 14: ID3 Algorithm Application](L6_3_14_explanation.md).

## Question 15

### Problem Statement
Consider CART's approach to regression problems.

| Feature1 | Feature2 | Target |
|----------|----------|--------|
| Low      | A        | 10.5   |
| High     | A        | 15.2   |
| Low      | B        | 12.8   |
| High     | B        | 18.1   |
| Medium   | A        | 13.0   |
| Medium   | B        | 16.5   |

#### Task
1. Calculate the variance of the entire dataset
2. Calculate variance reduction for splitting on Feature1 (Low vs {Medium, High})
3. What would be the predicted value for each leaf node after this split?
4. How does CART's regression criterion differ from classification criteria?

For a detailed explanation of this question, see [Question 15: CART Regression Trees](L6_3_15_explanation.md).

## Question 16

### Problem Statement
Which of the following scenarios would benefit most from each algorithm? Choose the best match.

#### Task
For each scenario, select ID3, C4.5, or CART and justify your choice:

1. Small dataset with only categorical features and no missing values
2. Large dataset with mixed feature types and 20% missing values  
3. Medical diagnosis requiring highly interpretable rules
4. Predicting house prices (continuous target variable)
5. Dataset with many categorical features having 10+ values each

For a detailed explanation of this question, see [Question 16: Algorithm Selection Scenarios](L6_3_16_explanation.md).

## Question 17

### Problem Statement
Consider pruning approaches across the three algorithms.

#### Task
1. Does ID3 include built-in pruning capabilities? Why or why not?
2. Describe C4.5's pessimistic error pruning in one sentence
3. What is the purpose of CART's cost-complexity pruning parameter $\alpha$?
4. If a subtree has training accuracy 90% but validation accuracy 75%, which algorithms would likely prune it?

For a detailed explanation of this question, see [Question 17: Pruning Strategies](L6_3_17_explanation.md).

## Question 18

### Problem Statement
Calculate gain ratio for this loan approval dataset:

| Income | Age_Group | Credit | Approved |
|--------|-----------|--------|----------|
| High   | Young     | Good   | Yes      |
| High   | Young     | Poor   | No       |
| Medium | Middle    | Good   | Yes      |
| Low    | Old       | Good   | Yes      |
| Low    | Young     | Poor   | No       |
| Medium | Old       | Good   | Yes      |

#### Task
1. Calculate entropy of the dataset
2. Calculate information gain for Income feature  
3. Calculate split information for Income feature
4. Calculate gain ratio and compare with information gain

For a detailed explanation of this question, see [Question 18: Gain Ratio Calculation](L6_3_18_explanation.md).

## Question 19

### Problem Statement
Consider the computational complexity of each algorithm.

#### Task
1. Which algorithm has the highest computational complexity for categorical features? Why?
2. How does handling continuous features affect C4.5's time complexity?
3. For a dataset with 1000 samples and 10 features (5 categorical with avg 4 values, 5 continuous), rank the algorithms by expected training time
4. What makes CART more computationally expensive than ID3 for categorical features?

For a detailed explanation of this question, see [Question 19: Computational Complexity](L6_3_19_explanation.md).

## Question 20

### Problem Statement
Analyze bias-variance trade-offs in decision tree algorithms.

#### Task
1. Which algorithm typically has the highest bias? Explain why
2. Which algorithm is most prone to overfitting without pruning?
3. How does CART's binary splitting strategy affect the bias-variance trade-off?
4. Which algorithm provides the best built-in protection against overfitting?

For a detailed explanation of this question, see [Question 20: Bias-Variance Analysis](L6_3_20_explanation.md).

## Question 21

### Problem Statement
Compare tree interpretability across algorithms.

#### Task
The geometric interpretation of decision trees helps understand their decision-making process. Which statement correctly describes decision boundaries?

A) ID3 creates axis-parallel rectangular regions in feature space
B) C4.5 can create diagonal decision boundaries due to continuous feature handling  
C) CART's binary splits always create more complex boundaries than multi-way splits
D) All decision tree algorithms create identical decision boundaries for the same dataset

For a detailed explanation of this question, see [Question 21: Decision Tree Interpretability](L6_3_21_explanation.md).

## Question 22

### Problem Statement
Consider CART's surrogate splits for missing value handling.

#### Task
1. Define surrogate splits in one sentence
2. Given primary split "Income > 50K" with 80% accuracy, rank these surrogates by quality:
   - "Education = Graduate": 70% agreement
   - "Age > 40": 65% agreement
   - "Experience > 8": 75% agreement
3. Why are surrogate splits more robust than simple imputation methods?
4. How would you use the best surrogate when Income is missing for a new sample?

For a detailed explanation of this question, see [Question 22: CART Surrogate Splits](L6_3_22_explanation.md).

## Question 23

### Problem Statement
Consider feature selection robustness across algorithms.

| Relevant_Feature | Noise_Feature1 | Noise_Feature2 | Target |
|------------------|----------------|----------------|--------|
| A                | X              | 1              | Yes    |
| A                | Y              | 2              | Yes    |
| B                | Z              | 3              | No     |
| B                | X              | 1              | No     |
| A                | Z              | 2              | Yes    |
| B                | Y              | 3              | No     |

#### Task
1. Calculate information gain for all three features
2. Calculate gain ratio for all three features  
3. Which algorithm is most likely to select the relevant feature first?
4. How do noise features affect each algorithm differently?

For a detailed explanation of this question, see [Question 23: Feature Selection Robustness](L6_3_23_explanation.md).

## Question 24

### Problem Statement
Write pseudocode for the ID3 algorithm and trace its execution.

#### Task
1. Write complete pseudocode for ID3 including stopping criteria
2. For this dataset, trace the first split decision:

| Feature1 | Feature2 | Class |
|----------|----------|-------|
| A        | X        | +     |
| A        | Y        | +     |
| B        | X        | -     |
| B        | Y        | -     |

3. Calculate information gain for both features
4. Show which feature ID3 would select and why

For a detailed explanation of this question, see [Question 24: ID3 Algorithm Implementation](L6_3_24_explanation.md).

## Question 25

### Problem Statement
Consider ID3's behavior when all features have been used but nodes remain impure.

#### Task
1. Describe the scenario where ID3 exhausts all features but has impure nodes
2. Given this partially constructed tree where all features are used:
   ```
   Root: Outlook
   ├── Sunny → $[Yes: 2, No: 3]$
   ├── Cloudy → $[Yes: 4, No: 0]$  
   └── Rain → $[Yes: 1, No: 2]$
   ```
3. How should ID3 handle the impure "Sunny" and "Rain" nodes?
4. What is the decision rule for leaf node class assignment in this case?

For a detailed explanation of this question, see [Question 25: ID3 Feature Exhaustion](L6_3_25_explanation.md).

## Question 26

### Problem Statement
Analyze overfitting susceptibility across the three decision tree algorithms.

#### Task
1. Which algorithm has the highest risk of overfitting and why?
2. How do the different splitting criteria affect overfitting tendency?
3. Which algorithm provides the best built-in overfitting protection?
4. Design a small example where one algorithm overfits but another doesn't

For a detailed explanation of this question, see [Question 26: Overfitting Susceptibility Analysis](L6_3_26_explanation.md).

## Question 27

### Problem Statement
Compare all three algorithms on the same dataset to understand their differences:

| Size | Location | Age | Price_Range |
|------|----------|-----|-------------|
| Small| Urban    | New | Low         |
| Large| Urban    | New | High        |
| Small| Rural    | Old | Low         |
| Large| Rural    | Old | High        |
| Medium| Urban   | New | Medium      |
| Medium| Rural   | Old | Medium      |

#### Task
1. **ID3 Analysis**: Calculate information gain for each feature
2. **C4.5 Analysis**: Calculate gain ratio for each feature  
3. **CART Analysis**: Find the best binary split for Size feature using Gini impurity
4. **Comparison**: Which feature would each algorithm choose as the root? Explain any differences

For a detailed explanation of this question, see [Question 27: Comprehensive Algorithm Comparison](L6_3_27_explanation.md).

## Question 28

### Problem Statement
Examine entropy calculation edge cases and mathematical properties.

#### Task
1. Calculate entropy for these distributions:
   - Pure node: $[10, 0]$
   - Balanced node: $[5, 5]$
   - Skewed node: $[9, 1]$
   - Empty node: $[0, 0]$
2. Explain how to handle the empty node case mathematically
3. Show that entropy is maximized for balanced distributions
4. Derive the maximum possible entropy for $k$ classes

For a detailed explanation of this question, see [Question 28: Entropy Edge Cases](L6_3_28_explanation.md).

## Question 29

### Problem Statement
Consider entropy edge cases and mathematical properties.

#### Task
1. Calculate entropy for these class distributions:
   - Pure node: $[8, 0]$
   - Balanced binary: $[4, 4]$  
   - Highly skewed: $[7, 1]$
2. What is the maximum possible entropy for a binary classification problem?
3. Prove that entropy is maximized when classes are equally distributed
4. How should you handle the log(0) case when calculating entropy?

For a detailed explanation of this question, see [Question 29: Entropy Mathematical Properties](L6_3_29_explanation.md).

## Question 30

### Problem Statement
Analyze stopping criteria across algorithms.

#### Task
1. List three stopping criteria used by ID3
2. What additional stopping criterion does C4.5 add beyond ID3's criteria?
3. Name two stopping criteria specific to CART
4. For a node with 5 samples (3 positive, 2 negative), should ID3 continue splitting? Consider minimum samples and purity thresholds

For a detailed explanation of this question, see [Question 30: Algorithm Stopping Criteria](L6_3_30_explanation.md).

## Question 31

### Problem Statement
Consider how modern machine learning libraries implement these classic algorithms.

#### Task
1. How does scikit-learn's DecisionTreeClassifier relate to CART?
2. What features from ID3 and C4.5 are preserved in modern implementations?
3. How have ensemble methods like Random Forest extended these basic algorithms?
4. What limitations of classic algorithms do modern methods address?

For a detailed explanation of this question, see [Question 31: Modern Algorithm Extensions](L6_3_31_explanation.md).

## Question 32

### Problem Statement
Analyze multi-way vs binary splits as a fundamental difference between algorithms.

#### Task
1. For feature "Grade" with values $\{A, B, C, D, F\}$, show all splits that:
   - ID3 would consider (1 multi-way split)
   - CART would consider (list all binary combinations)
2. Calculate the number of binary splits for a categorical feature with $k$ values
3. Discuss advantages and disadvantages of each approach
4. When might binary splits be preferred over multi-way splits?

For a detailed explanation of this question, see [Question 32: Multi-way vs Binary Splits](L6_3_32_explanation.md).

## Question 33

### Problem Statement
Consider CART's cost function approach to optimization.

#### Task
1. Write the cost function that CART minimizes when choosing splits
2. For a categorical feature "Color" with values $\{Red, Blue, Green, Yellow\}$, list all possible binary splits
3. Given class distributions: Red(2,1), Blue(1,2), Green(3,0), Yellow(1,1), find the optimal binary split using Gini impurity
4. Compare this result with what information gain would choose

For a detailed explanation of this question, see [Question 33: CART Cost Function](L6_3_33_explanation.md).

## Question 34

### Problem Statement
Analyze computational complexity across ID3, C4.5, and CART algorithms.

#### Task
1. Derive the time complexity for ID3 given $n$ samples, $m$ features, and average branching factor $b$
2. How does C4.5's complexity differ due to continuous feature handling?
3. Analyze CART's complexity considering binary splits and surrogate computation
4. For a dataset with 1000 samples, 20 features (10 categorical with avg 4 values, 10 continuous), estimate relative computation time

For a detailed explanation of this question, see [Question 34: Algorithm Complexity Analysis](L6_3_34_explanation.md).