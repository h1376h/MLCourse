# Lecture 6.3: Decision Tree Algorithms (ID3, C4.5, CART) Quiz

## Overview
This quiz contains 35 questions covering different topics from section 6.3 of the lectures on Decision Tree Algorithms, including ID3 foundations, C4.5 improvements, CART capabilities, algorithm comparisons, practical implementations, numerical examples, and comprehensive coverage of all three algorithms.

## Question 1

### Problem Statement
The ID3 algorithm follows a recursive approach to build decision trees.

#### Task
1. [🔍] What are the main steps of the ID3 algorithm?
2. [🔍] How does ID3 choose the best feature for splitting at each node?
3. [🔍] What is the base case for stopping recursion?
4. [🔍] Why is ID3 considered a greedy algorithm?

For a detailed explanation of this question, see [Question 1: ID3 Algorithm Overview](L6_3_1_explanation.md).

## Question 2

### Problem Statement
Consider a dataset with the following class distribution:

| Class | Count |
|-------|-------|
| Yes   | 8     |
| No    | 4     |

#### Task
1. [📚] Calculate the entropy of this dataset
2. [📚] If a feature splits this into two branches with distributions [6,2] and [2,2], calculate the information gain
3. [📚] Would this be a good split according to ID3?
4. [📚] What is the next step in ID3 after finding the best split?

For a detailed explanation of this question, see [Question 2: ID3 Split Selection](L6_3_2_explanation.md).

## Question 3

### Problem Statement
ID3 uses stopping criteria to prevent infinite recursion.

#### Task
1. [🔍] What are the three main stopping criteria in ID3?
2. [🔍] Why is it important to have stopping criteria?
3. [🔍] What happens when all features have been used?
4. [🔍] How do you handle cases where no features remain but the node is not pure?

For a detailed explanation of this question, see [Question 3: ID3 Stopping Criteria](L6_3_3_explanation.md).

## Question 4

### Problem Statement
Consider building a decision tree for a weather dataset with features:

| Feature | Values |
|---------|--------|
| Outlook | Sunny, Rainy, Cloudy |
| Temperature | Hot, Mild, Cool |
| Humidity | High, Normal |
| Windy | True, False |

#### Task
1. [📚] How many possible leaf nodes could this tree have?
2. [📚] What is the maximum depth of the tree?
3. [📚] How would ID3 handle categorical features with many values?
4. [📚] What are the limitations of ID3 for this dataset?

For a detailed explanation of this question, see [Question 4: ID3 Tree Construction](L6_3_4_explanation.md).

## Question 5

### Problem Statement
The ID3 algorithm is the foundation of decision tree learning.

#### Task
1. [🔍] What are the main steps of the ID3 algorithm?
2. [🔍] How does ID3 choose the best feature for splitting at each node?
3. [🔍] What is the base case for stopping recursion in ID3?
4. [🔍] Why is ID3 considered a greedy algorithm?

For a detailed explanation of this question, see [Question 5: ID3 Algorithm Foundations](L6_3_5_explanation.md).

## Question 6

### Problem Statement
C4.5 addresses several key limitations of the ID3 algorithm.

#### Task
1. [🔍] What are the main improvements that C4.5 makes over ID3?
2. [🔍] How does C4.5 address the bias toward features with many values?
3. [🔍] What is gain ratio and how does it differ from information gain?
4. [🔍] How does C4.5 handle continuous features differently from ID3?

For a detailed explanation of this question, see [Question 6: C4.5 Improvements](L6_3_6_explanation.md).

## Question 7

### Problem Statement
CART (Classification and Regression Trees) represents a significant advancement in decision tree algorithms.

#### Task
1. [🔍] What does CART stand for and what makes it unique compared to ID3 and C4.5?
2. [🔍] How does CART's binary split strategy differ from multi-way splits?
3. [🔍] What splitting criterion does CART use and why?
4. [🔍] How can CART handle both classification and regression problems?

For a detailed explanation of this question, see [Question 7: CART Algorithm Features](L6_3_7_explanation.md).

## Question 8

### Problem Statement
**Algorithm Comparison**: Compare ID3, C4.5, and CART using a concrete dataset:

| Feature1 | Feature2 | Feature3 | Target |
|----------|----------|----------|--------|
| A        | X        | 1.2      | Class1 |
| B        | Y        | 2.1      | Class2 |
| A        | Z        | 1.8      | Class1 |
| C        | X        | 3.0      | Class2 |

#### Task
1. [📚] **Feature handling**: Show how each algorithm would handle the categorical and continuous features
2. [📚] **Splitting criteria**: Calculate the splitting metrics each algorithm would use
3. [📚] **Tree structure**: Compare the resulting tree structures from each algorithm
4. [📚] **Performance analysis**: Discuss the advantages and disadvantages of each approach

For a detailed explanation of this question, see [Question 8: Algorithm Comparison](L6_3_8_explanation.md).

## Question 9

### Problem Statement
Consider a dataset with the following class distribution:

| Class | Count |
|-------|-------|
| Yes   | 8     |
| No    | 4     |

#### Task
1. [📚] Calculate the entropy of this dataset
2. [📚] If a feature splits this into two branches with distributions $[6,2]$ and $[2,2]$, calculate the information gain
3. [📚] Calculate the gain ratio for this split
4. [📚] How would CART evaluate this split using Gini impurity?

For a detailed explanation of this question, see [Question 9: Splitting Metrics Comparison](L6_3_9_explanation.md).

## Question 10

### Problem Statement
**Continuous Feature Handling**: Each algorithm handles continuous features differently.

| Algorithm | Approach | Example |
|-----------|----------|---------|
| ID3       | Does not handle directly | N/A |
| C4.5      | Binary splits with thresholds | Age ≤ 25 vs Age > 25 |
| CART      | Binary splits optimized | Income ≤ 50K vs Income > 50K |

#### Task
1. [🔍] Why can't ID3 handle continuous features directly?
2. [🔍] How does C4.5 find optimal split points for continuous features?
3. [🔍] How does CART's approach to continuous features differ from C4.5?
4. [🔍] What are the computational complexities of finding optimal splits?

For a detailed explanation of this question, see [Question 10: Continuous Feature Handling](L6_3_10_explanation.md).

## Question 11

### Problem Statement
**Missing Value Strategies**: Different algorithms handle missing values differently.

#### Task
1. [🔍] How does ID3 typically handle missing values?
2. [🔍] What is C4.5's "fractional instance" method for missing values?
3. [🔍] How does CART use surrogate splits for missing values?
4. [🔍] Which approach is most robust and why?

For a detailed explanation of this question, see [Question 11: Missing Value Strategies](L6_3_11_explanation.md).

## Question 12

### Problem Statement
**Splitting Criteria Comparison**: Compare the different splitting criteria used by each algorithm.

| Algorithm | Primary Criterion | Secondary Features |
|-----------|-------------------|-------------------|
| ID3       | Information Gain  | None |
| C4.5      | Gain Ratio        | Handles bias |
| CART      | Gini Impurity     | Binary splits only |

#### Task
1. [📚] Calculate information gain, gain ratio, and Gini impurity for the same split
2. [📚] Explain when each criterion might give different results
3. [📚] Which criterion is most suitable for different types of data?
4. [📚] How do computational costs compare between these criteria?

For a detailed explanation of this question, see [Question 12: Splitting Criteria Analysis](L6_3_12_explanation.md).

## Question 13

### Problem Statement
**Pruning Capabilities**: Compare pruning approaches across algorithms.

#### Task
1. [🔍] Does ID3 include built-in pruning capabilities?
2. [🔍] How does C4.5 implement post-pruning?
3. [🔍] What is CART's cost-complexity pruning approach?
4. [🔍] Which algorithm provides the most sophisticated pruning?

For a detailed explanation of this question, see [Question 13: Pruning Approaches](L6_3_13_explanation.md).

## Question 14

### Problem Statement
**Implementation Complexity**: Analyze the implementation requirements for each algorithm.

#### Task
1. [📚] What are the core data structures needed for ID3 implementation?
2. [📚] What additional complexity does C4.5 add over ID3?
3. [📚] What makes CART implementation more complex than ID3 and C4.5?
4. [📚] Compare the time and space complexities of all three algorithms?

For a detailed explanation of this question, see [Question 14: Implementation Analysis](L6_3_14_explanation.md).

## Question 15

### Problem Statement
**Regression Capabilities**: Only CART can handle regression problems directly.

#### Task
1. [🔍] Why can't ID3 handle regression problems?
2. [🔍] How could C4.5 be adapted for regression (theoretical approach)?
3. [🔍] How does CART handle regression problems?
4. [🔍] What splitting criterion does CART use for regression?

For a detailed explanation of this question, see [Question 15: Regression Handling](L6_3_15_explanation.md).

## Question 16

### Problem Statement
**Algorithm Selection Criteria**: When should you choose each algorithm?

#### Task
1. [📚] In what scenarios is ID3 still the best choice?
2. [📚] When should you prefer C4.5 over CART?
3. [📚] When is CART the most appropriate choice?
4. [📚] How do dataset characteristics influence algorithm selection?

For a detailed explanation of this question, see [Question 16: Algorithm Selection](L6_3_16_explanation.md).

## Question 17

### Problem Statement
**Performance Comparison**: Analyze computational performance across algorithms.

#### Task
1. [📚] Compare training time complexity for all three algorithms
2. [📚] Compare prediction time complexity
3. [📚] Compare memory requirements
4. [📚] How does performance scale with dataset size and dimensionality?

For a detailed explanation of this question, see [Question 17: Performance Analysis](L6_3_17_explanation.md).

## Question 18

### Problem Statement
**Bias and Variance**: Different algorithms have different bias-variance characteristics.

#### Task
1. [🔍] Which algorithm typically has the highest bias?
2. [🔍] Which algorithm is most prone to overfitting (high variance)?
3. [🔍] How do the different splitting criteria affect bias-variance tradeoff?
4. [🔍] Which algorithm generalizes best and why?

For a detailed explanation of this question, see [Question 18: Bias-Variance Analysis](L6_3_18_explanation.md).

## Question 19

### Problem Statement
**Feature Selection Bias**: Analyze how each algorithm handles features with different characteristics.

#### Task
1. [📚] How does ID3's information gain bias toward features with many values?
2. [📚] How does C4.5's gain ratio address this bias?
3. [📚] How does CART's binary splitting strategy affect feature selection?
4. [📚] Which algorithm is most robust to irrelevant features?

For a detailed explanation of this question, see [Question 19: Feature Selection Bias](L6_3_19_explanation.md).

## Question 20

### Problem Statement
**Interpretability Comparison**: Compare the interpretability of trees produced by each algorithm.

#### Task
1. [🔍] Which algorithm produces the most interpretable trees?
2. [🔍] How does tree depth typically compare across algorithms?
3. [🔍] Which algorithm's decision rules are easiest to understand?
4. [🔍] How does splitting strategy affect interpretability?

For a detailed explanation of this question, see [Question 20: Interpretability Analysis](L6_3_20_explanation.md).

## Question 21

### Problem Statement
**Modern Extensions**: How have these classic algorithms influenced modern decision tree methods?

#### Task
1. [📚] How do modern libraries (scikit-learn, XGBoost) build on these algorithms?
2. [📚] What features from each algorithm are preserved in modern implementations?
3. [📚] How have ensemble methods extended these basic algorithms?
4. [📚] What are the current limitations that modern algorithms address?

For a detailed explanation of this question, see [Question 21: Modern Extensions](L6_3_21_explanation.md).

## Question 22

### Problem Statement
**Comprehensive Implementation**: Implement and compare all three algorithms on a real dataset.

#### Task
1. [🔍] **Implementation**: Create working implementations of ID3, C4.5, and CART
2. [🔍] **Dataset**: Apply all three to the same classification dataset
3. [🔍] **Evaluation**: Compare accuracy, tree size, training time, and interpretability
4. [🔍] **Analysis**: Provide recommendations for when to use each algorithm

For a detailed explanation of this question, see [Question 22: Comprehensive Implementation](L6_3_22_explanation.md).

## Question 23

### Problem Statement
**C4.5 Gain Ratio Calculation**: Consider the following dataset for predicting loan approval:

| Income Level | Age Group | Credit Score | Loan Approved |
|--------------|-----------|--------------|---------------|
| High         | Young     | Good         | Yes           |
| High         | Young     | Poor         | No            |
| Medium       | Middle    | Good         | Yes           |
| Low          | Old       | Good         | Yes           |
| Low          | Young     | Poor         | No            |
| Medium       | Old       | Good         | Yes           |
| High         | Middle    | Poor         | No            |
| Low          | Middle    | Good         | Yes           |

#### Task
1. [📚] Calculate the entropy of the entire dataset
2. [📚] Calculate information gain for each feature using ID3 method
3. [📚] Calculate gain ratio for each feature using C4.5 method
4. [📚] Compare which feature ID3 vs C4.5 would choose and explain why

For a detailed explanation of this question, see [Question 23: C4.5 Gain Ratio Analysis](L6_3_23_explanation.md).

## Question 24

### Problem Statement
**C4.5 Continuous Feature Handling**: Consider this dataset with a continuous age feature:

| Age | Income | Credit | Loan Approved |
|-----|--------|---------|---------------|
| 22  | Low    | Poor    | No            |
| 25  | High   | Good    | Yes           |
| 35  | Medium | Good    | Yes           |
| 45  | High   | Poor    | No            |
| 30  | Low    | Good    | No            |
| 55  | Medium | Good    | Yes           |
| 28  | High   | Good    | Yes           |
| 40  | Low    | Poor    | No            |

#### Task
1. [📚] **Threshold identification**: Find all possible threshold values for the Age feature
2. [📚] **Binary split evaluation**: Calculate information gain for Age ≤ 30 vs Age > 30
3. [📚] **Optimal threshold**: Find the optimal threshold that maximizes information gain
4. [📚] **C4.5 advantage**: Explain how C4.5 handles this better than ID3

For a detailed explanation of this question, see [Question 24: C4.5 Continuous Features](L6_3_24_explanation.md).

## Question 25

### Problem Statement
**CART Gini Impurity Calculation**: Using the same loan dataset from Question 23:

| Income Level | Age Group | Credit Score | Loan Approved |
|--------------|-----------|--------------|---------------|
| High         | Young     | Good         | Yes           |
| High         | Young     | Poor         | No            |
| Medium       | Middle    | Good         | Yes           |
| Low          | Old       | Good         | Yes           |
| Low          | Young     | Poor         | No            |
| Medium       | Old       | Good         | Yes           |
| High         | Middle    | Poor         | No            |
| Low          | Middle    | Good         | Yes           |

#### Task
1. [📚] Calculate the Gini impurity of the entire dataset
2. [📚] For Income Level feature, calculate Gini impurity for each possible binary split
3. [📚] Find the binary split with lowest Gini impurity
4. [📚] Compare CART's choice with ID3 and C4.5 results from previous questions

For a detailed explanation of this question, see [Question 25: CART Gini Impurity](L6_3_25_explanation.md).

## Question 26

### Problem Statement
**CART Binary Split Strategy**: Consider a feature "City" with values [NYC, LA, Chicago, Miami, Boston]. CART must create binary splits.

#### Task
1. [🔍] How many possible binary splits can CART create for this feature?
2. [📚] Calculate the number of binary splits for a categorical feature with n values
3. [📚] If the target distribution for each city is: NYC(3,1), LA(2,2), Chicago(1,3), Miami(4,0), Boston(2,2), find the optimal binary split
4. [🔍] Why does CART prefer binary splits over multi-way splits?

For a detailed explanation of this question, see [Question 26: CART Binary Splits](L6_3_26_explanation.md).

## Question 27

### Problem Statement
**Algorithm Efficiency Comparison**: Compare the computational efficiency of ID3, C4.5, and CART.

#### Task
1. [🔍] Compare the time complexity of evaluating all possible splits for a categorical feature with k values
2. [🔍] How does the complexity change when handling continuous features?
3. [📚] Calculate the number of operations needed for each algorithm on a dataset with 1000 samples and 10 features
4. [🔍] Which algorithm scales best with increasing feature cardinality and why?

For a detailed explanation of this question, see [Question 27: Algorithm Efficiency](L6_3_27_explanation.md).

## Question 28

### Problem Statement
**Missing Value Handling Comparison**: Consider this dataset with missing values (shown as ?):

| Feature1 | Feature2 | Feature3 | Target |
|----------|----------|----------|--------|
| A        | X        | 1        | Yes    |
| B        | ?        | 2        | No     |
| ?        | Y        | 3        | Yes    |
| A        | Z        | ?        | No     |
| C        | X        | 1        | Yes    |

#### Task
1. [🔍] How would ID3 typically handle these missing values?
2. [🔍] Explain C4.5's fractional instance method for the sample with Feature2 = ?
3. [🔍] How would CART use surrogate splits for missing values?
4. [📚] Demonstrate one missing value handling approach with calculations

For a detailed explanation of this question, see [Question 28: Missing Value Strategies](L6_3_28_explanation.md).

## Question 29

### Problem Statement
**C4.5 vs ID3 Bias Demonstration**: Consider features with different cardinalities:

| Sample | Feature_A (2 values) | Feature_B (4 values) | Feature_C (8 values) | Target |
|--------|---------------------|---------------------|---------------------|---------|
| 1      | 0                   | 0                   | 0                   | Class1  |
| 2      | 1                   | 1                   | 1                   | Class2  |
| 3      | 0                   | 2                   | 2                   | Class1  |
| 4      | 1                   | 3                   | 3                   | Class2  |
| 5      | 0                   | 0                   | 4                   | Class1  |
| 6      | 1                   | 1                   | 5                   | Class2  |
| 7      | 0                   | 2                   | 6                   | Class1  |
| 8      | 1                   | 3                   | 7                   | Class2  |

#### Task
1. [📚] Calculate information gain for each feature using ID3
2. [📚] Calculate gain ratio for each feature using C4.5
3. [📚] Demonstrate the bias toward high-cardinality features in ID3
4. [📚] Show how C4.5's gain ratio corrects this bias

For a detailed explanation of this question, see [Question 29: Feature Cardinality Bias](L6_3_29_explanation.md).

## Question 30

### Problem Statement
**CART Regression Tree**: CART can handle regression problems. Consider this dataset:

| Feature1 | Feature2 | Target (continuous) |
|----------|----------|-------------------|
| Low      | A        | 10.5              |
| High     | A        | 15.2              |
| Low      | B        | 12.8              |
| High     | B        | 18.1              |
| Medium   | A        | 13.0              |
| Medium   | B        | 16.5              |

#### Task
1. [📚] Calculate the variance of the entire dataset
2. [📚] Calculate variance reduction for splitting on Feature1
3. [📚] Calculate variance reduction for splitting on Feature2
4. [🔍] How does CART determine the predicted value for each leaf node in regression?

For a detailed explanation of this question, see [Question 30: CART Regression](L6_3_30_explanation.md).

## Question 31

### Problem Statement
**Multi-Algorithm Decision Tree Construction**: Apply all three algorithms to the same small dataset:

| Weather | Temperature | Humidity | Wind | Play Tennis |
|---------|-------------|----------|------|-------------|
| Sunny   | Hot         | High     | Weak | No          |
| Sunny   | Hot         | High     | Strong | No        |
| Overcast| Hot         | High     | Weak | Yes         |
| Rain    | Mild        | High     | Weak | Yes         |
| Rain    | Cool        | Normal   | Weak | Yes         |
| Rain    | Cool        | Normal   | Strong | No        |

#### Task
1. [📚] **ID3 approach**: Build the first level using information gain
2. [📚] **C4.5 approach**: Build the first level using gain ratio
3. [📚] **CART approach**: Build the first level using Gini impurity with binary splits
4. [📚] **Comparison**: Compare the root node choices and explain differences

For a detailed explanation of this question, see [Question 31: Multi-Algorithm Comparison](L6_3_31_explanation.md).

## Question 32

### Problem Statement
**Pruning Capabilities Across Algorithms**: Compare pruning approaches in ID3, C4.5, and CART.

#### Task
1. [🔍] Why doesn't ID3 include built-in pruning capabilities?
2. [🔍] Explain C4.5's pessimistic error pruning approach
3. [🔍] Describe CART's cost-complexity pruning with the alpha parameter
4. [📚] Given a tree with 5 nodes and training errors [0, 1, 0, 2, 1], demonstrate one pruning calculation

For a detailed explanation of this question, see [Question 32: Pruning Comparison](L6_3_32_explanation.md).

## Question 33

### Problem Statement
**Feature Selection Robustness**: Analyze how each algorithm handles irrelevant features.

Dataset with one relevant and two irrelevant features:

| Relevant_Feature | Noise_Feature1 | Noise_Feature2 | Target |
|-----------------|----------------|----------------|--------|
| A               | X              | 1              | Yes    |
| A               | Y              | 2              | Yes    |
| B               | Z              | 3              | No     |
| B               | X              | 1              | No     |
| A               | Z              | 2              | Yes    |
| B               | Y              | 3              | No     |

#### Task
1. [📚] Calculate information gain for all features using ID3
2. [📚] Calculate gain ratio for all features using C4.5
3. [🔍] Which algorithm is most likely to select the relevant feature first?
4. [🔍] How do the splitting criteria affect robustness to noise?

For a detailed explanation of this question, see [Question 33: Feature Selection Robustness](L6_3_33_explanation.md).

## Question 34

### Problem Statement
**Tree Interpretability Analysis**: Compare the interpretability of trees produced by each algorithm.

#### Task
1. [🔍] Which algorithm typically produces the most compact trees?
2. [🔍] How does CART's binary splitting affect rule interpretability compared to ID3's multi-way splits?
3. [🔍] Why might C4.5 trees be easier to interpret than ID3 trees?
4. [📚] Given a business context (loan approval), rank the algorithms by interpretability and justify

For a detailed explanation of this question, see [Question 34: Tree Interpretability](L6_3_34_explanation.md).

## Question 35

### Problem Statement
**Overfitting Susceptibility**: Analyze which algorithm is most prone to overfitting.

#### Task
1. [🔍] Which algorithm has the highest risk of overfitting and why?
2. [🔍] How do the different splitting criteria affect overfitting tendency?
3. [🔍] Which algorithm provides the best built-in overfitting protection?
4. [📚] Design a small example where one algorithm overfits but another doesn't

For a detailed explanation of this question, see [Question 35: Overfitting Analysis](L6_3_35_explanation.md).
