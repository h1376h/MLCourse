# Lecture 6.3: Decision Tree Algorithms (ID3, C4.5, CART) Quiz

## Overview
This quiz contains 18 questions covering different topics from section 6.3 of the lectures on Decision Tree Algorithms, including ID3 foundations, C4.5 improvements, CART capabilities, algorithm comparisons, and practical implementations.

## Question 1

### Problem Statement
The ID3 algorithm is the foundation of decision tree learning.

#### Task
1. [🔍] What are the main steps of the ID3 algorithm?
2. [🔍] How does ID3 choose the best feature for splitting at each node?
3. [🔍] What is the base case for stopping recursion in ID3?
4. [🔍] Why is ID3 considered a greedy algorithm?

For a detailed explanation of this question, see [Question 1: ID3 Algorithm Foundations](L6_3_1_explanation.md).

## Question 2

### Problem Statement
C4.5 addresses several key limitations of the ID3 algorithm.

#### Task
1. [🔍] What are the main improvements that C4.5 makes over ID3?
2. [🔍] How does C4.5 address the bias toward features with many values?
3. [🔍] What is gain ratio and how does it differ from information gain?
4. [🔍] How does C4.5 handle continuous features differently from ID3?

For a detailed explanation of this question, see [Question 2: C4.5 Improvements](L6_3_2_explanation.md).

## Question 3

### Problem Statement
CART (Classification and Regression Trees) represents a significant advancement in decision tree algorithms.

#### Task
1. [🔍] What does CART stand for and what makes it unique compared to ID3 and C4.5?
2. [🔍] How does CART's binary split strategy differ from multi-way splits?
3. [🔍] What splitting criterion does CART use and why?
4. [🔍] How can CART handle both classification and regression problems?

For a detailed explanation of this question, see [Question 3: CART Algorithm Features](L6_3_3_explanation.md).

## Question 4

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

For a detailed explanation of this question, see [Question 4: Algorithm Comparison](L6_3_4_explanation.md).

## Question 5

### Problem Statement
Consider a dataset with the following class distribution:

| Class | Count |
|-------|-------|
| Yes   | 8     |
| No    | 4     |

#### Task
1. [📚] Calculate the entropy of this dataset
2. [📚] If a feature splits this into two branches with distributions [6,2] and [2,2], calculate the information gain
3. [📚] Calculate the gain ratio for this split
4. [📚] How would CART evaluate this split using Gini impurity?

For a detailed explanation of this question, see [Question 5: Splitting Metrics Comparison](L6_3_5_explanation.md).

## Question 6

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

For a detailed explanation of this question, see [Question 6: Continuous Feature Handling](L6_3_6_explanation.md).

## Question 7

### Problem Statement
**Missing Value Strategies**: Different algorithms handle missing values differently.

#### Task
1. [🔍] How does ID3 typically handle missing values?
2. [🔍] What is C4.5's "fractional instance" method for missing values?
3. [🔍] How does CART use surrogate splits for missing values?
4. [🔍] Which approach is most robust and why?

For a detailed explanation of this question, see [Question 7: Missing Value Strategies](L6_3_7_explanation.md).

## Question 8

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

For a detailed explanation of this question, see [Question 8: Splitting Criteria Analysis](L6_3_8_explanation.md).

## Question 9

### Problem Statement
**Pruning Capabilities**: Compare pruning approaches across algorithms.

#### Task
1. [🔍] Does ID3 include built-in pruning capabilities?
2. [🔍] How does C4.5 implement post-pruning?
3. [🔍] What is CART's cost-complexity pruning approach?
4. [🔍] Which algorithm provides the most sophisticated pruning?

For a detailed explanation of this question, see [Question 9: Pruning Approaches](L6_3_9_explanation.md).

## Question 10

### Problem Statement
**Implementation Complexity**: Analyze the implementation requirements for each algorithm.

#### Task
1. [📚] What are the core data structures needed for ID3 implementation?
2. [📚] What additional complexity does C4.5 add over ID3?
3. [📚] What makes CART implementation more complex than ID3 and C4.5?
4. [📚] Compare the time and space complexities of all three algorithms?

For a detailed explanation of this question, see [Question 10: Implementation Analysis](L6_3_10_explanation.md).

## Question 11

### Problem Statement
**Regression Capabilities**: Only CART can handle regression problems directly.

#### Task
1. [🔍] Why can't ID3 handle regression problems?
2. [🔍] How could C4.5 be adapted for regression (theoretical approach)?
3. [🔍] How does CART handle regression problems?
4. [🔍] What splitting criterion does CART use for regression?

For a detailed explanation of this question, see [Question 11: Regression Handling](L6_3_11_explanation.md).

## Question 12

### Problem Statement
**Algorithm Selection Criteria**: When should you choose each algorithm?

#### Task
1. [📚] In what scenarios is ID3 still the best choice?
2. [📚] When should you prefer C4.5 over CART?
3. [📚] When is CART the most appropriate choice?
4. [📚] How do dataset characteristics influence algorithm selection?

For a detailed explanation of this question, see [Question 12: Algorithm Selection](L6_3_12_explanation.md).

## Question 13

### Problem Statement
**Performance Comparison**: Analyze computational performance across algorithms.

#### Task
1. [📚] Compare training time complexity for all three algorithms
2. [📚] Compare prediction time complexity
3. [📚] Compare memory requirements
4. [📚] How does performance scale with dataset size and dimensionality?

For a detailed explanation of this question, see [Question 13: Performance Analysis](L6_3_13_explanation.md).

## Question 14

### Problem Statement
**Bias and Variance**: Different algorithms have different bias-variance characteristics.

#### Task
1. [🔍] Which algorithm typically has the highest bias?
2. [🔍] Which algorithm is most prone to overfitting (high variance)?
3. [🔍] How do the different splitting criteria affect bias-variance tradeoff?
4. [🔍] Which algorithm generalizes best and why?

For a detailed explanation of this question, see [Question 14: Bias-Variance Analysis](L6_3_14_explanation.md).

## Question 15

### Problem Statement
**Feature Selection Bias**: Analyze how each algorithm handles features with different characteristics.

#### Task
1. [📚] How does ID3's information gain bias toward features with many values?
2. [📚] How does C4.5's gain ratio address this bias?
3. [📚] How does CART's binary splitting strategy affect feature selection?
4. [📚] Which algorithm is most robust to irrelevant features?

For a detailed explanation of this question, see [Question 15: Feature Selection Bias](L6_3_15_explanation.md).

## Question 16

### Problem Statement
**Interpretability Comparison**: Compare the interpretability of trees produced by each algorithm.

#### Task
1. [🔍] Which algorithm produces the most interpretable trees?
2. [🔍] How does tree depth typically compare across algorithms?
3. [🔍] Which algorithm's decision rules are easiest to understand?
4. [🔍] How does splitting strategy affect interpretability?

For a detailed explanation of this question, see [Question 16: Interpretability Analysis](L6_3_16_explanation.md).

## Question 17

### Problem Statement
**Modern Extensions**: How have these classic algorithms influenced modern decision tree methods?

#### Task
1. [📚] How do modern libraries (scikit-learn, XGBoost) build on these algorithms?
2. [📚] What features from each algorithm are preserved in modern implementations?
3. [📚] How have ensemble methods extended these basic algorithms?
4. [📚] What are the current limitations that modern algorithms address?

For a detailed explanation of this question, see [Question 17: Modern Extensions](L6_3_17_explanation.md).

## Question 18

### Problem Statement
**Comprehensive Implementation**: Implement and compare all three algorithms on a real dataset.

#### Task
1. [🔍] **Implementation**: Create working implementations of ID3, C4.5, and CART
2. [🔍] **Dataset**: Apply all three to the same classification dataset
3. [🔍] **Evaluation**: Compare accuracy, tree size, training time, and interpretability
4. [🔍] **Analysis**: Provide recommendations for when to use each algorithm

For a detailed explanation of this question, see [Question 18: Comprehensive Implementation](L6_3_18_explanation.md).
