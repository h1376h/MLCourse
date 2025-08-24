# Lecture 6.3: Decision Tree Algorithms (ID3, C4.5, CART) Quiz

## Overview
This quiz contains 47 comprehensive questions covering decision tree algorithms ID3, C4.5, and CART. Topics include algorithm foundations, splitting criteria, feature handling, missing values, pruning, complexity analysis, practical implementations, edge cases, cost functions, overfitting analysis, modern extensions, visual tree construction, algorithm selection strategies, detailed comparisons between CART using Gini impurity vs Entropy with detailed numerical examples, advanced tree construction challenges, and subtle algorithmic properties designed to test deep understanding.

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
1. Calculate the entropy of this dataset
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
1. [📚] Calculate the maximum number of possible leaf nodes this tree could have
2. [📚] Determine the maximum depth the tree could reach
3. [📚] Explain how ID3 handles categorical features with different numbers of values
4. [📚] Identify the main limitations of ID3 when applied to this type of dataset

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
10. CART's binary splits (using Gini impurity) always result in more interpretable trees than multi-way splits

For a detailed explanation of this question, see [Question 5: Decision Tree Algorithm Properties](L6_3_5_explanation.md).

## Question 6

### Problem Statement
Consider C4.5's improvement over ID3 in handling feature selection bias.

#### Task
1. [📚] What is the main problem with ID3's information gain regarding features with many values?
2. For a feature with values $\{A, B, C\}$ splitting a dataset of $12$ samples into subsets of size $\{3, 5, 4\}$, calculate the split information using:
   $$\text{Split Info} = -\sum_{i=1}^{k} \frac{|S_i|}{|S|} \log_2\left(\frac{|S_i|}{|S|}\right)$$
3. If the information gain for this split is $0.8$, calculate the gain ratio using:
   $$\text{Gain Ratio} = \frac{\text{Information Gain}}{\text{Split Information}}$$
4. [📚] Explain in one sentence why split information corrects the bias
5. Calculate gain ratio for a binary feature splitting the same dataset into $\{7, 5\}$ with information gain $0.6$. Which feature would C4.5 prefer?

For a detailed explanation of this question, see [Question 6: C4.5 Gain Ratio Analysis](L6_3_6_explanation.md).

## Question 7

### Problem Statement
Design a simple "Algorithm Selection Game" where you must choose the most appropriate decision tree algorithm for different scenarios.

#### Task
For each scenario below, select the most suitable algorithm (ID3, C4.5, or CART) and explain your reasoning in 1-2 sentences:

1. **Small educational dataset**: $50$ samples, $4$ categorical features ($2$-$3$ values each), no missing data, interpretability is crucial
2. **Mixed-type dataset**: $1000$ samples, $6$ categorical features, $4$ continuous features, $15\%$ missing values
3. **High-cardinality problem**: $500$ samples, features include customer ID, zip code, and product category with $50+$ unique values
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
3. **CART approach (using Gini impurity)**: For the Cuisine feature, evaluate all possible binary splits using Gini impurity
4. **CART approach (using Entropy)**: For the Cuisine feature, evaluate all possible binary splits using entropy-based information gain
5. Which feature would each algorithm choose as the root? Explain any differences
6. Compare the results between CART using Gini vs CART using Entropy - are they the same? Why or why not?
7. Draw the first level of the decision tree that each algorithm would construct

For a detailed explanation of this question, see [Question 8: Multi-Algorithm Construction Trace](L6_3_8_explanation.md).

## Question 9

### Problem Statement
CART's binary splitting strategy differs fundamentally from ID3 and C4.5.

#### Task
1. For a categorical feature "Grade" with values $\{A, B, C, D\}$, list all possible binary splits CART (using Gini impurity) would consider
2. Calculate the number of binary splits for a categorical feature with $k$ values (Formula: $2^{k-1} - 1$)
3. What does CART stand for and why can it handle regression problems?
4. Given class distributions: A$(3,1)$, B$(2,2)$, C$(1,3)$, D$(4,0)$, find the optimal binary split using Gini impurity

For a detailed explanation of this question, see [Question 9: CART Binary Splitting Strategy](L6_3_9_explanation.md).

## Question 10

### Problem Statement
Compare splitting criteria used by different decision tree algorithms.

#### Task
1. For class distribution $[6, 2]$, calculate both entropy $$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$ and Gini impurity $$\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2$$
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
3. Calculate information gain for the threshold Age $\leq 27.5$
4. How does C4.5's approach to continuous features differ from manual discretization?
5. Find the optimal threshold that maximizes information gain for this age dataset

For a detailed explanation of this question, see [Question 12: Continuous Feature Handling](L6_3_12_explanation.md).

## Question 13

### Problem Statement
Consider missing value handling strategies across different algorithms.

#### Task
1. How does ID3 typically handle missing values in practice?
2. Describe C4.5's "fractional instance" method in one sentence
3. What are CART's surrogate splits (using Gini impurity) and why are they useful?
4. Given a dataset where $30\%$ of samples have missing values for Feature A, which algorithm would be most robust?

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
Consider CART's approach (using Gini impurity for classification, MSE for regression) to regression problems.

| Feature1 | Feature2 | Target |
|----------|----------|--------|
| Low      | A        | 10.5   |
| High     | A        | 15.2   |
| Low      | B        | 12.8   |
| High     | B        | 18.1   |
| Medium   | A        | 13.0   |
| Medium   | B        | 16.5   |

#### Task
1. Calculate the variance of the entire dataset using $$\text{Var}(S) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$
2. Calculate variance reduction for splitting on Feature1 (Low vs {Medium, High}) using:
   $$\text{Variance Reduction} = \text{Var}(S) - \sum_{i} \frac{|S_i|}{|S|} \text{Var}(S_i)$$
3. What would be the predicted value for each leaf node after this split?
4. How does CART's regression criterion (MSE) differ from classification criteria (Gini impurity)?

For a detailed explanation of this question, see [Question 15: CART Regression Trees](L6_3_15_explanation.md).

## Question 16

### Problem Statement
Which of the following scenarios would benefit most from each algorithm? Choose the best match.

#### Task
For each scenario, select ID3, C4.5, or CART and justify your choice:

1. Small dataset with only categorical features and no missing values
2. Large dataset with mixed feature types and $20\%$ missing values  
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
3. What is the purpose of CART's cost-complexity pruning parameter $\alpha$ (using Gini impurity)?
4. If a subtree has training accuracy $90\%$ but validation accuracy $75\%$, which algorithms would likely prune it?

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
3. For a dataset with $1000$ samples and $10$ features ($5$ categorical with avg $4$ values, $5$ continuous), rank the algorithms by expected training time
4. What makes CART (using Gini impurity) more computationally expensive than ID3 for categorical features?

For a detailed explanation of this question, see [Question 19: Computational Complexity](L6_3_19_explanation.md).

## Question 20

### Problem Statement
Analyze bias-variance trade-offs in decision tree algorithms.

#### Task
1. Which algorithm typically has the highest bias? Explain why
2. Which algorithm is most prone to overfitting without pruning?
3. How does CART's binary splitting strategy (using Gini impurity) affect the bias-variance trade-off?
4. Which algorithm provides the best built-in protection against overfitting?

For a detailed explanation of this question, see [Question 20: Bias-Variance Analysis](L6_3_20_explanation.md).

## Question 21

### Problem Statement
Compare tree interpretability across algorithms.

#### Task
The geometric interpretation of decision trees helps understand their decision-making process. Which statement correctly describes decision boundaries?

**A)** ID3 creates axis-parallel rectangular regions in feature space
**B)** C4.5 can create diagonal decision boundaries due to continuous feature handling  
**C)** CART's binary splits (using Gini impurity) always create more complex boundaries than multi-way splits
**D)** All decision tree algorithms create identical decision boundaries for the same dataset

For a detailed explanation of this question, see [Question 21: Decision Tree Interpretability](L6_3_21_explanation.md).

## Question 22

### Problem Statement
Consider CART's surrogate splits (using Gini impurity) for missing value handling.

#### Task
1. Define surrogate splits in one sentence
2. Given primary split "Income $> \$50K$" with $80\%$ accuracy, rank these surrogates by quality:
   - "Education = Graduate": $70\%$ agreement
   - "Age $> 40$": $65\%$ agreement
   - "Experience $> 8$": $75\%$ agreement
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

3. [📚] Calculate information gain for both features
4. [📚] Show which feature ID3 would select and why

For a detailed explanation of this question, see [Question 24: ID3 Algorithm Implementation](L6_3_24_explanation.md).

## Question 25

### Problem Statement
Consider ID3's behavior when all features have been used but nodes remain impure.

#### Task
1. Describe the scenario where ID3 exhausts all features but has impure nodes
2. Given this partially constructed tree where all features are used:

```
Root: Outlook
├── Sunny → [Yes: 2, No: 3]
├── Cloudy → [Yes: 4, No: 0]  
└── Rain → [Yes: 1, No: 2]
```

3. How should ID3 handle the impure "Sunny" and "Rain" nodes?
4. What is the decision rule for leaf node class assignment in this case?
5. Calculate the entropy of each impure leaf and determine which majority class rule to apply

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
3. **CART Analysis (using Gini impurity)**: Find the best binary split for Size feature using Gini impurity
4. **CART Analysis (using Entropy)**: Find the best binary split for Size feature using entropy-based information gain
5. **Comparison**: Which feature would each algorithm choose as the root? Explain any differences
6. **CART Comparison**: Compare the binary splits chosen by CART using Gini vs CART using Entropy. Are they identical? Explain any differences.

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
2. [📚] Explain how to handle the empty node case mathematically
3. Show that entropy is maximized for balanced distributions
4. [📚] Derive the maximum possible entropy for $k$ classes

**Hint:** Maximum entropy = $\log_2(k)$

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
3. [📚] Prove that entropy is maximized when classes are equally distributed
4. [📚] How should you handle the $\log(0)$ case when calculating entropy?

For a detailed explanation of this question, see [Question 29: Entropy Mathematical Properties](L6_3_29_explanation.md).

## Question 30

### Problem Statement
Analyze stopping criteria across algorithms.

#### Task
1. List three stopping criteria used by ID3
2. [📚] What additional stopping criterion does C4.5 add beyond ID3's criteria?
3. Name two stopping criteria specific to CART
4. For a node with $5$ samples ($3$ positive, $2$ negative), should ID3 continue splitting? Consider minimum samples and purity thresholds

For a detailed explanation of this question, see [Question 30: Algorithm Stopping Criteria](L6_3_30_explanation.md).

## Question 31

### Problem Statement
Compare CART algorithm performance using different impurity measures: Gini impurity vs Entropy.

#### Task
1. **Dataset Analysis**: Given a binary classification dataset with features A, B, C and target Y:
   - Feature A: 3 values, splits data into $[8,2]$, $[5,5]$, $[2,8]$
   - Feature B: 2 values, splits data into $[10,5]$, $[5,10]$
   - Feature C: 4 values, splits data into $[4,1]$, $[3,2]$, $[2,3]$, $[4,4]$

2. **CART with Gini Impurity**: Calculate Gini impurity for each possible binary split and find the optimal split
3. **CART with Entropy**: Calculate entropy-based information gain for each possible binary split and find the optimal split
4. **Comparison**: Are the optimal splits identical? If not, explain why they differ
5. **Theoretical Analysis**: When would Gini impurity and entropy produce different optimal splits?
6. **Practical Implications**: Which impurity measure would you recommend for this dataset and why?

For a detailed explanation of this question, see [Question 31: CART Gini vs Entropy Comparison](L6_3_31_explanation.md).

## Question 32

### Problem Statement
Consider CART's cost function approach to optimization.

#### Task
1. Write the cost function that CART (using Gini impurity) minimizes when choosing splits:
   $$\text{Cost}(T) = \sum_{\text{leaves}} N_t \cdot \text{Impurity}(t) + \alpha \cdot |\text{leaves}|$$
2. For a categorical feature "Color" with values $\{Red, Blue, Green, Yellow\}$, list all possible binary splits
3. Given class distributions: Red$(2,1)$, Blue$(1,2)$, Green$(3,0)$, Yellow$(1,1)$, find the optimal binary split using Gini impurity (CART's default criterion)
4. **CART with Entropy**: Find the optimal binary split using entropy-based information gain
5. **Comparison**: Are the optimal splits identical? If not, explain why they differ
6. **Cost Function Analysis**: How would the cost function change if we used entropy instead of Gini impurity?

For a detailed explanation of this question, see [Question 32: CART Cost Function](L6_3_32_explanation.md).

## Question 33

### Problem Statement
Analyze multi-way vs binary splits as a fundamental difference between algorithms.

#### Task
1. For feature "Grade" with values $\{A, B, C, D, F\}$, show all splits that:
   - ID3 would consider (1 multi-way split)
   - CART (using Gini impurity) would consider (list all binary combinations)
2. Calculate the number of binary splits for a categorical feature with $k$ values
3. Discuss advantages and disadvantages of each approach
4. When might binary splits be preferred over multi-way splits?

For a detailed explanation of this question, see [Question 33: Multi-way vs Binary Splits](L6_3_33_explanation.md).

## Question 34

### Problem Statement
Design a "Decision Tree Card Game" where you must build the best tree using limited information and strategic choices.

**Game Setup**: You have a deck of "Data Cards" and must build a decision tree by making strategic splitting choices. Each card represents a training sample, and you can only look at one feature at a time before deciding.

**Your Hand of Data Cards**:

| Card | Weather | Temperature | Humidity | Activity |
|------|---------|-------------|----------|----------|
| 1    | Sunny   | Warm        | Low      | Hike     |
| 2    | Sunny   | Cool        | High     | Read     |
| 3    | Rainy   | Cool        | High     | Read     |
| 4    | Cloudy  | Warm        | Low      | Hike     |
| 5    | Rainy   | Warm        | Low      | Read     |
| 6    | Cloudy  | Cool        | Low      | Hike     |

#### Task
1. **Feature Analysis**: Without calculating entropy, rank the three features by how "useful" they appear for predicting Activity. Explain your intuitive reasoning.
2. **Split Strategy**: If you could only use ONE feature to split the data, which would you choose and why? Draw the resulting tree.
3. **Verification**: Now calculate the information gain for your chosen feature to verify your intuition was correct.
4. **Tree Construction**: Build the complete decision tree using ID3 algorithm (show your work for the first two levels).
5. **CART Comparison**: Now build the tree using CART algorithm with BOTH Gini impurity and entropy. Compare the resulting trees - are they identical? Explain any differences.
6. **Creative Challenge**: Design a new data card that would make your tree misclassify. What does this reveal about decision tree limitations?

For a detailed explanation of this question, see [Question 34: Decision Tree Card Game](L6_3_34_explanation.md).

## Question 35

### Problem Statement
Analyze computational complexity across ID3, C4.5, and CART algorithms.

#### Task
1. Derive the time complexity for ID3 given $n$ samples, $m$ features, and average branching factor $b$ (Answer should be in the form $O(...)$)
2. How does C4.5's complexity differ due to continuous feature handling?
3. Analyze CART's complexity (using Gini impurity) considering binary splits and surrogate computation
4. For a dataset with $1000$ samples, $20$ features ($10$ categorical with avg $4$ values, $10$ continuous), estimate relative computation time

For a detailed explanation of this question, see [Question 35: Algorithm Complexity Analysis](L6_3_35_explanation.md).

## Question 36

### Problem Statement
Evaluate whether each of the following statements about advanced decision tree concepts is TRUE or FALSE. Provide a brief justification for each answer.

#### Task
1. CART's surrogate splits (using Gini impurity) help maintain tree performance when primary splitting features are unavailable
2. C4.5's pessimistic error pruning uses validation data to determine which subtrees to remove
3. Information gain can be negative when a split reduces the overall purity of child nodes
4. Decision trees with deeper maximum depth always achieve better training accuracy than shallower trees
5. CART's cost-complexity pruning parameter $\alpha$ (using Gini impurity) controls the trade-off between tree complexity and training error
6. Multi-way splits in ID3 always create more interpretable decision boundaries than binary splits
7. C4.5's gain ratio is always less than or equal to the corresponding information gain value
8. Regression trees use mean squared error reduction as their default splitting criterion
9. Feature bagging in Random Forest reduces correlation between individual trees in the ensemble
10. Decision tree algorithms guarantee finding the globally optimal tree structure

For a detailed explanation of this question, see [Question 36: Advanced Decision Tree Properties](L6_3_36_explanation.md).

## [⭐] Question 37

### Problem Statement
You are tasked with analyzing customer purchase behavior using decision tree algorithms. Create a "Decision Tree Construction Race" where you manually trace through the first split decision for all three algorithms on the provided dataset.

**Dataset: Customer Purchase Behavior**

| Product_Category | Purchase_Amount | Customer_Type | Service_Rating | Buy_Again |
|------------------|-----------------|---------------|----------------|-----------|
| Sports          | \$51-100         | Regular       | Excellent      | Yes       |
| Electronics     | \$200+           | Regular       | Excellent      | Yes       |
| Books           | \$200+           | Regular       | Excellent      | Yes       |
| Books           | \$101-200        | New           | Fair           | No        |
| Electronics     | \$200+           | Premium       | Good           | No        |
| Sports          | \$10-50          | Frequent      | Excellent      | Yes       |
| Clothing        | \$200+           | Premium       | Good           | Yes       |
| Clothing        | \$200+           | Premium       | Good           | Yes       |

#### Task
1. **ID3 approach**: Calculate information gain for each feature and identify the best split
2. **C4.5 approach**: Calculate gain ratio for each feature and compare with ID3's choice
3. **CART approach (using Gini impurity)**: For the Product_Category feature, evaluate all possible binary splits using Gini impurity
4. **CART approach (using Entropy)**: For the Product_Category feature, evaluate all possible binary splits using entropy-based information gain
5. Which feature would each algorithm choose as the root? Explain any differences
6. Compare the results between CART using Gini vs CART using Entropy - are they the same? Why or why not?
7. Draw the first level of the decision tree that each algorithm would construct
8. Analyze the impact of feature encoding on decision tree performance. How would you handle the categorical features (Product_Category, Customer_Type, Service_Rating) and the ordinal feature (Purchase_Amount) to ensure optimal tree construction?

For a detailed explanation of this question, see [Question 37: Customer Purchase Behavior Analysis](L6_3_37_explanation.md).

## Question 38

### Problem Statement
Design a "Split Quality Detective" game where you analyze suspicious splitting decisions.

**Scenario**: You're auditing a decision tree and found these three competing splits for the root node:

**Dataset**: $16$ samples with classes $[+: 10, -: 6]$

**Split A** (Feature: Weather)
- Sunny: $[+: 4, -: 1]$
- Cloudy: $[+: 3, -: 2]$  
- Rainy: $[+: 3, -: 3]$

**Split B** (Feature: Customer_ID)
- ID_001-100: $[+: 2, -: 0]$
- ID_101-200: $[+: 2, -: 0]$
- ID_201-300: $[+: 2, -: 0]$
- ID_301-400: $[+: 2, -: 0]$
- ID_401-500: $[+: 2, -: 6]$

**Split C** (Feature: Purchase_Amount $\leq \$50$)
- $\leq \$50$: $[+: 6, -: 4]$
- $> \$50$: $[+: 4, -: 2]$

#### Task
1. [📚] Calculate information gain for each split
2. [📚] Calculate gain ratio for Split A and Split B
3. Which split would each algorithm (ID3, C4.5, CART using Gini impurity) prefer? Explain your reasoning
4. Which split would CART using entropy prefer? Is it the same as CART using Gini impurity?
5. Compare the preferences of all algorithms. Which ones agree and which ones disagree? Why?
6. [📚] Identify which split shows signs of overfitting and explain why
7. [📚] What makes Split B problematic for real-world deployment?
8. [📚] Given the analysis, which split should actually be chosen for production deployment and why?

For a detailed explanation of this question, see [Question 38: Split Quality Analysis](L6_3_38_explanation.md).

## Question 39

### Problem Statement
Create a "Tree Surgery" simulation where you practice pruning decisions.

**Given Tree Structure**:

```
Root: Age ≤ 30 (Training Acc: 85%, Validation Acc: 78%)
├── Left: Income ≤ $40K (Training Acc: 90%, Validation Acc: 72%)
│   ├── Low Risk (Leaf): [Safe: 8, Risk: 1]
│   └── Medium Risk (Leaf): [Safe: 3, Risk: 4]
└── Right: Experience > 2 years (Training Acc: 88%, Validation Acc: 81%)
    ├── High Risk (Leaf): [Safe: 2, Risk: 6]
    └── Safe (Leaf): [Safe: 7, Risk: 1]
```

**Validation Performance**:
- Full tree: $75\%$ accuracy
- Pruning left subtree: $79\%$ accuracy  
- Pruning right subtree: $71\%$ accuracy
- Pruning both subtrees (root only): $73\%$ accuracy

#### Task
1. Calculate the training accuracy for each subtree and the full tree
2. Which pruning decision would **reduced error pruning** make? Explain
3. If using **cost-complexity pruning** with $\alpha = 0.1$, calculate the cost-complexity for:
   - Full tree
   - Tree with left subtree pruned
4. What does the validation performance pattern suggest about overfitting?
5. Write the final decision rule after optimal pruning
6. Compute the misclassification cost for each pruning option if $Safe=0$ cost, $Risk=10$ cost

For a detailed explanation of this question, see [Question 39: Tree Pruning Simulation](L6_3_39_explanation.md).

## Question 40

### Problem Statement
You are teaching decision trees to a friend using only pen and paper. Draw and explain how each algorithm would structure its first split differently.

#### Task
Given this simple dataset about movie preferences:

| Person | Age_Group | Genre_Preference | Has_Netflix | Likes_Movie |
|--------|-----------|------------------|-------------|-------------|
| Alice  | Young     | Action          | Yes         | Yes         |
| Bob    | Young     | Comedy          | No          | No          |
| Carol  | Old       | Drama           | Yes         | Yes         |
| David  | Old       | Action          | No          | Yes         |
| Eve    | Young     | Drama           | Yes         | No          |
| Frank  | Old       | Comedy          | Yes         | Yes         |

1. **Visual Design**: Draw four simple tree diagrams showing how ID3, C4.5, CART (using Gini impurity), and CART (using Entropy) would approach the first split. Use boxes for nodes and arrows for branches.

2. **Algorithm Personality**: If these algorithms were people, describe their "splitting personality" in one sentence each:
   - ID3's personality: ________
   - C4.5's personality: ________  
   - CART (Gini) personality: ________
   - CART (Entropy) personality: ________

3. **Teaching Moment**: Without any calculations, explain to your friend why CART (using Gini impurity) might split "Age_Group" into {Young} vs {Old} while ID3 might split it into {Young, Old} branches.
4. **CART Comparison**: Would CART using entropy make the same split as CART using Gini impurity for "Age_Group"? Explain why or why not.

4. **Quick Check**: Circle the correct answer: If a feature has 10 possible values, how many splits would CART consider?
   - A) 1 split (binary only)
   - B) 10 splits (one per value)  
   - C) $511$ splits ($2^9 - 1$)
   - D) $1023$ splits ($2^{10} - 1$)

For a detailed explanation of this question, see [Question 40: Visual Tree Construction](L6_3_40_explanation.md).

## Question 41

### Problem Statement
You run a consulting company that recommends the best decision tree algorithm for different clients. Match each client scenario with the most suitable algorithm using only logical reasoning.

#### Task

**Your Client Portfolio:**

**Client A - Elementary School Teacher**: 
- Wants to teach kids how computers make decisions
- Needs the simplest possible explanation
- Dataset: $20$ students, $3$ features (all yes/no), no missing data
- Priority: Maximum interpretability

**Client B - Medical Researcher**:
- Studying patient diagnosis patterns  
- Has mixed data types and some missing patient records
- Needs to handle uncertainty gracefully
- Priority: Robust handling of real-world messiness

**Client C - Tech Startup**:
- Building a recommendation system
- Large dataset with user IDs, timestamps, and continuous ratings
- Needs both classification AND regression capabilities
- Priority: Versatility and performance

**Client D - Insurance Company**:
- Analyzing risk factors with many categorical variables
- Some features have $50+$ categories (like zip codes)
- Worried about overfitting to irrelevant details
- Priority: Avoiding bias toward high-cardinality features

1. **Matching Game**: Connect each client (A, B, C, D) with their ideal algorithm (ID3, C4.5, CART using Gini impurity, CART using Entropy). Draw lines or write pairs.
2. **Consultation Notes**: For each match, write a one-sentence business justification that you'd tell the client.
3. **CART Comparison**: For clients who could use either CART approach, explain when you'd recommend Gini impurity vs entropy and why.
4. **The Plot Twist**: Client D mentions they also need to predict continuous insurance claim amounts. Does this change your recommendation? Explain.
5. **Elevator Pitch**: You have 30 seconds in an elevator to explain to a non-technical CEO why different algorithms exist. Write your pitch (2-3 sentences max).

For a detailed explanation of this question, see [Question 41: Algorithm Matchmaker](L6_3_41_explanation.md).

## Question 42

### Problem Statement
Deep dive into CART algorithm: Compare and contrast Gini impurity vs Entropy as splitting criteria.

#### Task
1. **Mathematical Foundation**: 
   - Write the formula for Gini impurity: $Gini(p) = 1 - \sum_{i=1}^{k} p_i^2$
   - Write the formula for Entropy: $H(p) = -\sum_{i=1}^{k} p_i \log_2(p_i)$
   - For a binary classification problem with class probabilities $[p, 1-p]$, show that both measures reach their maximum at $p = 0.5$

2. **Numerical Comparison**: Given a dataset split into two groups:
   - Group 1: $[8 positive, 2 negative]$
   - Group 2: $[3 positive, 7 negative]$
   Calculate both Gini impurity and Entropy for each group and for the overall split.

3. **Binary Split Analysis**: For a categorical feature "Size" with values {Small, Medium, Large} and target distribution:
   - Small: $[2, 8]$ (2 positive, 8 negative)
   - Medium: $[5, 5]$ (5 positive, 5 negative)  
   - Large: $[8, 2]$ (8 positive, 2 negative)
   
   Find the optimal binary split using:
   - CART with Gini impurity
   - CART with Entropy
   
4. **Theoretical Differences**: 
   - When would Gini impurity and entropy produce different optimal splits?
   - What are the computational advantages of Gini impurity over entropy?
   - In what scenarios might entropy be preferred over Gini impurity?

5. **Practical Implications**: 
   - For a real-world dataset with 1000 samples and 20 features, which impurity measure would you choose and why?
   - How would your choice affect training time, tree interpretability, and final performance?

For a detailed explanation of this question, see [Question 42: CART Impurity Measures Deep Dive](L6_3_42_explanation.md).

## Question 43

### Problem Statement
You are a "Decision Tree Detective" investigating a mysterious dataset about student study habits. Your mission is to solve the case using only ID3 and C4.5 algorithms to see how they approach the same evidence differently.

**Dataset: Student Study Habits Mystery**

| Study_Time | Study_Location | Coffee_Consumption | Exam_Result |
|------------|----------------|-------------------|-------------|
| Short      | Library        | None              | Fail        |
| Short      | Home           | High              | Fail        |
| Short      | Cafe           | None              | Fail        |
| Short      | Dorm           | High              | Fail        |
| Long       | Office         | None              | Pass        |
| Long       | Park           | High              | Pass        |
| Long       | Lab            | None              | Pass        |
| Long       | Study_Room     | High              | Fail        |

#### Task
1. [📚] Calculate information gain for each feature using ID3's approach. Which feature would ID3 choose as the root node?
2. [📚] Calculate gain ratio for each feature using C4.5's approach. Which feature would C4.5 choose as the root node?
3. [📚] Draw the first level of the decision tree that each algorithm would construct. Use boxes for nodes and arrows for branches.
4. Create a simple table showing:
   - Feature name
   - Information gain (ID3)
   - Split information
   - Gain ratio (C4.5)
   - Which algorithm prefers it
5. [📚] If you were a student trying to maximize your chances of passing, which algorithm's advice would you follow and why?
6. [📚] Design a new student record that would make both algorithms agree on the root split. What does this reveal about the fundamental differences between ID3 and C4.5?

For a detailed explanation of this question, see [Question 43: Student Study Habits Mystery](L6_3_43_explanation.md).

## [⭐] Question 44

### Problem Statement
Using the following dataset, we want to construct a decision tree that classifies Y without any error on the training set:

| A | B | C | Y |
|---|---|---|---|
| F | F | F | F |
| T | F | T | T |
| T | T | F | T |
| T | T | T | F |

#### Task
1. Calculate the entropy of the entire dataset and explain what this value tells you about the classification difficulty
2. Calculate information gain for each feature (A, B, C) and identify the optimal root split. Show your calculations step-by-step
3. Draw the complete decision tree structure that achieves zero training error. Label each node with its feature and each leaf with its class
4. Is your tree optimal? Can you construct another tree with less height that achieves zero error? Prove your answer mathematically
5. What is the minimum possible depth for a decision tree that perfectly classifies this dataset? Justify your answer
6. How would ID3, C4.5, and CART (using both Gini impurity and entropy) approach this dataset differently? Which would produce the most interpretable tree?

For a detailed explanation of this question, see [Question 44: Decision Tree Construction and Optimization](L6_3_44_explanation.md).

## [⭐] Question 45

### Problem Statement
Based on the following table, answer the questions:

| Weight | Eye Color | Num. Eyes | Output |
|--------|-----------|-----------|--------|
| N      | A         | 2         | L      |
| N      | V         | 2         | L      |
| N      | V         | 2         | L      |
| U      | V         | 3         | L      |
| U      | V         | 3         | L      |
| U      | A         | 4         | D      |
| N      | A         | 4         | D      |
| N      | V         | 4         | D      |
| U      | A         | 3         | D      |
| U      | A         | 3         | D      |

### Task
1. What is the conditional entropy of $H(\text{eye color}|\text{weight}=N)$?
2. What attribute would the ID3 algorithm select as root of the tree?
3. Draw the full decision tree learned from these data
4. What is the training set error of this tree?

For a detailed explanation of this question, see [Question 45: Decision Tree Construction with Conditional Entropy](L6_2_45_explanation.md).

## [⭐] Question 46

### Problem Statement
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

#### Task
1. The maximum depth of the decision tree must be less than $m+1$. 
2. Suppose data has $R$ records, the maximum depth of the decision tree must be less than $1 + \log_2 R$
3. Suppose one of the attributes has $R$ distinct values, and it has a unique value in each record. Then the decision tree will certainly have depth $0$ or $1$ (i.e. will be a single node, or else a root node directly connected to a set of leaves)
4. If a decision tree has depth $d$, then the number of internal nodes is at most $2^d - 1$.
5. For a dataset with $n$ samples and $m$ features, if each feature has at most $k$ distinct values, then the maximum possible number of leaf nodes in any decision tree is $\min(n, k^m)$.
6. In a perfectly balanced binary decision tree with $n$ leaf nodes, the tree depth is always exactly $\lceil \log_2 n \rceil$, regardless of the splitting strategy used during construction.

For a detailed explanation of this question, see [Question 46: Decision Tree Depth Analysis](L6_3_46_explanation.md).

## Question 47

### Problem Statement
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

#### Task

1. In C4.5, the gain ratio can exceed the information gain when the split information is less than 1.0, indicating that gain ratio provides a "boost" to splits with fewer branches.
2. CART's binary splitting strategy always produces trees with better generalization than ID3's multi-way splits because binary decisions are inherently more robust to noise in the data.
3. When ID3 encounters a tie in information gain between multiple features, it should randomly select one of the tied features to ensure unbiased tree construction.
4. C4.5's handling of continuous features requires sorting the feature values, which increases the time complexity from O(n) to O(n log n) for each continuous feature evaluation.
5. CART's surrogate splits are primarily designed to handle missing values during tree construction, not during prediction on new data.
6. ID3's recursive partitioning guarantees that each split reduces the overall dataset entropy, making it impossible for the algorithm to create a split that increases entropy at any node.
7. Since CART uses only binary splits, it cannot represent the same decision boundaries as ID3's multi-way splits, meaning some classification problems that ID3 can solve perfectly cannot be solved by CART with the same accuracy.
8. C4.5's gain ratio normalization by split information can occasionally prefer features with moderate information gain over features with high information gain, especially when dealing with high-cardinality categorical features.
9. In CART, when using Gini impurity vs entropy for the same dataset, the choice of impurity measure can affect which features are selected for splitting, but the final tree performance is typically very similar.
10. The theoretical maximum depth of an ID3 decision tree is bounded by the number of features when each feature is used at most once per path from root to leaf.

For a detailed explanation of this question, see [Question 47: Algorithm Properties and Behavior](L6_3_47_explanation.md).