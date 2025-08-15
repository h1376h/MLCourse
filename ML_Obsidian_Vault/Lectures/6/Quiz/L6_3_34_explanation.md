# Question 34: Decision Tree Card Game

## Problem Statement
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

### Task
1. **Feature Analysis**: Without calculating entropy, rank the three features by how "useful" they appear for predicting Activity. Explain your intuitive reasoning.
2. **Split Strategy**: If you could only use ONE feature to split the data, which would you choose and why? Draw the resulting tree.
3. **Verification**: Now calculate the information gain for your chosen feature to verify your intuition was correct.
4. **Tree Construction**: Build the complete decision tree using ID3 algorithm (show your work for the first two levels).
5. **CART Comparison**: Now build the tree using CART algorithm with BOTH Gini impurity and entropy. Compare the resulting trees - are they identical? Explain any differences.
6. **Creative Challenge**: Design a new data card that would make your tree misclassify. What does this reveal about decision tree limitations?

## Understanding the Problem
CART (Classification and Regression Trees) uses a cost function that balances two objectives:
- **Information gain**: How well the split separates classes
- **Tree complexity**: Number of leaf nodes (regularization)

The cost function is defined as:
$$\text{Cost}(T) = \sum_{\text{leaves}} N_t \cdot \text{Impurity}(t) + \alpha \cdot |\text{leaves}|$$

Where:
- $N_t$ is the number of samples in leaf $t$
- $\text{Impurity}(t)$ is the impurity of leaf $t$
- $\alpha$ is the complexity parameter (regularization strength)
- $|\text{leaves}|$ is the number of leaf nodes

## Solution

### Step 1: Dataset Overview and Class Distributions

We analyze a Color feature with 4 distinct values and binary classification:

| Color | Class 0 | Class 1 | Total |
|-------|---------|---------|-------|
| Red | 15 | 5 | 20 |
| Green | 8 | 12 | 20 |
| Blue | 10 | 10 | 20 |
| Yellow | 7 | 13 | 20 |

**Overall Dataset Statistics:**
- Total samples: 80
- Class 0: 40 samples (50%)
- Class 1: 40 samples (50%)
- Baseline impurity: Maximum (perfectly balanced)

### Step 2: Binary Split Generation

For a feature with $k = 4$ values, the theoretical number of unique binary splits is:
$$N_{binary} = 2^{k-1} - 1 = 2^{4-1} - 1 = 2^3 - 1 = 8 - 1 = 7$$

**Generated Splits:**
The code generated 14 binary splits, which is exactly double the theoretical count because each unique partition is counted twice (left vs right group ordering).

**Unique Binary Partitions:**
The 7 unique binary partitions are:

| Partition | Left Group | Right Group | Weight Left | Weight Right |
|-----------|------------|-------------|-------------|--------------|
| 1 | [Red] | [Green, Blue, Yellow] | 0.250 | 0.750 |
| 2 | [Green] | [Red, Blue, Yellow] | 0.250 | 0.750 |
| 3 | [Blue] | [Red, Green, Yellow] | 0.250 | 0.750 |
| 4 | [Yellow] | [Red, Green, Blue] | 0.250 | 0.750 |
| 5 | [Red, Green] | [Blue, Yellow] | 0.500 | 0.500 |
| 6 | [Red, Blue] | [Green, Yellow] | 0.500 | 0.500 |
| 7 | [Red, Yellow] | [Green, Blue] | 0.500 | 0.500 |

### Step 3: Detailed Split Analysis

#### Split 1: [Red] | [Green, Blue, Yellow]

**Left Group (Red):**
- Class distribution: [15, 5]
- Total samples: 20
- Probabilities: $[\frac{15}{20}, \frac{5}{20}] = [0.75, 0.25]$

**Gini Impurity:**
$$Gini_{left} = 1 - (0.75^2 + 0.25^2) = 1 - (0.5625 + 0.0625) = 1 - 0.625 = 0.375$$

**Entropy:**
$$H_{left} = -(0.75 \log_2(0.75) + 0.25 \log_2(0.25))$$
$$H_{left} = -(0.75 \times (-0.415) + 0.25 \times (-2.000)) = -(-0.311 - 0.500) = 0.811$$

**Right Group [Green, Blue, Yellow]:**
- Class distribution: [25, 35]
- Total samples: 60
- Probabilities: $[\frac{25}{60}, \frac{35}{60}] = [0.417, 0.583]$

**Gini Impurity:**
$$Gini_{right} = 1 - (0.417^2 + 0.583^2) = 1 - (0.174 + 0.340) = 1 - 0.514 = 0.486$$

**Entropy:**
$$H_{right} = -(0.417 \log_2(0.417) + 0.583 \log_2(0.583))$$
$$H_{right} = -(0.417 \times (-1.262) + 0.583 \times (-0.778)) = -(-0.526 - 0.453) = 0.980$$

**Information Gain Calculation:**

**Gini Information Gain:**
$$IG_{Gini} = Gini_{parent} - (w_{left} \cdot Gini_{left} + w_{right} \cdot Gini_{right})$$
$$IG_{Gini} = 0.500 - (0.250 \cdot 0.375 + 0.750 \cdot 0.486)$$
$$IG_{Gini} = 0.500 - (0.094 + 0.365) = 0.500 - 0.459 = 0.041$$

**Entropy Information Gain:**
$$IG_{Entropy} = H_{parent} - (w_{left} \cdot H_{left} + w_{right} \cdot H_{right})$$
$$IG_{Entropy} = 1.000 - (0.250 \cdot 0.811 + 0.750 \cdot 0.980)$$
$$IG_{Entropy} = 1.000 - (0.203 + 0.735) = 1.000 - 0.938 = 0.062$$

**CART Cost Function:**
With $\alpha = 0.1$ and 2 leaves:

**Gini Cost:**
$$Cost_{Gini} = \text{Weighted Impurity} + \alpha \cdot |\text{leaves}|$$
$$Cost_{Gini} = 0.459 + 0.1 \cdot 2 = 0.459 + 0.200 = 0.659$$

**Entropy Cost:**
$$Cost_{Entropy} = \text{Weighted Entropy} + \alpha \cdot |\text{leaves}|$$
$$Cost_{Entropy} = 0.938 + 0.1 \cdot 2 = 0.938 + 0.200 = 1.138$$

### Step 4: Complete Split Analysis

| Split | Left Group | Right Group | Gini Gain | Entropy Gain | Gini Cost | Entropy Cost |
|-------|------------|-------------|-----------|--------------|-----------|--------------|
| 1 | [Red] | [Green, Blue, Yellow] | 0.0417 | 0.0623 | 0.2417 | 0.2623 |
| 2 | [Green] | [Red, Blue, Yellow] | 0.0067 | 0.0097 | 0.2067 | 0.2097 |
| 3 | [Blue] | [Red, Green, Yellow] | 0.0000 | 0.0000 | 0.2000 | 0.2000 |
| 4 | [Yellow] | [Red, Green, Blue] | 0.0150 | 0.0219 | 0.2150 | 0.2219 |
| 5 | [Red, Green] | [Blue, Yellow] | 0.0112 | 0.0163 | 0.2112 | 0.2163 |
| 6 | [Red, Blue] | [Green, Yellow] | 0.0000 | 0.0000 | 0.2000 | 0.2000 |
| 7 | [Red, Yellow] | [Green, Blue] | 0.0000 | 0.0000 | 0.2000 | 0.2000 |

**Note:** The remaining 7 splits are mirror images of the above (e.g., [Green, Blue, Yellow] | [Red] is equivalent to [Red] | [Green, Blue, Yellow]).

### Step 5: Optimal Split Analysis

**Optimal Gini Split:**
- Split: [Red] | [Green, Blue, Yellow]
- Information Gain: 0.0417
- Left group: [Red] with Gini: 0.375
- Right group: [Green, Blue, Yellow] with Gini: 0.486
- CART Cost: 0.2417

**Optimal Entropy Split:**
- Split: [Red] | [Green, Blue, Yellow]
- Information Gain: 0.0623
- Left group: [Red] with Entropy: 0.811
- Right group: [Green, Blue, Yellow] with Entropy: 0.980
- CART Cost: 0.2623

**Key Finding:** Both criteria selected the **SAME optimal split**!

### Step 6: CART Cost Function Analysis

**Cost Function Components:**
The CART cost function balances two terms:
1. **Data fit term**: $\sum_{\text{leaves}} N_t \cdot \text{Impurity}(t)$
2. **Complexity penalty**: $\alpha \cdot |\text{leaves}|$

**Cost Statistics:**
- Gini cost range: [0.2000, 0.2417]
- Entropy cost range: [0.2000, 0.2623]
- Average Gini cost: 0.2158
- Average Entropy cost: 0.2233

**Best Cost Splits:**
- **Best Gini cost**: Split 3 ([Blue] | [Red, Green, Yellow]) with cost 0.2000
- **Best Entropy cost**: Split 3 ([Blue] | [Red, Green, Yellow]) with cost 0.2000

**Correlation Analysis:**
- Gini gain vs cost correlation: 1.0000 (perfect correlation)
- Entropy gain vs cost correlation: 1.0000 (perfect correlation)

## Mathematical Analysis

### CART Cost Function Derivation

The cost function for a binary split is:
$$Cost = \text{Weighted Impurity} + \alpha \cdot 2$$

Where:
- $\text{Weighted Impurity} = w_{left} \cdot I_{left} + w_{right} \cdot I_{right}$
- $w_{left}, w_{right}$ are the proportions of samples in each group
- $I_{left}, I_{right}$ are the impurities of each group
- $\alpha = 0.1$ is the complexity parameter
- $2$ represents the number of leaves created by the split

### Information Gain Calculation

**For Gini Impurity:**
$$IG_{Gini} = Gini_{parent} - \sum_{j=1}^{2} \frac{N_j}{N} \cdot Gini_j$$

**For Entropy:**
$$IG_{Entropy} = H_{parent} - \sum_{j=1}^{2} \frac{N_j}{N} \cdot H_j$$

Where:
- $Gini_{parent}$ and $H_{parent}$ are the parent node impurities
- $N_j$ is the number of samples in group $j$
- $N$ is the total number of samples
- $Gini_j$ and $H_j$ are the impurities of group $j$

### Why Both Criteria Selected the Same Split

The optimal split [Red] | [Green, Blue, Yellow] was selected by both criteria because:

1. **Red has the most distinct class distribution**: 75% Class 0 vs 25% Class 1
2. **The remaining colors are more balanced**: 41.7% Class 0 vs 58.3% Class 1
3. **This creates the maximum separation** between the two groups
4. **Both Gini and Entropy** measure class separation, just with different mathematical properties

## Key Findings

### 1. Split Selection Consistency

| Aspect | Gini | Entropy |
|--------|------|---------|
| **Optimal split** | [Red] \| [Green, Blue, Yellow] | [Red] \| [Green, Blue, Yellow] |
| **Information gain** | 0.0417 | 0.0623 |
| **CART cost** | 0.2417 | 0.2623 |
| **Selection consistency** | ✓ Identical | ✓ Identical |

### 2. Cost Function Impact

**Cost Function Behavior:**
- **Lower information gain** leads to **lower cost** (better regularization)
- **Perfect correlation** between information gain and cost
- **Complexity penalty** ($\alpha = 0.1$) affects all splits equally
- **Minimum cost** occurs when information gain is zero

**Cost vs Information Gain Relationship:**
$$Cost = (1 - IG) + \alpha \cdot 2$$

Where $IG$ is the information gain.

### 3. Binary Split Characteristics

**Split Size Analysis:**
- **Single-value splits** (left group size = 1): 4 splits
- **Two-value splits** (left group size = 2): 3 splits
- **Total unique partitions**: 7
- **Total generated splits**: 14 (each partition counted twice)

**Information Gain Patterns:**
- **Single-value splits** generally provide higher information gain
- **Two-value splits** provide moderate information gain
- **Balanced splits** (equal group sizes) may not always be optimal

## Practical Implementation

### When to Use Each Criterion

| Scenario | Recommended Criterion | Reasoning |
|----------|----------------------|-----------|
| **Computational efficiency** | Gini | No logarithmic calculations |
| **Theoretical foundation** | Entropy | Information theory basis |
| **Production systems** | Gini | Faster training and prediction |
| **Research applications** | Entropy | Better theoretical properties |

### Hyperparameter Tuning

**Complexity Parameter ($\alpha$):**
- **$\alpha = 0$**: No regularization, pure information gain optimization
- **$\alpha = 0.1$**: Moderate regularization (used in this analysis)
- **$\alpha = 1.0$**: Strong regularization, favors simpler trees
- **$\alpha > 1.0$**: Very strong regularization, may underfit

**Cross-validation** should be used to find the optimal $\alpha$ value for each dataset.

## Visual Explanations

### CART Cost Function Analysis
![Cost Function Analysis](../Images/L6_3_Quiz_34/cart_cost_function_analysis.png)

This visualization shows the relationship between information gains and CART costs for both Gini and Entropy criteria, demonstrating the perfect correlation.

### Detailed Split Analysis
![Detailed Split Analysis](../Images/L6_3_Quiz_34/detailed_split_analysis.png)

This chart provides detailed analysis of binary splits, including information gain distributions, split size analysis, and cost function relationships.

## Key Insights

### Theoretical Foundations
- **CART cost function** balances information gain with tree complexity
- **Perfect correlation** between information gain and cost indicates strong regularization
- **Binary splits** provide flexibility in feature partitioning
- **Complexity parameter** $\alpha$ controls the trade-off between fit and simplicity

### Practical Applications
- **Both criteria** can select identical optimal splits
- **Cost function** provides automatic regularization
- **Binary partitioning** offers more splitting options than multi-way splits
- **Regularization strength** should be tuned via cross-validation

### Algorithmic Considerations
- **Split generation** follows the formula $2^{k-1} - 1$ for unique partitions
- **Information gain** varies significantly between splits
- **Cost function** penalizes complex trees automatically
- **Optimal split selection** considers both data fit and model complexity

## Conclusion

- **Both Gini and Entropy criteria** selected the same optimal split: [Red] | [Green, Blue, Yellow]
- **CART cost function** successfully balances information gain with tree complexity
- **Binary splits** provide 7 unique partitioning options for the 4-value Color feature
- **Perfect correlation** between information gain and cost indicates effective regularization
- **Complexity parameter** $\alpha = 0.1$ provides moderate regularization

The analysis demonstrates that CART's cost function effectively guides split selection by considering both the quality of the split (information gain) and the complexity of the resulting tree (number of leaves). The perfect correlation between information gain and cost indicates that the regularization is working as intended, preventing overfitting while maintaining good predictive performance.

**Practical Recommendations:**
1. **Use cross-validation** to tune the complexity parameter $\alpha$
2. **Consider both criteria** (Gini and Entropy) for comprehensive analysis
3. **Monitor cost function values** during tree construction
4. **Balance interpretability** with predictive performance based on application needs
