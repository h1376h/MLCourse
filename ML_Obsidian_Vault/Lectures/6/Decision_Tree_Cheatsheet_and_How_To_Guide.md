# Decision Tree Cheatsheet and "How To" Guide for Pen & Paper Exams

## 📋 Quick Reference Cheatsheet

### Core Decision Tree Formulas

**Entropy (Binary Classification):**
$$H(S) = -p_1 \log_2(p_1) - p_2 \log_2(p_2)$$
where $p_1, p_2$ are class probabilities

**Entropy (Multi-class):**
$$H(S) = -\sum_{i=1}^{k} p_i \log_2(p_i)$$

**Information Gain:**
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

**Gain Ratio:**
$$\text{Gain Ratio}(S, A) = \frac{IG(S, A)}{\text{Split Info}(S, A)}$$

**Split Information:**
$$\text{Split Info}(S, A) = -\sum_{v \in Values(A)} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)$$

**Gini Index:**
$$Gini(S) = 1 - \sum_{i=1}^{k} p_i^2$$

**Classification Error:**
$$Error(S) = 1 - \max_i(p_i)$$

### Key Concepts

**Tree Structure:**
- **Root Node**: Top node of the tree
- **Internal Nodes**: Decision nodes with splits
- **Leaf Nodes**: Terminal nodes with predictions
- **Depth**: Maximum path length from root to leaf
- **Branches**: Paths from root to leaves

**Splitting Criteria:**
- **Information Gain**: Reduction in entropy
- **Gain Ratio**: Normalized information gain
- **Gini Index**: Alternative impurity measure
- **Classification Error**: Simple error rate

**Pruning:**
- **Pre-pruning**: Stop growing before overfitting
- **Post-pruning**: Remove branches after full growth
- **Cost-complexity**: $R_\alpha(T) = R(T) + \alpha|T|$

---

## 🎯 Question Type 1: Tree Structure Analysis

### How to Approach:

**Step 1: Understand Tree Components**
- Count nodes (root, internal, leaf)
- Calculate depth and breadth
- Identify decision rules and paths

**Step 2: Analyze Tree Properties**
- Maximum possible leaf nodes = $\prod_{i} |Values(feature_i)|$
- Maximum depth = number of features
- Number of edges = number of nodes - 1

**Step 3: Trace Decision Paths**
- Follow branches from root to leaf
- Apply decision rules sequentially
- Identify prediction for given input

**Step 4: Calculate Tree Metrics**
- **Depth**: Longest path from root to leaf
- **Size**: Total number of nodes
- **Complexity**: Number of decision rules

**Step 5: Draw Tree Diagram**
- Use clear hierarchical structure
- Label nodes with features and values
- Show decision paths and predictions

### Example Template:
```
Given: Decision tree with features [list]
1. Tree structure:
   - Root: [feature name]
   - Internal nodes: [count] nodes
   - Leaf nodes: [count] nodes
   - Maximum depth: [number]
2. Decision paths:
   - Path 1: [feature1] = [value1] → [feature2] = [value2] → [prediction]
   - Path 2: [feature1] = [value1] → [prediction]
3. For input [values]:
   - Follow path: [description]
   - Prediction: [result]
4. Tree metrics:
   - Total nodes: [number]
   - Edges: [number]
   - Decision rules: [number]
```

---

## 🎯 Question Type 2: Entropy and Information Gain Calculation

### How to Approach:

**Step 1: Calculate Original Entropy**
- Count samples in each class
- Calculate class probabilities: $p_i = \frac{count_i}{total}$
- Apply entropy formula: $H(S) = -\sum p_i \log_2(p_i)$

**Step 2: Analyze Feature Split**
- Create contingency table for feature vs class
- Calculate entropy for each feature value
- Use weighted average: $H(S|A) = \sum \frac{|S_v|}{|S|} H(S_v)$

**Step 3: Calculate Information Gain**
- $IG(S, A) = H(S) - H(S|A)$
- Compare with other features
- Higher IG = better split

**Step 4: Evaluate Split Quality**
- **Good split**: IG > 0.1 (or significant reduction)
- **Poor split**: IG < 0.05 (minimal reduction)
- **Perfect split**: IG = original entropy (pure leaves)

**Step 5: Consider Gain Ratio**
- Calculate split information
- $Gain Ratio = \frac{IG}{Split Info}$
- Use when features have many values

### Example Template:
```
Given: Dataset with [n] samples, classes [list]
1. Original entropy:
   - Class distribution: [counts]
   - Probabilities: p₁ = [value], p₂ = [value]
   - H(S) = -p₁log₂(p₁) - p₂log₂(p₂) = [value]
2. Feature split analysis:
   - Value 1: [counts] → H₁ = [value]
   - Value 2: [counts] → H₂ = [value]
   - Weighted average: H(S|A) = [calculation] = [value]
3. Information gain:
   - IG = H(S) - H(S|A) = [value] - [value] = [value]
4. Evaluation: [good/poor/perfect] split because [reasoning]
```

---

## 🎯 Question Type 3: Decision Tree Algorithm Analysis (ID3, C4.5, CART)

### How to Approach:

**Step 1: Understand Algorithm Differences**
- **ID3**: Uses information gain, categorical features only
- **C4.5**: Uses gain ratio, handles missing values
- **CART**: Uses Gini index, binary splits, handles regression

**Step 2: Analyze Algorithm Steps**
- **ID3**: Calculate IG for all features → choose best → split → recurse
- **C4.5**: Calculate gain ratio → choose best → handle missing values → recurse
- **CART**: Calculate Gini for all possible splits → choose best → recurse

**Step 3: Identify Stopping Criteria**
- **Pure node**: All samples same class
- **No features**: All features used
- **Empty dataset**: No samples remain
- **Minimum samples**: Pre-pruning threshold

**Step 4: Handle Algorithm Limitations**
- **ID3**: No continuous features, overfitting prone
- **C4.5**: Computationally expensive
- **CART**: Binary splits may be less interpretable

**Step 5: Compare Algorithm Choices**
- **Small dataset**: ID3 or C4.5
- **Large dataset**: CART
- **Missing values**: C4.5
- **Regression**: CART only

### Example Template:
```
Given: Dataset with features [list], algorithm [name]
1. Algorithm characteristics:
   - Splitting criterion: [information gain/gain ratio/gini]
   - Feature handling: [categorical/continuous/both]
   - Split type: [multi-way/binary]
2. Algorithm steps:
   - Step 1: [description]
   - Step 2: [description]
   - Step 3: [description]
3. Stopping criteria:
   - [criterion 1]: [condition]
   - [criterion 2]: [condition]
4. Limitations:
   - [limitation 1]: [explanation]
   - [limitation 2]: [explanation]
5. Algorithm choice: [recommendation] because [reasoning]
```

---

## 🎯 Question Type 4: Tree Pruning and Overfitting Analysis

### How to Approach:

**Step 1: Identify Overfitting**
- **Training accuracy**: Increases with depth
- **Validation accuracy**: Peaks then decreases
- **Overfitting point**: Where validation starts declining
- **Optimal depth**: Peak validation performance

**Step 2: Analyze Bias-Variance Trade-off**
- **Low depth**: High bias (underfitting), low variance
- **Medium depth**: Balanced bias and variance
- **High depth**: Low bias, high variance (overfitting)

**Step 3: Apply Pre-pruning Techniques**
- **Minimum samples per leaf**: Prevent tiny leaves
- **Maximum depth**: Limit tree complexity
- **Minimum impurity decrease**: Stop if split doesn't help
- **Maximum leaf nodes**: Control tree size

**Step 4: Apply Post-pruning Methods**
- **Reduced error pruning**: Remove branches that don't improve validation
- **Cost-complexity pruning**: $R_\alpha(T) = R(T) + \alpha|T|$
- **Cross-validation**: Choose optimal $\alpha$

**Step 5: Calculate Pruning Costs**
- **Error cost**: $R(T) = \sum \text{error rate} \times \text{node weight}$
- **Complexity cost**: $\alpha|T|$ where $|T|$ = number of nodes
- **Total cost**: $R_\alpha(T) = R(T) + \alpha|T|$

### Example Template:
```
Given: Tree performance data across depths
1. Overfitting analysis:
   - Training accuracy: [increases/decreases] with depth
   - Validation accuracy: peaks at depth [number]
   - Overfitting begins at depth [number]
2. Bias-variance trade-off:
   - Low depth: [high/low] bias, [high/low] variance
   - Optimal depth: [number] (balanced)
   - High depth: [high/low] bias, [high/low] variance
3. Pruning strategy:
   - Pre-pruning: [technique] with threshold [value]
   - Post-pruning: [method] with α = [value]
4. Cost analysis:
   - Error cost: R(T) = [value]
   - Complexity cost: α|T| = [value]
   - Total cost: R_α(T) = [value]
5. Optimal tree: depth [number] with [number] nodes
```

---

## 🎯 Question Type 5: Impurity Measures Comparison

### How to Approach:

**Step 1: Calculate Different Impurity Measures**
- **Entropy**: $H(S) = -\sum p_i \log_2(p_i)$
- **Gini Index**: $Gini(S) = 1 - \sum p_i^2$
- **Classification Error**: $Error(S) = 1 - \max_i(p_i)$

**Step 2: Compare Properties**
- **Range**: Entropy [0, log₂(k)], Gini [0, 1-1/k], Error [0, 1-1/k]
- **Sensitivity**: Entropy most sensitive, Error least sensitive
- **Computation**: Error simplest, Entropy most complex

**Step 3: Analyze Behavior**
- **Pure node**: All measures = 0
- **Maximum impurity**: All measures at maximum
- **Sensitivity to small changes**: Entropy > Gini > Error

**Step 4: Choose Appropriate Measure**
- **Classification**: Gini or Entropy
- **Interpretability**: Error (simple)
- **Sensitivity**: Entropy (fine-grained)

**Step 5: Apply to Feature Selection**
- Calculate impurity reduction for each feature
- Choose feature with maximum reduction
- Consider computational efficiency

### Example Template:
```
Given: Dataset with class distribution [counts]
1. Impurity calculations:
   - Probabilities: p₁ = [value], p₂ = [value]
   - Entropy: H(S) = [calculation] = [value]
   - Gini: Gini(S) = [calculation] = [value]
   - Error: Error(S) = [calculation] = [value]
2. Property comparison:
   - Range: Entropy [range], Gini [range], Error [range]
   - Sensitivity: [ranking from most to least]
   - Computation: [ranking from simplest to most complex]
3. Feature selection:
   - Feature A: IG = [value], Gini reduction = [value]
   - Feature B: IG = [value], Gini reduction = [value]
   - Best feature: [name] using [criterion]
4. Measure choice: [recommendation] because [reasoning]
```

---

## 🎯 Question Type 6: Missing Values and Special Cases

### How to Approach:

**Step 1: Identify Missing Value Patterns**
- **Missing at random**: No pattern in missingness
- **Missing not at random**: Pattern in missingness
- **Missing completely at random**: Independent of all variables

**Step 2: Apply Missing Value Strategies**
- **Ignore**: Remove samples with missing values
- **Imputation**: Fill with mean, median, or mode
- **Surrogate splits**: Use alternative features
- **Fractional instances**: Distribute sample across branches

**Step 3: Handle Special Cases**
- **Pure nodes**: All samples same class
- **Empty nodes**: No samples after split
- **Tie-breaking**: Multiple features with same IG
- **Continuous features**: Find optimal split point

**Step 4: Evaluate Impact**
- **Data loss**: How much data is discarded
- **Bias**: How imputation affects results
- **Performance**: Impact on accuracy

**Step 5: Choose Best Strategy**
- **Small dataset**: Surrogate splits
- **Large dataset**: Imputation
- **Critical applications**: Ignore with validation

### Example Template:
```
Given: Dataset with missing values in [feature]
1. Missing value analysis:
   - Pattern: [random/not random/completely random]
   - Percentage missing: [value]%
   - Impact: [description]
2. Strategy options:
   - Ignore: [number] samples lost
   - Imputation: [method] with value [value]
   - Surrogate splits: [alternative feature]
3. Implementation:
   - Chosen strategy: [method]
   - Modified dataset: [description]
   - Tree construction: [changes needed]
4. Evaluation:
   - Data loss: [percentage]
   - Bias introduced: [description]
   - Performance impact: [better/worse/same]
5. Recommendation: [strategy] because [reasoning]
```

---

## 🎯 Question Type 7: Cost-Sensitive Learning and Business Applications

### How to Approach:

**Step 1: Define Cost Matrix**
- **False Positive cost**: Cost of incorrect positive prediction
- **False Negative cost**: Cost of incorrect negative prediction
- **True Positive/True Negative**: Usually cost = 0

**Step 2: Calculate Expected Costs**
- **Expected cost**: $\sum_{i,j} P(i,j) \times Cost(i,j)$
- **Cost per prediction**: Average cost across all predictions
- **Total cost**: Cost per prediction × number of predictions

**Step 3: Optimize for Cost**
- **Threshold adjustment**: Change decision threshold
- **Class weights**: Weight classes by cost
- **Cost-sensitive splitting**: Use cost in split criterion

**Step 4: Analyze Business Impact**
- **Revenue impact**: How costs affect business metrics
- **Risk assessment**: Probability of high-cost errors
- **ROI calculation**: Return on investment in model

**Step 5: Make Recommendations**
- **Optimal threshold**: Minimize expected cost
- **Model selection**: Choose model with lowest cost
- **Monitoring**: Track cost performance over time

### Example Template:
```
Given: Cost matrix with FP cost = [value], FN cost = [value]
1. Cost analysis:
   - False Positive cost: [value] per error
   - False Negative cost: [value] per error
   - Cost ratio: FN/FP = [value]
2. Expected costs:
   - Current model: [cost calculation] = [value]
   - Alternative model: [cost calculation] = [value]
   - Cost difference: [value]
3. Optimization:
   - Optimal threshold: [value] (minimizes expected cost)
   - Class weights: [positive weight], [negative weight]
   - Modified splitting: [criterion with cost]
4. Business impact:
   - Annual cost: [value] × [predictions] = [value]
   - Cost savings: [value] with optimization
   - ROI: [percentage] improvement
5. Recommendation: [action] because [business justification]
```

---

## 🎯 Question Type 8: Cross-Validation and Model Selection

### How to Approach:

**Step 1: Choose Validation Strategy**
- **K-fold CV**: Divide data into K equal parts
- **Stratified CV**: Maintain class distribution in folds
- **Leave-one-out**: K = number of samples
- **Hold-out**: Single train/validation split

**Step 2: Design Validation Experiment**
- **Parameter grid**: Range of values to test
- **Evaluation metric**: Accuracy, cost, F1-score
- **Number of folds**: Typically 5 or 10
- **Randomization**: Shuffle data before splitting

**Step 3: Analyze Results**
- **Mean performance**: Average across folds
- **Standard deviation**: Measure of stability
- **Confidence intervals**: Statistical significance
- **Best parameters**: Highest mean performance

**Step 4: Handle Bias and Variance**
- **Selection bias**: Performance overestimated
- **Variance**: High standard deviation
- **Overfitting**: Training >> validation performance
- **Underfitting**: Both training and validation low

**Step 5: Make Final Selection**
- **Best model**: Highest validation performance
- **Robust model**: Low variance across folds
- **Practical model**: Balance of performance and complexity

### Example Template:
```
Given: Dataset with [n] samples, [k]-fold cross-validation
1. Validation setup:
   - Folds: [k] folds with [n/k] samples each
   - Parameter grid: [list of values]
   - Evaluation metric: [metric name]
2. Results analysis:
   - Mean performance: [value] ± [standard deviation]
   - Best parameters: [parameter values]
   - Performance range: [min] to [max]
3. Bias-variance analysis:
   - Selection bias: [high/medium/low]
   - Variance: [high/medium/low] (std = [value])
   - Overfitting: [yes/no] (train = [value], val = [value])
4. Model selection:
   - Best model: [parameter values] with performance [value]
   - Robust model: [parameter values] with std [value]
   - Final choice: [recommendation] because [reasoning]
5. Confidence: [percentage] confidence in selection
```

---

## 🎯 Question Type 9: Decision Tree Visualization and Interpretation

### How to Approach:

**Step 1: Create Tree Diagram**
- **Hierarchical structure**: Root at top, leaves at bottom
- **Node labels**: Feature names and split values
- **Edge labels**: Feature values or conditions
- **Leaf labels**: Predictions and sample counts

**Step 2: Analyze Decision Paths**
- **Important paths**: High sample count or high accuracy
- **Critical decisions**: Early splits with high IG
- **Redundant paths**: Similar predictions from different paths

**Step 3: Calculate Feature Importance**
- **Information gain**: Total IG contribution
- **Frequency**: How often feature is used
- **Depth**: Average depth of feature usage
- **Impact**: Effect on final predictions

**Step 4: Interpret Business Rules**
- **Simple rules**: Easy to understand paths
- **Complex rules**: Multiple conditions
- **Risk factors**: Features indicating high-risk cases
- **Protective factors**: Features indicating low-risk cases

**Step 5: Communicate Results**
- **Key insights**: Most important findings
- **Actionable recommendations**: What to do with results
- **Limitations**: What the model can't tell us

### Example Template:
```
Given: Decision tree with [number] nodes
1. Tree visualization:
   - Root: [feature] with split [condition]
   - Left branch: [feature] → [prediction] ([count] samples)
   - Right branch: [feature] → [prediction] ([count] samples)
2. Decision paths:
   - Path 1: [conditions] → [prediction] (accuracy [value])
   - Path 2: [conditions] → [prediction] (accuracy [value])
3. Feature importance:
   - [Feature 1]: IG = [value], used [number] times
   - [Feature 2]: IG = [value], used [number] times
4. Business rules:
   - Rule 1: If [condition], then [action]
   - Rule 2: If [condition], then [action]
5. Key insights:
   - [insight 1]: [explanation]
   - [insight 2]: [explanation]
   - Recommendations: [action items]
```

---

## 🎯 Question Type 10: Advanced Topics (Multi-output, Online Learning)

### How to Approach:

**Step 1: Multi-output Trees**
- **Multiple targets**: Predict several variables simultaneously
- **Modified splitting**: Use multi-output impurity measures
- **Vector predictions**: Each leaf contains vector of predictions
- **Applications**: Multi-label classification, multi-target regression

**Step 2: Online Learning**
- **Incremental updates**: Update tree with new data
- **Concept drift**: Handle changing data distributions
- **Memory constraints**: Limited storage for historical data
- **Hoeffding trees**: Statistical bounds for split decisions

**Step 3: Streaming Data**
- **Infinite data**: Process data without storing all
- **Window-based**: Use sliding window of recent data
- **Adaptive**: Adjust tree structure over time
- **VFDT**: Very Fast Decision Trees for streaming

**Step 4: Interpretability**
- **Rule extraction**: Convert tree to if-then rules
- **Feature importance**: Rank features by contribution
- **Path analysis**: Analyze decision paths
- **Visualization**: Create interpretable plots

**Step 5: Real-world Applications**
- **Medical diagnosis**: Interpretable predictions
- **Credit scoring**: Explainable decisions
- **Fraud detection**: Real-time classification
- **Recommendation systems**: Personalized predictions

### Example Template:
```
Given: [advanced topic] with [specific requirements]
1. Problem characteristics:
   - Data type: [streaming/online/multi-output]
   - Constraints: [memory/time/interpretability]
   - Requirements: [accuracy/speed/explainability]
2. Algorithm adaptation:
   - Modified splitting: [criterion changes]
   - Update mechanism: [how to handle new data]
   - Memory management: [storage strategy]
3. Implementation details:
   - Data structure: [tree representation]
   - Update frequency: [how often to update]
   - Performance metrics: [evaluation criteria]
4. Challenges and solutions:
   - Challenge 1: [description] → Solution: [approach]
   - Challenge 2: [description] → Solution: [approach]
5. Applications:
   - Domain: [application area]
   - Benefits: [advantages over other methods]
   - Limitations: [constraints and drawbacks]
```

---

## 📝 General Exam Strategy

### Before Starting:
1. **Read the entire question** - identify all parts
2. **Identify question type** - use appropriate approach
3. **Plan your time** - allocate based on question complexity
4. **Gather formulas** - write down relevant equations

### During Solution:
1. **Show all steps** - even if you can do mental math
2. **Use clear notation** - define variables explicitly
3. **Draw diagrams** - visualization helps understanding
4. **Check units** - ensure dimensional consistency
5. **Verify results** - plug back into original equations

### Common Mistakes to Avoid:
- **Forgetting log base 2** in entropy calculations
- **Mixing up information gain and gain ratio**
- **Not considering class imbalance** in impurity measures
- **Ignoring overfitting** in tree depth analysis
- **Forgetting cost considerations** in business applications
- **Not checking stopping criteria** in algorithm analysis
- **Miscalculating weighted averages** in split evaluation

### Time Management:
- **Simple calculations**: 2-3 minutes
- **Medium complexity**: 5-8 minutes  
- **Complex derivations**: 10-15 minutes
- **Multi-part problems**: 15-20 minutes

### Final Checklist:
- [ ] All parts of question addressed
- [ ] Units and signs correct
- [ ] Results make intuitive sense
- [ ] Diagrams labeled clearly
- [ ] Key steps explained

---

## 🎯 Quick Reference: Decision Trees

### Which Impurity Measure?
```
What's the problem type?
├─ Classification → Entropy or Gini
├─ Regression → MSE or MAE
└─ Multi-output → Vector impurity measures
```

### Which Algorithm?
```
What are the requirements?
├─ Simple, categorical → ID3
├─ Missing values, continuous → C4.5
├─ Binary splits, regression → CART
└─ Large datasets → CART or Random Forest
```

### Which Pruning Method?
```
What's the goal?
├─ Prevent overfitting → Pre-pruning
├─ Optimize complexity → Cost-complexity pruning
├─ Maximize validation → Reduced error pruning
└─ Business constraints → Cost-sensitive pruning
```

### Which Validation Strategy?
```
What's the dataset size?
├─ Small (<1000) → Leave-one-out or 10-fold
├─ Medium (1000-10000) → 5-fold or 10-fold
└─ Large (>10000) → Hold-out or 5-fold
```

---

*This guide covers the most common Decision Tree question types. Practice with each approach and adapt based on specific problem requirements. Remember: understanding the concepts is more important than memorizing formulas!*
