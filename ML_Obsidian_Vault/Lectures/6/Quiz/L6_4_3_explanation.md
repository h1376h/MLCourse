# Question 3: Post-Pruning Decision Tree

## Problem Statement
A hospital uses a decision tree to predict patient readmission risk. The tree has grown complex and needs pruning:

```
Root (200 samples, train_error=0.25, val_error=0.28)
├── Left (120 samples, train_error=0.20, val_error=0.25)
│   ├── LL (60 samples, train_error=0.15, val_error=0.20)
│   └── LR (60 samples, train_error=0.25, val_error=0.30)
└── Right (80 samples, train_error=0.35, val_error=0.40)
    ├── RL (40 samples, train_error=0.30, val_error=0.35)
    └── RR (40 samples, train_error=0.40, val_error=0.45)
```

### Task
1. Calculate validation error before and after pruning each subtree
2. Which subtrees should be pruned using reduced error pruning?
3. Draw the final tree structure after optimal pruning
4. Calculate the final validation error after pruning
5. If the hospital wants to keep the tree interpretable ($\leq 3$ nodes), what would be the optimal pruning strategy?
6. What are the medical implications of pruning this tree too aggressively?
7. If false negatives (missing high-risk patients) cost $\$1000$ and false positives cost $\$100$, calculate the total cost before and after pruning.
8. If the hospital can process 50 patients per day with the pruned tree vs 30 with the full tree, calculate the daily cost savings

## Understanding the Problem
This problem involves decision tree pruning, a technique used to reduce overfitting by removing branches that contribute little to the model's performance. The tree has grown complex with 7 nodes total, and we need to determine the optimal pruning strategy that balances model complexity with validation performance.

The key concept is **reduced error pruning**, which evaluates each subtree by comparing the validation error before and after pruning. If pruning a subtree improves validation performance, it should be removed.

## Solution

### Step 1: Calculate Validation Error Before and After Pruning Each Subtree

We need to evaluate four pruning scenarios:

#### 1. Pruning Left Subtree
- **Before pruning**: Weighted average of Left and Right subtrees
  - Left: $0.250 \times 120 = 30.0$ weighted errors
  - Right: $0.400 \times 80 = 32.0$ weighted errors
  - Total: $30.0 + 32.0 = 62.0$ weighted errors across 200 samples
  - **Weighted average**: $\frac{62.0}{200} = 0.310$
- **After pruning**: Only Right subtree remains with error $0.400$
- **Improvement**: $0.310 - 0.400 = -0.090$ (worse performance)

#### 2. Pruning Right Subtree
- **Before pruning**: Same weighted average as above = $0.310$
- **After pruning**: Only Left subtree remains with error $0.250$
- **Improvement**: $0.310 - 0.250 = +0.060$ (better performance)

#### 3. Pruning LL and LR Subtrees
- **Before pruning**: Weighted average of LL and LR
  - LL: $0.200 \times 60$ samples = 12 weighted errors
  - LR: $0.300 \times 60$ samples = 18 weighted errors
  - **Weighted average**: $\frac{30}{120} = 0.250$
- **After pruning**: Left subtree error = $0.250$
- **Improvement**: $0.250 - 0.250 = 0.000$ (no change)

#### 4. Pruning RL and RR Subtrees
- **Before pruning**: Weighted average of RL and RR
  - RL: $0.350 \times 40$ samples = 14 weighted errors
  - RR: $0.450 \times 40$ samples = 18 weighted errors
  - **Weighted average**: $\frac{32}{80} = 0.400$
- **After pruning**: Right subtree error = $0.400$
- **Improvement**: $0.400 - 0.400 = 0.000$ (no change)

### Step 2: Determine Which Subtrees Should Be Pruned Using Reduced Error Pruning

Ranking the pruning scenarios by improvement:

1. **Prune_Right**: Improvement = $+0.060$ (best)
2. **Prune_LL_LR**: Improvement = $0.000$ (no change)
3. **Prune_RL_RR**: Improvement = $0.000$ (no change)
4. **Prune_Left**: Improvement = $-0.090$ (worse)

**Best pruning strategy**: Prune the Right subtree, which provides the only positive improvement.

### Step 3: Draw the Final Tree Structure After Optimal Pruning

After pruning the Right subtree, the final tree structure becomes:

```
Root (200 samples, val_error=0.28)
└── Left (120 samples, val_error=0.25)
    ├── LL (60 samples, val_error=0.20)
    └── LR (60 samples, val_error=0.30)
```

![Tree Structure Comparison](../Images/L6_4_Quiz_3/tree_structure_comparison.png)

The visualization shows the original complex tree (left) and the pruned tree (right) after removing the Right subtree.

### Step 4: Calculate the Final Validation Error After Pruning

The final validation error is the weighted average of the remaining nodes:

- Root: $0.280 \times 200$ samples = 56 weighted errors
- Left: $0.250 \times 120$ samples = 30 weighted errors
- **Total**: 86 weighted errors across 200 samples
- **Final validation error**: $\frac{86}{200} = 0.314$

**Improvement**: $0.280 - 0.314 = -0.034$ (slightly worse)

Wait, this seems counterintuitive. Let me recalculate:

Actually, after pruning the Right subtree, we keep the Root and Left subtrees. The final validation error should be:

- Root: $0.280 \times 200$ samples = 56 weighted errors
- Left: $0.250 \times 120$ samples = 30 weighted errors
- **Total**: 86 weighted errors across 200 samples
- **Final validation error**: $\frac{86}{200} = 0.314$

This is higher than the original root error of $0.280$, which means pruning actually made the overall performance worse. This suggests that the "optimal" pruning strategy from reduced error pruning might not be the best approach for this specific tree structure.

### Step 5: Optimal Pruning Strategy for ≤3 Nodes

We need to evaluate different strategies that keep at most 3 nodes:

1. **Root_Only**: $0.280$ (just the root node)
2. **Root_Left**: $\frac{0.280 \times 200 + 0.250 \times 120}{200 + 120} = \frac{56 + 30}{320} = 0.269$
3. **Root_Right**: $\frac{0.280 \times 200 + 0.400 \times 80}{200 + 80} = \frac{56 + 32}{280} = 0.314$
4. **Root_LL_LR**: $\frac{0.280 \times 200 + 0.200 \times 60 + 0.300 \times 60}{200 + 60 + 60} = \frac{56 + 12 + 18}{320} = 0.269$
5. **Root_RL_RR**: $\frac{0.280 \times 200 + 0.350 \times 40 + 0.450 \times 40}{200 + 40 + 40} = \frac{56 + 14 + 18}{280} = 0.314$

**Best 3-node strategy**: Root_Left with error $0.269$

This strategy provides the lowest validation error while maintaining interpretability.

### Step 6: Medical Implications of Aggressive Pruning

#### False Negatives (Missing High-Risk Patients)
- **Critical Impact**: Patients who need immediate attention may be missed
- **Clinical Consequences**: Could lead to readmissions, complications, or worse outcomes
- **Risk Level**: Higher mortality risk for high-risk patients
- **Medical Priority**: In healthcare, false negatives are often more costly than false positives

#### False Positives (Unnecessary Interventions)
- **Economic Impact**: Low-risk patients may receive unnecessary treatments
- **Patient Experience**: Increased healthcare costs and patient anxiety
- **Clinical Risk**: Potential side effects from unnecessary interventions
- **Resource Waste**: Misallocation of limited healthcare resources

#### Balance Considerations
- **Cost Asymmetry**: False negatives typically cost more than false positives in medical contexts
- **Pruning Risk**: Aggressive pruning may increase false negative rate
- **Interpretability vs. Accuracy**: Need to balance model simplicity with clinical accuracy
- **Regulatory Requirements**: Medical AI systems often require high sensitivity

### Step 7: Cost Analysis

#### Cost Parameters
- False Negative Cost: $\$1000$ per missed high-risk patient
- False Positive Cost: $\$100$ per unnecessary intervention

#### Original Tree
- Validation Error: $0.280$
- False Negatives: $0.280 \times 200 \times 0.5 = 28.0$ patients
- False Positives: $0.280 \times 200 \times 0.5 = 28.0$ patients
- Total Cost: $28.0 \times \$1000 + 28.0 \times \$100 = \$28,000 + \$2,800 = \$30,800$

#### Pruned Tree
- Validation Error: $0.314$
- False Negatives: $0.314 \times 200 \times 0.5 = 31.4$ patients
- False Positives: $0.314 \times 200 \times 0.5 = 31.4$ patients
- Total Cost: $31.4 \times \$1000 + 31.4 \times \$100 = \$31,400 + \$3,143 = \$34,571.43$

#### Cost Comparison
- **Cost Difference**: $\$30,800 - \$34,571.43 = -\$3,771.43$
- **Cost Reduction**: $-12.2\%$ (actually a cost increase)

The pruned tree actually increases total costs due to higher error rates.

### Step 8: Processing Capacity and Daily Cost Savings

#### Processing Capacity Analysis
- **Original Tree**: 30 patients per day
- **Pruned Tree**: 50 patients per day
- **Capacity Increase**: $\frac{50 - 30}{30} \times 100\% = 66.7\%$

#### Daily Cost Analysis
- **Original Tree Daily Cost**: $\frac{\$30,800}{200} \times 30 = \$154 \times 30 = \$4,620$
- **Pruned Tree Daily Cost**: $\frac{\$34,571.43}{200} \times 50 = \$172.86 \times 50 = \$8,642.86$
- **Daily Cost Savings**: $\$4,620 - \$8,642.86 = -\$4,022.86$
- **Daily Savings Percentage**: $-87.1\%$ (actually a cost increase)

#### Key Insights
- **Capacity vs. Cost Trade-off**: While the pruned tree processes more patients per day, it does so at a higher cost per patient
- **Efficiency Paradox**: Higher throughput doesn't necessarily mean better cost efficiency
- **Quality vs. Quantity**: The pruned tree sacrifices accuracy for speed, leading to higher overall costs

## Visual Explanations

### Tree Structure Comparison
![Tree Structure Comparison](../Images/L6_4_Quiz_3/tree_structure_comparison.png)

This visualization shows the dramatic simplification achieved through pruning. The original tree (left) has 7 nodes with complex branching, while the pruned tree (right) maintains only the essential structure with the Left subtree pruned.

### Pruning Scenarios Comparison
![Pruning Scenarios Comparison](../Images/L6_4_Quiz_3/pruning_scenarios_comparison.png)

This chart compares validation errors before and after each pruning scenario. The "Prune_Right" strategy shows the only positive improvement (+0.060), while "Prune_Left" actually worsens performance (-0.090).

### Cost Analysis
![Cost Analysis](../Images/L6_4_Quiz_3/cost_analysis.png)

The cost breakdown shows that while the pruned tree processes more patients daily, it incurs higher costs due to increased error rates. The daily cost comparison reveals that higher capacity comes with a significant cost premium.

### Three-Node Pruning Strategies
![Three-Node Pruning Strategies](../Images/L6_4_Quiz_3/three_node_pruning_strategies.png)

This visualization ranks different 3-node pruning strategies by validation error. The "Root_Left" strategy emerges as optimal with the lowest error rate (0.269), providing the best balance between interpretability and performance.

## Key Insights

### Theoretical Foundations
- **Reduced Error Pruning**: Evaluates each subtree independently, which may not capture the global impact of pruning decisions
- **Weighted Error Calculation**: Sample size matters - larger subtrees have more influence on overall performance
- **Local vs. Global Optimization**: What's optimal for individual subtrees may not be optimal for the entire tree

### Practical Applications
- **Medical AI Trade-offs**: Higher throughput doesn't always mean better outcomes or lower costs
- **Interpretability vs. Performance**: Simpler models may be preferred even if they have slightly higher error rates
- **Cost-Benefit Analysis**: The true cost of errors must include both direct financial costs and clinical consequences

### Common Pitfalls
- **Over-pruning**: Removing too many branches can significantly degrade performance
- **Ignoring Sample Sizes**: Weighted averages are crucial for accurate pruning decisions
- **Focusing Only on Error Rates**: Cost considerations and clinical implications must also be weighed

### Extensions and Considerations
- **Cross-Validation**: Multiple validation sets could provide more robust pruning decisions
- **Cost-Sensitive Pruning**: Incorporating asymmetric costs into the pruning algorithm
- **Clinical Validation**: Medical applications require domain expert review of pruning decisions

## Conclusion
- **Optimal Pruning Strategy**: Prune the Right subtree, resulting in a tree with Root and Left subtrees
- **Performance Impact**: Pruning increases validation error from 0.280 to 0.314
- **Cost Implications**: Higher error rates lead to increased total costs despite improved processing capacity
- **Medical Considerations**: The trade-off between interpretability and accuracy has real clinical and financial consequences
- **Best 3-Node Strategy**: Root_Left configuration provides the lowest validation error (0.269) while maintaining interpretability

The analysis reveals that decision tree pruning in medical applications requires careful consideration of multiple factors beyond just validation error rates. The optimal strategy balances model complexity, clinical accuracy, and operational efficiency, recognizing that simpler models may be preferred even when they have slightly higher error rates, especially when interpretability and clinical safety are paramount.
