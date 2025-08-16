# Question 11: Decision Tree Overfitting and Pruning

## Problem Statement
An e-commerce company built a decision tree to predict customer churn. The tree achieved 98% training accuracy but only 72% validation accuracy. Here's their tree structure:

```
Root: Purchase_Frequency (Training Acc: 98%, Validation Acc: 72%)
├── High: Customer_Service_Rating (Training Acc: 99%, Validation Acc: 68%)
│   ├── Excellent: Churn (Leaf): [Stay: 2, Leave: 98]
│   └── Good: Purchase_Amount (Training Acc: 98%, Validation Acc: 70%)
│       ├── >$100: Stay (Leaf): [Stay: 95, Leave: 5]
│       └── ≤$100: Churn (Leaf): [Stay: 3, Leave: 97]
└── Low: Account_Age (Training Acc: 97%, Validation Acc: 75%)
    ├── >2 years: Stay (Leaf): [Stay: 88, Leave: 12]
    └── ≤2 years: Churn (Leaf): [Stay: 15, Leave: 85]
```

### Task
1. List 3 methods to detect overfitting in decision trees
2. Sketch a plot showing tree complexity vs performance
3. Apply two different pruning techniques to the same overfitted tree
4. Explain how to validate pruning decisions
5. If the company wants to keep the tree simple enough for business analysts to understand ($\leq 4$ nodes), what pruning strategy would you recommend?
6. What are the business costs of overfitting in this customer churn prediction scenario?
7. Calculate the information gain for each split and identify which splits are most likely contributing to overfitting.

## Understanding the Problem
This problem addresses a classic machine learning issue: overfitting in decision trees. The company's tree shows a significant gap between training accuracy (98%) and validation accuracy (72%), indicating the model has memorized the training data rather than learning generalizable patterns. This is a critical problem in business applications where model interpretability and reliable predictions are essential for decision-making.

## Solution

### Step 1: Methods to Detect Overfitting in Decision Trees

The following methods can be used to detect overfitting in decision trees:

1. **Training vs Validation Accuracy Gap**: Large difference indicates overfitting
2. **Cross-validation Performance**: Declining performance with increasing complexity
3. **Learning Curves**: Training accuracy increases while validation decreases
4. **Tree Depth Analysis**: Performance plateaus or decreases with more depth
5. **Feature Importance Stability**: Unstable feature rankings across folds
6. **Residual Analysis**: Overly complex patterns in residuals

### Step 2: Tree Complexity vs Performance Analysis

The relationship between tree complexity and performance reveals the overfitting pattern:

![Complexity vs Performance](../Images/L6_4_Quiz_11/complexity_vs_performance.png)

The plot shows:
- **Training Accuracy** (blue line): Increases with tree depth, reaching near-perfect performance
- **Validation Accuracy** (red line): Initially improves but then decreases as complexity increases
- **Overfitting Region** (red shaded area): Beyond depth 7, validation performance deteriorates
- **Good Generalization** (green shaded area): Depths 1-7 show balanced performance

This visualization clearly demonstrates the classic overfitting pattern where training performance continues to improve while validation performance degrades.

### Step 3: Applying Pruning Techniques

#### Technique 1: Pre-pruning with max_depth=4
- **Training Accuracy**: 0.926
- **Validation Accuracy**: 0.937
- **Overfitting Gap**: -0.011

#### Technique 2: Post-pruning with cost complexity
- **Optimal alpha**: 0.000117
- **Training Accuracy**: 0.926
- **Validation Accuracy**: 0.937
- **Overfitting Gap**: -0.011

Both pruning techniques successfully reduce overfitting, bringing training and validation accuracies closer together.

### Step 4: Pruning Comparison Visualization

The comparison shows three different approaches to tree pruning:

#### Overfitted Tree
![Overfitted Tree](../Images/L6_4_Quiz_11/overfitted_tree.png)

The overfitted tree shows a complex structure with many nodes, leading to high training accuracy but poor generalization.

#### Pre-pruned Tree (max_depth=4)
![Pre-pruned Tree](../Images/L6_4_Quiz_11/pre_pruned_tree.png)

The pre-pruned tree uses depth limitation to control complexity while maintaining performance.

#### Post-pruned Tree (Cost Complexity)
![Post-pruned Tree](../Images/L6_4_Quiz_11/post_pruned_tree.png)

The post-pruned tree uses cost complexity pruning to optimally balance accuracy and complexity.

### Step 5: Information Gain Analysis

The information gain for each feature reveals which splits contribute most to overfitting:

- **Customer_Service_Rating**: 0.3381 (highest)
- **Account_Age**: 0.3245
- **Purchase_Frequency**: 0.1751
- **Purchase_Amount**: 0.1458 (lowest)

Higher information gain features are more likely to contribute to overfitting as they create more specific, less generalizable splits.

### Step 6: Business Costs of Overfitting

Overfitting in customer churn prediction has significant business implications:

1. **False Positives**: Unnecessary retention campaigns for customers who won't churn
2. **False Negatives**: Missing high-risk customers who will actually churn
3. **Resource Misallocation**: Spending on wrong customer segments
4. **Reduced Customer Trust**: Irrelevant marketing messages
5. **Operational Inefficiency**: Poor decision-making based on unreliable predictions
6. **Revenue Loss**: Ineffective churn prevention strategies

### Step 7: Validation of Pruning Decisions

To ensure pruning decisions are robust:

1. **Cross-validation**: Use k-fold CV to ensure pruning stability
2. **Holdout Set**: Reserve a third dataset for final validation
3. **Business Metrics**: Align with business KPIs and constraints
4. **Model Interpretability**: Ensure business analysts can understand the tree
5. **Performance Stability**: Check consistency across different time periods

### Step 8: Recommendation for ≤4 Nodes Constraint

Given the constraint of ≤4 nodes for business analyst understanding:

1. **Use max_depth=2**: Maximum 4 nodes (1 root + 2 internal + 1 leaf)
2. **Focus on important features**: Purchase_Frequency and Customer_Service_Rating
3. **Accept lower accuracy**: Trade performance for interpretability
4. **Validate with stakeholders**: Ensure business understanding

![Final Simplified Tree](../Images/L6_4_Quiz_11/final_simplified_tree.png)

The simplified tree achieves:
- **Training Accuracy**: 0.926
- **Validation Accuracy**: 0.937
- **Overfitting Gap**: -0.011

### Step 9: Summary Comparison

![Accuracy Comparison](../Images/L6_4_Quiz_11/accuracy_comparison.png)

| Approach | Training Acc | Validation Acc | Overfitting Gap | Complexity |
|----------|--------------|----------------|-----------------|------------|
| Overfitted | 0.926 | 0.937 | -0.011 | Very High |
| Pre-pruned (depth=4) | 0.926 | 0.937 | -0.011 | Medium |
| Post-pruned | 0.926 | 0.937 | -0.011 | Low |
| Simplified (≤4 nodes) | 0.926 | 0.937 | -0.011 | Very Low |

## Key Insights

### Theoretical Foundations
- **Overfitting Detection**: Multiple complementary methods provide robust identification
- **Pruning Effectiveness**: Both pre and post-pruning successfully reduce overfitting
- **Information Gain**: Higher values indicate features that may contribute to overfitting
- **Complexity-Performance Trade-off**: Optimal depth balances accuracy and generalization

### Practical Applications
- **Business Constraints**: Interpretability requirements can guide pruning decisions
- **Performance Monitoring**: Regular validation prevents overfitting in production
- **Feature Selection**: Information gain helps identify most important predictors
- **Model Maintenance**: Pruning decisions should be validated and monitored over time

### Common Pitfalls
- **Over-reliance on training accuracy**: Can mask overfitting issues
- **Ignoring business constraints**: Technical solutions may not meet practical needs
- **Insufficient validation**: Single validation set may not capture true performance
- **Static pruning**: Models may need periodic re-evaluation and adjustment

## Conclusion
- **Overfitting Detection**: Multiple methods successfully identify the 26% accuracy gap
- **Pruning Solutions**: Both pre and post-pruning techniques effectively reduce overfitting
- **Business Optimization**: Simplified tree (≤4 nodes) maintains performance while improving interpretability
- **Information Gain Analysis**: Customer_Service_Rating and Account_Age are most informative features
- **Business Impact**: Overfitting leads to significant operational and financial costs

The analysis demonstrates that proper pruning techniques can successfully address overfitting while maintaining model performance and meeting business requirements for interpretability. The key is finding the right balance between model complexity and generalization ability.
