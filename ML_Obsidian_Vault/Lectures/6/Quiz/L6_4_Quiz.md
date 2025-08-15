# Lecture 6.4: Tree Pruning and Regularization Quiz

## Overview
This quiz contains 22 comprehensive questions covering Tree Pruning and Regularization, including overfitting detection, pre-pruning techniques, post-pruning methods, cost-complexity pruning, cross-validation, and regularization strategies. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Netflix uses a decision tree to recommend movies. Their engineers noticed that as the tree grew deeper, training accuracy increased but user satisfaction (validation metric) decreased. Here's their data:

| Tree Depth | Training Accuracy | User Satisfaction Score | User Complaints |
|------------|-------------------|------------------------|-----------------|
| 1          | 0.65             | 0.68                  | 15%            |
| 2          | 0.78             | 0.75                  | 12%            |
| 3          | 0.89             | 0.82                  | 8%             |
| 4          | 0.95             | 0.79                  | 11%            |
| 5          | 0.98             | 0.76                  | 18%            |
| 6          | 0.99             | 0.74                  | 25%            |

#### Task
1. Identify the depth at which overfitting begins and justify your answer
2. Determine the optimal tree depth for this dataset
3. Explain the bias-variance tradeoff demonstrated in this data
4. Sketch a graph showing training vs validation accuracy
5. If Netflix wants to keep user complaints below 10%, what's the maximum acceptable tree depth?
6. What business implications does this overfitting have for Netflix?

For a detailed explanation of this question, see [Question 1: Overfitting Detection Game](L6_4_1_explanation.md).

## Question 2

### Problem Statement
You're a decision tree gardener trying to control tree growth. Your dataset has 1000 samples and 8 binary features, but you want to prevent your trees from becoming too "bushy."

#### Task
1. Calculate the maximum number of leaf nodes if each leaf must contain at least 50 samples
2. Determine the theoretical maximum depth before pre-pruning
3. For a binary classification problem, suggest an appropriate minimum Gini impurity threshold
4. Given training accuracy 0.95 and validation accuracy 0.82, which pre-pruning parameter should be adjusted first?
5. If you want exactly 20 leaf nodes, what minimum samples per leaf threshold would you need?
6. You're building a medical diagnosis tree. What additional pre-pruning constraints would you consider?

For a detailed explanation of this question, see [Question 2: Pre-Pruning Threshold Puzzle](L6_4_2_explanation.md).

## Question 3

### Problem Statement
A hospital uses a decision tree to predict patient readmission risk. The tree has grown complex and needs pruning:

```
Root (200 patients, train_error=0.25, val_error=0.28)
├── Left: Age ≤ 65 (120 patients, train_error=0.20, val_error=0.25)
│   ├── LL: Income ≤ $50K (60 patients, train_error=0.15, val_error=0.20)
│   └── LR: Income > $50K (60 patients, train_error=0.25, val_error=0.30)
└── Right: Age > 65 (80 patients, train_error=0.35, val_error=0.40)
    ├── RL: Chronic Disease = Yes (40 patients, train_error=0.30, val_error=0.35)
    └── RR: Chronic Disease = No (40 patients, train_error=0.40, val_error=0.45)
```

#### Task
1. Calculate validation error before and after pruning each subtree
2. Determine which subtrees should be pruned using reduced error pruning
3. Draw the final tree structure after optimal pruning
4. Calculate the final validation error after pruning
5. If the hospital wants to keep the tree interpretable (≤3 nodes), what would be the optimal pruning strategy?
6. What are the medical implications of pruning this tree too aggressively?

For a detailed explanation of this question, see [Question 3: Post-Pruning Decision Tree](L6_4_3_explanation.md).

## Question 4

### Problem Statement
A bank is implementing CART's cost-complexity pruning for their fraud detection system. The cost function is $R_\alpha(T) = R(T) + \alpha|T|$, where false positives cost $10 and false negatives cost $100.

#### Task
1. For $\alpha = 0.1$, calculate the cost for a tree with 7 nodes and total error 0.3
2. If pruning a subtree reduces nodes from 7 to 3 but increases error from 0.3 to 0.4, find the critical $\alpha$ value
3. For $\alpha = 0.05$, compare a tree with 5 nodes and error 0.35 vs 3 nodes and error 0.40
4. Explain the relationship between $\alpha$ and tree complexity
5. If the bank wants to minimize total cost including operational costs of $5 per node, what's the optimal $\alpha$?
6. What are the business implications of choosing different $\alpha$ values for fraud detection?

For a detailed explanation of this question, see [Question 4: Cost-Complexity Pruning Calculation](L6_4_4_explanation.md).

## Question 5

### Problem Statement
You're investigating the best way to use cross-validation for pruning parameter selection. Your dataset has 1000 samples, and you need to be careful about bias.

#### Task
1. Recommend the number of folds for decision tree pruning and justify your choice
2. Calculate the number of samples in each validation fold for 5-fold CV
3. Design a reasonable grid of $\alpha$ values to test
4. Explain how to handle bias introduced by parameter selection
5. If you use 10-fold CV and find that $\alpha = 0.1$ works best, but then test on a held-out test set and find $\alpha = 0.05$ works better, what does this suggest about your validation strategy?
6. You're working with a small dataset (200 samples). How would you modify your cross-validation strategy?

For a detailed explanation of this question, see [Question 5: Cross-Validation Strategy](L6_4_5_explanation.md).

## Question 6

### Problem Statement
You're judging a competition between different pruning approaches. Each method claims to be the best, but you need to evaluate them systematically.

#### Task
1. Rank these pruning methods by expected tree size (smallest to largest):
   - Pre-pruning with max_depth=3
   - Reduced error pruning
   - Cost-complexity pruning with α=0.1
2. Identify which method is most robust to noisy data and explain why
3. Compare computational speed of different pruning methods
4. Evaluate which method produces the most interpretable trees
5. If you have a dataset where noise increases with feature values, which pruning method would you expect to perform worst and why?
6. You're building a real-time recommendation system. Which pruning method would you choose and why?

For a detailed explanation of this question, see [Question 6: Pruning Method Comparison](L6_4_6_explanation.md).

## Question 7

### Problem Statement
You're experimenting with different ways to prevent decision trees from overfitting. You have various techniques at your disposal.

#### Task
1. Determine if a split reducing Gini impurity from 0.5 to 0.45 should be allowed with threshold 0.1
2. For a dataset with 10 features, calculate how many features to randomly select at each split
3. Compare limiting max_depth=3 vs post-pruning for a tree that naturally grows to depth 6
4. Explain how L1/L2 regularization concepts could be applied to decision trees
5. If you randomly select 3 features at each split from a pool of 10 features, what's the probability that the same feature is selected at both the root and its left child?
6. You're building a tree for a mobile app with limited memory. What regularization strategy would you prioritize?

For a detailed explanation of this question, see [Question 7: Regularization Techniques](L6_4_7_explanation.md).

## Question 8

### Problem Statement
You're analyzing learning curves from a decision tree experiment. The curves tell a story about overfitting, but you need to interpret them carefully.

#### Task
1. Interpret what a large gap between training and validation curves indicates
2. Identify the point in the learning curve where overfitting begins
3. Predict how learning curves would change after applying cost-complexity pruning
4. Explain how to use learning curves to select optimal tree complexity
5. If your learning curves show training accuracy increasing but validation accuracy decreasing, and then both start decreasing together, what might be happening to your data?
6. You're working with a dataset that has seasonal patterns. How would this affect your learning curve interpretation?

For a detailed explanation of this question, see [Question 8: Learning Curves Analysis](L6_4_8_explanation.md).

## Question 9

### Problem Statement
You're applying the Minimum Description Length principle to decision trees. This principle suggests that the best model is the one that can be described most concisely while still explaining the data well.

#### Task
1. Explain how MDL balances model complexity and accuracy
2. Estimate the description length for a tree with 5 nodes
3. Describe how MDL penalizes overly complex trees
4. List the main advantages of MDL-based pruning
5. If you have two trees with identical accuracy but different description lengths, and one tree has a leaf node that splits on a feature with only 2 unique values, what does MDL suggest about this split?
6. You're building a tree for a system with limited bandwidth. How would MDL help you optimize for transmission efficiency?

For a detailed explanation of this question, see [Question 9: MDL-Based Pruning](L6_4_9_explanation.md).

## Question 10

### Problem Statement
You're investigating C4.5's confidence factor pruning mechanism. This method uses statistical confidence to decide when to prune, but it has some interesting quirks.

#### Task
1. Explain how C4.5's confidence factor pruning works
2. Identify the statistical assumptions underlying confidence-based pruning
3. Describe how to choose an appropriate confidence level
4. Identify scenarios where confidence-based pruning might fail
5. If you set a confidence level of 95% and your tree has 100 nodes, approximately how many nodes would you expect to be pruned by chance alone?
6. You're building a tree for a safety-critical system (e.g., autonomous driving). What confidence level would you choose and why?

For a detailed explanation of this question, see [Question 10: Confidence-Based Pruning](L6_4_10_explanation.md).

## Question 11

### Problem Statement
An e-commerce company built a decision tree to predict customer churn. The tree achieved 98% training accuracy but only 72% validation accuracy. Here's their tree structure:

```
Root: Purchase_Frequency ≤ 2/month (1000 customers)
├── Left: Age ≤ 35 (600 customers, train_acc=0.99, val_acc=0.70)
│   ├── LL: Income ≤ $50K (300 customers, train_acc=0.99, val_acc=0.65)
│   └── LR: Income > $50K (300 customers, train_acc=0.99, val_acc=0.75)
└── Right: Age > 35 (400 customers, train_acc=0.97, val_acc=0.75)
    ├── RL: Location = Urban (200 customers, train_acc=0.98, val_acc=0.70)
    └── RR: Location = Rural (200 customers, train_acc=0.96, val_acc=0.80)
```

#### Task
1. List three methods to detect overfitting in decision trees
2. Sketch a plot showing the relationship between tree complexity and performance
3. Apply two different pruning techniques to the same overfitted tree
4. Explain how to validate pruning decisions
5. If the company wants to keep the tree simple enough for business analysts to understand (≤4 nodes), what pruning strategy would you recommend?
6. What are the business costs of overfitting in this customer churn prediction scenario?

For a detailed explanation of this question, see [Question 11: Overfitting Detection and Mitigation](L6_4_11_explanation.md).

## Question 12

### Problem Statement
You're conducting research on pruning effectiveness and need to design a comprehensive study.

#### Task
1. Explain how to test pruning methods on datasets with different characteristics
2. Design a systematic comparison of major pruning techniques
3. Identify appropriate metrics for evaluating pruning effectiveness
4. Provide evidence-based recommendations for pruning method selection
5. If you have limited computational resources and can only test 3 pruning methods on 2 datasets, which combinations would you choose to maximize insights?
6. You're writing a research paper. What statistical tests would you use to determine if pruning method differences are significant?

For a detailed explanation of this question, see [Question 12: Comprehensive Pruning Study](L6_4_12_explanation.md).

## Question 13

### Problem Statement
You're building a decision tree for a high-frequency trading system where early stopping is crucial.

#### Task
1. List three different early stopping criteria for decision trees
2. Explain how to monitor validation performance during tree construction
3. Compare early stopping vs post-pruning approaches
4. Describe how to implement patience-based early stopping
5. If your validation performance fluctuates due to market volatility, how would you modify your early stopping criteria?
6. What are the financial implications of stopping too early vs too late in this trading scenario?

For a detailed explanation of this question, see [Question 13: Early Stopping Strategies](L6_4_13_explanation.md).

## Question 14

### Problem Statement
You're building an automated system to find optimal pruning parameters.

#### Task
1. Design a grid search strategy for pruning parameters
2. Explain how to use nested cross-validation for unbiased parameter selection
3. Compare different metrics for pruning evaluation
4. Design an automated pipeline for optimal pruning
5. If you have 1000 samples and want to test 5 α values with 5-fold CV, how many total model fits will you need to perform?
6. You're building this for a company that needs results within 1 hour. How would you modify your optimization strategy?

For a detailed explanation of this question, see [Question 14: Pruning Parameter Optimization](L6_4_14_explanation.md).

## Question 15

### Problem Statement
You're working with data from IoT sensors that have varying levels of noise depending on environmental conditions.

#### Task
1. Explain how noise affects decision tree overfitting
2. Identify which pruning methods are most robust to noise
3. Describe how to modify pruning criteria for noisy data
4. Explain the role of outlier detection in tree regularization
5. If noise increases exponentially with feature values, how would you modify your pruning thresholds?
6. You're building a tree for a smart home system. What are the safety implications of pruning too aggressively with noisy sensor data?

For a detailed explanation of this question, see [Question 15: Pruning with Noisy Data](L6_4_15_explanation.md).

## Question 16

### Problem Statement
You're building a decision tree that must balance accuracy, interpretability, and computational efficiency.

#### Task
1. Explain how to add randomness to tree construction
2. Describe how pruning criteria can adapt to local data characteristics
3. Explain how to balance multiple objectives (accuracy, interpretability, size)
4. Describe pruning strategies for streaming/online learning scenarios
5. If you have a budget constraint of 1000 total node evaluations, how would you distribute this budget across different pruning methods?
6. You're building a tree for a medical device. How would you weight the different objectives given patient safety requirements?

For a detailed explanation of this question, see [Question 16: Advanced Regularization](L6_4_16_explanation.md).

## Question 17

### Problem Statement
Practice pruning decisions on a complex tree structure used by an insurance company:

```
Root: Age ≤ 30 (Training Acc: 88%, Validation Acc: 75%)
├── Left: Income ≤ $40K (Training Acc: 92%, Validation Acc: 70%)
│   ├── Low Risk (Leaf): [Safe: 10, Risk: 2]
│   └── Medium Risk (Leaf): [Safe: 4, Risk: 6]
└── Right: Experience > 2 years (Training Acc: 90%, Validation Acc: 78%)
    ├── High Risk (Leaf): [Safe: 3, Risk: 8]
    └── Safe (Leaf): [Safe: 9, Risk: 2]
```

#### Task
1. Calculate training accuracy for each subtree and the full tree
2. Determine which pruning decision reduced error pruning would make
3. For α = 0.15, calculate cost-complexity for full tree vs pruned versions
4. Analyze what the validation performance pattern suggests about overfitting
5. Write the final decision rule after optimal pruning
6. If the insurance company wants to minimize false negatives (missing high-risk customers) while keeping false positives below 20%, what pruning strategy would you recommend?
7. What are the regulatory implications of pruning this risk assessment tree?

For a detailed explanation of this question, see [Question 17: Tree Surgery Simulation](L6_4_17_explanation.md).

## Question 18

### Problem Statement
You're a consultant helping companies choose pruning strategies. Create a decision matrix for evaluating different approaches.

#### Task
1. Design a 4×4 matrix comparing pruning methods vs evaluation criteria
2. Select 4 key criteria for evaluating pruning methods
3. Design a 1-5 scoring system for each method-criteria combination
4. Recommend the best pruning method for a medical diagnosis application
5. If you have to add a cost constraint to your matrix, how would you weight it relative to accuracy and interpretability?
6. You're presenting this to a non-technical CEO. How would you explain the trade-offs in business terms?

For a detailed explanation of this question, see [Question 18: Pruning Decision Matrix](L6_4_18_explanation.md).

## Question 19

### Problem Statement
Find the optimal α value for cost-complexity pruning given these tree options:
- Full tree: 9 nodes, error = 0.25
- Pruned tree 1: 5 nodes, error = 0.30
- Pruned tree 2: 3 nodes, error = 0.35
- Pruned tree 3: 1 node, error = 0.45

#### Task
1. Write the cost-complexity function for each option
2. Find the critical α value where pruning tree 1 becomes beneficial
3. Determine the range of α values where pruned tree 2 is optimal
4. If you want a tree with ≤4 nodes, what α value should you use?
5. If you have a budget constraint that limits you to trees with ≤6 nodes, what's the optimal α range?
6. You're building this for a mobile app with limited memory. How would you modify your α selection strategy?

For a detailed explanation of this question, see [Question 19: Alpha Selection Game](L6_4_19_explanation.md).

## Question 20

### Problem Statement
You're building a decision tree for a credit risk assessment system that must satisfy multiple constraints:
- Interpretability is crucial (regulatory requirement)
- Accuracy must be ≥85%
- Tree size should be ≤10 nodes
- Training time should be ≤5 minutes
- False positive rate must be ≤15%

#### Task
1. Identify which constraints are most likely to conflict
2. Design a regularization strategy that satisfies all constraints
3. Determine which parameters to tune first and explain why
4. Design a validation plan for your strategy
5. If you can only satisfy 4 out of 5 constraints, which one would you relax and why?
6. What are the legal implications if your tree violates the interpretability requirement?

For a detailed explanation of this question, see [Question 20: Regularization Trade-off Puzzle](L6_4_20_explanation.md).

## Question 21

### Problem Statement
Understand when different pruning methods should be applied during the tree development timeline:
Tree construction → Training → Validation → Deployment

#### Task
1. Place each pruning method on the appropriate timeline position:
   - Pre-pruning
   - Post-pruning
   - Cost-complexity pruning
   - Reduced error pruning
2. Explain why the timing of pruning is important
3. Identify which pruning method is most computationally efficient
4. Determine which method offers the most flexibility for parameter tuning
5. If you discover during deployment that your tree is overfitting, which pruning methods can you still apply?
6. You're working in an agile development environment with weekly deployments. How does this affect your pruning strategy?

For a detailed explanation of this question, see [Question 21: Pruning Timeline Challenge](L6_4_21_explanation.md).

## Question 22

### Problem Statement
Design a comprehensive tool for measuring tree complexity that can be used across different domains.

#### Task
1. Design three different metrics for measuring tree complexity
2. Explain how to normalize these metrics for fair comparison
3. Set reasonable thresholds for each metric
4. Apply your metrics to a tree with 7 nodes and depth 4
5. If you want to create a single "complexity score" that combines all three metrics, how would you weight them?
6. You're building this tool for a company that builds trees for different industries (finance, healthcare, retail). How would you adapt your metrics for each domain?

For a detailed explanation of this question, see [Question 22: Tree Complexity Calculator](L6_4_22_explanation.md).

## Question 23

### Problem Statement
You encounter a strange situation where pruning a decision tree actually increases 
both training and validation accuracy. This seems impossible at first glance.

#### Task
1. Explain how this paradoxical situation could occur
2. Design a simple example dataset and tree structure where this happens
3. What does this reveal about the relationship between tree complexity and performance?
4. How would you detect and handle such cases in practice?
5. If pruning increases both accuracies, what does this suggest about the original tree's structure?

For a detailed explanation of this question, see [Question 23: Tree Pruning Paradox](L6_4_23_explanation.md).