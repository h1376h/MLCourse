# Lecture 6.4: Tree Pruning and Regularization Quiz

## Overview
This quiz contains 36 comprehensive questions covering Tree Pruning and Regularization, including overfitting detection, pre-pruning techniques, post-pruning methods, cost-complexity pruning, cross-validation, and regularization strategies. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

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
1. [📚] At what depth does overfitting begin? Justify your answer
2. [📚] What is the optimal tree depth for this dataset?
3. [📚] Explain the bias-variance tradeoff demonstrated in this data
4. Sketch a graph showing training vs validation accuracy
5. [📚] If Netflix wants to keep user complaints below 10%, what's the maximum acceptable tree depth?
6. [📚] If Netflix has $100$ million users and each complaint costs $2 in customer service, calculate the total cost of overfitting at depth $6$

For a detailed explanation of this question, see [Question 1: Netflix Decision Tree](L6_4_1_explanation.md).

## Question 2

### Problem Statement
You're a decision tree gardener trying to control tree growth. Your dataset has 1000 samples and 8 binary features, but you want to prevent your trees from becoming too "bushy."

#### Task
1. [📚] If you want each leaf to have at least $50$ samples, what's the maximum number of leaf nodes possible?
2. [📚] If the dataset has $8$ binary features, what's the theoretical maximum depth before pre-pruning?
3. For a binary classification problem, suggest an appropriate minimum Gini impurity threshold
4. [📚] Given training accuracy $0.95$ and validation accuracy $0.82$, which pre-pruning parameter should be adjusted first?
5. [📚] If you want exactly $20$ leaf nodes, what minimum samples per leaf threshold would you need?
6. You're building a medical diagnosis tree. What additional pre-pruning constraints would you consider?
7. Calculate the minimum impurity decrease threshold that would prevent splitting a node with $100$ samples into two groups of $45$ and $55$ samples.

For a detailed explanation of this question, see [Question 2: Pre-Pruning Threshold Puzzle](L6_4_2_explanation.md).

## Question 3

### Problem Statement
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

#### Task
1. [📚] Calculate validation error before and after pruning each subtree
2. [📚] Which subtrees should be pruned using reduced error pruning?
3. [📚] Draw the final tree structure after optimal pruning
4. [📚] Calculate the final validation error after pruning
5. [📚] If the hospital wants to keep the tree interpretable ($\leq 3$ nodes), what would be the optimal pruning strategy?
6. [📚] What are the medical implications of pruning this tree too aggressively?
7. [📚] If false negatives (missing high-risk patients) cost $1000 and false positives cost $100, calculate the total cost before and after pruning.
8. [📚] If the hospital can process $50$ patients per day with the pruned tree vs $30$ with the full tree, calculate the daily cost savings

For a detailed explanation of this question, see [Question 3: Post-Pruning Decision Tree](L6_4_3_explanation.md).

## Question 4

### Problem Statement
A bank is implementing CART's cost-complexity pruning for their fraud detection system. The cost function is $R_\alpha(T) = R(T) + \alpha|T|$, where false positives cost $10 and false negatives cost $100.

#### Task
1. Write the cost-complexity function: $R_\alpha(T) = R(T) + \alpha|T|$
2. For $\alpha = 0.1$, calculate the cost for a tree with $7$ nodes and total error $0.3$
3. For $\alpha = 0.05$, compare a tree with $5$ nodes and error $0.35$ vs $3$ nodes and error $0.40$
4. Explain the relationship between $\alpha$ and tree complexity
5. If the bank wants to minimize total cost including operational costs of $5 per node, what's the optimal $\alpha$?
6. What are the business implications of choosing different $\alpha$ values for fraud detection?
7. Design a cost matrix for a medical diagnosis system where false negatives are $10\times$ more expensive than false positives.
8. If the bank processes $10,000$ transactions per day, calculate the daily fraud detection cost for different $\alpha$ values

For a detailed explanation of this question, see [Question 4: Cost-Complexity Pruning Calculation](L6_4_4_explanation.md).

## Question 5

### Problem Statement
You're investigating the best way to use cross-validation for pruning parameter selection. Your dataset has 1000 samples, and you need to be careful about bias.

#### Task
1. How many folds would you use for decision tree pruning? Justify your choice
2. If using $5$-fold CV, how many samples are in each validation fold?
3. Design a reasonable grid of $\alpha$ values to test
4. Explain how to handle bias introduced by parameter selection
5. If you use $10$-fold CV and find that $\alpha = 0.1$ works best, but then test on a held-out test set and find $\alpha = 0.05$ works better, what does this suggest about your validation strategy?
6. You're working with a small dataset ($200$ samples). How would you modify your cross-validation strategy?
7. Calculate the minimum number of samples needed per fold to ensure statistical significance with $95\%$ confidence.

For a detailed explanation of this question, see [Question 5: Cross-Validation Strategy](L6_4_5_explanation.md).

## Question 6

### Problem Statement
You're judging a competition between different pruning approaches. Each method claims to be the best, but you need to evaluate them systematically.

#### Task
1. Rank these pruning methods by expected tree size (smallest to largest):
   - Pre-pruning with max_depth = $3$
   - Reduced error pruning
   - Cost-complexity pruning with $\alpha = 0.1$
2. Which method is most robust to noisy data? Why?
3. Compare computational speed of different pruning methods
4. Evaluate which method produces the most interpretable trees
5. If you have a dataset where noise increases with feature values, which pruning method would you expect to perform worst and why?
6. You're building a real-time recommendation system. Which pruning method would you choose and why?
7. Design an experiment to measure the computational efficiency of each pruning method.

For a detailed explanation of this question, see [Question 6: Pruning Method Comparison](L6_4_6_explanation.md).

## Question 7

### Problem Statement
You're experimenting with different ways to prevent decision trees from overfitting. You have various techniques at your disposal.

#### Task
1. If a split reduces Gini impurity from $0.5$ to $0.45$, should it be allowed with threshold $0.1$?
2. For a dataset with $10$ features, how many features would you randomly select at each split?
3. Compare limiting max_depth = $3$ vs post-pruning for a tree that naturally grows to depth $6$
4. Explain how L1/L2 regularization concepts could be applied to decision trees
5. If you randomly select $3$ features at each split from a pool of $10$ features, what's the probability that the same feature is selected at both the root and its left child?
6. You're building a tree for a mobile app with limited memory. What regularization strategy would you prioritize?
7. Calculate the expected number of unique features used in a tree with $7$ splits if you randomly select $3$ features per split.

For a detailed explanation of this question, see [Question 7: Regularization Techniques](L6_4_7_explanation.md).

## Question 8

### Problem Statement
You're analyzing learning curves from a decision tree experiment. The curves tell a story about overfitting, but you need to interpret them carefully.

#### Task
1. What does a large gap between training and validation curves indicate?
2. At what point in the learning curve does overfitting begin?
3. Predict how learning curves would change after applying cost-complexity pruning
4. Explain how to use learning curves to select optimal tree complexity
5. If your learning curves show training accuracy increasing but validation accuracy decreasing, and then both start decreasing together, what might be happening to your data?
6. You're working with a dataset that has seasonal patterns. How would this affect your learning curve interpretation?
7. Sketch a learning curve that shows both underfitting and overfitting phases.

For a detailed explanation of this question, see [Question 8: Learning Curves Analysis](L6_4_8_explanation.md).

## Question 9

### Problem Statement
You're applying the Minimum Description Length principle to decision trees. This principle suggests that the best model is the one that can be described most concisely while still explaining the data well.

#### Task
1. Explain how MDL balances model complexity and accuracy
2. For a tree with $5$ nodes, estimate the description length
3. Describe how MDL penalizes overly complex trees
4. List the main advantages of MDL-based pruning
5. If you have two trees with identical accuracy but different description lengths, and one tree has a leaf node that splits on a feature with only $2$ unique values, what does MDL suggest about this split?
6. You're building a tree for a system with limited bandwidth. How would MDL help you optimize for transmission efficiency?
7. Calculate the description length penalty for a tree that grows from $3$ to $7$ nodes.
8. Using the bias-variance decomposition formula $$\text{Variance} = E[(\hat{f}(x) - E[\hat{f}(x)])^2]$$, explain how MDL-based pruning affects the variance component of a decision tree's prediction error.

For a detailed explanation of this question, see [Question 9: MDL-Based Pruning](L6_4_9_explanation.md).

## Question 10

### Problem Statement
You're investigating C4.5's confidence factor pruning mechanism. This method uses statistical confidence to decide when to prune, but it has some interesting quirks.

#### Task
1. How does C4.5's confidence factor pruning work?
2. What statistical assumptions underlie confidence-based pruning?
3. Describe how to choose an appropriate confidence level
4. Identify scenarios where confidence-based pruning might fail
5. If you set a confidence level of $95\%$ and your tree has $100$ nodes, approximately how many nodes would you expect to be pruned by chance alone?
6. You're building a tree for a safety-critical system (e.g., autonomous driving). What confidence level would you choose and why?
7. Calculate the minimum confidence level needed to prune a node with $50$ samples and $80\%$ accuracy.

For a detailed explanation of this question, see [Question 10: Confidence-Based Pruning](L6_4_10_explanation.md).

## Question 11

### Problem Statement
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

#### Task
1. List 3 methods to detect overfitting in decision trees
2. Sketch a plot showing tree complexity vs performance
3. Apply two different pruning techniques to the same overfitted tree
4. Explain how to validate pruning decisions
5. If the company wants to keep the tree simple enough for business analysts to understand ($\leq 4$ nodes), what pruning strategy would you recommend?
6. What are the business costs of overfitting in this customer churn prediction scenario?
7. Calculate the information gain for each split and identify which splits are most likely contributing to overfitting.

For a detailed explanation of this question, see [Question 11: Overfitting Detection and Mitigation](L6_4_11_explanation.md).

## Question 12

### Problem Statement
You're conducting research on pruning effectiveness and need to design a comprehensive study.

#### Task
1. How would you test pruning methods on datasets with different characteristics?
2. Design a systematic comparison of all major pruning techniques
3. Identify appropriate metrics for evaluating pruning effectiveness
4. Provide evidence-based recommendations for pruning method selection
5. If you have limited computational resources and can only test 3 pruning methods on 2 datasets, which combinations would you choose to maximize insights?
6. You're writing a research paper. What statistical tests would you use to determine if pruning method differences are significant?
7. Design a hypothesis test to determine if one pruning method is significantly better than another.

For a detailed explanation of this question, see [Question 12: Comprehensive Pruning Study](L6_4_12_explanation.md).

## Question 13

### Problem Statement
You're building a decision tree for a high-frequency trading system where early stopping is crucial.

#### Task
1. List 3 different early stopping criteria for decision trees
2. How do you monitor validation performance during tree construction?
3. Compare early stopping vs post-pruning approaches
4. Describe how to implement patience-based early stopping
5. If your validation performance fluctuates due to market volatility, how would you modify your early stopping criteria?
6. Design an adaptive patience mechanism that increases patience when validation performance is stable.

For a detailed explanation of this question, see [Question 13: Early Stopping Strategies](L6_4_13_explanation.md).

## Question 14

### Problem Statement
You're building an automated system to find optimal pruning parameters.

#### Task
1. Design a grid search strategy for pruning parameters
2. How would you use nested cross-validation for unbiased parameter selection?
3. Compare different metrics for pruning evaluation
4. Design an automated pipeline for optimal pruning
5. If you have $1000$ samples and want to test $5$ $\alpha$ values with $5$-fold CV, how many total model fits will you need to perform?
6. You're building this for a company that needs results within 1 hour. How would you modify your optimization strategy?
7. Calculate the optimal grid spacing for $\alpha$ values if you want to test values between $0.01$ and $1.0$ with logarithmic spacing.

For a detailed explanation of this question, see [Question 14: Pruning Parameter Optimization](L6_4_14_explanation.md).

## Question 15

### Problem Statement
You're working with data from IoT sensors that have varying levels of noise depending on environmental conditions. Consider a decision tree trained on sensor data with the following characteristics:

**Dataset Information:**
- Total samples: $1000$
- Features: Temperature ($x_1$), Humidity ($x_2$), Pressure ($x_3$)
- True underlying pattern: $f(x) = \text{sign}(2x_1 + x_2 - 3)$
- Noise model: $\epsilon \sim \mathcal{N}(0, \sigma^2)$ where $\sigma^2 = 0.25$
- Training accuracy: $95\%$
- Validation accuracy: $72\%$

**Tree Structure:**
- Root split: $x_1 \leq 1.5$ (Training: $95\%$, Validation: $72\%$)
- Left subtree: $x_2 \leq 2.0$ (Training: $98\%$, Validation: $68\%$)
- Right subtree: $x_3 \leq 1.8$ (Training: $92\%$, Validation: $76\%$)

#### Task
1. Calculate the overfitting gap $\Delta = \text{Training Acc} - \text{Validation Acc}$ and explain why noise causes this gap to widen. Use the bias-variance decomposition $E[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$ to show how noise affects each component.
2. Given these pruning options, calculate which is most robust using the generalization gap metric $G = \frac{\text{Training Acc} - \text{Test Acc}}{\text{Tree Complexity}}$ where complexity is measured by $\log(\text{Depth} \times \text{Leaves})$:
   - No pruning: Training Acc = $95\%$, Test Acc = $72\%$, Depth = $8$, Leaves = $25$
   - Depth pruning (max_depth = $4$): Training Acc = $87\%$, Test Acc = $78\%$, Depth = $4$, Leaves = $12$
   - Sample pruning (min_samples = $50$): Training Acc = $89\%$, Test Acc = $75\%$, Depth = $6$, Leaves = $18$
   - Combined pruning: Training Acc = $85\%$, Test Acc = $80\%$, Depth = $3$, Leaves = $8$
3. Design mathematical functions that adjust pruning thresholds based on noise level $\sigma$. If $\sigma = 0.25$, derive optimal values for:
   - min_samples_split = $f_1(\sigma) = \max(10, \lceil 50\sigma^2 \rceil)$
   - max_depth = $f_2(\sigma) = \lfloor 8 - 4\sigma \rfloor$
   - min_impurity_decrease = $f_3(\sigma) = 0.01 + 0.1\sigma$
4. If $p = 10\%$ of the data are outliers that shift the decision boundary by $\Delta = 0.5$, calculate:
   - Expected change in training accuracy: $\Delta_{\text{train}} = p \cdot \Delta \cdot \text{Training Acc}$
   - Expected change in validation accuracy: $\Delta_{\text{val}} = p \cdot \Delta \cdot \text{Validation Acc}$
   - Optimal outlier removal threshold: $\tau = \arg\min_{\tau} \left|\Delta_{\text{train}}(\tau) - \Delta_{\text{val}}(\tau)\right|$
5. If noise increases exponentially as $\sigma(x_1) = 0.1 \cdot e^{x_1/2}$, derive the optimal pruning function that minimizes expected error:
   - Find optimal tree depth: $d^*(x_1) = \arg\min_d \left(\text{Bias}(d) + \text{Variance}(d, \sigma(x_1))\right)$
   - Calculate expected error: $E[\text{Error}] = \int_0^3 \left(\text{Bias}^2(d^*(x_1)) + \sigma^2(x_1)\right) dx_1$

6. For a fire detection system with cost matrix $C = \begin{bmatrix} 0 & 1000 \\ 100000 & 0 \end{bmatrix}$ where:
   - False negative cost = $100,000 (missed fire)
   - False positive cost = $1,000 (false alarm)
   - Base detection rate = $95\%$
   - Noise level = $0.3$
   Calculate the optimal pruning threshold $\alpha^*$ that minimizes expected cost: $\alpha^* = \arg\min_{\alpha} \sum_{i,j} C_{ij} \cdot P_{ij}(\alpha)$
7. Design a mathematical function to estimate local noise in a neighborhood of size $k = 50$. If the local variance in a region is $\sigma^2_{\text{local}} = 0.4$, calculate the optimal pruning parameters for that region:
   - Local noise estimate: $\hat{\sigma}_{\text{local}} = \sqrt{\frac{1}{k-1} \sum_{i=1}^k (x_i - \bar{x})^2}$
   - Adaptive minsamples: $n{\text{min}} = \max(10, \lceil 25\hat{\sigma}_{\text{local}}^2 \rceil)$
   - Adaptive maxdepth: $d{\text{max}} = \lfloor 6 - 3\hat{\sigma}_{\text{local}} \rfloor$
8. For a pruned tree with error decomposition:
   - Bias = $0.08$
   - Variance = $0.12$
   - Irreducible error = $0.15$
   Calculate:
   - Expected prediction error: $E[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$
   - If you reduce bias by $50\%$ to $0.04$, what's the new expected error?
   - What variance reduction $\Delta V$ is needed to achieve expected error $\leq 0.2$? Solve: $0.04^2 + (0.12 - \Delta V) + 0.15 \leq 0.2$

For a detailed explanation of this question, see [Question 15: Pruning with Noisy Data](L6_4_15_explanation.md).

## Question 16

### Problem Statement
You're building a decision tree that must balance accuracy, interpretability, and computational efficiency.

#### Task
1. How can you add randomness to tree construction?
2. How can pruning criteria adapt to local data characteristics?
3. Explain how to balance multiple objectives (accuracy, interpretability, size)
4. Describe pruning strategies for streaming/online learning scenarios
5. If you have a budget constraint of 1000 total node evaluations, how would you distribute this budget across different pruning methods?
6. You're building a tree for a medical device. How would you weight the different objectives given patient safety requirements?
7. Design a multi-objective optimization function that balances accuracy, interpretability, and efficiency.

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
1. Calculate the training accuracy for each subtree and the full tree
2. Which pruning decision would reduced error pruning make?
3. For $\alpha = 0.15$, calculate the cost-complexity for the full tree vs pruned versions
4. Analyze what the validation performance pattern suggests about overfitting
5. Write the final decision rule after optimal pruning
6. If the insurance company wants to minimize false negatives (missing high-risk customers) while keeping false positives below $20\%$, what pruning strategy would you recommend?
7. What are the regulatory implications of pruning this risk assessment tree?
8. Using the bias formula $$\text{Bias} = E[\hat{f}(x)] - f(x)$$, estimate the bias of the full tree vs pruned versions if training accuracy represents $E[\hat{f}(x)]$ and validation accuracy represents $f(x)$.

For a detailed explanation of this question, see [Question 17: Tree Surgery Simulation](L6_4_17_explanation.md).

## Question 18

### Problem Statement
You're a consultant helping companies choose pruning strategies. Create a decision matrix for evaluating different approaches.

#### Task
1. Create a $4 \times 4$ matrix comparing pruning methods vs evaluation criteria
2. Choose 4 key criteria for evaluating pruning methods
3. Design a 1-5 scoring system for each method-criteria combination
4. Recommend the best pruning method for a medical diagnosis application
5. If you have to add a cost constraint to your matrix, how would you weight it relative to accuracy and interpretability?
6. Calculate the weighted score for each pruning method if interpretability is twice as important as accuracy.

For a detailed explanation of this question, see [Question 18: Pruning Decision Matrix](L6_4_18_explanation.md).

## Question 19

### Problem Statement
Find the optimal $\alpha$ value for cost-complexity pruning given these tree options:
- Full tree: 9 nodes, error = 0.25
- Pruned tree 1: 5 nodes, error = 0.30
- Pruned tree 2: 3 nodes, error = 0.35
- Pruned tree 3: 1 node, error = 0.45

#### Task
1. Write the cost-complexity function for each option
2. Find the critical $\alpha$ value where pruning tree $1$ becomes beneficial
3. Determine the range of $\alpha$ values where pruned tree $2$ is optimal
4. If you want a tree with $\leq 4$ nodes, what $\alpha$ value should you use?
5. If you have a budget constraint that limits you to trees with $\leq 6$ nodes, what's the optimal $\alpha$ range?
6. You're building this for a mobile app with limited memory. How would you modify your $\alpha$ selection strategy?
7. Calculate the total cost for each tree option if operational costs are $2 per node per month.

For a detailed explanation of this question, see [Question 19: Alpha Selection Game](L6_4_19_explanation.md).

## Question 20

### Problem Statement
You're building a decision tree for a credit risk assessment system that must satisfy multiple constraints:
- Interpretability is crucial (regulatory requirement)
- Accuracy must be $\geq 85\%$
- Tree size should be $\leq 10$ nodes
- Training time should be $\leq 5$ minutes

#### Task
1. Which constraints are most likely to conflict?
2. Design a regularization strategy that satisfies all constraints
3. Determine which parameters to tune first and explain why
4. Design a validation plan for your strategy
5. If you can only satisfy $4$ out of $5$ constraints, which one would you relax and why?
6. Calculate the minimum training time needed if you want to test $5$ different pruning strategies.

For a detailed explanation of this question, see [Question 20: Regularization Trade-off Puzzle](L6_4_20_explanation.md).

## Question 21

### Problem Statement
Understand when different pruning methods should be applied during the tree development timeline:
Tree construction → Training → Validation → Deployment

#### Task
1. Place each pruning method on the timeline:
   - Pre-pruning
   - Post-pruning
   - Cost-complexity pruning
   - Reduced error pruning
2. Why is the timing of pruning important?
3. Identify which pruning method is most computationally efficient
4. Determine which method offers the most flexibility for parameter tuning
5. If you discover during deployment that your tree is overfitting, which pruning methods can you still apply?

For a detailed explanation of this question, see [Question 21: Pruning Timeline Challenge](L6_4_21_explanation.md).

## Question 22

### Problem Statement
Design a comprehensive tool for measuring tree complexity that can be used across different domains.

#### Task
1. Design $3$ different metrics for measuring tree complexity
2. How would you normalize these metrics for fair comparison?
3. Set reasonable thresholds for each metric
4. Apply your metrics to a tree with $7$ nodes and depth $4$
5. If you want to create a single "complexity score" that combines all three metrics, how would you weight them?
6. Calculate the complexity score for a tree that grows from $3$ to $7$ nodes over time.

For a detailed explanation of this question, see [Question 22: Tree Complexity Calculator](L6_4_22_explanation.md).

## Question 23

### Problem Statement
A research team is comparing different validation strategies for pruning decision trees. They've collected data on validation accuracy and computational cost for various approaches.

| Validation Method | Folds | Validation Accuracy | Computational Cost (minutes) | Bias Estimate | Variance Estimate |
|------------------|-------|-------------------|------------------------------|---------------|-------------------|
| Hold-out ($70/30$) | $1$     | $0.82$             | $5$                            | $0.03$          | $0.08$             |
| $5$-fold CV        | $5$     | $0.85$             | $25$                           | $0.01$          | $0.05$             |
| $10$-fold CV       | $10$    | $0.87$             | $50$                           | $0.005$         | $0.03$             |
| Leave-one-out      | $1000$  | $0.89$             | $500$                          | $0.001$         | $0.02$             |
| Nested $5$-fold    | $25$    | $0.86$             | $125$                          | $0.008$         | $0.04$             |

#### Task
1. Calculate the total error (bias² + variance) for each validation method
2. Rank the methods by validation accuracy per computational minute
3. If you have $30$ minutes and need accuracy $\geq 0.85$, which method would you choose?
4. Plot validation accuracy vs computational cost and identify the Pareto frontier
5. If you use nested cross-validation, how would you correct for the bias introduced by parameter selection?
6. Calculate the minimum number of samples needed for each fold to ensure statistical significance with $90\%$ confidence
7. Using the formula $$E[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$, calculate the expected prediction error for each validation method. Assume irreducible error = $0.02$. Which method has the lowest expected prediction error?
8. For each validation method, calculate the bias-to-variance ratio. What does this ratio tell you about the method's performance characteristics?

For a detailed explanation of this question, see [Question 23: Pruning Validation Strategy Analysis](L6_4_23_explanation.md).

## Question 24

### Problem Statement
A machine learning engineer is tuning regularization parameters for a decision tree system. They've collected performance data across different parameter combinations.

| Max Depth | Min Samples Leaf | Min Impurity Decrease | Training Accuracy | Validation Accuracy | Tree Size (nodes) | Training Time (sec) |
|-----------|------------------|----------------------|-------------------|-------------------|-------------------|---------------------|
| $3$         | $10$               | $0.01$                 | $0.78$             | $0.76$             | $7$                 | $2.1$                 |
| $3$         | $20$               | $0.05$                 | $0.75$             | $0.74$             | $5$                 | $1.8$                 |
| $5$         | $10$               | $0.01$                 | $0.89$             | $0.82$             | $15$                | $4.2$                 |
| $5$         | $20$               | $0.05$                 | $0.85$             | $0.80$             | $11$                | $3.9$                 |
| $7$         | $10$               | $0.01$                 | $0.95$             | $0.78$             | $23$                | $6.8$                 |
| $7$         | $20$               | $0.05$                 | $0.91$             | $0.76$             | $17$                | $6.1$                 |
| $10$        | $10$               | $0.01$                 | $0.98$             | $0.75$             | $31$                | $9.5$                 |
| $10$        | $20$               | $0.05$                 | $0.94$             | $0.73$             | $25$                | $8.7$                 |

#### Task
1. Identify which parameter combinations show clear signs of overfitting
2. Find the parameter combination that maximizes validation accuracy while keeping training time under $5$ seconds
3. Calculate the validation accuracy per node for each configuration
4. Which parameter has the strongest effect on validation accuracy?
5. Create a scatter plot of tree size vs validation accuracy and identify the optimal region
6. For a real-time system that needs accuracy $\geq 0.80$ and response time $\leq 100$ms, which configuration would you choose?
7. If you want to add a new regularization parameter (feature subsampling ratio), how would you modify this analysis to include it?
8. For the overfitting configurations (max_depth=$7$ and max_depth=$10$), estimate the bias and variance components. Use the formula $$\text{Bias} = E[\hat{f}(x)] - f(x)$$ where $f(x)$ is the true function. If training accuracy represents $E[\hat{f}(x)]$ and validation accuracy represents $f(x)$, calculate bias and variance for each overfitting case.
9. Explain how increasing tree complexity affects the bias-variance trade-off in your decision trees, using the data from the table.

For a detailed explanation of this question, see [Question 24: Regularization Parameter Tuning Analysis](L6_4_24_explanation.md).

## Question 25

### Problem Statement
A financial services company needs to build an interpretable decision tree for loan approval that can be explained to regulators. They've collected data on how different pruning strategies affect both accuracy and interpretability.

| Pruning Method | Tree Size (nodes) | Max Depth | Training Accuracy | Validation Accuracy | Interpretability Score | Regulatory Compliance |
|----------------|-------------------|-----------|-------------------|-------------------|----------------------|----------------------|
| No Pruning     | $31$                | $8$         | $0.98$             | $0.82$             | $2.1$                  | Non-compliant        |
| Pre-pruning    | $15$                | $4$         | $0.89$             | $0.85$             | $7.8$                  | Fully compliant      |
| Post-pruning   | $19$                | $6$         | $0.92$             | $0.84$             | $6.2$                  | Needs review         |
| Cost-complexity| $12$                | $3$         | $0.85$             | $0.86$             | $8.5$                  | Fully compliant      |
| Reduced Error  | $22$                | $5$         | $0.90$             | $0.83$             | $5.9$                  | Needs review         |

*Interpretability Score: 1-10 scale (10 = most interpretable), Regulatory Compliance: Fully compliant, Needs review, Non-compliant*

#### Task
1. Which pruning methods meet regulatory requirements for interpretability?
2. Plot tree size vs validation accuracy and identify the optimal trade-off point
3. If non-compliance costs $50,000 per violation, calculate the risk cost for each method
4. Find the pruning method that maximizes validation accuracy while maintaining full regulatory compliance
5. If the company processes $1000$ loans per month, calculate the monthly cost of false decisions for each method
6. Design a pruning strategy that can adapt to changing regulatory requirements
7. If interpretability score decreases exponentially with tree depth, what's the optimal depth for regulatory compliance?

For a detailed explanation of this question, see [Question 25: Pruning for Interpretability Analysis](L6_4_25_explanation.md).

## Question 26

### Problem Statement
You're a decision tree gardener with a special challenge: you have exactly 100 seeds (training samples) and must grow a tree that's both beautiful (accurate) and manageable (interpretable). Each seed can grow into a leaf, and each split costs 2 seeds.

#### Task
1. If you want a tree with maximum depth $3$, what's the maximum number of leaves possible?
2. Design a pruning strategy that ensures each leaf has at least $8$ seeds
3. Calculate the minimum impurity decrease threshold needed to justify splitting a node with $25$ seeds
4. If your tree naturally grows to $15$ leaves but you can only maintain $8$, which leaves would you prune first?
5. Create a "gardening schedule" showing when to apply pre-pruning vs post-pruning techniques

For a detailed explanation of this question, see [Question 26: Tree Gardener's Dilemma](L6_4_26_explanation.md).

## Question 27

### Problem Statement
You're a detective investigating suspicious tree behavior. A company's decision tree suddenly started making terrible predictions after a "routine update." Here are the clues:

**Before Update:**
- Tree size: $12$ nodes, Validation accuracy: $87\%$
- Training accuracy: $89\%$

**After Update:**
- Tree size: $31$ nodes, Validation accuracy: $72\%$
- Training accuracy: $98\%$

#### Task
1. What crime has been committed? (Identify the problem)
2. List 3 suspects (pruning methods that could have failed)
3. What evidence suggests overfitting?
4. Design an investigation plan to restore the tree's performance
5. If you had to choose between pre-pruning and post-pruning to fix this, which would you recommend and why?

For a detailed explanation of this question, see [Question 27: Pruning Detective Mystery](L6_4_27_explanation.md).

## Question 28

### Problem Statement
Your decision tree is competing in a "Tree Fitness Challenge" where it must balance multiple performance metrics. The scoring system is:
- **Accuracy Score**: 40% of total score
- **Interpretability Score**: 30% of total score  
- **Efficiency Score**: 20% of total score
- **Robustness Score**: 10% of total score

#### Task
1. Design a fitness function that combines all four metrics
2. If your tree has $85\%$ accuracy, $7/10$ interpretability, $8/10$ efficiency, and $6/10$ robustness, calculate its total fitness score
3. Your tree is too complex ($15$ nodes). Design a pruning strategy to improve its fitness
4. Calculate the expected fitness improvement after pruning
5. What's the optimal tree size for maximum fitness in this competition?

For a detailed explanation of this question, see [Question 28: Tree Fitness Challenge](L6_4_28_explanation.md).

## Question 29

### Problem Statement
You're a "Tree Chef" creating pruning recipes for different scenarios. Each recipe must include the right ingredients (pruning methods) and cooking instructions (parameters).

#### Task
1. **Recipe 1**: Create a pruning recipe for a medical diagnosis tree that must be interpretable and accurate
2. **Recipe 2**: Create a pruning recipe for a real-time fraud detection system with limited memory
3. **Recipe 3**: Create a pruning recipe for an educational tool that teaches students about decision trees
4. For each recipe, specify the main pruning method, key parameters, and expected outcome
5. Which recipe would be most expensive to implement and why?

For a detailed explanation of this question, see [Question 29: Pruning Recipe Creation](L6_4_29_explanation.md).

## Question 30

### Problem Statement
You're simulating the evolution of decision trees over time. Each generation, trees can either grow (add nodes) or prune (remove nodes) based on their "fitness" in the environment.

#### Task
1. Design an evolution rule: trees with validation accuracy $> 85\%$ should grow, trees with validation accuracy $< 75\%$ should prune
2. If a tree starts with $5$ nodes and $80\%$ validation accuracy, predict its size after $3$ generations
3. What happens to trees that are "just right" (validation accuracy between $75\%$-$85\%$)?
4. Design a mutation mechanism that occasionally tries different pruning strategies
5. If the environment becomes more complex (more noise), how would this affect the optimal tree size?

For a detailed explanation of this question, see [Question 30: Tree Evolution Simulation](L6_4_30_explanation.md).

## Question 31

### Problem Statement
Evaluate whether each of the following statements about tree pruning and regularization is TRUE or FALSE. Provide a brief justification for each answer.

#### Task
1. Pre-pruning always produces smaller trees than post-pruning for the same dataset
2. Cost-complexity pruning with $\alpha = 0$ will always result in the full unpruned tree
3. Reduced error pruning requires a separate validation set to make pruning decisions
4. Cross-validation can completely eliminate bias in pruning parameter selection
5. L1 regularization can be directly applied to decision tree nodes like in linear models
6. Post-pruning is computationally more expensive than pre-pruning during training
7. The minimum impurity decrease threshold should always be set to $0.01$ for optimal results
8. Early stopping based on validation accuracy will always prevent overfitting
9. Tree depth limits are more effective than leaf count limits for controlling complexity
10. Pruning decisions made on training data are always reliable for generalization

For a detailed explanation of this question, see [Question 31: Tree Pruning Properties](L6_4_31_explanation.md).

## Question 32

### Problem Statement
Match each pruning technique on the left with its correct characteristic on the right:

#### Task
1. Pre-pruning with max_depth                    A) Most computationally expensive
2. Reduced error pruning                         B) Requires validation set
3. Cost-complexity pruning                       C) Stops tree growth early
4. Confidence-based pruning                      D) Balances error and complexity
5. MDL-based pruning                             E) Uses statistical significance
6. Post-pruning with validation                  F) Minimizes description length

For a detailed explanation of this question, see [Question 32: Pruning Technique Matching](L6_4_32_explanation.md).

## Question 33

### Problem Statement
Design a "Pruning Strategy Selection Game" where you must choose the most appropriate pruning approach for different scenarios.

#### Task
For each scenario below, select the most suitable pruning method and explain your reasoning in 1-2 sentences:

1. **Real-time fraud detection**: System must make predictions in $< 10$ms, memory is limited to $100$MB, interpretability is not required
2. **Medical diagnosis tool**: Doctors need to understand the decision process, accuracy is critical, training time can be up to $1$ hour
3. **Educational demonstration**: Students need to see how pruning affects tree structure, dataset is small ($< 100$ samples), visual clarity is important
4. **Production recommendation system**: Must handle $1$M+ users, accuracy is important but not critical, maintenance costs should be minimized
5. **Research prototype**: Exploring different pruning approaches, computational resources are unlimited, need to compare multiple methods

For a detailed explanation of this question, see [Question 33: Pruning Strategy Selection](L6_4_33_explanation.md).

## Question 34

### Problem Statement
Apply pruning techniques to this decision tree dataset about loan approval:

| Income | Age | Credit_Score | Employment_Years | Loan_Approved |
|--------|-----|--------------|------------------|---------------|
| High   | 25  | 750          | 2                | Yes           |
| High   | 35  | 680          | 8                | Yes           |
| Medium | 28  | 720          | 3                | No            |
| Low    | 45  | 650          | 15               | No            |
| Medium | 32  | 700          | 5                | Yes           |
| High   | 29  | 780          | 1                | No            |
| Low    | 38  | 600          | 12               | No            |
| Medium | 41  | 690          | 7                | No            |

#### Task
1. Calculate the Gini impurity of the entire dataset
2. If the tree splits on Income first, calculate the weighted Gini impurity after the split
3. For a pre-pruning threshold of min_samples_leaf = $2$, would this split be allowed?
4. If using cost-complexity pruning with $\alpha = 0.1$, calculate the cost for a tree with $5$ nodes and error rate $0.25$
5. Which pruning method would be most appropriate for this dataset and why?

For a detailed explanation of this question, see [Question 34: Loan Approval Pruning Application](L6_4_34_explanation.md).

## Question 35

### Problem Statement
Which of the following scenarios would benefit most from each pruning approach? Choose the best match.

#### Task
For each scenario, select the most suitable pruning method and justify your choice:

1. **Financial risk assessment**: Must comply with regulatory requirements, decisions must be explainable, false positives are expensive
2. **Real-time gaming**: Player behavior changes rapidly, predictions must be made in milliseconds, accuracy can be sacrificed for speed
3. **Academic research**: Comparing different pruning methods, need statistical rigor, computational resources are available
4. **Customer service chatbot**: Must explain decisions to users, training data is limited, response time should be under 2 seconds
5. **Industrial quality control**: High-stakes decisions, false negatives are very expensive, model updates are infrequent

For a detailed explanation of this question, see [Question 35: Pruning Method Scenarios](L6_4_35_explanation.md).

## [⭐] Question 36

### Problem Statement
A university is using a decision tree to predict student grades based on class participation and assignment completion. The dataset shows how different combinations of factors affect final grades:

| Class | A | B | C | Grade |
|-------|---|---|---|-------|
| ALL   | T | F | T | 74    |
| ALL   | T | T | F | 23    |
| SOME  | T | F | T | 61    |
| SOME  | T | F | F | 74    |
| SOME  | F | T | T | 25    |
| SOME  | F | T | F | 61    |
| NONE  | F | T | T | 54    |
| NONE  | F | F | F | 42    |

*Note: A, B, C are binary features (T=True, F=False), Class has three values (ALL, SOME, NONE), and Grade is the target variable*

#### Task
1. Calculate the mean grade for each class participation level (ALL, SOME, NONE) and identify which level has the highest average performance
2. For a decision tree using Gini impurity, calculate the information gain for splitting on feature A vs feature B vs feature C. Which feature would be chosen as the root node?
3. If you build a decision tree with max_depth = $2$ and min_samples_leaf = $1$, what would be the tree structure? Draw the tree showing the splits and leaf node predictions
4. If the tree achieves $100\%$ training accuracy but only $65\%$ validation accuracy on new student data, what pruning strategy would you recommend?
5. For $\alpha = 0.1$, calculate the cost-complexity function $R_\alpha(T) = R(T) + \alpha|T|$ for:
   - Full tree: $7$ nodes, training error = $0.0$
   - Pruned tree: $3$ nodes, training error = $0.125$
6. If the university needs to explain grade predictions to students and parents, what maximum tree depth would you recommend? Justify your answer considering the trade-off between accuracy and interpretability
7. Design a cross-validation approach for this dataset that accounts for the small sample size ($8$ samples) while still providing reliable pruning parameter selection
8. If the university processes $1000$ students per semester and incorrect grade predictions cost $50 in administrative overhead, calculate the potential cost savings from implementing an optimal pruning strategy

For a detailed explanation of this question, see [Question 36: Student Grade Prediction Pruning](L6_4_36_explanation.md).