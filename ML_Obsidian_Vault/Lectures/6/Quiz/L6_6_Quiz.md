# Lecture 6.6: Tree Pruning Techniques Quiz

## Overview
This quiz contains 5 questions covering different topics from section 6.6 of the lectures on Tree Pruning Techniques, including pruning methods, cost-complexity pruning, reduced error pruning, and pruning evaluation.

## Question 1

### Problem Statement
Pruning removes unnecessary branches from decision trees to improve generalization.

#### Task
1. [🔍] What is the main goal of tree pruning?
2. [🔍] What is the difference between pre-pruning and post-pruning?
3. [🔍] When in the tree-building process does each type occur?
4. [🔍] Why is post-pruning generally preferred over pre-pruning?

For a detailed explanation of this question, see [Question 1: Pruning Overview](L6_6_1_explanation.md).

## Question 2

### Problem Statement
Cost-complexity pruning balances tree accuracy and complexity.

#### Task
1. [📚] What is the cost-complexity function?
2. [📚] How does the $\alpha$ parameter control pruning?
3. [📚] What happens when $\alpha = 0$?
4. [📚] What happens when $\alpha$ is very large?

For a detailed explanation of this question, see [Question 2: Cost-Complexity Pruning](L6_6_2_explanation.md).

## Question 3

### Problem Statement
Reduced error pruning uses validation data to guide pruning decisions.

#### Task
1. [🔍] How does reduced error pruning work?
2. [🔍] What is the pruning criterion in this method?
3. [🔍] Why is a separate validation set needed?
4. [🔍] What are the advantages of reduced error pruning?

For a detailed explanation of this question, see [Question 3: Reduced Error Pruning](L6_6_3_explanation.md).

## Question 4

### Problem Statement
Consider a subtree that can be pruned to a leaf node:

| Node Type | Training Samples | Misclassifications | Error Rate |
|-----------|------------------|-------------------|------------|
| Subtree    | 100             | 15                | 15%        |
| Leaf       | 100             | 20                | 20%        |

#### Task
1. [📚] Should this subtree be pruned according to reduced error pruning?
2. [📚] What is the error reduction if pruning occurs?
3. [📚] How would cost-complexity pruning evaluate this decision?
4. [📚] What factors besides error rate should be considered?

For a detailed explanation of this question, see [Question 4: Pruning Decision Analysis](L6_6_4_explanation.md).

## Question 5

### Problem Statement
Different pruning methods have different characteristics:

| Method | Validation Required | Computational Cost | Pruning Quality |
|--------|-------------------|-------------------|-----------------|
| Pre-pruning | No | Low | Variable |
| Post-pruning | Yes | High | Good |
| Cost-complexity | Yes | Medium | Good |

#### Task
1. [📚] **Method A**: When would you use pre-pruning?
2. [📚] **Method B**: When would you use post-pruning?
3. [📚] **Method C**: When would you use cost-complexity pruning?
4. [📚] How do you choose the best pruning method for your problem?

For a detailed explanation of this question, see [Question 5: Pruning Method Selection](L6_6_5_explanation.md).
