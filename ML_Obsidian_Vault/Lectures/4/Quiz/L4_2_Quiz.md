# Lecture 4.2: The Perceptron Algorithm Quiz

## Overview
This quiz contains 16 questions covering different topics from section 4.2 of the lectures on The Perceptron Algorithm, including the model structure, learning rules, convergence, limitations, and applications of the perceptron.

## Question 1

### Problem Statement
Consider the historical context and basic structure of the perceptron model.

#### Task
1. [🔍] Define the perceptron model in one sentence
2. How does the perceptron relate to biological neurons? Describe in two sentences
3. What are the key limitations of the perceptron as described by Minsky and Papert?
4. [🔍] Why is the perceptron still studied today despite these limitations? Answer in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Perceptron History and Structure](L4_2_1_explanation.md).

## Question 2

### Problem Statement
Consider a binary perceptron with weight vector $w = [w_1, w_2, w_0]^T = [2, -1, -3]^T$, where $w_0$ is the bias term.

#### Task
1. [📚] Write the equation of the decision boundary in the 2D feature space
2. [📚] Sketch this decision boundary
3. [📚] For the points $(2, 1)$ and $(1, 3)$, determine which class each point would be assigned to
4. [📚] If the perceptron makes a mistake on the point $(1, 3)$ with true label $y = 1$, what would the updated weight vector be if the learning rate $\eta = 1$?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Binary Perceptron](L4_2_2_explanation.md).

## Question 3

### Problem Statement
Consider the geometric interpretation of the perceptron learning process.

#### Task
1. [🔍] Explain geometrically what happens when the perceptron weights are updated. Answer in two sentences
2. [🔍] What geometric condition must be satisfied for the perceptron algorithm to converge? Answer in one sentence
3. [🔍] Draw a simple 2D example of a dataset that is not linearly separable
4. [🔍] For the dataset you drew, explain what would happen if you ran the perceptron algorithm on it. Answer in two sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Geometric Interpretation](L4_2_3_explanation.md).

## Question 4

### Problem Statement
Consider the perceptron learning algorithm applied to the following dataset:

| $x_1$ | $x_2$ | $y$ (target) |
|-------|-------|--------------|
| 1     | 1     | 1            |
| 2     | 2     | 1            |
| 1     | -1    | -1           |
| -1    | 1     | -1           |
| -2    | -1    | -1           |

#### Task
1. [📚] Plot these points in a 2D coordinate system and visually verify that they are linearly separable
2. [📚] Initialize the weight vector to $w = [w_1, w_2, w_0]^T = [0, 0, 0]^T$ and apply the perceptron algorithm with learning rate $\eta = 1$
3. [📚] Show your work step-by-step for at least 3 iterations or until convergence, indicating which points are misclassified at each step and how the weights are updated
4. [📚] Draw the final decision boundary obtained after the algorithm converges

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Perceptron Algorithm](L4_2_4_explanation.md).

## Question 5

### Problem Statement
Consider the perceptron convergence theorem.

#### Task
1. [🔍] State the perceptron convergence theorem in one sentence
2. [🔍] What conditions must be satisfied for the perceptron convergence theorem to apply?
3. [📚] If a dataset with $n$ points is linearly separable with a margin $\gamma$ and the features are bounded by $\|x\| \leq R$, what is the maximum number of updates the perceptron algorithm will make?
4. [🔍] Why does the perceptron algorithm not converge for linearly non-separable data? Answer in one or two sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Convergence Theorem](L4_2_5_explanation.md).

## Question 6

### Problem Statement
Consider different initializations for the perceptron algorithm.

#### Task
1. [🔍] How does the choice of initial weights affect the final solution of the perceptron algorithm? Answer in one sentence
2. [🔍] List two common strategies for initializing the weights of a perceptron
3. [📚] For a linearly separable dataset, if we initialize the weights to $w = [1, 1, 0]^T$ instead of zeros, would the perceptron algorithm still converge to a solution? Explain why or why not
4. [🔍] Does the choice of learning rate affect the final solution of the perceptron algorithm (assuming convergence)? Answer in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Initialization Strategies](L4_2_6_explanation.md).

## Question 7

### Problem Statement
Consider the famous XOR problem, where we have the following dataset:

| $x_1$ | $x_2$ | $y$ (target) |
|-------|-------|--------------|
| 0     | 0     | 0            |
| 0     | 1     | 1            |
| 1     | 0     | 1            |
| 1     | 1     | 0            |

#### Task
1. [📚] Plot these points in a 2D coordinate system
2. [📚] Prove that this dataset is not linearly separable by showing that no linear decision boundary can correctly classify all points
3. [🔍] Explain why the perceptron algorithm would fail on this dataset. Answer in one sentence
4. [🔍] Suggest two approaches that could be used to solve the XOR problem with a perceptron-like model. Briefly explain each approach in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Perceptron Limitations](L4_2_7_explanation.md).

## Question 8

### Problem Statement
Consider the perceptron learning rule update:
$$w_{t+1} = w_t + \eta \cdot y \cdot x$$
where $w_t$ is the weight vector at time $t$, $\eta$ is the learning rate, $y$ is the true label, and $x$ is the feature vector.

#### Task
1. [📚] If $w = [1, 2, -1]^T$, $x = [2, 0, 1]^T$ (including a bias term), $y = -1$, and $\eta = 0.5$, calculate the updated weight vector after one update
2. [📚] For the same example, would the perceptron make a prediction error before the update? Show your calculation
3. [🔍] How does the magnitude of the learning rate affect the learning process? Answer in two sentences
4. [🔍] Can the perceptron learning rule be viewed as a form of gradient descent? Explain why or why not in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Perceptron Learning Rule](L4_2_8_explanation.md).

## Question 9

### Problem Statement
Consider the kernelized perceptron, which allows the algorithm to learn non-linear decision boundaries.

#### Task
1. [🔍] Explain the "kernel trick" in one or two sentences
2. [🔍] What is the advantage of using a kernelized perceptron over standard perceptron? Answer in one sentence
3. [🔍] Give an example of a kernel function and explain what type of decision boundary it can produce
4. [📚] If the original perceptron decision function is $f(x) = \text{sign}(w^T x + b)$, what is the kernelized version of this function?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Kernelized Perceptron](L4_2_9_explanation.md).

## Question 10

### Problem Statement
Consider applying the perceptron algorithm to the following dataset:

| $x_1$ | $x_2$ | $y$ (target) |
|-------|-------|--------------|
| 1     | 2     | 1            |
| 2     | 1     | 1            |
| 3     | 3     | 1            |
| 6     | 4     | -1           |
| 5     | 6     | -1           |
| 7     | 5     | -1           |

#### Task
1. [📚] Plot these points in a 2D coordinate system
2. [📚] Starting with $w = [0, 0, 0]^T$ and $\eta = 1$, perform the first two updates of the perceptron algorithm, showing your calculations
3. [📚] Draw an approximate final decision boundary that would separate these classes
4. [🔍] Would a single perceptron be able to learn a circular decision boundary? Explain why or why not in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Perceptron Examples](L4_2_10_explanation.md).

## Question 11

### Problem Statement
Consider the effect of learning rate on the perceptron algorithm's convergence and performance.

#### Task
1. [🔍] How does choosing an excessively high learning rate affect the perceptron algorithm? Answer in two sentences
2. [📚] For the dataset below, apply the perceptron algorithm with two different learning rates: $\eta = 0.1$ and $\eta = 2.0$. Show the first three iterations for each case.

| $x_1$ | $x_2$ | $y$ (target) |
|-------|-------|--------------|
| 1     | 1     | 1            |
| 3     | 1     | 1            |
| 2     | 4     | 1            |
| -1    | 1     | -1           |
| 0     | -2    | -1           |
| -2    | -1    | -1           |

3. [📚] Compare the trajectories of the weight vectors for both learning rates and explain which one would likely converge faster to a stable solution
4. [🔍] Is there a maximum learning rate beyond which the perceptron is guaranteed to not converge, even for linearly separable data? Explain your answer in one or two sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: Learning Rate Effects](L4_2_11_explanation.md).

## Question 12

### Problem Statement
Consider the differences between online (stochastic) perceptron and batch perceptron algorithms.

#### Task
1. [🔍] Explain the key difference between online and batch perceptron algorithms in one or two sentences
2. [🔍] List one advantage and one disadvantage of online perceptron compared to batch perceptron
3. [📚] For the dataset below, perform the first 2 iterations of both online perceptron and batch perceptron with initial weights $w = [0, 0, 0]^T$ and learning rate $\eta = 1$. Show your calculations for each.

| $x_1$ | $x_2$ | $y$ (target) |
|-------|-------|--------------|
| 2     | 1     | 1            |
| 0     | 3     | 1            |
| -1    | 0     | -1           |
| -2    | -2    | -1           |

4. [🔍] In what scenarios would you prefer to use online perceptron over batch perceptron? Answer in one or two sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: Online vs Batch Perceptron](L4_2_12_explanation.md).

## Question 13

### Problem Statement
Consider the Voted Perceptron algorithm, an extension to the standard perceptron.

#### Task
1. [🔍] Explain how the Voted Perceptron algorithm differs from the standard perceptron in one or two sentences
2. [🔍] What problem does the Voted Perceptron algorithm attempt to solve? Answer in one sentence
3. [📚] If we have the following sequence of perceptron weight vectors during training with their survival times (number of correct predictions before a mistake):
   - $w_1 = [1, 1, 0]^T$ with survival time $c_1 = 2$
   - $w_2 = [1, 2, -1]^T$ with survival time $c_2 = 4$
   - $w_3 = [2, 2, -1]^T$ with survival time $c_3 = 1$
   
   How would the Voted Perceptron make a prediction for a new point $x = [2, 1, 1]^T$? Show your calculations.

4. [🔍] How does the Voted Perceptron potentially improve generalization compared to the standard perceptron? Answer in one or two sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 13: Voted Perceptron](L4_2_13_explanation.md).

## Question 14

### Problem Statement
Consider the Averaged Perceptron algorithm, another popular variant of the perceptron.

#### Task
1. [🔍] Describe the Averaged Perceptron algorithm in one or two sentences
2. [🔍] How does the Averaged Perceptron differ from the Voted Perceptron? Answer in one sentence
3. [📚] Given the following sequence of weight vectors from an online perceptron training process:
   - $w_1 = [0, 0, 0]^T$
   - $w_2 = [1, 0, 0]^T$
   - $w_3 = [1, 1, 0]^T$
   - $w_4 = [1, 1, -1]^T$
   - $w_5 = [2, 1, -1]^T$
   
   Calculate the averaged weight vector. Show your work.

4. [🔍] Why might the Averaged Perceptron be more robust than the standard perceptron in practice? Answer in one or two sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 14: Averaged Perceptron](L4_2_14_explanation.md).

## Question 15

### Problem Statement
Consider the Pocket Algorithm, a modification of the perceptron algorithm for dealing with non-linearly separable data.

#### Task
1. [🔍] Describe how the Pocket Algorithm works in one or two sentences
2. [🔍] What problem does the Pocket Algorithm attempt to solve? Answer in one sentence
3. [📚] For the following non-linearly separable dataset:

| $x_1$ | $x_2$ | $y$ (target) |
|-------|-------|--------------|
| 1     | 1     | 1            |
| 2     | 2     | 1            |
| 3     | 1     | 1            |
| 2     | 3     | -1           |
| 1     | 2     | -1           |
| 3     | 3     | 1            |

   Run 3 iterations of the Pocket Algorithm with initial weights $w = [0, 0, 0]^T$ and learning rate $\eta = 1$. Show your work, including which weight vector is kept in the "pocket" at each step.

4. [🔍] In what practical scenarios would the Pocket Algorithm be particularly useful? Answer in one or two sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 15: Pocket Algorithm](L4_2_15_explanation.md).

## Question 16

### Problem Statement
Consider the Margin Perceptron, a variant that tries to find a decision boundary with a large margin.

#### Task
1. [🔍] Explain how the Margin Perceptron differs from the standard perceptron in one or two sentences
2. [🔍] How does the update rule for the Margin Perceptron differ from the standard perceptron? Answer using mathematical notation
3. [📚] Given a dataset where all points are correctly classified but some points are very close to the decision boundary, would a standard perceptron continue to update its weights? Explain why or why not and contrast this with the behavior of the Margin Perceptron
4. [📚] For the dataset below with margin parameter $\gamma = 1.5$, perform 2 iterations of the Margin Perceptron algorithm with initial weights $w = [1, 0, 0]^T$ and learning rate $\eta = 1$. Show your calculations.

| $x_1$ | $x_2$ | $y$ (target) |
|-------|-------|--------------|
| 2     | 2     | 1            |
| 3     | 1     | 1            |
| -2    | -1    | -1           |
| -1    | -2    | -1           |

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 16: Margin Perceptron](L4_2_16_explanation.md). 