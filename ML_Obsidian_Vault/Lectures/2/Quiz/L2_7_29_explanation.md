# Question 29: Minimizing Expected Loss in Classification

## Problem Statement
Consider a simple classification problem where we need to predict whether a student will pass (class 1) or fail (class 0) an exam based on their study hours. Our model produces a probability $p$ that the student will pass.

We can make decisions using different loss functions:
- 0-1 Loss: $L(y, \hat{y}) = \mathbf{1}(y \neq \hat{y})$ (equal penalty for all errors)
- Asymmetric Loss: $L(y, \hat{y}) = \begin{cases} 2 & \text{if } y=1, \hat{y}=0 \text{ (missed opportunity)} \\ 1 & \text{if } y=0, \hat{y}=1 \text{ (wasted effort)} \\ 0 & \text{if } y=\hat{y} \text{ (correct decision)} \end{cases}$

### Task
1. For the 0-1 loss function, derive the decision rule that minimizes the expected loss (Bayes risk). At what probability threshold should we predict class 1?
2. For the asymmetric loss function, derive the decision rule that minimizes the expected loss. At what probability threshold should we predict class 1?
3. Explain why these thresholds differ and what this means in practical terms for the student.
4. If our model gives a probability $p = 0.4$ that a student will pass, what decision would minimize expected loss under each loss function?

## Understanding the Problem

In this problem, we are faced with a binary classification task where we need to decide whether to predict if a student will pass or fail an exam. Instead of simply having class labels, we have a probabilistic model that outputs $p$, the probability that the student will pass.

The crucial aspect of this problem is that we need to make decisions that minimize the expected loss (also known as Bayes risk) under different loss functions:

1. The 0-1 loss function assigns an equal penalty (1) for any misclassification and no penalty (0) for correct classifications. This is the most common loss function used in classification problems.

2. The asymmetric loss function assigns different penalties for different types of errors:
   - A penalty of 2 for missed opportunities (predicting fail when the student would have passed)
   - A penalty of 1 for wasted effort (predicting pass when the student would have failed)
   - No penalty for correct predictions

The asymmetric loss reflects real-world scenarios where different types of errors have different consequences. In this case, missing an opportunity (not encouraging a student who would have passed) is considered worse than wasting effort (encouraging a student who ends up failing).

To find the optimal decision rules, we need to calculate the expected loss for each possible action and choose the action that minimizes this expected loss.

## Solution

### Step 1: Deriving the Decision Rule for 0-1 Loss Function

Let's start by clearly defining the 0-1 loss function in tabular form:

| True Class (y) | Predicted Class (ŷ) | Loss L(y,ŷ) |
|----------------|---------------------|-------------|
| 0 (fail)       | 0 (fail)            | 0           |
| 0 (fail)       | 1 (pass)            | 1           |
| 1 (pass)       | 0 (fail)            | 1           |
| 1 (pass)       | 1 (pass)            | 0           |

Under the 0-1 loss function, the loss is 1 if our prediction is incorrect and 0 if it's correct. We want to minimize the expected loss, which depends on our prediction and the true class.

Let's denote:
- $y$ as the true class (1 for pass, 0 for fail)
- $\hat{y}$ as our prediction
- $p = P(y=1)$ as the probability that the student will pass
- $1-p = P(y=0)$ as the probability that the student will fail

For predicting pass ($\hat{y}=1$):
- Case 1: Student fails (y=0) but we predicted pass ($\hat{y}=1$)
  - Probability: $P(y=0) = 1-p$
  - Loss: $L(0,1) = 1$
- Case 2: Student passes (y=1) and we predicted pass ($\hat{y}=1$)
  - Probability: $P(y=1) = p$
  - Loss: $L(1,1) = 0$

The expected loss (Bayes risk) when we predict pass ($\hat{y}=1$) is:
$$R(\hat{y}=1) = P(y=0) \cdot L(0,1) + P(y=1) \cdot L(1,1)$$
$$R(\hat{y}=1) = (1-p) \cdot 1 + p \cdot 0 = 1-p$$

For predicting fail ($\hat{y}=0$):
- Case 1: Student fails (y=0) and we predicted fail ($\hat{y}=0$)
  - Probability: $P(y=0) = 1-p$
  - Loss: $L(0,0) = 0$
- Case 2: Student passes (y=1) but we predicted fail ($\hat{y}=0$)
  - Probability: $P(y=1) = p$
  - Loss: $L(1,0) = 1$

The expected loss when we predict fail ($\hat{y}=0$) is:
$$R(\hat{y}=0) = P(y=0) \cdot L(0,0) + P(y=1) \cdot L(1,0)$$
$$R(\hat{y}=0) = (1-p) \cdot 0 + p \cdot 1 = p$$

To minimize the expected loss, we compare the two risks and choose the action with the lower risk:
- Predict pass ($\hat{y}=1$) if $R(\hat{y}=1) < R(\hat{y}=0)$
- Predict fail ($\hat{y}=0$) if $R(\hat{y}=0) < R(\hat{y}=1)$

This gives us:
- Predict pass ($\hat{y}=1$) if $1-p < p$
- Predict fail ($\hat{y}=0$) if $p < 1-p$

Solving the first inequality:
$$1-p < p$$
$$1 < 2p$$
$$0.5 < p$$

Therefore, the decision rule for the 0-1 loss function is: Predict that the student will pass ($\hat{y}=1$) if $p > 0.5$, otherwise predict that the student will fail ($\hat{y}=0$).

At exactly $p = 0.5$, both decisions have equal expected loss (0.5), so either decision is optimal.

![Expected Loss for 0-1 Loss Function](../Images/L2_7_Quiz_29/zero_one_loss.png)

In this graph:
- The blue line represents the expected loss when predicting pass ($\hat{y}=1$), which is $1-p$
- The red line represents the expected loss when predicting fail ($\hat{y}=0$), which is $p$
- The vertical dashed line marks the decision threshold at $p = 0.5$
- The red shaded region (left) indicates that it's optimal to predict fail when $p < 0.5$
- The blue shaded region (right) indicates that it's optimal to predict pass when $p > 0.5$

### Step 2: Deriving the Decision Rule for Asymmetric Loss Function

Now let's analyze the asymmetric loss function in tabular form:

| True Class (y) | Predicted Class (ŷ) | Loss L(y,ŷ) | Interpretation     |
|----------------|---------------------|-------------|-------------------|
| 0 (fail)       | 0 (fail)            | 0           | Correct prediction |
| 0 (fail)       | 1 (pass)            | 1           | Wasted effort     |
| 1 (pass)       | 0 (fail)            | 2           | Missed opportunity |
| 1 (pass)       | 1 (pass)            | 0           | Correct prediction |

For this asymmetric loss function, we follow a similar approach but use the different loss values specified in the problem.

For predicting pass ($\hat{y}=1$):
- Case 1: Student fails (y=0) but we predicted pass ($\hat{y}=1$)
  - Probability: $P(y=0) = 1-p$
  - Loss: $L(0,1) = 1$ (wasted effort)
- Case 2: Student passes (y=1) and we predicted pass ($\hat{y}=1$)
  - Probability: $P(y=1) = p$
  - Loss: $L(1,1) = 0$

The expected loss when we predict pass ($\hat{y}=1$) is:
$$R(\hat{y}=1) = P(y=0) \cdot L(0,1) + P(y=1) \cdot L(1,1)$$
$$R(\hat{y}=1) = (1-p) \cdot 1 + p \cdot 0 = 1-p$$

For predicting fail ($\hat{y}=0$):
- Case 1: Student fails (y=0) and we predicted fail ($\hat{y}=0$)
  - Probability: $P(y=0) = 1-p$
  - Loss: $L(0,0) = 0$
- Case 2: Student passes (y=1) but we predicted fail ($\hat{y}=0$)
  - Probability: $P(y=1) = p$
  - Loss: $L(1,0) = 2$ (missed opportunity)

The expected loss when we predict fail ($\hat{y}=0$) is:
$$R(\hat{y}=0) = P(y=0) \cdot L(0,0) + P(y=1) \cdot L(1,0)$$
$$R(\hat{y}=0) = (1-p) \cdot 0 + p \cdot 2 = 2p$$

Again, to minimize the expected loss, we compare the two risks:
- Predict pass ($\hat{y}=1$) if $1-p < 2p$
- Predict fail ($\hat{y}=0$) if $2p < 1-p$

Solving the first inequality:
$$1-p < 2p$$
$$1 < 2p + p$$
$$1 < 3p$$
$$\frac{1}{3} < p$$

Therefore, the decision rule for the asymmetric loss function is: Predict that the student will pass ($\hat{y}=1$) if $p > \frac{1}{3}$, otherwise predict that the student will fail ($\hat{y}=0$).

At exactly $p = \frac{1}{3}$, both decisions have equal expected loss ($\frac{2}{3}$), so either decision is optimal.

![Expected Loss for Asymmetric Loss Function](../Images/L2_7_Quiz_29/asymmetric_loss.png)

In this graph:
- The blue line represents the expected loss when predicting pass ($\hat{y}=1$), which is $1-p$
- The red line represents the expected loss when predicting fail ($\hat{y}=0$), which is $2p$
- The vertical dashed line marks the decision threshold at $p = \frac{1}{3}$
- The red shaded region (left) indicates that it's optimal to predict fail when $p < \frac{1}{3}$
- The blue shaded region (right) indicates that it's optimal to predict pass when $p > \frac{1}{3}$

### Step 3: Explaining the Difference in Thresholds

The thresholds differ due to the asymmetric nature of the second loss function. We can derive a general formula for the optimal threshold $p^*$ in binary classification problems:

For a general binary classification with cost $C_{FP}$ for false positives and $C_{FN}$ for false negatives:
- Expected loss for predicting class 1 = $(1-p) \times C_{FP}$
- Expected loss for predicting class 0 = $p \times C_{FN}$

The optimal threshold $p^*$ is where these expected losses are equal:
$$(1-p^*) \times C_{FP} = p^* \times C_{FN}$$
$$C_{FP} - p^* \times C_{FP} = p^* \times C_{FN}$$
$$C_{FP} = p^* \times (C_{FP} + C_{FN})$$
$$p^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$

Applying this formula:
- For 0-1 loss: $C_{FP} = C_{FN} = 1$, so $p^* = \frac{1}{1+1} = 0.5$
- For asymmetric loss: $C_{FP} = 1$, $C_{FN} = 2$, so $p^* = \frac{1}{1+2} = \frac{1}{3}$

In practical terms for the student, this means:

1. With the 0-1 loss, our approach is balanced: we simply predict the more likely outcome. If the probability of passing is greater than 50%, we predict pass, otherwise, we predict fail.

2. With the asymmetric loss, we're more inclined to be optimistic and predict that the student will pass even when the probability is lower (as long as it's above 33.3%). This reflects a teaching philosophy that it's worse to discourage a student who could have passed (missed opportunity) than to encourage a student who ultimately fails (wasted effort).

The lower threshold means we would provide more students with encouragement and resources to pass, potentially at the cost of some wasted effort, but with the benefit of not missing opportunities for students who are on the edge but could succeed with the right support.

![Comparison of Loss Functions](../Images/L2_7_Quiz_29/loss_function_comparison.png)

This comparison plot shows both loss functions together:
- Solid blue line: Expected loss when predicting pass (0-1 loss)
- Solid red line: Expected loss when predicting fail (0-1 loss)
- Dashed blue line: Expected loss when predicting pass (asymmetric loss)
- Dashed red line: Expected loss when predicting fail (asymmetric loss)
- Vertical dashed line: 0-1 loss threshold at $p = 0.5$
- Vertical dotted line: Asymmetric loss threshold at $p = \frac{1}{3}$
- Vertical green line: Example case with $p = 0.4$
- The markers at $p = 0.4$ show the expected losses for each combination of loss function and decision

### Step 4: Decision for p = 0.4

If our model gives a probability $p = 0.4$ that a student will pass, we can calculate the expected losses for each decision under both loss functions:

For the 0-1 loss function:
- Expected loss if we predict pass ($\hat{y}=1$):
  - $R(\hat{y}=1) = (1-p) \times 1 + p \times 0 = (1-0.4) \times 1 + 0.4 \times 0 = 0.6$
- Expected loss if we predict fail ($\hat{y}=0$):
  - $R(\hat{y}=0) = (1-p) \times 0 + p \times 1 = (1-0.4) \times 0 + 0.4 \times 1 = 0.4$
- Since $0.4 < 0.6$, the expected loss is lower when predicting fail
- Therefore, with the 0-1 loss function, we would predict that the student will fail ($\hat{y}=0$)

For the asymmetric loss function:
- Expected loss if we predict pass ($\hat{y}=1$):
  - $R(\hat{y}=1) = (1-p) \times 1 + p \times 0 = (1-0.4) \times 1 + 0.4 \times 0 = 0.6$
- Expected loss if we predict fail ($\hat{y}=0$):
  - $R(\hat{y}=0) = (1-p) \times 0 + p \times 2 = (1-0.4) \times 0 + 0.4 \times 2 = 0.8$
- Since $0.6 < 0.8$, the expected loss is lower when predicting pass
- Therefore, with the asymmetric loss function, we would predict that the student will pass ($\hat{y}=1$)

This illustrates how the choice of loss function directly affects our decisions. With $p = 0.4$, we get opposite decisions depending on which loss function we use:
- With 0-1 loss: Predict fail (ŷ=0)
- With asymmetric loss: Predict pass (ŷ=1)

![Decision Regions](../Images/L2_7_Quiz_29/decision_regions.png)

This visualization shows the decision regions for both loss functions:
- Top bar: Decision regions for 0-1 loss
  - Red region (0 to 0.5): Predict fail
  - Blue region (0.5 to 1): Predict pass
- Bottom bar: Decision regions for asymmetric loss
  - Red region (0 to 1/3): Predict fail
  - Blue region (1/3 to 1): Predict pass
- The vertical green line shows our example case $p = 0.4$
- The red circle on the top bar indicates that with 0-1 loss and $p = 0.4$, we predict fail
- The blue circle on the bottom bar indicates that with asymmetric loss and $p = 0.4$, we predict pass

## Key Insights

### Theoretical Foundations
- The optimal decision rule for minimizing expected loss depends on the specific loss function used.
- For symmetric loss functions like 0-1 loss, the decision threshold is 0.5 (predict the most likely class).
- For asymmetric loss functions, the threshold shifts to reflect the relative costs of different types of errors.
- The general formula for the optimal threshold in binary classification is $p^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$, where $C_{FP}$ is the cost of false positives and $C_{FN}$ is the cost of false negatives.

### Practical Applications
- In educational settings, asymmetric loss functions better reflect real-world priorities, where missing an opportunity for a student might be considered worse than wasted effort.
- Medical diagnostics often use asymmetric loss functions, where failing to diagnose a serious condition (false negative) may be much costlier than a false alarm (false positive).
- Business decisions frequently involve asymmetric costs, such as in fraud detection where missing a fraud case (false negative) may be much more expensive than investigating a legitimate transaction (false positive).

### Common Pitfalls
- Using the standard 0.5 threshold when the costs of different errors are actually asymmetric.
- Failing to explicitly consider the cost structure of errors in a specific domain.
- Not communicating the expected loss to stakeholders, which is necessary for informed decision-making.
- Assuming that model accuracy is the only relevant metric, when in fact expected loss may be more directly relevant to the business or application context.

## Conclusion

- The 0-1 loss function leads to a decision threshold of $p = 0.5$, meaning we predict the most likely class.
- The asymmetric loss function, which penalizes missed opportunities more than wasted effort, leads to a lower threshold of $p = \frac{1}{3}$, making us more inclined to predict that students will pass.
- For a probability of $p = 0.4$, the 0-1 loss function would lead us to predict fail, while the asymmetric loss function would lead us to predict pass.
- This demonstrates how incorporating domain-specific costs into our loss functions can lead to different, and potentially more appropriate, decisions.

By understanding how to derive and apply optimal decision rules for different loss functions, we can make better predictions that align with the specific priorities and cost structures of our application domain. 