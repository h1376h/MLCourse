# Question 27: One-Hot Encoding and Multinomial Classification

## Problem Statement
A natural language processing engineer is building a text classification model for customer support tickets. The engineer has a dataset with 500 support tickets that have been manually labeled with one of five categories:

1. Billing Issues
2. Technical Problems
3. Account Access
4. Feature Requests
5. General Inquiries

To prepare the data for a logistic regression model, the engineer one-hot encodes the category labels. After training, the model predicts probabilities for each category. For a new unlabeled ticket, the model outputs the following probabilities:

- $P(\text{Billing Issues}) = 0.15$
- $P(\text{Technical Problems}) = 0.35$
- $P(\text{Account Access}) = 0.20$
- $P(\text{Feature Requests}) = 0.10$
- $P(\text{General Inquiries}) = 0.20$

The original training data had the following distribution of categories:

- Billing Issues: 100 tickets
- Technical Problems: 150 tickets
- Account Access: 80 tickets
- Feature Requests: 70 tickets
- General Inquiries: 100 tickets

### Task
1. Explain the relationship between one-hot encoding and the multinomial distribution in this context
2. Calculate the maximum likelihood estimates for the prior probabilities of each category based on the training data
3. Using Bayes' theorem and assuming the model outputs are accurate likelihood estimates, calculate the posterior probability that the ticket belongs to the "Technical Problems" category if we know that 60% of all support tickets are about technical issues (regardless of what the model predicts)
4. The engineer decides to use a classification threshold of 0.30, meaning a ticket is assigned to a category if its probability exceeds this threshold. Based on the model outputs for the new ticket:
   a. Which category would be assigned to the ticket?
   b. What are the potential issues with using a fixed threshold for multinomial classification?
5. Calculate the cross-entropy loss between the true one-hot encoded label $[0,1,0,0,0]$ (Technical Problems) and the model's predicted probabilities using:
   $$H(y, \hat{y}) = -\sum_{i=1}^{5} y_i \log(\hat{y}_i)$$

## Understanding the Problem
This problem explores fundamental concepts in machine learning classification:
- One-hot encoding for representing categorical variables
- The multinomial distribution for modeling categorical data
- Maximum Likelihood Estimation (MLE) for determining prior probabilities
- Bayes' theorem for calculating posterior probabilities
- Threshold-based classification decisions
- Cross-entropy loss for evaluating prediction accuracy

These concepts are essential for understanding how classification models work with categorical data, particularly in natural language processing applications.

## Solution

### Step 1: Relationship Between One-Hot Encoding and Multinomial Distribution

One-hot encoding represents categorical variables as binary vectors where only one element is 1 (hot), and all others are 0 (cold). In our five-category problem, the encodings are:

- Billing Issues: [1, 0, 0, 0, 0]
- Technical Problems: [0, 1, 0, 0, 0]
- Account Access: [0, 0, 1, 0, 0]
- Feature Requests: [0, 0, 0, 1, 0]
- General Inquiries: [0, 0, 0, 0, 1]

This encoding scheme directly relates to the multinomial distribution through the following connections:

- Each support ticket represents one trial in the multinomial distribution
- Each ticket must belong to exactly one of the five categories (mutual exclusivity)
- One-hot encoding maps each category to a unique binary vector
- The multinomial distribution models the probability of observing different counts across categories

For a logistic regression model with softmax activation (multinomial logistic regression), the model outputs probabilities that sum to 1 across all categories, which corresponds to the parameters of the multinomial distribution.

![One-Hot Encoding and Multinomial Distribution](../Codes/images/step1_one_hot_multinomial.png)

### Step 2: Maximum Likelihood Estimates for Prior Probabilities

The Maximum Likelihood Estimate (MLE) for prior probabilities is the proportion of training examples that belong to each category.

With a total of 500 training tickets, the MLE calculations are:

- Billing Issues: 100/500 = 0.2000 (20.0%)
- Technical Problems: 150/500 = 0.3000 (30.0%)
- Account Access: 80/500 = 0.1600 (16.0%)
- Feature Requests: 70/500 = 0.1400 (14.0%)
- General Inquiries: 100/500 = 0.2000 (20.0%)

These proportions maximize the likelihood function:
$$L(p_1, p_2, \ldots, p_5) = \frac{n!}{n_1!n_2!\ldots n_5!} \times p_1^{n_1} \times p_2^{n_2} \times \ldots \times p_5^{n_5}$$

Subject to the constraint that $\sum_{i=1}^{5} p_i = 1$

![MLE Prior Probabilities](../Codes/images/step2_prior_probabilities.png)

### Step 3: Posterior Probability Using Bayes' Theorem

We need to calculate the posterior probability that the ticket belongs to the "Technical Problems" category given that:
- The model predicts P(Technical Problems) = 0.35
- We know that 60% of all support tickets are about technical issues

Using Bayes' theorem:
$$P(\text{Technical} | \text{Model}) = \frac{P(\text{Model} | \text{Technical}) \times P(\text{Technical})}{P(\text{Model})}$$

Step-by-step calculation:

1. Identify the values we know:
   - Model probability for Technical Problems: 0.35
   - Prior probability P(Technical): 0.60
   - Prior probability P(Non-Technical): 0.40

2. For P(Model | Technical), we use the model output directly:
   - P(Model | Technical) = 0.35

3. Calculate P(Model | Non-Technical):
   - Non-technical probabilities: [0.15, 0.20, 0.10, 0.20]
   - Sum of non-technical probabilities: 0.6500
   - Average (P(Model | Non-Technical)): 0.6500 / 4 = 0.1625

4. Calculate P(Model) using the law of total probability:
   - P(Model) = P(Model | Technical) × P(Technical) + P(Model | Non-Technical) × P(Non-Technical)
   - P(Model) = 0.35 × 0.60 + 0.1625 × 0.40
   - P(Model) = 0.2100 + 0.0650
   - P(Model) = 0.2750

5. Finally, calculate the posterior probability:
   - P(Technical | Model) = P(Model | Technical) × P(Technical) / P(Model)
   - P(Technical | Model) = (0.35 × 0.60) / 0.2750
   - P(Technical | Model) = 0.2100 / 0.2750
   - P(Technical | Model) = 0.7636 or 76.4%

The posterior probability that the ticket belongs to "Technical Problems" is approximately 76.4%, which is significantly higher than the model's original prediction of 35%.

![Bayes' Theorem Application](../Codes/images/step3_bayes_theorem.png)

### Step 4: Classification with Threshold

Using a classification threshold of 0.30, a ticket is assigned to a category if its probability exceeds this threshold.

Step 1: Compare each category's probability to the threshold:
- Billing Issues: 0.15 (Below threshold)
- Technical Problems: 0.35 (Exceeds threshold)
- Account Access: 0.20 (Below threshold)
- Feature Requests: 0.10 (Below threshold)
- General Inquiries: 0.20 (Below threshold)

Step 2: Determine the classification result:
- Only 'Technical Problems' exceeds the threshold, so the ticket is classified as 'Technical Problems'

Potential issues with using a fixed threshold for multinomial classification include:

1. Loss of probability information (discards model confidence levels)
2. Ambiguity when multiple categories exceed the threshold
3. Ambiguity when no category exceeds the threshold
4. Different categories may need different optimal thresholds
5. Doesn't account for class imbalance
6. Ignores varying costs of misclassification between categories

![Classification Threshold](../Codes/images/step4_classification_threshold.png)

### Step 5: Cross-Entropy Loss

The cross-entropy loss between the true one-hot encoded label [0,1,0,0,0] (Technical Problems) and the model's predicted probabilities is calculated using:

$$H(y, \hat{y}) = -\sum_{i=1}^{5} y_i \log(\hat{y}_i)$$

Step-by-step calculation:

First, identifying the values:
- True label (y): [0, 1, 0, 0, 0] (one-hot encoding for Technical Problems)
- Predicted probabilities (ŷ): [0.15, 0.35, 0.20, 0.10, 0.20]

Calculating each term in the summation:
1. Billing Issues: -0 × log(0.15) = 0 (since y₁ = 0)
2. Technical Problems: -1 × log(0.35) = -1 × (-1.0498) = 1.0498
3. Account Access: -0 × log(0.20) = 0 (since y₃ = 0)
4. Feature Requests: -0 × log(0.10) = 0 (since y₄ = 0)
5. General Inquiries: -0 × log(0.20) = 0 (since y₅ = 0)

Final cross-entropy loss = 1.0498

Note that only the term for the true class (Technical Problems) contributes to the loss, as all other y values are 0. The cross-entropy loss would be lower if the model had assigned a higher probability to the correct class, and higher if it had assigned a lower probability.

![Cross-Entropy Loss](../Codes/images/step5_cross_entropy_loss.png)

## Key Insights

### Theoretical Foundations
- One-hot encoding creates a direct mapping between categorical variables and the multinomial distribution parameters
- Maximum Likelihood Estimation provides a principled approach for estimating prior probabilities from observed frequencies
- Bayes' theorem allows us to incorporate prior knowledge into our predictions, significantly improving classification decisions
- Cross-entropy loss quantifies the difference between predicted probabilities and true labels, serving as a measure of model performance

### Practical Applications
- Incorporating known prior probabilities can dramatically alter classification decisions. In our example, the posterior probability increased from 35% to 76.4% when we incorporated prior knowledge
- Threshold-based classification simplifies decision-making but loses valuable probability information and can lead to ambiguities
- Cross-entropy loss provides a natural training objective for models that output probability distributions, encouraging high confidence in correct classifications

### Common Pitfalls
- Ignoring prior probabilities can lead to suboptimal classifications, especially when training data distribution differs from real-world distribution
- Fixed thresholds may not be appropriate for all categories, especially in imbalanced classification problems
- Focusing solely on accuracy metrics without considering probability calibration can hide important information about model uncertainty

## Conclusion

This problem demonstrates several key concepts in classification modeling:

1. One-hot encoding provides an effective representation for categorical data that aligns with the multinomial distribution
2. MLE provides a straightforward way to estimate prior probabilities from training data
3. Bayes' theorem allows us to update probabilities based on prior knowledge, potentially leading to more accurate classifications
4. While threshold-based classification is practical, it has several limitations that should be considered in real-world applications
5. Cross-entropy loss provides a principled way to evaluate and optimize classification models that output probability distributions

These concepts form the foundation of many machine learning classification techniques, particularly in natural language processing applications where categorical outcomes are common. 