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

These concepts are essential for understanding how classification models work with categorical data, particularly in natural language processing applications like support ticket categorization.

## Solution

### Step 1: Relationship Between One-Hot Encoding and Multinomial Distribution

One-hot encoding represents categorical variables as binary vectors where only one element is 1 (hot), and all others are 0 (cold). In our five-category problem, the encodings are:

- Billing Issues: [1, 0, 0, 0, 0]
- Technical Problems: [0, 1, 0, 0, 0]
- Account Access: [0, 0, 1, 0, 0]
- Feature Requests: [0, 0, 0, 1, 0]
- General Inquiries: [0, 0, 0, 0, 1]

This encoding scheme directly relates to the multinomial distribution through several key connections:

1. **Multinomial Distribution Fundamentals**
   - The multinomial distribution models the probability of observing specific counts across $k$ categories
   - The probability mass function is:
     $$P(X_1=n_1, X_2=n_2, \ldots, X_k=n_k) = \frac{n!}{n_1!n_2!\ldots n_k!} \times p_1^{n_1} \times p_2^{n_2} \times \ldots \times p_k^{n_k}$$
   - Where $n = n_1 + n_2 + \ldots + n_k$ is the total number of trials
   - And $p_1, p_2, \ldots, p_k$ are the probabilities for each category (with $\sum p_i = 1$)

2. **Connection to One-Hot Encoding**
   - Each support ticket represents one trial in the multinomial distribution
   - Each ticket must belong to exactly one of the five categories (mutual exclusivity)
   - One-hot encoding maps each category to a unique binary vector
   - The position of the '1' in the vector indicates which category a ticket belongs to
   - The sum of all positions equals 1 (preserving the mutual exclusivity property)

3. **Mathematical Connection**
   - For a single ticket, let $y = [y_1, y_2, \ldots, y_5]$ be the one-hot encoded vector
   - Only one $y_i$ equals 1, and all others are 0
   - Let $p = [p_1, p_2, \ldots, p_5]$ be the predicted probabilities from the model
   - The probability of observing $y$ is: $P(y) = p_1^{y_1} \times p_2^{y_2} \times \ldots \times p_5^{y_5} = p_i$ (where $y_i = 1$)
   - For multiple tickets, the joint probability follows the multinomial distribution

4. **Connection to Logistic Regression**
   - Multinomial logistic regression uses softmax activation to output probabilities:
     $$P(\text{category }i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$
     where $z_i$ are the model's raw outputs
   - These probabilities sum to 1 across all categories
   - The model's output directly represents the parameters of the multinomial distribution
   - Cross-entropy loss ($-\sum y_i \log(p_i)$) is derived from the negative log-likelihood of the multinomial distribution

![One-Hot Encoding and Multinomial Distribution](../Images/L2_4_Quiz_27/step1_one_hot_multinomial.png)

### Step 2: Maximum Likelihood Estimates for Prior Probabilities

The Maximum Likelihood Estimate (MLE) for prior probabilities is the proportion of training examples that belong to each category. We can derive this systematically using the principles of maximum likelihood estimation.

#### Derivation of the MLE for Multinomial Parameters:

1. **Likelihood Function:**
   - For the multinomial distribution with $k=5$ categories, we have:
     $$L(p_1, p_2, \ldots, p_5) = \frac{n!}{n_1!n_2!\ldots n_5!} \times p_1^{n_1} \times p_2^{n_2} \times \ldots \times p_5^{n_5}$$
   - Where $n = n_1 + n_2 + \ldots + n_5$ is the total number of tickets

2. **Log-Likelihood Function:**
   - Taking the logarithm simplifies the calculation:
     $$\ln(L) = \ln\left(\frac{n!}{n_1!n_2!\ldots n_5!}\right) + n_1\ln(p_1) + n_2\ln(p_2) + \ldots + n_5\ln(p_5)$$
   - The first term is constant with respect to $p_i$, so we can focus on:
     $$\ln(L) \propto n_1\ln(p_1) + n_2\ln(p_2) + \ldots + n_5\ln(p_5)$$

3. **Constraint:**
   - The probabilities must sum to 1:
     $$p_1 + p_2 + \ldots + p_5 = 1$$

4. **Lagrangian Formulation:**
   - To maximize $\ln(L)$ subject to the constraint, we use a Lagrangian:
     $$\mathcal{L} = n_1\ln(p_1) + n_2\ln(p_2) + \ldots + n_5\ln(p_5) - \lambda(p_1 + p_2 + \ldots + p_5 - 1)$$

5. **Taking Partial Derivatives:**
   - $\frac{\partial\mathcal{L}}{\partial p_i} = \frac{n_i}{p_i} - \lambda$
   - Setting all $\frac{\partial\mathcal{L}}{\partial p_i} = 0$:
     $$\frac{n_i}{p_i} = \lambda \text{ for all } i$$
     $$p_i = \frac{n_i}{\lambda} \text{ for all } i$$

6. **Using the Constraint:**
   - $p_1 + p_2 + \ldots + p_5 = 1$
   - $\frac{n_1}{\lambda} + \frac{n_2}{\lambda} + \ldots + \frac{n_5}{\lambda} = 1$
   - $\frac{n_1 + n_2 + \ldots + n_5}{\lambda} = 1$
   - $\frac{n}{\lambda} = 1$
   - $\lambda = n$ (where $n$ is the total count)

7. **Substituting Back:**
   - $p_i = \frac{n_i}{\lambda} = \frac{n_i}{n}$

Based on our training data with 500 total tickets, we can calculate the MLE for each category:

| Category | Count ($n_i$) | Calculation | MLE ($p_i$) |
|----------|---------------|-------------|-------------|
| Billing Issues | 100 | $\frac{100}{500}$ | 0.2000 (20.0%) |
| Technical Problems | 150 | $\frac{150}{500}$ | 0.3000 (30.0%) |
| Account Access | 80 | $\frac{80}{500}$ | 0.1600 (16.0%) |
| Feature Requests | 70 | $\frac{70}{500}$ | 0.1400 (14.0%) |
| General Inquiries | 100 | $\frac{100}{500}$ | 0.2000 (20.0%) |

We can verify that the probabilities sum to 1: $0.2000 + 0.3000 + 0.1600 + 0.1400 + 0.2000 = 1.0000$

![MLE Prior Probabilities](../Images/L2_4_Quiz_27/step2_prior_probabilities.png)

### Step 3: Posterior Probability Using Bayes' Theorem

We need to calculate the posterior probability that the ticket belongs to the "Technical Problems" category given that:
- The model predicts P(Technical Problems) = 0.35
- We know that 60% of all support tickets are about technical issues

Using Bayes' theorem:
$$P(\text{Technical}|\text{Model}) = \frac{P(\text{Model}|\text{Technical}) \times P(\text{Technical})}{P(\text{Model})}$$

#### Step-by-step calculation:

1. **Identify the values we know:**
   - Model probability for Technical Problems: 0.35
   - Prior probability P(Technical): 0.60
   - Prior probability P(Non-Technical): 0.40 = 1 - 0.60

2. **Calculate the model's behavior for non-technical tickets:**
   - Non-technical probabilities from model output:
     [0.15, 0.20, 0.10, 0.20] (for Billing, Account, Feature, General respectively)
   - Sum of non-technical probabilities: 0.15 + 0.20 + 0.10 + 0.20 = 0.6500
   - Average (P(Model|Non-Technical)): 0.6500 / 4 = 0.1625

3. **Calculate P(Model) using the law of total probability:**
   - P(Model) = P(Model|Technical) × P(Technical) + P(Model|Non-Technical) × P(Non-Technical)
   - P(Model) = 0.35 × 0.60 + 0.1625 × 0.40
   - P(Model) = 0.2100 + 0.0650
   - P(Model) = 0.2750

4. **Apply Bayes' theorem to calculate the posterior probability:**
   - P(Technical|Model) = P(Model|Technical) × P(Technical) / P(Model)
   - P(Technical|Model) = (0.35 × 0.60) / 0.2750
   - P(Technical|Model) = 0.2100 / 0.2750
   - P(Technical|Model) = 0.7636 or 76.4%

5. **Interpret the result:**
   - Original model probability for Technical Problems: 35.0%
   - Posterior probability after incorporating prior knowledge: 76.4%
   - This represents a 41.4 percentage point increase
   - Or a 118.2% relative increase in probability

The posterior probability is significantly higher than the model's original prediction, demonstrating the powerful impact of incorporating prior knowledge through Bayes' theorem.

![Bayes' Theorem Application](../Images/L2_4_Quiz_27/step3_bayes_theorem.png)

### Step 4: Classification with Threshold

Using a classification threshold of 0.30, a ticket is assigned to a category if its probability exceeds this threshold.

#### Threshold-based classification procedure:

1. **Compare each category's probability to the threshold:**
   - Billing Issues: 0.15 ≤ 0.30 (Below threshold)
   - Technical Problems: 0.35 > 0.30 (Exceeds threshold)
   - Account Access: 0.20 ≤ 0.30 (Below threshold)
   - Feature Requests: 0.10 ≤ 0.30 (Below threshold)
   - General Inquiries: 0.20 ≤ 0.30 (Below threshold)

2. **Determine final classification decision:**
   - Only 'Technical Problems' exceeds the threshold with probability 0.35
   - Classification decision: Ticket is classified as 'Technical Problems'

3. **Potential issues with using fixed thresholds in multinomial classification:**
   1. Loss of probability information - discards model confidence levels
   2. Ambiguity when multiple categories exceed the threshold
   3. Ambiguity when no category exceeds the threshold
   4. Different categories may need different optimal thresholds
   5. Doesn't account for class imbalance in the data
   6. Ignores varying costs of misclassification between categories

4. **Alternative approaches to threshold-based classification:**
   1. Always selecting the highest probability category (argmax)
   2. Using category-specific thresholds
   3. Incorporating prior probabilities through Bayes' theorem (as we did in Step 3)
   4. Using a reject option for low-confidence predictions
   5. Applying calibration techniques to improve probability estimates

![Classification Threshold](../Images/L2_4_Quiz_27/step4_classification_threshold.png)

### Step 5: Cross-Entropy Loss

The cross-entropy loss between the true one-hot encoded label [0,1,0,0,0] (Technical Problems) and the model's predicted probabilities is calculated using:

$$H(y, \hat{y}) = -\sum_{i=1}^{5} y_i \log(\hat{y}_i)$$

#### Step-by-step calculation:

1. **Understanding cross-entropy loss:**
   - Cross-entropy loss measures the difference between two probability distributions
   - It quantifies how well predicted probabilities match the true labels
   - Lower values indicate better model performance
   - Perfect predictions (probability 1.0 for correct class) would give loss = 0

2. **Review the input values:**
   - True label (y): [0, 1, 0, 0, 0] (one-hot encoding for Technical Problems)
   - Predicted probabilities (ŷ): [0.15, 0.35, 0.20, 0.10, 0.20]
   - When using one-hot encoding, only the term for the true class contributes to the loss

3. **Calculate each term in the summation:**
   - Billing Issues (y₁ = 0): -0 × log(0.15) = 0
   - Technical Problems (y₂ = 1): -1 × log(0.35) = -1 × (-1.049822) = 1.049822
   - Account Access (y₃ = 0): -0 × log(0.20) = 0
   - Feature Requests (y₄ = 0): -0 × log(0.10) = 0
   - General Inquiries (y₅ = 0): -0 × log(0.20) = 0

4. **Sum all terms to get the final cross-entropy loss:**
   - H(y, ŷ) = 0 + 1.049822 + 0 + 0 + 0 = 1.049822

5. **Interpret the cross-entropy loss value:**
   - The loss quantifies the "surprise" of seeing the true label given the model's predictions
   - The model assigned probability 0.35 to the correct class
   - Perfect predictions would give a cross-entropy of 0
   - Random guessing (probability 0.2 for each class) would give a cross-entropy of 1.609
   - Our loss is 1.0498, indicating the model performs better than random
   - But there's still room for improvement (loss > 0)

6. **Understanding how to improve cross-entropy loss:**
   - The loss would be lower if the model assigned higher probability to the correct class
   - For example, if P(Technical) = 0.5, loss would be 0.693
   - If P(Technical) = 0.9, loss would be 0.105
   - Cross-entropy loss encourages high confidence in correct predictions
   - It heavily penalizes being confidently wrong (if true class gets very low probability)

![Cross-Entropy Loss](../Images/L2_4_Quiz_27/step5_cross_entropy_loss.png)

## Key Insights

### Theoretical Foundations
- One-hot encoding creates a direct mapping between categorical variables and the multinomial distribution parameters
- Maximum Likelihood Estimation provides a principled approach for estimating prior probabilities from observed frequencies
- Bayes' theorem allows us to incorporate prior knowledge into our predictions, significantly improving classification decisions
- Cross-entropy loss quantifies the difference between predicted probabilities and true labels, serving as a natural training objective derived from the negative log-likelihood

### Practical Applications
- Incorporating known prior probabilities can dramatically alter classification decisions. In our example, the posterior probability increased from 35% to 76.4% when we incorporated prior knowledge
- Threshold-based classification simplifies decision-making but loses valuable probability information and can lead to ambiguities
- Cross-entropy loss encourages models to assign high probabilities to correct classes, improving classification performance
- Alternative classification approaches like argmax selection or category-specific thresholds can mitigate some of the issues with fixed thresholds

### Common Pitfalls
- Ignoring prior probabilities can lead to suboptimal classifications, especially when training data distribution differs from real-world distribution
- Fixed thresholds may not be appropriate for all categories, especially in imbalanced classification problems
- Focusing solely on accuracy metrics without considering probability calibration can hide important information about model uncertainty
- Not understanding the connection between the loss function and the probabilistic interpretation can lead to misinterpretations of model outputs

## Conclusion

This problem demonstrates several key concepts in classification modeling:

1. One-hot encoding provides an effective representation for categorical data that aligns with the multinomial distribution, with a direct mathematical relationship through the probability mass function
2. MLE provides a straightforward way to estimate prior probabilities from training data, following directly from the principles of maximum likelihood
3. Bayes' theorem allows us to update probabilities based on prior knowledge, potentially leading to dramatic improvements in classification accuracy
4. While threshold-based classification is practical, it has several limitations that should be considered in real-world applications
5. Cross-entropy loss provides a principled way to evaluate and optimize classification models that output probability distributions

These concepts form the foundation of many machine learning classification techniques, particularly in natural language processing applications where categorical outcomes are common. 