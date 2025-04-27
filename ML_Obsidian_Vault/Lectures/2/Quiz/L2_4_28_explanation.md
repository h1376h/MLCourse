# Question 28: Maximum Likelihood Estimation & Classification Thresholds

## Problem Statement
A machine learning engineer is training a multinomial classifier that predicts three categories: {Cat, Dog, Bird}. The engineer represents each label using one-hot encoding: Cat = $[1,0,0]$, Dog = $[0,1,0]$, and Bird = $[0,0,1]$. 

After training, the model outputs the following probability vectors for 5 test samples:
- Sample 1: $[0.7, 0.2, 0.1]$
- Sample 2: $[0.3, 0.5, 0.2]$
- Sample 3: $[0.1, 0.3, 0.6]$
- Sample 4: $[0.4, 0.4, 0.2]$
- Sample 5: $[0.2, 0.1, 0.7]$

The true labels are: Sample 1: Cat, Sample 2: Dog, Sample 3: Bird, Sample 4: Cat, Sample 5: Bird.

### Task
1. Write down the log-likelihood of the observed data given the model's predictions. If $y_i$ represents the true label and $\hat{y}_i$ represents the predicted probabilities, the log-likelihood can be written as:
   $$\log L = \sum_{i=1}^{n} \log(\hat{y}_{i,c_i})$$
   where $c_i$ is the correct class index for sample $i$

2. Using MLE principles, what threshold would you set for classification to maximize the likelihood of the observed data?

3. If you had to pick one fixed probability threshold ($p > \text{threshold} \rightarrow$ classify as that class), what value would maximize accuracy on this dataset?


## Understanding the Problem

This problem explores several key concepts in probabilistic classification:

- **Log-likelihood**: A measure of how well the model's predicted probabilities match the observed data. Higher (less negative) values indicate better model performance.
- **Maximum likelihood estimation (MLE)**: A statistical method that finds parameter values that maximize the likelihood function, making the observed data most probable.
- **Classification thresholds**: Criteria used to convert probabilistic outputs into discrete class predictions.
- **One-hot encoding**: A representation where each class is encoded as a binary vector with a single 1 and the rest 0s.

The problem requires understanding how these concepts interconnect and apply them to make optimal classification decisions.

## Solution

### Step 1: Calculate the Log-Likelihood of the Observed Data

The log-likelihood measures how well the model's predicted probabilities match the observed data. For a multinomial classifier with one-hot encoded targets, the log-likelihood is calculated as:

$$\log L = \sum_{i=1}^{n} \log(P(y_i|x_i))$$

Where:
- $n$ is the number of samples
- $y_i$ is the true class for sample $i$
- $P(y_i|x_i)$ is the predicted probability for the true class of sample $i$

Let's calculate this step by step:

1. **Sample 1** (Cat):
   - True class: Cat (index 0)
   - Probability assigned to true class: 0.7
   - $\log(0.7) = -0.3567$

2. **Sample 2** (Dog):
   - True class: Dog (index 1)
   - Probability assigned to true class: 0.5
   - $\log(0.5) = -0.6931$

3. **Sample 3** (Bird):
   - True class: Bird (index 2)
   - Probability assigned to true class: 0.6
   - $\log(0.6) = -0.5108$

4. **Sample 4** (Cat):
   - True class: Cat (index 0)
   - Probability assigned to true class: 0.4
   - $\log(0.4) = -0.9163$

5. **Sample 5** (Bird):
   - True class: Bird (index 2)
   - Probability assigned to true class: 0.7
   - $\log(0.7) = -0.3567$

Total log-likelihood:
$$\log L = -0.3567 + (-0.6931) + (-0.5108) + (-0.9163) + (-0.3567) = -2.8336$$

This value of -2.8336 quantifies how well the model's predictions match the true labels. Higher (less negative) values indicate better predictive performance.

![Log-Likelihood Calculation](../Images/L2_4_Quiz_28/step1_log_likelihood.png)

### Step 2: Optimal Threshold from MLE Perspective

From a Maximum Likelihood Estimation (MLE) perspective, the optimal classification strategy is to assign each sample to the class with the highest probability. This maximizes the likelihood of the observed data.

This approach corresponds to using a relative threshold rather than an absolute fixed value. The decision rule is:

$$\text{Predict class } j \text{ where } j = \arg\max_i P(y_i|x)$$

Let's analyze each sample's classification using this maximum probability rule:

1. **Sample 1**:
   - Probabilities: [0.7, 0.2, 0.1]
   - Maximum probability: 0.7 for class 'Cat'
   - True class: 'Cat'
   - Result: **Correct**

2. **Sample 2**:
   - Probabilities: [0.3, 0.5, 0.2]
   - Maximum probability: 0.5 for class 'Dog'
   - True class: 'Dog'
   - Result: **Correct**

3. **Sample 3**:
   - Probabilities: [0.1, 0.3, 0.6]
   - Maximum probability: 0.6 for class 'Bird'
   - True class: 'Bird'
   - Result: **Correct**

4. **Sample 4**:
   - Probabilities: [0.4, 0.4, 0.2]
   - Maximum probability: 0.4 for both 'Cat' and 'Dog'
   - In case of a tie, we typically take the first class (Cat in this case)
   - True class: 'Cat'
   - Result: **Correct**

5. **Sample 5**:
   - Probabilities: [0.2, 0.1, 0.7]
   - Maximum probability: 0.7 for class 'Bird'
   - True class: 'Bird'
   - Result: **Correct**

Using the MLE approach (selecting the class with highest probability), we achieve 5/5 correct classifications, giving an accuracy of 100%.

**Mathematical Justification**:
1. For one-hot encoded labels y = [0,0,...,1,0,...], only one position has y_j = 1
2. Log-likelihood for a sample is log(p_j) where j is the true class
3. To maximize likelihood across all samples, we choose the class with the highest probability for each sample

![MLE Classification](../Images/L2_4_Quiz_28/step2_mle_threshold.png)

### Step 3: Determine a Fixed Probability Threshold for Maximum Accuracy

Now, we'll determine a fixed probability threshold that maximizes accuracy on this dataset. With a fixed threshold approach, we only classify a sample to a class if the probability exceeds the threshold value.

This introduces three possible scenarios:
1. Only one class exceeds the threshold → Classify as that class
2. Multiple classes exceed the threshold → Classify as the class with highest probability
3. No class exceeds the threshold → The sample remains unclassified

We tested threshold values from 0.05 to 0.95 in increments of 0.05 and analyzed their performance:

| Threshold | Accuracy | % Classified | Correctly Classified |
|-----------|----------|--------------|----------------------|
| 0.05      | 100%     | 100%         | 5/5                  |
| 0.10      | 100%     | 100%         | 5/5                  |
| 0.15      | 100%     | 100%         | 5/5                  |
| 0.20      | 100%     | 100%         | 5/5                  |
| 0.25      | 100%     | 100%         | 5/5                  |
| 0.30      | 100%     | 100%         | 5/5                  |
| 0.35      | 100%     | 100%         | 5/5                  |
| 0.40      | 80%      | 80%          | 4/5                  |
| 0.45      | 80%      | 80%          | 4/5                  |
| 0.50      | 60%      | 60%          | 3/5                  |
| 0.55      | 60%      | 60%          | 3/5                  |
| 0.60      | 40%      | 40%          | 2/5                  |
| 0.65      | 40%      | 40%          | 2/5                  |
| 0.70+     | 0%       | 0%           | 0/5                  |

Our analysis reveals:
- The best accuracy of 100% is achieved with threshold values from 0.05 to 0.35
- The optimal threshold is 0.05, which is the lowest threshold that achieves maximum accuracy
- Lower thresholds are preferable when multiple thresholds give the same accuracy because they classify more samples

As the threshold increases:
- Fewer samples exceed the threshold, leading to more unclassified samples
- The classification rate decreases
- Beyond a threshold of 0.7, no sample is classified

**Key insight**: The optimal threshold balances accuracy and classification coverage. While multiple thresholds achieve 100% accuracy in this example, a lower threshold (0.05) is preferred as it ensures all samples are classified.

![Fixed Threshold Analysis](../Images/L2_4_Quiz_28/step3_fixed_threshold.png)

## Key Insights

### The Relationship Between Log-Likelihood and Model Performance

The log-likelihood of -2.8336 indicates how well the model's predictions align with the true labels. This value can be:
- Used to compare different models on the same dataset
- Optimized during training to improve model performance
- Converted to perplexity or other metrics for interpretation
- Related to cross-entropy loss (negative log-likelihood)

### MLE Classification Principle

The MLE approach demonstrates that:
- Choosing the class with highest probability is mathematically optimal for maximizing likelihood
- This approach achieved perfect accuracy on this dataset (5/5 correct)
- No absolute threshold is needed in this framework—only the relative ordering of probabilities matters
- This is equivalent to using softmax with cross-entropy loss in neural networks

### Fixed Threshold Trade-offs

The fixed threshold analysis reveals important trade-offs:
- Lower threshold → More samples classified → Potentially more errors
- Higher threshold → Fewer samples classified → Higher confidence in classifications
- The optimal fixed threshold (0.05) achieved 100% accuracy with 100% classification rate
- Fixed thresholds can be useful when:
  - The cost of misclassification is high
  - We want to ensure high confidence in our predictions
  - We can tolerate some samples remaining unclassified

### Practical Applications

These concepts have wide-ranging applications:
1. **Medical diagnosis**: Setting higher thresholds to ensure high confidence before diagnosing serious conditions
2. **Fraud detection**: Balancing false positives and false negatives through threshold selection
3. **Content moderation**: Setting appropriate confidence thresholds for automated content filtering
4. **Quality control**: Ensuring high confidence in defect detection while maintaining throughput

## Summary Visualization

The following visualization summarizes our key findings:

![Summary of Findings](../Images/L2_4_Quiz_28/step4_summary.png)

## Conclusion

This analysis demonstrates several fundamental concepts in probabilistic classification:

1. **Log-Likelihood Calculation**: The log-likelihood of -2.8336 measures how well the model's predictions align with the true labels, with higher (less negative) values indicating better model performance.

2. **MLE Threshold**: From a maximum likelihood perspective, the optimal strategy is to classify each sample to the class with the highest probability, which achieved 100% accuracy on this dataset.

3. **Fixed Probability Threshold**: The optimal fixed threshold is 0.05, which achieves 100% accuracy while ensuring all samples are classified. Higher thresholds result in fewer classifications and potentially lower overall accuracy.

4. **Threshold Trade-offs**: Fixed thresholds introduce a trade-off between classification confidence and coverage, allowing practitioners to balance these factors based on the specific requirements of their application.

The principles explored in this problem form the foundation of probabilistic classification and inform the design and evaluation of machine learning models across diverse domains. 