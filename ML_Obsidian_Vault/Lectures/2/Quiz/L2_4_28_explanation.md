# Question 28: One-Hot Encoding and Maximum Likelihood Estimation

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
This problem explores key concepts in classification, particularly how to make decisions based on predicted probabilities:

- One-hot encoding represents categorical variables as binary vectors
- Log-likelihood measures how well a model's probability predictions match the true labels
- Maximum Likelihood Estimation (MLE) provides principles for optimal classification
- Threshold-based classification requires determining an appropriate cutoff value

We need to understand how to evaluate the model's performance using log-likelihood and determine optimal classification thresholds from both theoretical (MLE) and practical (accuracy-based) perspectives.

## Solution

### Step 1: Log-Likelihood of the Observed Data

The log-likelihood measures how well the model's predicted probabilities match the observed data. For our multinomial classifier, we calculate the sum of the logarithms of the predicted probabilities for the correct classes across all samples:

$$\log L = \sum_{i=1}^{n} \log(\hat{y}_{i,c_i})$$

Where:
- $\hat{y}_{i,c_i}$ is the predicted probability for the correct class of sample $i$
- $c_i$ is the index of the correct class for sample $i$

Let's calculate this step-by-step for each sample:

| Sample | True Class | Probabilities | Probability of True Class | Log Probability |
|--------|------------|---------------|---------------------------|----------------|
| 1      | Cat        | [0.7, 0.2, 0.1] | 0.70 | -0.3567 |
| 2      | Dog        | [0.3, 0.5, 0.2] | 0.50 | -0.6931 |
| 3      | Bird       | [0.1, 0.3, 0.6] | 0.60 | -0.5108 |
| 4      | Cat        | [0.4, 0.4, 0.2] | 0.40 | -0.9163 |
| 5      | Bird       | [0.2, 0.1, 0.7] | 0.70 | -0.3567 |

Summing these values, we get the log-likelihood:
$$\log L = -0.3567 + (-0.6931) + (-0.5108) + (-0.9163) + (-0.3567) = -2.8336$$

This log-likelihood value of -2.8336 quantifies how well the model's predictions align with the true labels. A higher (less negative) value indicates better predictive performance.

![Log-Likelihood Calculation](../Codes/images/step1_log_likelihood.png)

### Step 2: MLE Threshold for Classification

From a Maximum Likelihood Estimation (MLE) perspective, the optimal classification strategy is to assign each sample to the class with the highest probability. This maximizes the likelihood of the observed data.

This approach corresponds to using a relative threshold (comparing probabilities within each sample) rather than an absolute fixed threshold value.

Let's analyze the classification of each sample using the maximum probability rule:

| Sample | Probabilities | Max Probability Class | True Class | Classification Result |
|--------|---------------|------------------------|------------|----------------------|
| 1      | [0.7, 0.2, 0.1] | Cat (0.70) | Cat | Correct |
| 2      | [0.3, 0.5, 0.2] | Dog (0.50) | Dog | Correct |
| 3      | [0.1, 0.3, 0.6] | Bird (0.60) | Bird | Correct |
| 4      | [0.4, 0.4, 0.2] | Cat (0.40)* | Cat | Correct |
| 5      | [0.2, 0.1, 0.7] | Bird (0.70) | Bird | Correct |

*Note: For Sample 4, there's a tie between Cat and Dog (both 0.40). We've chosen Cat as the prediction since it's the first class with the maximum probability.

Using the maximum probability classification rule gives us perfect accuracy (5/5 correct classifications) on this dataset. This demonstrates that from an MLE perspective, the optimal strategy is to classify each sample according to the class with the highest probability.

![MLE Classification Approach](../Codes/images/step2_mle_threshold.png)

### Step 3: Fixed Probability Threshold for Maximum Accuracy

If we must use a fixed probability threshold, where a class is assigned only if its probability exceeds that threshold, we need to determine the optimal value that maximizes accuracy on this dataset.

For a fixed threshold strategy:
- If no class exceeds the threshold, the sample remains unclassified
- If exactly one class exceeds the threshold, we assign the sample to that class
- If multiple classes exceed the threshold, we assign the sample to the class with the highest probability

Let's analyze the performance of different thresholds from 0.05 to 0.95:

| Threshold | Correctly Classified | Total Classified | Accuracy |
|-----------|----------------------|------------------|----------|
| 0.05 - 0.35 | 5/5 | 5/5 | 100% |
| 0.40 - 0.45 | 4/5 | 4/5 | 80% |
| 0.50 - 0.55 | 3/5 | 3/5 | 60% |
| 0.60 - 0.65 | 2/5 | 2/5 | 40% |
| 0.70 - 0.95 | 0/5 | 0/5 | 0% |

Based on this analysis:
- The best accuracy (100%) is achieved with thresholds from 0.05 to 0.35
- The optimal threshold is 0.05, which is the lowest threshold that achieves maximum accuracy
- Lower thresholds are preferable when multiple thresholds give the same accuracy because they classify more samples while maintaining the same level of accuracy

As the threshold increases beyond 0.35:
- Fewer samples can be classified (because fewer probabilities exceed the threshold)
- Accuracy decreases accordingly
- At thresholds of 0.70 and above, no samples can be classified at all

![Fixed Threshold Analysis](../Codes/images/step3_fixed_threshold.png)

## Key Insights

### Theoretical Foundations
- Log-likelihood provides a principled way to evaluate probabilistic predictions without requiring classification decisions
- MLE principles suggest classifying according to the highest probability class, regardless of absolute values
- From an information-theoretic perspective, using all probability information (rather than thresholded decisions) typically leads to better performance
- Fixed thresholds introduce a trade-off between classification confidence and the number of classified samples

### Practical Applications
- MLE-based classification (choosing the highest probability class) is often preferable to fixed thresholds
- When multiple thresholds yield the same accuracy, selecting the lowest reasonable threshold maximizes coverage
- The perfect accuracy achieved in this example (with MLE approach) suggests the model has successfully learned to distinguish between the three classes
- Probability outputs contain more information than binary decisions, and preserving this information can be valuable for downstream tasks

### Common Pitfalls
- Using too high a threshold can result in many unclassified samples or even no classifications
- Using too low a threshold might classify samples with low confidence that should be rejected
- Fixed thresholds don't account for class-specific considerations (some classes might need higher confidence)
- Ties in probabilities (as in Sample 4) require additional tie-breaking rules

## Conclusion

This problem demonstrates important principles in classification decision-making:

1. The log-likelihood (-2.8336 in this case) provides a comprehensive measure of how well a model's probabilistic predictions match observed data.
2. From an MLE perspective, the optimal classification strategy is to assign each sample to its highest-probability class (achieving 100% accuracy in this example).
3. When using fixed thresholds, there's a clear trade-off between classification confidence and coverage.
4. The optimal fixed threshold for this dataset is 0.05, which achieves 100% accuracy while maximizing the number of classified samples.
5. As thresholds increase, fewer samples can be classified, eventually leading to no classifications at all (at thresholds ≥ 0.70).

These principles apply broadly to probabilistic classification models, helping practitioners make informed decisions about how to convert probabilities into class assignments while considering the specific requirements of their applications. 