# Lecture 4.6: Multi-class Classification Strategies Quiz

## Overview
This quiz contains 10 questions covering different topics from section 4.6 of the lectures on Multi-class Classification Strategies, including One-vs-All (OVA), One-vs-One (OVO), Error-Correcting Output Codes, Multiclass Perceptron, Softmax Regression, and more.

## Question 1

### Problem Statement
Compare different strategies for extending binary classifiers to handle multi-class problems.

#### Task
1. [🔍] Define the One-vs-All (OVA) and One-vs-One (OVO) approaches in one sentence each
2. [🔍] If we have a dataset with $10$ classes, how many binary classifiers would we need to train for OVA and OVO approaches?
3. [🔍] List one advantage and one disadvantage of OVA compared to OVO
4. [🔍] For what types of base classifiers would the OVO approach be particularly beneficial? Explain in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Multi-class Classification Strategies](L4_6_1_explanation.md).

## Question 2

### Problem Statement
Consider a multi-class classification problem with $4$ classes: A, B, C, and D. You decide to use the One-vs-All (OVA) approach with logistic regression as the base classifier.

#### Task
1. [📚] If you train $4$ logistic regression models with the following score functions, 
   - $f_A(x) = 2.1$
   - $f_B(x) = 1.7$
   - $f_C(x) = -0.5$
   - $f_D(x) = 0.8$
   
   which class would be predicted for a new data point $x$?
2. [📚] Convert these scores to probabilities using the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ and verify that your prediction is the class with highest probability
3. [🔍] When might the OVA approach fail to provide a clear decision for a data point? Explain in one sentence
4. [🔍] How would you resolve ambiguities or conflicts in predictions from different binary classifiers in OVA? Suggest one approach

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: One-vs-All Strategy](L4_6_2_explanation.md).

## Question 3

### Problem Statement
Consider a multi-class classification problem with $4$ classes: A, B, C, and D. You decide to use the One-vs-One (OVO) approach with Logistic Regression as the base classifier.

#### Task
1. [📚] How many binary classifiers will you need to train for the OVO approach? List all pairs
2. [📚] Suppose you get the following predictions from each binary classifier:
   - A vs B: A wins
   - A vs C: C wins
   - A vs D: A wins
   - B vs C: C wins
   - B vs D: B wins
   - C vs D: C wins
   
   Using a voting scheme, which class would you predict?

3. [🔍] What is a potential issue with the voting scheme in OVO when the number of classes is large? Answer in one sentence
4. [🔍] How does OVO handle class imbalance compared to OVA? Explain in one or two sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: One-vs-One Strategy](L4_6_3_explanation.md).

## Question 4

### Problem Statement
Consider a $4$-class classification problem using Error-Correcting Output Codes (ECOC). You design a code matrix as follows:

| Class | Bit 1 | Bit 2 | Bit 3 | Bit 4 | Bit 5 |
|-------|-------|-------|-------|-------|-------|
| A     | +1    | +1    | +1    | -1    | -1    |
| B     | +1    | -1    | -1    | +1    | -1    |
| C     | -1    | +1    | -1    | +1    | +1    |
| D     | -1    | -1    | +1    | -1    | +1    |

#### Task
1. How many binary classifiers need to be trained for this ECOC scheme?
2. For a new data point, the binary classifiers output: $[+1, +1, -1, +1, -1]$. Calculate the Hamming distance between this output and each class codeword
3. Based on the Hamming distances, which class would be predicted for this data point?
4. How does ECOC provide error correction capability? Explain in 1-2 sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Error-Correcting Output Codes](L4_6_4_explanation.md).

## Question 5

### Problem Statement
Consider the multi-class perceptron algorithm, which directly extends the binary perceptron to handle multiple classes.

#### Task
1. [🔍] How does the multi-class perceptron differ from the binary perceptron? Explain in one sentence
2. [🔍] If we have $K$ classes and $d$ features, how many weight vectors and parameters do we need to learn?
3. [📚] For a $3$-class problem with $2$ features, if the weight vectors are:
   - $w_1 = [1, 2]^T$
   - $w_2 = [3, -1]^T$
   - $w_3 = [0, 1]^T$
   
   which class would be predicted for a new point $x = [2, 2]^T$?
4. [📚] If the true label for this point is class $1$, write the update rule for the multi-class perceptron (which weight vectors would be updated and how)

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Multi-class Perceptron](L4_6_5_explanation.md).

## Question 6

### Problem Statement
Consider softmax regression for multi-class classification.

#### Task
1. [🔍] What is the difference between softmax regression and logistic regression? Answer in one sentence
2. [📚] Write the softmax function that converts raw scores to class probabilities
3. [📚] For a $3$-class problem with scores $z_1 = 2$, $z_2 = 0$, and $z_3 = 1$, calculate the softmax probabilities
4. [🔍] How does softmax regression ensure that the predicted probabilities sum to $1$? Explain in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Softmax Regression](L4_6_6_explanation.md).

## Question 7

### Problem Statement
Consider the multi-class cross-entropy loss function for softmax regression:

$$L(w) = -\sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(p_{ik})$$

where $y_{ik}$ is $1$ if example $i$ belongs to class $k$ and $0$ otherwise, and $p_{ik}$ is the predicted probability of example $i$ belonging to class $k$.

#### Task
1. [📚] For a single data point with true class $2$ out of $3$ classes, and predicted probabilities $p_1 = 0.2$, $p_2 = 0.5$, $p_3 = 0.3$, calculate the cross-entropy loss
2. [📚] Derive the gradient of the cross-entropy loss with respect to the model parameters for a single data point
3. [🔍] Why is cross-entropy a more appropriate loss function for multi-class classification than squared error? Answer in one or two sentences
4. [🔍] How does the multi-class cross-entropy loss reduce to binary cross-entropy when there are only two classes? Explain in one or two sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Multi-class Cross Entropy](L4_6_7_explanation.md).

## Question 8

### Problem Statement
Consider Maximum Likelihood Estimation (MLE) for softmax regression in multi-class classification problems.

#### Task
1. [🔍] How does MLE relate to minimizing cross-entropy loss? Answer in one sentence
2. [📚] Write the likelihood function for a multi-class classification problem with softmax regression
3. [📚] Write the log-likelihood function for the same problem
4. [🔍] Why is maximizing the log-likelihood equivalent to minimizing the cross-entropy loss? Answer in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: MLE for Softmax Regression](L4_6_8_explanation.md).

## Question 9

### Problem Statement
Compare different multi-class classification strategies in terms of various metrics.

#### Task
1. [🔍] Complete the following table by marking with X the strategies that have the specified properties:

| Property | OVA | OVO | ECOC | Direct Multi-class (e.g., Softmax) |
|----------|-----|-----|------|----------------------------------|
| Handles large number of classes efficiently |     |     |      |                                  |
| Provides probability estimates naturally |     |     |      |                                  |
| Most computationally efficient during training |     |     |      |                                  |
| Most computationally efficient during prediction |     |     |      |                                  |
| Most robust to class imbalance |     |     |      |                                  |

2. [🔍] Which approach would you recommend for a problem with $100$ classes but limited training data? Explain why in one sentence
3. [🔍] Which approach would you recommend for a problem with $3$ classes and abundant training data? Explain why in one sentence
4. [🔍] How does the choice of base classifier (e.g., Perceptron vs. logistic regression) affect the selection of multi-class strategy? Explain in two sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Strategy Comparison](L4_6_9_explanation.md).

## Question 10

### Problem Statement
Consider multi-class Linear Discriminant Analysis (LDA).

#### Task
1. [🔍] How does LDA naturally handle multi-class problems? Answer in one sentence
2. [🔍] What assumption does LDA make about the class-conditional densities? Answer in one sentence
3. [📚] For a $3$-class problem with class means $\mu_1 = [1, 0]^T$, $\mu_2 = [0, 2]^T$, $\mu_3 = [2, 1]^T$, and shared covariance matrix $\Sigma = I$ (identity matrix), write the discriminant function for each class
4. [📚] For a new data point $x = [1, 1]^T$, determine which class LDA would assign to it

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Multi-class LDA](L4_6_10_explanation.md).