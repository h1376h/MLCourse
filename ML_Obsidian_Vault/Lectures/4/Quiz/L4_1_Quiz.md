# Lecture 4.1: Foundations of Linear Classification Quiz

## Overview
This quiz contains 10 questions covering different topics from section 4.1 of the lectures on Foundations of Linear Classification, including the difference between classification and regression, decision boundaries, feature space, and evaluation metrics.

## Question 1

### Problem Statement
Explain the fundamental differences between classification and regression tasks in machine learning.

#### Task
1. [🔍] Define classification and regression in one sentence each
2. [🔍] List two key differences between classification and regression in terms of output
3. [🔍] Give an example of when you would use classification instead of regression, and vice versa
4. Explain how loss functions differ between classification and regression problems

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Classification vs Regression](L4_1_1_explanation.md).

## Question 2

### Problem Statement
Consider a linear decision boundary in a 2D feature space defined by the equation $2x_1 + 3x_2 - 6 = 0$.

#### Task
1. [📚] Sketch this decision boundary in the $(x_1, x_2)$ plane
2. [📚] Identify the regions in the feature space where points would be classified as positive and negative
3. [📚] For the points $(1, 2)$, $(3, 0)$, and $(2, 1)$, determine which class each point would be assigned to
4. [📚] Calculate the distance from the point $(4, 5)$ to the decision boundary

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Decision Boundaries in 2D](L4_1_2_explanation.md).

## Question 3

### Problem Statement
Consider the following dataset for binary classification:

| Feature 1 | Feature 2 | Class |
|-----------|-----------|-------|
| 1         | 2         | 0     |
| 2         | 3         | 0     |
| 3         | 2         | 0     |
| 5         | 3         | 1     |
| 6         | 2         | 1     |
| 5         | 1         | 1     |

#### Task
1. [📚] Plot the data points in a 2D feature space
2. [📚] Draw a possible linear decision boundary that separates the two classes
3. [📚] Write the equation of your proposed decision boundary in the form $w_1x_1 + w_2x_2 + b = 0$
4. [📚] Can this dataset be perfectly separated by a linear decision boundary? Explain why or why not

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Linear Separability](L4_1_3_explanation.md).

## Question 4

### Problem Statement
Consider a binary classification task with the confusion matrix below:

|                 | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| Actual Positive | 85                 | 15                 |
| Actual Negative | 25                 | 75                 |

#### Task
1. Calculate the following metrics:
   - Accuracy
   - Precision
   - Recall (Sensitivity)
   - Specificity
   - F1 score
2. If the cost of a false negative is twice the cost of a false positive, calculate the total cost
3. Would you consider this a good classifier for a medical diagnosis test? Explain why or why not in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Evaluation Metrics](L4_1_4_explanation.md).

## Question 5

### Problem Statement
A binary classifier outputs probabilities rather than hard classifications. The table below shows the predicted probabilities and true labels for 8 test instances:

| Instance | True Label | Predicted Probability (of class 1) |
|----------|------------|-----------------------------------|
| 1        | 1          | 0.8                               |
| 2        | 0          | 0.3                               |
| 3        | 1          | 0.6                               |
| 4        | 0          | 0.2                               |
| 5        | 1          | 0.7                               |
| 6        | 0          | 0.4                               |
| 7        | 1          | 0.5                               |
| 8        | 0          | 0.1                               |

#### Task
1. [📚] Calculate the confusion matrix using a threshold of 0.5
2. [📚] Calculate accuracy, precision, recall, and F1 score using a threshold of 0.5
3. [📚] If we change the threshold to 0.6, how would the confusion matrix change?
4. [📚] Sketch the ROC curve for this classifier by calculating true positive and false positive rates at thresholds of 0.9, 0.7, 0.5, 0.3, and 0.1

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: ROC Curves](L4_1_5_explanation.md).

## Question 6

### Problem Statement
Consider a binary classification problem in a 3D feature space, where the decision boundary is given by the equation $x_1 + 2x_2 - x_3 + 5 = 0$.

#### Task
1. [🔍] What is the dimensionality of the decision boundary? Explain in one sentence
2. [📚] Write the discriminant function $f(x)$ corresponding to this decision boundary
3. [📚] For a point $x = (2, 1, 3)$, determine which class it would be assigned to
4. [🔍] How would the decision boundary change if we applied the transformation $x'_1 = 2x_1$, $x'_2 = x_2/2$, and $x'_3 = x_3$?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Discriminant Functions](L4_1_6_explanation.md).

## Question 7

### Problem Statement
Consider two different feature representations for a binary classification problem:
- Feature set A: Original features $(x_1, x_2)$
- Feature set B: Transformed features $(x_1, x_2, x_1^2, x_2^2, x_1x_2)$

#### Task
1. [🔍] Explain how the choice of feature representation affects the decision boundary
2. [🔍] For the XOR problem with points at $(0,0)$, $(0,1)$, $(1,0)$, and $(1,1)$ (alternate labels), can a linear classifier solve this with feature set A? Explain why or why not in one sentence
3. [🔍] Can a linear classifier solve the XOR problem with feature set B? Explain why or why not in one sentence
4. [🔍] What is the trade-off when using more complex feature representations? Answer in two sentences

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Feature Spaces](L4_1_7_explanation.md).

## Question 8

### Problem Statement
Below is a table showing the predictions of two different binary classifiers (A and B) on the same test set:

| True Label | Classifier A | Classifier B |
|------------|--------------|--------------|
| 1          | 1            | 1            |
| 1          | 1            | 0            |
| 0          | 0            | 0            |
| 1          | 0            | 1            |
| 0          | 0            | 1            |
| 1          | 1            | 1            |
| 0          | 1            | 0            |
| 1          | 0            | 0            |
| 0          | 0            | 0            |
| 1          | 1            | 1            |

#### Task
1. [📚] Construct the confusion matrix for each classifier
2. [📚] Calculate accuracy, precision, recall, and F1 score for each classifier
3. [🔍] Which classifier would you recommend if false positives are more costly than false negatives? Explain in one sentence
4. [🔍] Which classifier would you recommend if recall is more important than precision? Explain in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Classifier Comparison](L4_1_8_explanation.md).

## Question 9

### Problem Statement
A linear classifier for binary classification is defined by $f(x) = w^Tx + b$, where $w = [2, -1]^T$ and $b = -3$.

#### Task
1. [📚] Write the equation of the decision boundary in the form $w_1x_1 + w_2x_2 + b = 0$
2. [📚] Sketch this decision boundary in the 2D feature space
3. [📚] For the point $(3, 2)$, calculate the value of $f(x)$ and determine its predicted class
4. [📚] Calculate the functional margin of the point $(3, 2)$ with respect to this classifier

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Linear Classifier Fundamentals](L4_1_9_explanation.md).

## Question 10

### Problem Statement
A binary classifier produces the following probabilities for 10 test instances:

| Instance | True Label | Predicted Probability |
|----------|------------|----------------------|
| 1        | 1          | 0.9                  |
| 2        | 1          | 0.8                  |
| 3        | 1          | 0.7                  |
| 4        | 1          | 0.6                  |
| 5        | 1          | 0.4                  |
| 6        | 0          | 0.3                  |
| 7        | 0          | 0.3                  |
| 8        | 0          | 0.2                  |
| 9        | 0          | 0.1                  |
| 10       | 0          | 0.05                 |

#### Task
1. [📚] Calculate the true positive rate (TPR) and false positive rate (FPR) for thresholds of 0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, and 0.02
2. [📚] Plot the ROC curve using these values
3. [📚] Calculate the area under the ROC curve (AUC) using the trapezoidal rule
4. [🔍] What would the ROC curve look like for a perfect classifier? For a random classifier? Sketch both

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: ROC Curve Analysis](L4_1_10_explanation.md). 