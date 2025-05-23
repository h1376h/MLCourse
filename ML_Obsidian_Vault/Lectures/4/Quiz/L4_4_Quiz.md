# Lecture 4.4: Linear Separability and Loss Functions Quiz

## Overview
This quiz contains 31 questions from different topics covered in section 4.4 of the lectures on Linear Separability and Loss Functions.

## Question 1

### Problem Statement
Consider a 2D feature space with the following four data points:
- Class A: $(1, 1)$, $(2, 3)$
- Class B: $(-1, 0)$, $(0, -2)$

#### Task
1. [📚] Sketch these points in a 2D coordinate system
2. [📚] Draw a linear decision boundary that separates these two classes
3. [📚] Write the equation of this decision boundary in the form $w_1x_1 + w_2x_2 + b = 0$
4. [📚] Is this dataset linearly separable? Explain why or why not in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Linear Separability in 2D](L4_4_1_explanation.md).

## Question 2

### Problem Statement
Consider the following loss functions for a classification problem where $y \in \{-1, 1\}$ and $f(x)$ is the model's prediction:

- **0-1 Loss**: 
$$L_{0-1}(y, f(x)) = \begin{cases} 
0 & \text{if } y \cdot f(x) > 0 \\ 
1 & \text{otherwise} 
\end{cases}$$

- **Hinge Loss**: 
$$L_{hinge}(y, f(x)) = \max(0, 1 - y \cdot f(x))$$

- **Logistic Loss**: 
$$L_{log}(y, f(x)) = \log(1 + e^{-y \cdot f(x)})$$

#### Task
1. For a data point with true label $y = 1$ and model prediction $f(x) = 0.5$, calculate all three loss values
2. For a data point with true label $y = -1$ and model prediction $f(x) = -2$, calculate all three loss values
3. Which of these loss functions is non-differentiable? Identify the point(s) of non-differentiability

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Loss Functions Comparison](L4_4_2_explanation.md).

## Question 3

### Problem Statement
Consider Linear Discriminant Analysis (LDA) for a binary classification problem with the following information:
- Class 1 has mean $\mu_1 = [1, 2]^T$
- Class 2 has mean $\mu_2 = [3, 0]^T$
- Both classes share the covariance matrix $\Sigma = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$

#### Task
1. [📚] Calculate the direction of the LDA projection ($w = \Sigma^{-1}(\mu_1 - \mu_2)$)
2. [📚] What is the threshold value for classification in the projected space?
3. [📚] For a new data point $x_1 = [2, 1]^T$, which class would LDA assign it to?
4. [📚] For another new data point $x_2 = [0, 3]^T$, which class would LDA assign it to?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Linear Discriminant Analysis](L4_4_3_explanation.md).

## Question 4

### Problem Statement
For a binary classification problem, we have a dataset with two classes that are non-linearly separable in their original feature space.

#### Task
1. Name two ways we could potentially make this data linearly separable
2. If we applied a quadratic feature transform $\phi(x_1, x_2) = [x_1, x_2, x_1^2, x_2^2, x_1x_2]^T$, explain in one sentence how this might help
3. What is the "kernel trick" and how does it relate to transforming features?
4. Name one advantage and one disadvantage of using feature transformations for linear classifiers

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Feature Transformations](L4_4_4_explanation.md).

## Question 5

### Problem Statement
Consider the margin concept in linear classifiers. The following statements relate to different linear classification methods.

#### Task
For each statement, identify whether it applies to: (a) Perceptron, (b) Logistic Regression, (c) Linear Discriminant Analysis (LDA), or (d) Support Vector Machine (SVM)

1. Finds a decision boundary that maximizes the margin between classes
2. [🔍] Uses a probabilistic approach based on class-conditional densities and Bayes' rule
3. [🔍] Simply tries to find any decision boundary that separates the classes
4. Directly models the posterior probability $P(y|x)$ using the sigmoid function
5. [🔍] Is a discriminative model that maximizes the ratio of between-class to within-class scatter

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Classifier Characteristics](L4_4_5_explanation.md).

## Question 6

### Problem Statement
The XOR problem is a classic example that demonstrates the limitations of linear classifiers. Consider the following binary classification dataset:
- Class A: $(0, 0)$, $(1, 1)$
- Class B: $(0, 1)$, $(1, 0)$

#### Task
1. Sketch these points in a 2D coordinate system
2. Prove that this dataset is not linearly separable in one sentence
3. [📚] If you add a new feature $x_3 = x_1 \cdot x_2$, write down the transformed data points
4. Show that the transformed data is now linearly separable by providing a separating hyperplane equation

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: XOR and Feature Transformation](L4_4_6_explanation.md).

## Question 7

### Problem Statement
Consider the Pocket Algorithm, which is an improvement over the standard Perceptron for non-separable data.

#### Task
1. [🔍] In one sentence, explain the key difference between the standard Perceptron and Pocket Algorithm
2. [🔍] Given the following sequence of weight vectors and their respective number of correctly classified points:
   - $w_1 = [1, 2]$, correctly classifies 6/10 points
   - $w_2 = [0, 3]$, correctly classifies 7/10 points
   - $w_3 = [2, 1]$, correctly classifies 5/10 points
   - $w_4 = [-1, 4]$, correctly classifies 8/10 points
   - $w_5 = [3, 0]$, correctly classifies 7/10 points
   
   What weight vector would the Pocket Algorithm retain after these iterations?
3. [🔍] What weight vector would the standard Perceptron have after these iterations?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Pocket Algorithm](L4_4_7_explanation.md).

## Question 8

### Problem Statement
Let's compare the convexity properties of different loss functions used in linear classification.

#### Task
1. Which of the following loss functions are convex? (Yes/No for each)
   - 0-1 Loss
   - Hinge Loss (SVM)
   - Logistic Loss
   - Squared Error Loss
2. Why is convexity an important property for optimization in machine learning? Answer in one sentence
3. Sketch the shape of the logistic loss function $L(z) = \log(1 + e^{-z})$ for $z \in [-3, 3]$

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Loss Function Properties](L4_4_8_explanation.md).

## Question 9

### Problem Statement
Linear Discriminant Analysis (LDA) makes several assumptions about the underlying data distributions.

#### Task
1. [🔍] List two key assumptions of LDA
2. [📚] Given two classes with equal covariance matrices $\Sigma = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$ and equal prior probabilities, if the means are $\mu_1 = [2, 3]^T$ and $\mu_2 = [4, 1]^T$, at what point would the posterior probabilities $P(C_1|x) = P(C_2|x) = 0.5$?
3. [📚] For a two-class LDA with shared covariance matrix $\Sigma = I$ (identity matrix), write the decision boundary equation in terms of the class means $\mu_1$ and $\mu_2$
4. [🔍] How does LDA differ from the Perceptron in terms of how it finds the decision boundary? Answer in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: LDA Assumptions and Decision Boundary](L4_4_9_explanation.md).

## Question 10

### Problem Statement
Consider a simple 2D dataset with points that are not linearly separable, and you need to apply the Pocket Algorithm.

#### Task
1. Explain the goal of the Pocket Algorithm in one sentence
2. [📚] If after 100 iterations, your Pocket weights are $w = [3, -1, 2]^T$ (including bias term), write the equation of the corresponding decision boundary
3. [🔍] For a perceptron with learning rate $\eta = 0.1$, calculate the weight update for a misclassified point $x = [2, 1]^T$ with true label $y = 1$
4. Why does the Pocket Algorithm perform better than the standard Perceptron for non-separable data? Explain in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Pocket Algorithm Applications](L4_4_10_explanation.md).

## Question 11

### Problem Statement
Consider combining concepts from earlier lectures with linear separability.

#### Task
1. How does the concept of bias-variance tradeoff relate to the choice of linear vs. non-linear decision boundaries?
2. If a dataset has high overlap between classes, which would typically perform better: a linear classifier with regularization or a linear classifier without regularization? Explain in one sentence
3. For a 2D dataset where the optimal Bayes decision boundary is a circle, would a linear classifier or a quadratic feature transform be more appropriate? Why?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: Combining Concepts](L4_4_11_explanation.md).

## Question 12

### Problem Statement
Linear Discriminant Analysis (LDA) approaches classification from a generative modeling perspective, unlike the discriminative approach of Logistic Regression.

#### Task
1. Given the between-class scatter matrix $S_B = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}$ and within-class scatter matrix $S_W = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$, find the direction that maximizes class separation in LDA
2. For binary classification with LDA, if the prior probabilities are equal, where is the decision boundary located relative to the two class means?
3. Compare and contrast how LDA and Logistic Regression would behave with outliers in the training data in one sentence
4. When would you prefer Logistic Regression over LDA? List one specific scenario

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: LDA vs. Logistic Regression](L4_4_12_explanation.md).

## Question 13

### Problem Statement
Consider a dataset with two features $(x_1, x_2)$ and binary labels $y \in \{-1, 1\}$. You're using a linear classifier with decision boundary $2x_1 - 3x_2 + 1 = 0$.

#### Task
1. Calculate the distance from point $(2, 3)$ to this decision boundary
2. For a new data point $(0, 1)$, determine which class the model will predict
3. If you normalize the weight vector to unit length, what would the new decision boundary equation be?
4. Sketch the decision boundary in a 2D coordinate system and indicate the positive and negative regions

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 13: Decision Boundary Geometry](L4_4_13_explanation.md).

## Question 14

### Problem Statement
You've trained two linear classifiers on the same dataset:
- Model A: Linear Perceptron with weights $w = [2, -1]^T$ and bias $b = 0.5$
- Model B: Linear Discriminant Analysis (LDA)

For a new data point, you want to understand how these models make their classification decisions.

#### Task
1. Write the decision boundary equation for Model A in the form $w_1x_1 + w_2x_2 + b = 0$
2. If the true data-generating distributions are Gaussian with equal covariance, which model is theoretically more appropriate? Explain why in one sentence
3. For a test point $(1, 2)$, determine which class Model A will predict
4. How does LDA's approach to finding the decision boundary differ from the Perceptron's approach? Explain in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 14: Comparing Classifiers](L4_4_14_explanation.md).

## Question 15

### Problem Statement
Consider a linear classifier with the following loss function:
$$L(y, f(x)) = \exp(-y \cdot f(x))$$
where $y \in \{-1, 1\}$ is the true label and $f(x) = w^T x + b$ is the model's prediction.

#### Task
1. Compute the gradient of this loss function with respect to $w$
2. Compare this exponential loss with the hinge loss in terms of how they penalize misclassified points
3. For a correctly classified point with margin $y \cdot f(x) = 2$, calculate the loss value
4. Is this loss function convex? Explain why or why not in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 15: Exponential Loss](L4_4_15_explanation.md).

## Question 16

### Problem Statement
You have a dataset with two perfectly linearly separable classes. Considering both the Perceptron algorithm and the Pocket algorithm:

#### Task
1. Would the Pocket algorithm give a different result than the standard Perceptron for this dataset? Explain why or why not in one sentence
2. If we add 5% random noise to the class labels (flipping some labels), which algorithm would be more robust? Explain in one sentence
3. Draw a simple 2D example where the Perceptron's final solution depends on the initialization of weights
4. How does the Pocket algorithm relate to the concept of empirical risk minimization from statistical learning theory? Answer in one sentence

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 16: Perceptron vs Pocket](L4_4_16_explanation.md).

## Question 17

### Problem Statement
Consider a linear classifier being trained on a dataset with the following properties:
- The true data distribution has significant class overlap
- There are a few extreme outliers in the minority class

#### Task
1. Which is likely to perform better on this dataset: a model trained with 0-1 loss or a model trained with hinge loss? Explain why in one sentence
2. How would the Pocket algorithm handle the outliers compared to the standard Perceptron? Explain in one sentence
3. Connect this scenario to the bias-variance tradeoff concept from earlier lectures
4. Draw a simple illustration of how LDA might place its decision boundary differently than a standard Perceptron in this scenario

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 17: Robust Classification](L4_4_17_explanation.md).

## [⭐] Question 18

### Problem Statement
Consider a medical dataset with tumor features and diagnostic outcomes. Each patient has data on age (years) and tumor size (mm), with the target variable $y$ indicating whether the tumor is malignant (1) or benign (0).

| Age (years) | Tumor Size (mm) | $y$ (Malignant) |
|-------------|-----------------|-----------------|
| 15          | 20              | 0               |
| 65          | 30              | 0               |
| 30          | 50              | 1               |
| 90          | 20              | 1               |
| 44          | 35              | 0               |
| 20          | 70              | 1               |
| 50          | 40              | 1               |
| 36          | 25              | 0               |

#### Task
1. Calculate the mean vectors for each class (malignant and benign)
2. Calculate the shared covariance matrix assuming equal covariance for both classes
3. Determine the LDA projection direction $w = \Sigma^{-1}(\mu_1 - \mu_2)$ where $\mu_1$ is the mean for class $y=1$ and $\mu_2$ is the mean for class $y=0$
4. Calculate the threshold value for classification in the projected space, assuming equal prior probabilities
5. For a new patient with age 50 years and tumor size 30mm, which diagnosis would LDA predict?
6. For another patient with age 60 years and tumor size 30mm, which diagnosis would LDA predict?

#### Solution Approaches
This problem can be tackled using multiple approaches:

For a detailed explanation using the shared covariance matrix formula, which provides a standard LDA implementation, see [Question 18: LDA for Medical Diagnosis](L4_4_18_explanation.md).

For an alternative approach using the within-class scatter matrix formulation, which was used in our class lectures, see [Alternative Solution: Within-Class Scatter Approach](L4_4_18_explanation_alternative.md).

Note that while the numerical values differ slightly between these approaches, the fundamental concepts and final classification decisions remain the same, as the relative relationships between variables are preserved.

## Question 19

### Problem Statement
A bank is using LDA to classify credit applications as approved (1) or denied (0) based on annual income (thousands of dollars) and debt-to-income ratio (percentage). The following table shows training data from previous applications:

| Income ($K) | Debt-to-Income (%) | Credit Approved (y) |
|-------------|---------------------|---------------------|
| 65          | 28                  | 1                   |
| 50          | 32                  | 0                   |
| 79          | 22                  | 1                   |
| 48          | 40                  | 0                   |
| 95          | 18                  | 1                   |
| 36          | 36                  | 0                   |
| 72          | 30                  | 1                   |
| 60          | 34                  | 0                   |
| 85          | 24                  | 1                   |
| 42          | 38                  | 0                   |

#### Task
1. Calculate the class means for approved and denied applications
2. Calculate the pooled within-class covariance matrix
3. Find the between-class covariance matrix $S_B$
4. Determine the optimal projection direction for the LDA by finding the eigenvector corresponding to the largest eigenvalue of $S_W^{-1}S_B$
5. Calculate the threshold for classification assuming the prior probabilities are $P(y=1) = 0.3$ and $P(y=0) = 0.7$
6. For a new applicant with income $55K and debt-to-income ratio 25%, which class would LDA predict? Will their credit application be approved or denied?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 19: LDA for Credit Approval](L4_4_19_explanation.md). 

## Question 20

### Problem Statement
You are given data from two classes with the following 2-dimensional feature vectors:

**Class 0:** $\mathbf{x}^{(1)}=\begin{bmatrix} 1 \\ 2 \end{bmatrix}$, $\mathbf{x}^{(2)}=\begin{bmatrix} 2 \\ 3 \end{bmatrix}$, $\mathbf{x}^{(3)}=\begin{bmatrix} 3 \\ 3 \end{bmatrix}$  
**Class 1:** $\mathbf{x}^{(1)}=\begin{bmatrix} 5 \\ 2 \end{bmatrix}$, $\mathbf{x}^{(2)}=\begin{bmatrix} 6 \\ 3 \end{bmatrix}$, $\mathbf{x}^{(3)}=\begin{bmatrix} 6 \\ 4 \end{bmatrix}$

Assume that the feature vectors in each class follow a multivariate Gaussian distribution.

#### Task
1. Calculate the mean vector for each class
2. Calculate the within-class scatter matrix $S_W$ by summing the individual class scatter matrices
3. Calculate the between-class scatter matrix $S_B = (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^T$
4. Determine the LDA projection direction by finding the eigenvector corresponding to the largest eigenvalue of $S_W^{-1}S_B$
5. Assuming equal prior probabilities, calculate the threshold for classification in the projected space
6. Using LDA, classify the new data point $\mathbf{x}_{\text{new}} = \begin{bmatrix} 4 \\ 3 \end{bmatrix}$ and explain how the decision boundary relates to linear separability

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 20: LDA with Scatter Matrices](L4_4_20_explanation.md).

## Question 21

### Problem Statement
Consider a binary classification problem with the following dataset:

| A    | B    | Class |
|------|------|-------|
| 3.5  | 4.0  | 1     |
| 2.0  | 4.0  | 1     |
| 2.0  | 6.0  | 1     |
| 1.5  | 7.0  | 1     |
| 7.0  | 6.5  | 1     |
| 2.1  | 2.5  | 0     |
| 8.0  | 4.0  | 0     |
| 9.1  | 4.5  | 0     |

An LDA model has been developed for this dataset with the following statistics:
- Sample means: $\boldsymbol{\mu}_1 = [3.2, 5.5]^T$ for Class 1 and $\boldsymbol{\mu}_0 = [6.4, 3.7]^T$ for Class 0
- Covariance matrices: 
$$\Sigma_1 = \begin{bmatrix} 5.08 & 0.5 \\ 0.5 & 2 \end{bmatrix}$$ 
for Class 1, and 
$$\Sigma_0 = \begin{bmatrix} 14.7 & 3.9 \\ 3.9 & 1.08 \end{bmatrix}$$ 
for Class 0

#### Task
1. Explain how having different covariance matrices for each class affects the LDA assumptions
2. Using the discriminant function approach, calculate the posterior probabilities for the new data point $\mathbf{x}_{\text{new}} = [4, 5]^T$
3. Determine the predicted class for the new data point $\mathbf{x}_{\text{new}} = [4, 5]^T$
4. Describe how the decision boundary would be different if we used a pooled covariance matrix (standard LDA)

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 21: LDA Prediction](L4_4_21_explanation.md).

## Question 22

### Problem Statement
Consider the following table with gender and height data:

| Gender | Height | Estimation |
|--------|--------|------------|
| F      | 160    |            |
| M      | 160    |            |
| F      | 170    |            |
| M      | 170    |            |
| M      | 170    |            |
| M      | 180    |            |

#### Task
1. Based on the LDA method, estimate the category (gender) for each person in the table
2. Show your work and explain the LDA classification process
3. What assumptions does LDA make about the data distribution?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 22: LDA Classification](L4_4_22_explanation.md).

## Question 23

### Problem Statement
Find the LDA classifier for a training set of eight points: 
- Class 1: $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$, $\begin{bmatrix} -1 \\ 0 \end{bmatrix}$, $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$, $\begin{bmatrix} 0 \\ -1 \end{bmatrix}$
- Class 2: $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$, $\begin{bmatrix} 0 \\ -1 \end{bmatrix}$, $\begin{bmatrix} 2 \\ 1 \end{bmatrix}$, $\begin{bmatrix} 2 \\ -1 \end{bmatrix}$

#### Task
1. Calculate the mean vector $\boldsymbol{\mu}_1$ and $\boldsymbol{\mu}_2$ for each class
2. Calculate the shared covariance matrix $\boldsymbol{\Sigma}$ assuming equal covariance for both classes
3. Determine the LDA projection direction $\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$
4. Calculate the threshold value for classification in the projected space, assuming equal prior probabilities
5. For a new point $\mathbf{x}_{\text{new}} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$, which class would LDA assign it to?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 23: LDA Classification](L4_4_23_explanation.md).

## Question 24

### Problem Statement
Compute LDA projection for the following two-dimensional dataset:
- $X_1=(x_1, x_2)=\{(4, 1), (2, 4), (2, 3), (3, 6), (4, 4)\}$
- $X_2=(x_1, x_2)=\{(9, 10), (6, 8), (9, 5), (8, 7), (10, 8)\}$

#### Task
1. Calculate the mean vectors $\mu_1$ and $\mu_2$ for each class
2. Compute the within-class scatter matrices $S_1$ and $S_2$ for each class
3. Determine the total within-class scatter matrix $S_W$
4. Calculate the between-class scatter matrix $S_B$
5. Find the optimal projection direction $\mathbf{w}$ by solving the generalized eigenvalue problem
6. For a new data point $(5, 5)$, determine which class it would be assigned to using LDA

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 24: LDA Projection for Two-Dimensional Dataset](L4_4_24_explanation.md).

## Question 25

### Problem Statement
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

![LDA Visualization](../Images/L4_4_Quiz_25/lda_visualization.png)

#### Task
1. [🔍]  LDA method projects p-dimensional data into a one-dimensional space and then compares it with a threshold to determine the class label
2. [🔍] LDA method is more appropriate for linearly separable data.
3. [🔍] In developing LDA, the mean values of both classes $m_1 = \sum_{i=1}^{N_1} x_i$ and $m_2 = \sum_{i=1}^{N_2} x_i$ play essential roles.
4. [🔍] The main objective of this approach is to transform data into a space such that the resulting data points demonstrate minimum within-class variations and maximum between-class variations.
5. [🔍] The resulting model using LDA is always equivalent to that of linear classification with LSE.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 25: Fisher's Linear Discriminant Analysis](L4_4_25_explanation.md).

## Question 26

### Problem Statement
Linear discriminant analysis has many applications, such as dimensionality reduction and feature extraction. In this problem, we consider a simple task with two classes expressed as follows:

- Class 0: $\mathbf{x}_1^{(0)} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$, $\mathbf{x}_2^{(0)} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

- Class 1: $\mathbf{x}_1^{(1)} = \begin{bmatrix} 4 \\ 3 \end{bmatrix}$, $\mathbf{x}_2^{(1)} = \begin{bmatrix} 5 \\ 4 \end{bmatrix}$

Note that in this problem we use column vectors for the data points to simplify the calculation.

#### Task
1. [📚] Compute the mean vector for each class, $\mu_0$ and $\mu_1$.
2. [📚] Compute the covariance matrix for each class, $\Sigma_0$ and $\Sigma_1$.

The Fisher's linear discriminant analysis is defined to maximize criterion function:

$$S(\mathbf{w}) = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{within}}} = \frac{(\mathbf{w}^{\top} \mu_0 - \mathbf{w}^{\top} \mu_1)^2}{\mathbf{w}^{\top} (\Sigma_0 + \Sigma_1)\mathbf{w}}$$

An optimal solution $\mathbf{w}^*$ is:

$$\mathbf{w}^* = (\Sigma_0 + \Sigma_1)^{-1}(\mu_0 - \mu_1)$$

3. [📚] Find the optimal $\mathbf{w}^*$ with unit length.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 26: Linear Discriminant Analysis with Simple Data](L4_4_26_explanation.md).

## Question 27

### Problem Statement
Linear discriminant analysis has many applications, such as dimensionality reduction and feature extraction. In this problem, we consider a special case with two classes expressed as follows:

- Class A: $\mathbf{x}_1^{(A)} = \begin{bmatrix} 1 \\ 3 \end{bmatrix}$, $\mathbf{x}_2^{(A)} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$

- Class B: $\mathbf{x}_1^{(B)} = \begin{bmatrix} 6 \\ 4 \end{bmatrix}$, $\mathbf{x}_2^{(B)} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}$

Note that in this problem we use column vectors for the data points to simplify the calculation.

#### Task
1. [📚] Compute the mean vector for each class, $\mu_A$ and $\mu_B$.
2. [📚] Compute the covariance matrix for each class, $\Sigma_A$ and $\Sigma_B$.

Fisher's linear discriminant analysis aims to maximize the criterion function:

$$S(\mathbf{w}) = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{within}}} = \frac{(\mathbf{w}^{\top} \mu_A - \mathbf{w}^{\top} \mu_B)^2}{\mathbf{w}^{\top} (\Sigma_A + \Sigma_B)\mathbf{w}}$$

An optimal solution $\mathbf{w}^*$ is:

$$\mathbf{w}^* = (\Sigma_A + \Sigma_B)^{-1}(\mu_A - \mu_B)$$

3. [📚] Find the optimal $\mathbf{w}^*$ with unit length.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 27: Linear Discriminant Analysis with Singular Scatter Matrix](L4_4_27_explanation.md).

## Question 28

### Problem Statement
You are a fruit vendor trying to automatically sort fruits. You have data for two types of fruits, "SweetApples" (label +1) and "SourOranges" (label -1), based on two features: "Sweetness" ($x_1$, scale 1-10) and "Sourness" ($x_2$, scale 1-10).

| Fruit ID     | Sweetness ($x_1$) | Sourness ($x_2$) | Class ($y$) |
|--------------|-------------------|------------------|-------------|
| SA1          | 8                 | 2                | +1          |
| SA2          | 7                 | 3                | +1          |
| SO1          | 3                 | 8                | -1          |
| SO2          | 2                 | 7                | -1          |

#### Task
1. Sketch these points in a 2D coordinate system (Sweetness as x-axis, Sourness as y-axis), labeling each point with its Fruit ID and class.
2. Draw a linear decision boundary that separates these two classes and write its equation in the form $w_1x_1 + w_2x_2 + b = 0$.
3. [📚] Consider a Perceptron algorithm with learning rate $\eta = 1$ and initial weights $w_1=0, w_2=0, b=0$. The Perceptron updates its weights if $y \cdot (w_1x_1 + w_2x_2 + b) \leq 0$. If the points are processed in the order SA1, SA2, SO1, SO2, what are the weights $(w_1, w_2, b)$ after the *first* update? Show your calculation for the first point processed.
4. Is this dataset linearly separable? Explain why or why not in one sentence.
5. Apply Fisher's Linear Discriminant Analysis (LDA) to find the optimal projection direction for this dataset.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 28: Fruit Classification with Perceptron and Fisher's LDA](L4_4_28_explanation.md).

## Question 29

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

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 29: Pocket Algorithm](L4_4_29_explanation.md).


## Question 30

### Problem Statement
Consider the Pocket Algorithm applied to a dataset with some outliers.

#### Task
1. [🔍] For a non-linearly separable dataset, explain how the Pocket Algorithm decides which weight vector to keep in the "pocket". Answer in one sentence.
2. [📚] Apply the Pocket Algorithm to the following dataset:

| $x_1$ | $x_2$ | $y$ (target) |
|-------|-------|--------------|
| -2    | 0     | -1           |
| -1    | -2    | -1           |
| 0     | -1    | -1           |
| 1     | 1     | 1            |
| 2     | 2     | 1            |
| 0     | 3     | 1            |
| 4     | 0     | -1           |

   Starting with weights $w = [0, 0, 0]^T$ and learning rate $\eta = 1$, run 4 iterations, showing which examples are misclassified at each step, the updated weights, and which weight vector is kept in the "pocket".

3. [📚] Calculate the classification accuracy of your final "pocket" weights versus the final perceptron weights.
4. [🔍] If you were to continue training, would the "pocket" weights necessarily improve? Explain your answer in one or two sentences.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 30: Pocket Algorithm with Outliers](L4_4_30_explanation.md).

## Question 31

### Problem Statement
Consider applying the Pocket Algorithm with different random initializations.

#### Task
1. [📚] For the following dataset:

| $x_1$ | $x_2$ | $y$ (target) |
|-------|-------|--------------|
| 2     | 0     | 1            |
| 0     | 2     | 1            |
| -2    | 0     | -1           |
| 0     | -2    | -1           |
| 2     | 2     | -1           |
| -2    | -2    | 1            |

   Run 3 iterations of the Pocket Algorithm with two different initializations:
   - Initialization 1: $w = [1, 0, 0]^T$
   - Initialization 2: $w = [0, 1, 0]^T$

   Use learning rate $\eta = 1$ for both runs. Show your work for each initialization including which weight vector is kept in the "pocket" at each step.

2. [📚] Compare the performance of the final "pocket" weights from both initializations. Which one classifies more points correctly?
3. [📚] Draw the decision boundaries corresponding to both final "pocket" weights in the same coordinate system.
4. [🔍] Based on this example, explain in one or two sentences why multiple runs with different initializations might be beneficial when using the Pocket Algorithm.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 31: Pocket Algorithm Initializations](L4_4_31_explanation.md). 