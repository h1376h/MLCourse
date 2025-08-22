# Lecture 5.3: Kernel Trick for Nonlinear Classification Quiz

## Overview
This quiz contains 4 questions covering different topics from section 5.3 of the lectures on Kernel Trick, Feature Space Transformation, Common Kernels, and Mercer's Theorem.

## Question 1

### Problem Statement
Consider a dataset that is not linearly separable in the original feature space but can be separated using a nonlinear decision boundary.

#### Task
1. Explain the concept of feature space transformation and how it enables linear separation of nonlinearly separable data
2. What is the "kernel trick" and why is it computationally advantageous?
3. Give a concrete example of a 2D dataset that is not linearly separable but becomes linearly separable after transformation to a higher-dimensional space
4. What would be the computational complexity of explicitly computing features in very high-dimensional spaces?

For a detailed explanation of this problem, see [Question 1: Kernel Trick Concept](L5_3_1_explanation.md).

## Question 2

### Problem Statement
Consider different types of kernel functions and their properties.

#### Task
1. List three common kernel functions used in SVMs and write their mathematical expressions
2. For the RBF (Gaussian) kernel $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2)$:
   - What does the parameter $\gamma$ control?
   - What happens when $\gamma$ is very large vs. very small?
   - How does $\gamma$ affect the bias-variance tradeoff?
3. For the polynomial kernel $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d$:
   - What do the parameters $c$ and $d$ represent?
   - What are the computational implications of increasing the degree $d$?

For a detailed explanation of this problem, see [Question 2: Common Kernel Functions](L5_3_2_explanation.md).

## Question 3

### Problem Statement
Consider the mathematical requirements for valid kernel functions and Mercer's theorem.

#### Task
1. What conditions must a function satisfy to be a valid kernel function?
2. State Mercer's theorem and explain its significance for kernel methods
3. What is a Gram matrix (kernel matrix) and what property must it satisfy?
4. Given two valid kernels $K_1(\mathbf{x}, \mathbf{z})$ and $K_2(\mathbf{x}, \mathbf{z})$, prove that the following are also valid kernels:
   - $K(\mathbf{x}, \mathbf{z}) = K_1(\mathbf{x}, \mathbf{z}) + K_2(\mathbf{x}, \mathbf{z})$
   - $K(\mathbf{x}, \mathbf{z}) = c \cdot K_1(\mathbf{x}, \mathbf{z})$ for $c > 0$
5. Why is it important to verify that a function is a valid kernel before using it in SVMs?

For a detailed explanation of this problem, see [Question 3: Mercer's Theorem and Valid Kernels](L5_3_3_explanation.md).

## Question 4

### Problem Statement
Consider the practical aspects of kernel selection and parameter tuning in SVMs.

#### Task
1. How would you approach selecting the most appropriate kernel for a given dataset?
2. Describe a systematic methodology for tuning kernel parameters (e.g., $\gamma$ for RBF, degree for polynomial)
3. What are the computational and memory considerations when working with different kernels?
4. Explain the concept of "kernel alignment" and how it can guide kernel selection
5. Design a custom kernel function for text classification where you want to measure similarity based on common n-grams. Verify that your proposed function satisfies the requirements of a valid kernel.

For a detailed explanation of this problem, see [Question 4: Kernel Selection and Parameter Tuning](L5_3_4_explanation.md).
