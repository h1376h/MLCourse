# Lecture 5.3: Kernel Trick for Nonlinear Classification Quiz

## Overview
This quiz contains 16 questions covering different topics from section 5.3 of the lectures on Kernel Trick, Feature Space Transformation, Common Kernels, RBF Kernels, Polynomial Kernels, Mercer's Theorem, and Kernel Selection.

## Question 1

### Problem Statement
Consider the classic XOR problem with four points:
- $(0, 0) \rightarrow y = -1$
- $(0, 1) \rightarrow y = +1$
- $(1, 0) \rightarrow y = +1$
- $(1, 1) \rightarrow y = -1$

#### Task
1. Prove that this dataset is not linearly separable in $\mathbb{R}^2$
2. Apply the feature transformation $\phi(x_1, x_2) = (x_1, x_2, x_1x_2)$ and show the transformed points
3. Find a linear hyperplane in the 3D feature space that separates the transformed data
4. Express the decision boundary in the original 2D space
5. Calculate the kernel function $K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^T\phi(\mathbf{z})$ for this transformation

For a detailed explanation of this problem, see [Question 1: XOR Problem and Feature Transformation](L5_3_1_explanation.md).

## Question 2

### Problem Statement
Analyze the computational complexity of explicit feature mapping versus the kernel trick.

#### Task
1. For a polynomial kernel of degree $d$ applied to $n$-dimensional input, derive the number of features in the explicit mapping
2. Calculate this for $n = 10, d = 3$ and $n = 100, d = 2$
3. What is the computational cost of computing $\phi(\mathbf{x})^T\phi(\mathbf{z})$ explicitly versus using $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + 1)^d$?
4. For what values of $n$ and $d$ does the kernel trick provide significant savings?
5. Analyze the memory requirements for storing the feature vectors versus kernel evaluations

For a detailed explanation of this problem, see [Question 2: Computational Complexity Analysis](L5_3_2_explanation.md).

## Question 3

### Problem Statement
Work with specific kernel calculations for the polynomial kernel $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + c)^d$.

Given vectors $\mathbf{x} = (2, -1, 3)$ and $\mathbf{z} = (1, 2, -1)$.

#### Task
1. Calculate $K(\mathbf{x}, \mathbf{z})$ for $c = 1, d = 2$
2. Calculate $K(\mathbf{x}, \mathbf{z})$ for $c = 0, d = 3$
3. Find the explicit feature mapping $\phi(\mathbf{x})$ for the case $c = 0, d = 2$ in 3D
4. Verify that $K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^T\phi(\mathbf{z})$ for your calculated mapping
5. How does the parameter $c$ affect the relative importance of different order terms?

For a detailed explanation of this problem, see [Question 3: Polynomial Kernel Calculations](L5_3_3_explanation.md).

## Question 4

### Problem Statement
Analyze the RBF (Gaussian) kernel: $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma ||\mathbf{x} - \mathbf{z}||^2)$.

#### Task
1. Calculate $K(\mathbf{x}, \mathbf{z})$ for $\mathbf{x} = (1, 0)$, $\mathbf{z} = (0, 1)$ with $\gamma = 0.5, 1, 2$
2. Show that $K(\mathbf{x}, \mathbf{x}) = 1$ for any $\mathbf{x}$
3. Prove that $0 \leq K(\mathbf{x}, \mathbf{z}) \leq 1$ for any $\mathbf{x}, \mathbf{z}$
4. Derive the behavior of $K(\mathbf{x}, \mathbf{z})$ as $||\mathbf{x} - \mathbf{z}|| \rightarrow \infty$
5. Show that the RBF kernel corresponds to an infinite-dimensional feature space

For a detailed explanation of this problem, see [Question 4: RBF Kernel Properties](L5_3_4_explanation.md).

## Question 5

### Problem Statement
Study the effect of the RBF kernel parameter $\gamma$ on decision boundaries.

#### Task
1. For a 1D dataset with points $x_1 = -1, x_2 = 1$ (different classes), sketch the decision boundary for $\gamma = 0.1, 1, 10$
2. Predict how $\gamma$ affects overfitting and underfitting
3. Derive the limit behavior as $\gamma \rightarrow 0$ and $\gamma \rightarrow \infty$
4. Design a synthetic 2D dataset where small $\gamma$ performs better than large $\gamma$
5. Calculate the effective "width" of influence for each data point as a function of $\gamma$

For a detailed explanation of this problem, see [Question 5: RBF Parameter Effects](L5_3_5_explanation.md).

## Question 6

### Problem Statement
Examine Mercer's theorem and the conditions for valid kernels.

#### Task
1. State Mercer's theorem precisely
2. For the kernel matrix $K = \begin{pmatrix} 1 & 0.5 \\ 0.5 & 1 \end{pmatrix}$, verify that it's positive semi-definite
3. Check if $K(\mathbf{x}, \mathbf{z}) = \exp(\mathbf{x}^T\mathbf{z})$ is a valid kernel
4. Prove that $K(\mathbf{x}, \mathbf{z}) = -||\mathbf{x} - \mathbf{z}||^2$ is not a valid kernel
5. Design a 2D example showing why non-PSD kernels lead to optimization problems

For a detailed explanation of this problem, see [Question 6: Mercer's Theorem and Kernel Validity](L5_3_6_explanation.md).

## Question 7

### Problem Statement
Explore kernel combinations and closure properties.

#### Task
1. Prove that if $K_1$ and $K_2$ are valid kernels, then $K(\mathbf{x}, \mathbf{z}) = K_1(\mathbf{x}, \mathbf{z}) + K_2(\mathbf{x}, \mathbf{z})$ is valid
2. Show that $K(\mathbf{x}, \mathbf{z}) = cK_1(\mathbf{x}, \mathbf{z})$ is valid for $c > 0$
3. Prove that $K(\mathbf{x}, \mathbf{z}) = K_1(\mathbf{x}, \mathbf{z}) \cdot K_2(\mathbf{x}, \mathbf{z})$ is valid
4. Design a combined kernel: $K(\mathbf{x}, \mathbf{z}) = \alpha K_{linear}(\mathbf{x}, \mathbf{z}) + \beta K_{RBF}(\mathbf{x}, \mathbf{z})$ and choose appropriate $\alpha, \beta$
5. Is $K(\mathbf{x}, \mathbf{z}) = \min(K_1(\mathbf{x}, \mathbf{z}), K_2(\mathbf{x}, \mathbf{z}))$ a valid kernel? Prove or find a counterexample

For a detailed explanation of this problem, see [Question 7: Kernel Combinations and Closure](L5_3_7_explanation.md).

## Question 8

### Problem Statement
Investigate feature space dimensionality and the kernel trick.

#### Task
1. Calculate the dimensionality of the feature space for polynomial kernels of degree $d = 1, 2, 3, 4$ in $n = 5$ dimensions
2. What is the dimensionality of the RBF kernel feature space?
3. How can SVMs handle infinite-dimensional feature spaces computationally?
4. Prove that the kernel trick allows us to work in high-dimensional spaces without explicit computation
5. Show that the decision function can be expressed entirely in terms of kernel evaluations

For a detailed explanation of this problem, see [Question 8: Feature Space Dimensionality](L5_3_8_explanation.md).

## Question 9

### Problem Statement
Design custom kernels for specific applications.

#### Task
1. Create a string kernel for DNA sequences that counts matching k-mers (subsequences of length k)
2. Design a graph kernel that measures similarity between graph structures
3. Develop a kernel for time series that is invariant to time shifts
4. Verify that your string kernel satisfies Mercer's conditions
5. Implement a normalized version of your kernels: $\tilde{K}(\mathbf{x}, \mathbf{z}) = \frac{K(\mathbf{x}, \mathbf{z})}{\sqrt{K(\mathbf{x}, \mathbf{x})K(\mathbf{z}, \mathbf{z})}}$

For a detailed explanation of this problem, see [Question 9: Custom Kernel Design](L5_3_9_explanation.md).

## Question 10

### Problem Statement
Develop a systematic kernel selection methodology.

#### Task
1. List the factors to consider when choosing between linear, polynomial, and RBF kernels
2. Design a decision tree for kernel selection based on dataset characteristics
3. How would you use cross-validation to compare different kernel families?
4. What is kernel alignment and how can it guide kernel selection?
5. Create a practical algorithm for automated kernel selection

For a detailed explanation of this problem, see [Question 10: Kernel Selection Methodology](L5_3_10_explanation.md).

## Question 11

### Problem Statement
Analyze the computational and storage complexity of different kernels.

#### Task
1. Compare the evaluation time for linear, polynomial (degree 3), and RBF kernels
2. Calculate the space complexity of storing the kernel matrix for $n = 10^3, 10^4, 10^5$ samples
3. Design strategies for reducing kernel matrix storage requirements
4. What is the trade-off between kernel complexity and classification accuracy?
5. How does the choice of kernel affect training vs prediction time?

For a detailed explanation of this problem, see [Question 11: Computational Analysis](L5_3_11_explanation.md).

## Question 12

### Problem Statement
Implement kernel approximation techniques for large-scale problems.

#### Task
1. Describe the Nyström method for low-rank kernel matrix approximation
2. For a rank-$r$ approximation of an $n \times n$ kernel matrix, what are the computational savings?
3. Implement random Fourier features for RBF kernel approximation
4. How does the approximation quality affect SVM performance?
5. Design an adaptive algorithm that chooses the approximation rank based on desired accuracy

For a detailed explanation of this problem, see [Question 12: Kernel Approximation](L5_3_12_explanation.md).

## Question 13

### Problem Statement
Study kernel parameter optimization using grid search and cross-validation.

#### Task
1. For an RBF kernel, design a grid search over $C \in [10^{-3}, 10^3]$ and $\gamma \in [10^{-4}, 10^1]$
2. Implement nested cross-validation for unbiased parameter selection
3. How many hyperparameter combinations should you test for reliable results?
4. Design an early stopping criterion for expensive parameter searches
5. Compare grid search vs random search vs Bayesian optimization for kernel parameter tuning

For a detailed explanation of this problem, see [Question 13: Parameter Optimization](L5_3_13_explanation.md).

## Question 14

### Problem Statement
Investigate kernel learning and adaptive kernels.

#### Task
1. Formulate the problem of learning kernel parameters jointly with SVM training
2. Design a gradient descent algorithm for optimizing RBF kernel parameters
3. How would you learn the optimal weights for a combination of multiple kernels?
4. What are the risks of overfitting when learning kernel parameters?
5. Implement cross-validation specifically for kernel parameter selection

For a detailed explanation of this problem, see [Question 14: Kernel Learning](L5_3_14_explanation.md).

## Question 15

### Problem Statement
Apply kernels to real-world scenarios and analyze performance.

#### Task
1. For text classification, compare linear vs polynomial vs RBF kernels
2. For image recognition, design appropriate kernels for different feature representations
3. How would you handle mixed data types (continuous + categorical) with kernels?
4. Design experiments to measure the effect of kernel choice on generalization
5. What preprocessing steps are essential for different kernel types?

For a detailed explanation of this problem, see [Question 15: Real-world Applications](L5_3_15_explanation.md).

## Question 16

### Problem Statement
Theoretical analysis of kernel methods and SVM generalization.

#### Task
1. Derive the VC dimension bound for kernel SVMs
2. How does the choice of kernel affect the complexity of the hypothesis space?
3. Prove that the margin-based generalization bound applies to kernel SVMs
4. What is the relationship between kernel complexity and overfitting?
5. Design a theoretical framework for comparing the generalization ability of different kernels

For a detailed explanation of this problem, see [Question 16: Theoretical Analysis](L5_3_16_explanation.md).