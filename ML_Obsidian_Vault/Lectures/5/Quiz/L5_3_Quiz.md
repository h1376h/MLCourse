# Lecture 5.3: Kernel Trick for Nonlinear Classification Quiz

## Overview
This quiz contains 22 questions covering different topics from section 5.3 of the lectures on Kernel Trick, Feature Space Transformation, Common Kernels, RBF Kernels, Polynomial Kernels, Mercer's Theorem, and Kernel Selection.

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
6. Design a puzzle game with four squares in a 2×2 grid. Valid patterns have exactly one colored square (positions $(0,1)$ or $(1,0)$), while invalid patterns have no squares colored $(0,0)$ or all squares colored $(1,1)$. Design a 3D thinking tool to help players visualize patterns and create a rule for determining if patterns are solvable using 3D separation.

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
6. Design a recommendation system using user genre preferences. User A: Action=8, Romance=2; User B: Action=2, Romance=8; User C: Action=5, Romance=5. Using similarity function $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma ||\mathbf{x} - \mathbf{z}||^2)$, calculate similarity scores for $\gamma = 0.5, 1, 2$ and design a recommendation confidence system. Determine the optimal $\gamma$ for 70% similarity threshold.

For a detailed explanation of this problem, see [Question 4: RBF Kernel Properties](L5_3_4_explanation.md).

## Question 5

### Problem Statement
Study the effect of the RBF kernel parameter $\gamma$ on decision boundaries.

#### Task
1. For a 1D dataset with points $x_1 = -1, x_2 = 1$ (different classes), sketch the decision boundary for $\gamma = 0.1, 1, 10$ (you can draw this by hand)
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
5. Design a normalized version of your kernels: $\tilde{K}(\mathbf{x}, \mathbf{z}) = \frac{K(\mathbf{x}, \mathbf{z})}{\sqrt{K(\mathbf{x}, \mathbf{x})K(\mathbf{z}, \mathbf{z})}}$

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
3. Design random Fourier features for RBF kernel approximation
4. How does the approximation quality affect SVM performance?
5. Design an adaptive algorithm that chooses the approximation rank based on desired accuracy

For a detailed explanation of this problem, see [Question 12: Kernel Approximation](L5_3_12_explanation.md).

## Question 13

### Problem Statement
Study kernel parameter optimization using grid search and cross-validation.

#### Task
1. For an RBF kernel, design a grid search over $C \in [10^{-3}, 10^3]$ and $\gamma \in [10^{-4}, 10^1]$
2. Design nested cross-validation for unbiased parameter selection
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
5. Design cross-validation specifically for kernel parameter selection

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

## Question 17

### Problem Statement
Given points $\mathbf{x}_1 = (1, 2)$, $\mathbf{x}_2 = (0, 1)$, $\mathbf{x}_3 = (2, 0)$, compute kernel matrices for different kernels.

#### Task
1. Compute the $3 \times 3$ Gram matrix $K_{ij} = \mathbf{x}_i^T \mathbf{x}_j$
2. Compute $K_{ij} = (\mathbf{x}_i^T \mathbf{x}_j + 1)^2$
3. Compute $K_{ij} = \exp(-0.5 ||\mathbf{x}_i - \mathbf{x}_j||^2)$
4. Verify that each matrix is positive semi-definite by checking eigenvalues
5. Determine the effective dimensionality for each kernel

For a detailed explanation of this problem, see [Question 17: Kernel Matrix Computations](L5_3_17_explanation.md).

## Question 18

### Problem Statement
For the polynomial kernel $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + c)^d$, analyze feature space properties.

#### Task
1. For 2D input and $d=2, c=1$, write out the explicit feature mapping $\phi(\mathbf{x})$
2. Calculate the feature space dimension for $n$-dimensional input with degree $d$
3. For $\mathbf{x} = (2, 1)$ and $\mathbf{z} = (1, 3)$ with $d=3, c=0$, compute $K(\mathbf{x}, \mathbf{z})$ two ways:
   - Using kernel trick: $(\mathbf{x}^T\mathbf{z})^3$
   - Using explicit feature mapping
4. Show how $c$ affects the relative importance of interaction terms
5. Compare complexity of kernel vs. explicit computation for $d=5$

For a detailed explanation of this problem, see [Question 18: Polynomial Kernel Analysis](L5_3_18_explanation.md).

## Question 19

### Problem Statement
Analyze the RBF kernel $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma ||\mathbf{x} - \mathbf{z}||^2)$ for classification.

#### Task
1. For points $(1, 0)$, $(0, 1)$, $(2, 2)$, calculate all pairwise kernel values with $\gamma = 0.5$
2. Sketch kernel value vs. distance for $\gamma = 0.1, 1, 10$ (you can draw this by hand)
3. Predict how the decision boundary complexity changes with $\gamma$
4. Interpret kernel values as similarity scores and rank point pairs
5. For a dataset with average pairwise distance $d_{avg} = 2$, estimate appropriate $\gamma$ range

For a detailed explanation of this problem, see [Question 19: RBF Parameter Analysis](L5_3_19_explanation.md).

## Question 20

### Problem Statement
Determine which functions are valid kernels using Mercer's theorem.

#### Task
1. Check validity of:
   - $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z})^2 + (\mathbf{x}^T\mathbf{z})^3$
   - $K(\mathbf{x}, \mathbf{z}) = \exp(\mathbf{x}^T\mathbf{z})$
   - $K(\mathbf{x}, \mathbf{z}) = \sin(\mathbf{x}^T\mathbf{z})$
2. For 3 points $(0, 0)$, $(1, 0)$, $(0, 1)$, compute Gram matrices and check PSD property
3. Show that $K(\mathbf{x}, \mathbf{z}) = 2K_1(\mathbf{x}, \mathbf{z}) + 3K_2(\mathbf{x}, \mathbf{z})$ is valid if $K_1, K_2$ are valid
4. Provide an example of an invalid kernel and show why it fails
5. Design a valid kernel for comparing sets of different sizes

For a detailed explanation of this problem, see [Question 20: Kernel Validity Testing](L5_3_20_explanation.md).

## Question 21

### Problem Statement
Analyze the geometry of feature spaces induced by different kernels.

#### Task
1. For 2D input, describe the geometry of the feature space
2. For $(\mathbf{x}^T\mathbf{z} + 1)^2$ with 2D input, visualize the 6D feature space structure
3. Explain why RBF kernels correspond to infinite-dimensional feature spaces
4. Show that linear kernels preserve angles but RBF kernels don't
5. Prove that any finite dataset becomes separable in sufficiently high dimensions

For a detailed explanation of this problem, see [Question 21: Feature Space Geometry](L5_3_21_explanation.md).

## Question 22

### Problem Statement
Compare computational costs of different kernel approaches.

#### Task
1. Calculate memory needed for kernel matrices with $n = 1000, 10000, 100000$ samples
2. Compare time complexity for computing:
   - Linear kernel matrix: $O(?)$
   - RBF kernel matrix: $O(?)$
   - Polynomial kernel matrix: $O(?)$
3. How does kernel choice affect SVM training time?
4. For $n_{SV}$ support vectors, compare prediction costs
5. Calculate speedup vs. accuracy loss for rank-$r$ kernel approximation

For a detailed explanation of this problem, see [Question 22: Computational Complexity](L5_3_22_explanation.md).