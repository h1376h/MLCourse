# Lecture 5.3: Kernel Trick for Nonlinear Classification Quiz

## Overview
This quiz contains 12 questions covering different topics from section 5.3 of the lectures on Kernel Trick, Feature Space Transformation, Common Kernels, RBF Kernels, Polynomial Kernels, Mercer's Theorem, and Kernel Selection.

## Question 1

### Problem Statement
Consider the XOR problem with four points:
- $(0, 0) \rightarrow$ Class -1
- $(0, 1) \rightarrow$ Class +1  
- $(1, 0) \rightarrow$ Class +1
- $(1, 1) \rightarrow$ Class -1

#### Task
1. [📚] Plot these points and try to draw a linear decision boundary that separates the classes
2. [🔍] Explain why this dataset is not linearly separable in the original 2D space
3. [📚] Apply the feature transformation $\phi(x_1, x_2) = (x_1, x_2, x_1 x_2)$ and show the transformed points
4. [📚] Verify that the transformed data is linearly separable in the 3D feature space

For a detailed explanation of this problem, see [Question 1: XOR Problem and Feature Transformation](L5_3_1_explanation.md).

## Question 2

### Problem Statement
Consider the computational challenge of explicit feature mapping versus the kernel trick.

#### Task
1. [🔍] For a polynomial kernel of degree $d$ applied to $n$-dimensional input, how many features are in the explicit feature space?
2. [📚] Calculate the number of features for a degree-3 polynomial kernel with 2-dimensional input
3. [🔍] What is the computational cost of computing the inner product $\phi(\mathbf{x})^T\phi(\mathbf{z})$ explicitly vs. using the kernel trick?
4. [📚] Explain why the kernel trick $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + 1)^3$ is more efficient than explicit feature mapping

For a detailed explanation of this problem, see [Question 2: Kernel Trick Computational Advantage](L5_3_2_explanation.md).

## Question 3

### Problem Statement
Consider the linear kernel $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T\mathbf{z}$ and polynomial kernel $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + c)^d$.

#### Task
1. [📚] For vectors $\mathbf{x} = (2, 1)$ and $\mathbf{z} = (1, 3)$, calculate the linear kernel value
2. [📚] Calculate the polynomial kernel value with $c = 1$ and $d = 2$
3. [📚] What feature transformation does the polynomial kernel with $d = 2$ and $c = 0$ correspond to in 2D?
4. [🔍] How does the parameter $c$ affect the importance of higher-order vs. lower-order terms?

For a detailed explanation of this problem, see [Question 3: Linear and Polynomial Kernels](L5_3_3_explanation.md).

## Question 4

### Problem Statement
Consider the RBF (Radial Basis Function) kernel: $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma ||\mathbf{x} - \mathbf{z}||^2)$.

#### Task
1. [📚] For points $\mathbf{x} = (1, 0)$ and $\mathbf{z} = (0, 1)$ with $\gamma = 1$, calculate the RBF kernel value
2. [📚] What is the kernel value when $\mathbf{x} = \mathbf{z}$ (identical points)?
3. [📚] What happens to the kernel value as the distance $||\mathbf{x} - \mathbf{z}||$ increases?
4. [🔍] How does the parameter $\gamma$ control the "width" or "spread" of the RBF kernel?
5. [📚] For $\gamma = 0.5$ vs. $\gamma = 2$, which creates a more localized (narrow) influence?

For a detailed explanation of this problem, see [Question 4: RBF Kernel Properties](L5_3_4_explanation.md).

## Question 5

### Problem Statement
Consider the effect of the RBF kernel parameter $\gamma$ on the decision boundary and model complexity.

#### Task
1. [🔍] When $\gamma$ is very large, what happens to the decision boundary complexity?
2. [🔍] When $\gamma$ is very small, what does the RBF kernel approximate?
3. [📚] How does increasing $\gamma$ affect the bias-variance tradeoff?
4. [🔍] In terms of overfitting and underfitting, what are the risks of choosing $\gamma$ too large or too small?
5. [📚] Describe the typical shape of the validation curve when plotting accuracy vs. $\gamma$

For a detailed explanation of this problem, see [Question 5: RBF Parameter Effects](L5_3_5_explanation.md).

## Question 6

### Problem Statement
Consider the requirements for a valid kernel function according to Mercer's theorem.

#### Task
1. [🔍] What does it mean for a kernel function to be "positive definite"?
2. [📚] What is the Gram matrix (kernel matrix) for a set of points $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$?
3. [📚] For three points with kernel values $K(\mathbf{x}_1, \mathbf{x}_1) = 1$, $K(\mathbf{x}_2, \mathbf{x}_2) = 1$, $K(\mathbf{x}_3, \mathbf{x}_3) = 1$, and $K(\mathbf{x}_i, \mathbf{x}_j) = 0.5$ for $i \neq j$, write the Gram matrix
4. [🔍] Why must the Gram matrix be positive semi-definite for a valid kernel?

For a detailed explanation of this problem, see [Question 6: Mercer's Theorem and Valid Kernels](L5_3_6_explanation.md).

## Question 7

### Problem Statement
Consider kernel combinations and operations that preserve validity.

#### Task
1. [📚] If $K_1(\mathbf{x}, \mathbf{z})$ and $K_2(\mathbf{x}, \mathbf{z})$ are valid kernels, show that $K(\mathbf{x}, \mathbf{z}) = K_1(\mathbf{x}, \mathbf{z}) + K_2(\mathbf{x}, \mathbf{z})$ is also valid
2. [📚] Show that $K(\mathbf{x}, \mathbf{z}) = c \cdot K_1(\mathbf{x}, \mathbf{z})$ is valid for any $c > 0$
3. [📚] Is $K(\mathbf{x}, \mathbf{z}) = K_1(\mathbf{x}, \mathbf{z}) \cdot K_2(\mathbf{x}, \mathbf{z})$ a valid kernel?
4. [🔍] Give an example of combining a linear and RBF kernel into a single valid kernel

For a detailed explanation of this problem, see [Question 7: Kernel Combinations](L5_3_7_explanation.md).

## Question 8

### Problem Statement
Consider the relationship between kernels and feature space dimensionality.

#### Task
1. [📚] What is the dimensionality of the feature space for a polynomial kernel of degree $d$ in $n$ dimensions?
2. [🔍] What is the dimensionality of the feature space for the RBF kernel?
3. [📚] How can SVMs work with infinite-dimensional feature spaces without running into computational issues?
4. [🔍] Why don't we need to explicitly know the feature mapping $\phi(\mathbf{x})$ when using kernels?

For a detailed explanation of this problem, see [Question 8: Feature Space Dimensionality](L5_3_8_explanation.md).

## Question 9

### Problem Statement
Consider a custom kernel design problem for specific applications.

#### Task
1. [📚] Design a kernel for comparing text documents based on the number of common words. Write the mathematical expression.
2. [🔍] Verify that your proposed kernel satisfies the properties of a valid kernel
3. [📚] How would you modify your kernel to give more weight to rare words vs. common words?
4. [🔍] What preprocessing steps might be important when using your text kernel?

For a detailed explanation of this problem, see [Question 9: Custom Kernel Design](L5_3_9_explanation.md).

## Question 10

### Problem Statement
Consider kernel selection strategies for different types of data and problems.

#### Task
1. [🔍] When would you choose a linear kernel over nonlinear kernels?
2. [🔍] When would you choose an RBF kernel over a polynomial kernel?
3. [📚] What are the main factors to consider when selecting a kernel (data size, dimensionality, noise level)?
4. [🔍] How can you use cross-validation to compare different kernel choices?
5. [📚] What is kernel alignment and how can it guide kernel selection?

For a detailed explanation of this problem, see [Question 10: Kernel Selection Strategies](L5_3_10_explanation.md).

## Question 11

### Problem Statement
Consider the computational and memory considerations when working with kernels.

#### Task
1. [📚] What is the space complexity of storing the full kernel matrix for $n$ training samples?
2. [🔍] Why can the kernel matrix become prohibitively large for very large datasets?
3. [📚] What are some strategies for handling large kernel matrices (approximations, caching, etc.)?
4. [🔍] How does the choice of kernel affect the training time and prediction time of SVMs?
5. [📚] Which kernels are generally fastest to compute: linear, polynomial, or RBF?

For a detailed explanation of this problem, see [Question 11: Computational Considerations](L5_3_11_explanation.md).

## Question 12

### Problem Statement
Consider practical kernel parameter tuning using grid search and cross-validation.

#### Task
1. [📚] For an RBF kernel SVM, what are the main hyperparameters you need to tune simultaneously?
2. [🔍] Describe a systematic grid search approach for tuning RBF kernel parameters $C$ and $\gamma$
3. [📚] What ranges would you typically search for $C$ and $\gamma$ parameters?
4. [🔍] How can you detect overfitting during the parameter tuning process?
5. [📚] What is the difference between nested cross-validation and simple cross-validation for parameter selection?

For a detailed explanation of this problem, see [Question 12: Kernel Parameter Tuning](L5_3_12_explanation.md).
