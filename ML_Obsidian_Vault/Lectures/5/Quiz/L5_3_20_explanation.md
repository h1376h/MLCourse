# Question 20: Kernel Validity Testing

## Problem Statement
Determine which functions are valid kernels using Mercer's theorem.

### Task
1. Check validity of:
   - $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z})^2 + (\mathbf{x}^T\mathbf{z})^3$
   - $K(\mathbf{x}, \mathbf{z}) = \exp(\mathbf{x}^T\mathbf{z})$
   - $K(\mathbf{x}, \mathbf{z}) = \sin(\mathbf{x}^T\mathbf{z})$
2. For 3 points $(0, 0)$, $(1, 0)$, $(0, 1)$, compute Gram matrices and check PSD property
3. Show that $K(\mathbf{x}, \mathbf{z}) = 2K_1(\mathbf{x}, \mathbf{z}) + 3K_2(\mathbf{x}, \mathbf{z})$ is valid if $K_1, K_2$ are valid
4. Provide an example of an invalid kernel and show why it fails
5. Design a valid kernel for comparing sets of different sizes

## Understanding the Problem
A kernel function $K(\mathbf{x}, \mathbf{z})$ is valid if and only if it satisfies Mercer's theorem, which requires that the corresponding Gram matrix $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ is positive semi-definite (PSD) for any finite set of points $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$. A matrix is PSD if all its eigenvalues are non-negative.

The key properties of valid kernels are:
- **Symmetry**: $K(\mathbf{x}, \mathbf{z}) = K(\mathbf{z}, \mathbf{x})$
- **Positive Semi-definiteness**: The Gram matrix has non-negative eigenvalues
- **Reproducing Property**: Valid kernels correspond to inner products in some feature space

## Solution

### Step 1: Check Validity of Specific Kernels

We test each kernel function on a set of 5 test points: $(0,0)$, $(1,0)$, $(0,1)$, $(1,1)$, and $(-1,0)$.

#### 1.1 $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z})^2 + (\mathbf{x}^T\mathbf{z})^3$

**Gram Matrix:**
```
[[ 0.  0.  0.  0.  0.]
 [ 0.  2.  0.  2.  0.]
 [ 0.  0.  2.  2.  0.]
 [ 0.  2.  2. 12.  0.]
 [ 0.  0.  0.  0.  2.]]
```

**Eigenvalues:** $[12.74, 1.26, 2.00, 0.00, 2.00]$

**Analysis:** All eigenvalues are non-negative (minimum eigenvalue = 0.000000), so this kernel is **VALID**.

**Explanation:** This is a polynomial kernel that combines quadratic and cubic terms. Since both $(\mathbf{x}^T\mathbf{z})^2$ and $(\mathbf{x}^T\mathbf{z})^3$ are valid kernels (polynomial kernels), their sum is also valid.

#### 1.2 $K(\mathbf{x}, \mathbf{z}) = \exp(\mathbf{x}^T\mathbf{z})$

**Gram Matrix:**
```
[[1.00  1.00  1.00  1.00  1.00]
 [1.00  2.72  1.00  2.72  0.37]
 [1.00  1.00  2.72  2.72  1.00]
 [1.00  2.72  2.72  7.39  0.37]
 [1.00  0.37  1.00  0.37  2.72]]
```

**Eigenvalues:** $[10.23, 3.33, 0.21, 1.72, 1.06]$

**Analysis:** All eigenvalues are positive (minimum eigenvalue = 0.209130), so this kernel is **VALID**.

**Explanation:** This is the exponential kernel, which is a valid kernel because it can be written as $\exp(\mathbf{x}^T\mathbf{z}) = \sum_{k=0}^{\infty} \frac{(\mathbf{x}^T\mathbf{z})^k}{k!}$, a sum of polynomial kernels with positive coefficients.

#### 1.3 $K(\mathbf{x}, \mathbf{z}) = \sin(\mathbf{x}^T\mathbf{z})$

**Gram Matrix:**
```
[[ 0.00  0.00  0.00  0.00  0.00]
 [ 0.00  0.84  0.00  0.84 -0.84]
 [ 0.00  0.00  0.84  0.84  0.00]
 [ 0.00  0.84  0.84  0.91 -0.84]
 [ 0.00 -0.84  0.00 -0.84  0.84]]
```

**Eigenvalues:** $[2.69, 1.11, 0.00, -0.37, 0.00]$

**Analysis:** One eigenvalue is negative (-0.367247), so this kernel is **INVALID**.

**Explanation:** The sine function is not a valid kernel because it can produce negative values and doesn't satisfy the positive semi-definiteness requirement. The sine function oscillates between positive and negative values, which leads to negative eigenvalues in the Gram matrix.

### Step 2: Gram Matrices for Specific 3 Points

For the points $(0,0)$, $(1,0)$, and $(0,1)$, we compute the Gram matrices:

#### Polynomial Kernel $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z})^2 + (\mathbf{x}^T\mathbf{z})^3$
```
[[0. 0. 0.]
 [0. 2. 0.]
 [0. 0. 2.]]
```
**Eigenvalues:** $[0, 2, 2]$ ✓ **PSD**

#### Exponential Kernel $K(\mathbf{x}, \mathbf{z}) = \exp(\mathbf{x}^T\mathbf{z})$
```
[[1.00  1.00  1.00]
 [1.00  2.72  1.00]
 [1.00  1.00  2.72]]
```
**Eigenvalues:** $[0.40, 4.32, 1.72]$ ✓ **PSD**

#### Sine Kernel $K(\mathbf{x}, \mathbf{z}) = \sin(\mathbf{x}^T\mathbf{z})$
```
[[0.00  0.00  0.00]
 [0.00  0.84  0.00]
 [0.00  0.00  0.84]]
```
**Eigenvalues:** $[0, 0.84, 0.84]$ ✓ **PSD**

**Note:** Interestingly, the sine kernel appears valid for this specific set of 3 points, but fails for the larger set of 5 points. This demonstrates that kernel validity must be checked for all possible finite sets of points.

### Step 3: Linear Combination of Valid Kernels

We demonstrate that if $K_1$ and $K_2$ are valid kernels, then $K = 2K_1 + 3K_2$ is also valid.

**Individual Kernels:**
- $K_1(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T\mathbf{z}$ (Linear kernel)
- $K_2(\mathbf{x}, \mathbf{z}) = \exp(-0.1\|\mathbf{x}-\mathbf{z}\|^2)$ (RBF kernel)

**Combined Kernel:** $K(\mathbf{x}, \mathbf{z}) = 2K_1(\mathbf{x}, \mathbf{z}) + 3K_2(\mathbf{x}, \mathbf{z})$

**Verification:**
- Both individual kernels are valid (all eigenvalues ≥ 0)
- The combined kernel Gram matrix equals $2 \times K_1 + 3 \times K_2$
- All eigenvalues of the combined kernel are non-negative (minimum = 0.020492)

**Result:** The linear combination is **VALID**.

**Theoretical Justification:** If $K_1$ and $K_2$ are valid kernels, then for any positive constants $a, b > 0$, the kernel $K = aK_1 + bK_2$ is also valid. This follows from the fact that:
1. The sum of PSD matrices is PSD
2. Scaling a PSD matrix by a positive constant preserves PSD property

### Step 4: Examples of Invalid Kernels

#### 4.1 $K(\mathbf{x}, \mathbf{z}) = -\|\mathbf{x}-\mathbf{z}\|^2$

**Gram Matrix:**
```
[[ 0. -1. -1. -2. -1.]
 [-1.  0. -2. -1. -4.]
 [-1. -2.  0. -1. -2.]
 [-2. -1. -1.  0. -5.]
 [-1. -4. -2. -5.  0.]]
```

**Eigenvalues:** $[-8.66, 6.05, 2.00, 0.00, 0.61]$

**Analysis:** One eigenvalue is negative (-8.66), so this kernel is **INVALID**.

**Explanation:** The negative sign makes this kernel violate the positive semi-definiteness requirement. Distance-based kernels must be positive to be valid.

#### 4.2 $K(\mathbf{x}, \mathbf{z}) = x_0 \cdot z_1$ (Asymmetric)

**Gram Matrix:**
```
[[ 0.  0.  0.  0.  0.]
 [ 0.  0.  1.  1.  0.]
 [ 0.  0.  0.  0.  0.]
 [ 0.  0.  1.  1.  0.]
 [ 0.  0. -1. -1.  0.]]
```

**Eigenvalues:** $[0, 0, 1, 0, 0]$

**Analysis:** All eigenvalues are non-negative, so this kernel is **VALID** for this specific set of points.

**Explanation:** While this kernel is asymmetric ($K(\mathbf{x}, \mathbf{z}) \neq K(\mathbf{z}, \mathbf{x})$), it still produces a PSD Gram matrix for this particular set of points. However, asymmetry is generally a red flag for kernel validity.

### Step 5: Kernel for Comparing Sets of Different Sizes

We design a valid kernel for comparing sets of different sizes using the average pairwise kernel approach.

**Set Kernel Definition:**
$$K(A, B) = \frac{1}{|A| \cdot |B|} \sum_{a \in A} \sum_{b \in B} k(a, b)$$

where $k(a, b)$ is a base kernel (e.g., RBF or linear).

**Test Sets:**
- Set A: $\{(0,0), (1,0)\}$ (2 points)
- Set B: $\{(0,1), (1,1), (0.5,0.5)\}$ (3 points)
- Set C: $\{(-1,0)\}$ (1 point)

#### RBF-based Set Kernel
**Kernel Values:**
- $K(A,B) = 0.3699$
- $K(A,C) = 0.1931$
- $K(B,C) = 0.0747$

**Gram Matrix:**
```
[[0.68  0.37  0.19]
 [0.37  0.68  0.07]
 [0.19  0.07  1.00]]
```

**Eigenvalues:** $[0.30, 1.22, 0.84]$ ✓ **PSD**

#### Linear-based Set Kernel
**Kernel Values:**
- $K(A,B) = 0.2500$
- $K(A,C) = -0.5000$
- $K(B,C) = -0.5000$

**Gram Matrix:**
```
[[ 0.25  0.25 -0.50]
 [ 0.25  0.94 -0.50]
 [-0.50 -0.50  1.00]]
```

**Eigenvalues:** $[1.68, 0.52, 0.00]$ ✓ **PSD**

**Result:** Both set kernels are **VALID**.

**Explanation:** The set kernel is valid because:
1. It inherits the PSD property from the base kernel
2. The averaging operation preserves the kernel properties
3. It can handle sets of different sizes naturally

## Visual Explanations

### Kernel Validity Analysis

![Kernel Validity Analysis](../Images/L5_3_Quiz_20/kernel_validity_analysis.png)

The visualization shows:
- **Top row:** Eigenvalue distributions for the three main kernels
- **Bottom row:** Heatmaps of the corresponding Gram matrices
- **Validity indicators:** Clear YES/NO labels for each kernel

Key observations:
1. **Polynomial kernel:** All eigenvalues are non-negative, confirming validity
2. **Exponential kernel:** All eigenvalues are positive, confirming validity  
3. **Sine kernel:** One negative eigenvalue (-0.37), confirming invalidity

### Set Kernel Analysis

![Set Kernel Analysis](../Images/L5_3_Quiz_20/set_kernel_analysis.png)

The visualization shows:
- **Left:** RBF-based set kernel Gram matrix
- **Right:** Linear-based set kernel Gram matrix
- Both matrices are symmetric and have positive eigenvalues

### Test Points Visualization

![Test Points](../Images/L5_3_Quiz_20/test_points.png)

The visualization shows:
- **Red circles:** All test points used for kernel evaluation
- **Blue squares:** Specific points (0,0), (1,0), (0,1) used in Task 2
- **Point labels:** Coordinates for easy reference

## Key Insights

### Theoretical Foundations
- **Mercer's Theorem:** A kernel is valid if and only if its Gram matrix is PSD for any finite set of points
- **Kernel Properties:** Valid kernels must be symmetric and positive semi-definite
- **Feature Space:** Every valid kernel corresponds to an inner product in some (possibly infinite-dimensional) feature space

### Practical Applications
- **Polynomial Kernels:** Valid for any positive integer degree
- **Exponential Kernels:** Valid because they can be expressed as infinite sums of polynomial kernels
- **Trigonometric Kernels:** Generally invalid due to oscillatory behavior
- **Linear Combinations:** Valid kernels can be combined with positive coefficients to create new valid kernels

### Common Pitfalls
- **Asymmetric Kernels:** May not be valid even if they produce PSD matrices for specific point sets
- **Negative Coefficients:** Can invalidate otherwise valid kernels
- **Oscillatory Functions:** Functions like sine and cosine typically produce invalid kernels
- **Distance-based Kernels:** Must be positive to be valid

### Extensions and Generalizations
- **Set Kernels:** Provide a framework for comparing collections of different sizes
- **Multiple Kernel Learning:** Combines multiple valid kernels for improved performance
- **Kernel Approximation:** Techniques for handling large-scale kernel computations

## Conclusion
- **Valid kernels:** $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z})^2 + (\mathbf{x}^T\mathbf{z})^3$ and $K(\mathbf{x}, \mathbf{z}) = \exp(\mathbf{x}^T\mathbf{z})$
- **Invalid kernel:** $K(\mathbf{x}, \mathbf{z}) = \sin(\mathbf{x}^T\mathbf{z})$ due to negative eigenvalues
- **Linear combinations:** Valid kernels can be combined with positive coefficients
- **Set kernels:** Provide a valid approach for comparing sets of different sizes
- **Mercer's theorem:** Provides the fundamental criterion for kernel validity through PSD Gram matrices

The analysis demonstrates the importance of checking kernel validity through eigenvalue analysis of Gram matrices, and shows how different mathematical properties (polynomial vs. trigonometric) affect kernel validity.
