# Question 40: Kernel Trick and Nonlinear SVM Hyperplane

## Problem Statement
Use kernel trick and find the equation for hyperplane using nonlinear SVM.

**Dataset:**
- **Positive Points**: $\{(1,0), (3,0), (5,0)\}$
- **Negative Points**: $\{(0,0), (2,0), (4,0), (6,0)\}$

### Task
1. Plot the original points in 1D space and show why they are not linearly separable
2. Apply an appropriate feature transformation to make the data linearly separable
3. Plot the transformed points in the new feature space
4. Find the equation for the separating hyperplane in the transformed space
5. Express the decision function in terms of the original input space
6. Verify that the hyperplane correctly separates all points
7. Calculate the margin of the separating hyperplane
8. Identify the support vectors in the transformed space

## Understanding the Problem
This problem demonstrates the core concept of the kernel trick in Support Vector Machines. We have a 1D dataset where positive and negative points alternate along the number line, making linear separation impossible in the original space. The kernel trick allows us to map these points to a higher-dimensional feature space where they become linearly separable.

The key insight is that while the points cannot be separated by any single threshold in 1D, we can find transformations that reveal the underlying pattern - in this case, the alternating nature of the labels suggests we need to capture the "parity" or "modular" structure of the data.

## Solution

We will explore multiple kernel transformations to find effective ways to separate this challenging dataset.

### Step 1: Analyze Original 1D Space
The original dataset consists of points on a line:
- Positive: $x \in \{1, 3, 5\}$ 
- Negative: $x \in \{0, 2, 4, 6\}$

These points alternate between positive and negative classes, making them impossible to separate with any single threshold. No matter where we place a decision boundary, we will misclassify some points.

### Step 2: Understanding the Primary Transformation

Let's focus on the **Primary Transformation**: $\phi(x) = [x^2, (x \bmod 2 - 0.5)x^2]$

This transformation has two components:
- $\phi_1(x) = x^2$: Quadratic scaling
- $\phi_2(x) = (x \bmod 2 - 0.5)x^2$: Parity-aware quadratic scaling

**Key Insight**: The modular arithmetic captures the alternating pattern:
- For odd $x$: $x \bmod 2 = 1$, so $\phi_2(x) = (1 - 0.5)x^2 = 0.5x^2$
- For even $x$: $x \bmod 2 = 0$, so $\phi_2(x) = (0 - 0.5)x^2 = -0.5x^2$

**Step-by-Step Transformation Calculations:**

For $x = 1$ (positive class):
$$\phi_1(1) = 1^2 = 1$$
$$\phi_2(1) = (1 \bmod 2 - 0.5) \times 1^2 = (1 - 0.5) \times 1 = 0.5$$
$$\phi(1) = [1, 0.5]$$

For $x = 3$ (positive class):
$$\phi_1(3) = 3^2 = 9$$
$$\phi_2(3) = (3 \bmod 2 - 0.5) \times 3^2 = (1 - 0.5) \times 9 = 4.5$$
$$\phi(3) = [9, 4.5]$$

For $x = 5$ (positive class):
$$\phi_1(5) = 5^2 = 25$$
$$\phi_2(5) = (5 \bmod 2 - 0.5) \times 5^2 = (1 - 0.5) \times 25 = 12.5$$
$$\phi(5) = [25, 12.5]$$

For $x = 0$ (negative class):
$$\phi_1(0) = 0^2 = 0$$
$$\phi_2(0) = (0 \bmod 2 - 0.5) \times 0^2 = (0 - 0.5) \times 0 = 0$$
$$\phi(0) = [0, 0]$$

For $x = 2$ (negative class):
$$\phi_1(2) = 2^2 = 4$$
$$\phi_2(2) = (2 \bmod 2 - 0.5) \times 2^2 = (0 - 0.5) \times 4 = -2$$
$$\phi(2) = [4, -2]$$

For $x = 4$ (negative class):
$$\phi_1(4) = 4^2 = 16$$
$$\phi_2(4) = (4 \bmod 2 - 0.5) \times 4^2 = (0 - 0.5) \times 16 = -8$$
$$\phi(4) = [16, -8]$$

For $x = 6$ (negative class):
$$\phi_1(6) = 6^2 = 36$$
$$\phi_2(6) = (6 \bmod 2 - 0.5) \times 6^2 = (0 - 0.5) \times 36 = -18$$
$$\phi(6) = [36, -18]$$

**Pattern Recognition**:
- All positive points (odd $x$) have $\phi_2(x) > 0$
- All negative points (even $x$) have $\phi_2(x) \leq 0$

This creates a clear linear separation in the 2D feature space!

### Step 3: SVM Hyperplane Derivation

The SVM finds the optimal separating hyperplane in the transformed space:
$$w_1\phi_1 + w_2\phi_2 + b = 0$$

From our analysis, the SVM computed:
- $w_1 = 0.999813 \approx 1$
- $w_2 = 1.999646 \approx 2$
- $b = -0.999757 \approx -1$

**Theoretical Hyperplane Equation:**
$$\phi_1 + 2\phi_2 - 1 = 0$$

Substituting our transformations:
$$x^2 + 2(x \bmod 2 - 0.5)x^2 - 1 = 0$$

Factoring:
$$x^2[1 + 2(x \bmod 2 - 0.5)] - 1 = 0$$

### Step 4: Decision Function Analysis

The decision function is:
$$f(x) = \text{sign}(x^2[1 + 2(x \bmod 2 - 0.5)] - 1)$$

**Case 1: Odd numbers** ($x \bmod 2 = 1$):
$$f(x) = \text{sign}(x^2[1 + 2(1 - 0.5)] - 1)$$
$$= \text{sign}(x^2[1 + 2(0.5)] - 1)$$
$$= \text{sign}(x^2[1 + 1] - 1)$$
$$= \text{sign}(2x^2 - 1)$$

For $x \geq 1$ (our positive points), $2x^2 - 1 > 0$, so $f(x) = +1$ ✓

**Case 2: Even numbers** ($x \bmod 2 = 0$):
$$f(x) = \text{sign}(x^2[1 + 2(0 - 0.5)] - 1)$$
$$= \text{sign}(x^2[1 + 2(-0.5)] - 1)$$
$$= \text{sign}(x^2[1 - 1] - 1)$$
$$= \text{sign}(0 - 1)$$
$$= \text{sign}(-1) = -1$$

All even numbers are classified as negative ✓

### Step 5: Verification of Each Point

Let's verify our decision function works for each point:

**Positive Points (Odd):**
- $x = 1$: $f(1) = \text{sign}(2(1)^2 - 1) = \text{sign}(1) = +1$ ✓
- $x = 3$: $f(3) = \text{sign}(2(3)^2 - 1) = \text{sign}(17) = +1$ ✓
- $x = 5$: $f(5) = \text{sign}(2(5)^2 - 1) = \text{sign}(49) = +1$ ✓

**Negative Points (Even):**
- $x = 0$: $f(0) = -1$ ✓
- $x = 2$: $f(2) = -1$ ✓
- $x = 4$: $f(4) = -1$ ✓
- $x = 6$: $f(6) = -1$ ✓

All points are correctly classified!

### Step 6: Detailed Decision Function Calculations

Let's calculate the decision function value for each point step by step:

**Decision Function**: $f(x) = w_1\phi_1(x) + w_2\phi_2(x) + b$

Using computed coefficients: $w_1 = 0.999813$, $w_2 = 1.999646$, $b = -0.999757$

**Point $x = 1$ (positive class):**
$$f(1) = 0.999813 \times 1 + 1.999646 \times 0.5 + (-0.999757)$$
$$= 0.999813 + 0.999823 - 0.999757 = 0.999879$$
Since $|f(1)| \approx 1$, this is a **support vector**.

**Point $x = 3$ (positive class):**
$$f(3) = 0.999813 \times 9 + 1.999646 \times 4.5 + (-0.999757)$$
$$= 8.998317 + 8.998408 - 0.999757 = 16.996968$$
Since $f(3) > 1$, this point is beyond the positive margin.

**Point $x = 5$ (positive class):**
$$f(5) = 0.999813 \times 25 + 1.999646 \times 12.5 + (-0.999757)$$
$$= 24.995325 + 24.995578 - 0.999757 = 48.991146$$
Since $f(5) > 1$, this point is beyond the positive margin.

**Point $x = 0$ (negative class):**
$$f(0) = 0.999813 \times 0 + 1.999646 \times 0 + (-0.999757)$$
$$= 0 + 0 - 0.999757 = -0.999757$$
Since $|f(0)| \approx 1$, this is a **support vector**.

**Point $x = 2$ (negative class):**
$$f(2) = 0.999813 \times 4 + 1.999646 \times (-2) + (-0.999757)$$
$$= 3.999252 - 3.999293 - 0.999757 = -0.999798$$
Since $|f(2)| \approx 1$, this is a **support vector**.

**Point $x = 4$ (negative class):**
$$f(4) = 0.999813 \times 16 + 1.999646 \times (-8) + (-0.999757)$$
$$= 15.997008 - 15.997170 - 0.999757 = -0.999919$$
Since $|f(4)| \approx 1$, this is a **support vector**.

**Point $x = 6$ (negative class):**
$$f(6) = 0.999813 \times 36 + 1.999646 \times (-18) + (-0.999757)$$
$$= 35.993269 - 35.993633 - 0.999757 = -1.000121$$
Since $|f(6)| \approx 1$, this is a **support vector**.

### Step 7: Support Vector Summary

**Total Support Vectors: 5 points**
- $x = 1$: $\phi(1) = [1, 0.5]$, $f(1) = 0.999879$ (positive margin)
- $x = 0$: $\phi(0) = [0, 0]$, $f(0) = -0.999757$ (negative margin)
- $x = 2$: $\phi(2) = [4, -2]$, $f(2) = -0.999798$ (negative margin)
- $x = 4$: $\phi(4) = [16, -8]$, $f(4) = -0.999919$ (negative margin)
- $x = 6$: $\phi(6) = [36, -18]$, $f(6) = -1.000121$ (negative margin)

### Step 8: Margin Calculation

The margin is the distance from the hyperplane to the support vectors:
$$\text{margin} = \frac{1}{||\mathbf{w}||} = \frac{1}{\sqrt{w_1^2 + w_2^2}}$$

$$\text{margin} = \frac{1}{\sqrt{(0.999813)^2 + (1.999646)^2}} = \frac{1}{\sqrt{4.998211}} = 0.447294$$

**Distance between margin boundaries**: $2 \times 0.447294 = 0.894588$

### Step 9: Kernel Function Analysis

The kernel function for our primary transformation is:
$$K(x, z) = \phi(x)^T\phi(z) = \phi_1(x)\phi_1(z) + \phi_2(x)\phi_2(z)$$

Substituting our transformations:
$$K(x, z) = x^2z^2 + (x \bmod 2 - 0.5)(z \bmod 2 - 0.5)x^2z^2$$
$$= x^2z^2[1 + (x \bmod 2 - 0.5)(z \bmod 2 - 0.5)]$$

**Kernel Matrix Calculation:**

For our dataset, the $7 \times 7$ kernel matrix is:

$$K = \begin{bmatrix}
1.25 & 11.25 & 31.25 & 0 & 3 & 12 & 27 \\
11.25 & 101.25 & 281.25 & 0 & 27 & 108 & 243 \\
31.25 & 281.25 & 781.25 & 0 & 75 & 300 & 675 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
3 & 27 & 75 & 0 & 20 & 80 & 180 \\
12 & 108 & 300 & 0 & 80 & 320 & 720 \\
27 & 243 & 675 & 0 & 180 & 720 & 1620
\end{bmatrix}$$

**Kernel Validity**: All eigenvalues are non-negative, confirming this is a valid Mercer kernel.

### Step 10: Alternative Successful Transformations

**Sign-based Transformation**: $\phi(x) = [x, \text{sign}(x \bmod 2 - 0.5)]$

**Detailed Analysis:**
- **All points are support vectors** (7 total)
- Hyperplane: $0.000095\phi_1 + 0.999762\phi_2 - 0.000428 = 0$ (essentially $\phi_2 = 0$)
- Creates perfect horizontal separation at $\phi_2 = 0$
- Margin: 1.000238 (distance between $\phi_2 = +1$ and $\phi_2 = -1$)

**Parity-weighted Transformation**: $\phi(x) = [(x \bmod 2)x, (1-x \bmod 2)x]$

**Detailed Analysis:**
- **5 support vectors**: $x \in \{0, 1, 2, 4, 6\}$
- Hyperplane: $1.999962\phi_1 - 0.000006\phi_2 - 0.999974 = 0$ (essentially $\phi_1 = 0.5$)
- Creates axis-aligned separation: odd numbers on x-axis, even numbers on y-axis
- Margin: 0.500010

**Trigonometric Transformation**: $\phi(x) = [\cos(\pi x), \sin(\pi x)]$

**Detailed Analysis:**
- **All points are support vectors** (7 total)
- Hyperplane: $-\phi_1 = 0$ (vertical line through origin)
- Perfect separation: odd numbers at $(-1, 0)$, even numbers at $(1, 0)$
- Margin: 1.000000 (maximum possible for this configuration)

## Visual Explanations

### Original 1D Dataset
![Original 1D Dataset](../Images/L5_3_Quiz_40/original_1d_dataset.png)

The plot shows the alternating pattern of positive (red circles) and negative (blue squares) points along the number line, demonstrating the impossibility of linear separation in 1D.

### Primary Transformation
![Primary Transformation](../Images/L5_3_Quiz_40/primary_transformation.png)

The primary transformation $\phi(x) = [x^2, (x \bmod 2 - 0.5)x^2]$ creates a diagonal separation in the 2D feature space. Positive points (red circles) have positive $\phi_2$ values, while negative points (blue squares) have non-positive $\phi_2$ values. The black line shows the optimal hyperplane, and green circles indicate support vectors.

### Sign-based Transformation
![Sign-based Transformation](../Images/L5_3_Quiz_40/sign_based_transformation.png)

The sign-based transformation $\phi(x) = [x, \text{sign}(x \bmod 2 - 0.5)]$ creates horizontal stripes. All positive points lie on the line $\phi_2 = +1$, while all negative points lie on $\phi_2 = -1$. The hyperplane is approximately horizontal at $\phi_2 = 0$.

### Parity-weighted Transformation
![Parity-weighted Transformation](../Images/L5_3_Quiz_40/parity_weighted_transformation.png)

The parity-weighted transformation $\phi(x) = [(x \bmod 2)x, (1-x \bmod 2)x]$ creates axis-aligned clusters. Positive points (odd numbers) lie on the x-axis, while negative points (even numbers) lie on the y-axis. The hyperplane is approximately vertical.

### Trigonometric Transformation
![Trigonometric Transformation](../Images/L5_3_Quiz_40/trigonometric_transformation.png)

The trigonometric transformation $\phi(x) = [\cos(\pi x), \sin(\pi x)]$ maps all points to the unit circle. Positive points (odd numbers) map to $(-1, 0)$, while negative points (even numbers) map to $(1, 0)$. The hyperplane is a vertical line through the origin.

## Key Insights

### Kernel Design Principles
- **Pattern Recognition**: Successful kernels capture the underlying data pattern (alternating/parity structure)
- **Dimensionality**: Higher dimensions don't guarantee better separation - the right transformation matters
- **Geometric Intuition**: Each kernel creates a different geometric arrangement in feature space

### Mathematical Properties
- All successful transformations produce positive semi-definite kernel matrices (verified by eigenvalue analysis)
- The kernel trick allows implicit computation without explicit feature mapping
- Support vectors determine the decision boundary regardless of the transformation used

### Practical Applications
- **Text Classification**: Similar alternating patterns appear in document classification
- **Signal Processing**: Periodic patterns in time series data
- **Computer Vision**: Checkerboard or alternating spatial patterns

## Conclusion

This comprehensive analysis demonstrates the power and mathematical elegance of the kernel trick for nonlinear classification:

### Detailed Mathematical Results

**Primary Transformation Analysis:**
- **Hyperplane Equation**: $x^2[1 + 2(x \bmod 2 - 0.5)] - 1 = 0$
- **Decision Function**: $f(x) = \text{sign}(2x^2 - 1)$ for odd $x$, $f(x) = -1$ for even $x$
- **Margin**: $0.447294$ in the transformed space
- **Support Vectors**: **5 points** $(x = 0, 1, 2, 4, 6)$ - more than initially apparent
- **Kernel Function**: $K(x,z) = x^2z^2[1 + (x \bmod 2 - 0.5)(z \bmod 2 - 0.5)]$

### Support Vector Distribution Analysis

**Key Finding**: The number of support vectors varies significantly across transformations:
- **Primary**: 5 support vectors (most points lie on margin boundaries)
- **Sign-based**: 7 support vectors (all points are support vectors)
- **Parity-weighted**: 5 support vectors
- **Trigonometric**: 7 support vectors (all points are support vectors)

This reveals that some transformations create "tighter" margins where more points become critical for the decision boundary.

### Transformation Comparison

| Transformation | Support Vectors | Margin | Geometric Insight |
|----------------|-----------------|--------|-------------------|
| **Primary** | 5/7 points | 0.447 | Diagonal separation with parity scaling |
| **Sign-based** | 7/7 points | 1.000 | Perfect horizontal stripes |
| **Parity-weighted** | 5/7 points | 0.500 | Axis-aligned clusters |
| **Trigonometric** | 7/7 points | 1.000 | Unit circle mapping |

### Step-by-Step Calculation Insights

**Critical Discovery**: Our detailed calculations revealed:
1. **Decision function values** must be computed individually for each point
2. **Support vectors** are identified by $|f(x)| \approx 1$, not just SVM indices
3. **Margin boundaries** contain more points than initially expected
4. **Kernel matrices** provide validation of transformation validity

### Fundamental Principles

**Pattern Recognition**: The key insight is that successful kernels must capture the **alternating/parity structure** inherent in the data. Simple polynomial transformations fail because they don't encode this critical pattern.

**Mathematical Rigor**: Every successful transformation satisfies:
- Positive semi-definite kernel matrices (Mercer's theorem)
- Perfect linear separability in feature space
- Consistent decision function calculations
- Proper margin geometry

### Practical Applications

This analysis demonstrates that:
- **Multiple valid solutions** exist for the same nonlinear problem
- **Geometric intuition** guides kernel design more than mathematical complexity
- **Support vector identification** requires careful analysis of decision boundaries
- **Kernel validity** can be verified through eigenvalue analysis

The kernel trick successfully transforms an impossible 1D classification problem into multiple elegant 2D solutions, each providing unique geometric insights while maintaining mathematical rigor.
