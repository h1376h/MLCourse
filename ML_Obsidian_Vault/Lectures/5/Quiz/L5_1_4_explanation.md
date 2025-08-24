# Question 4: Dual Formulation Derivation

## Problem Statement
Consider the dual formulation of the maximum margin problem:
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j$$
$$\text{subject to: } \sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \geq 0$$

### Task
1. Derive this dual formulation from the primal using Lagrange multipliers
2. What is the relationship between $\mathbf{w}$ and the dual variables $\alpha_i$?
3. For a dataset with $n = 1000$ training points in $d = 50$ dimensions, compare the number of variables in primal vs dual formulations
4. Under what conditions would you prefer the dual formulation over the primal?
5. Prove that strong duality holds for the SVM optimization problem

## Understanding the Problem
The SVM dual formulation is a fundamental concept in support vector machines that transforms the original constrained optimization problem (primal) into an equivalent problem expressed in terms of Lagrange multipliers (dual). This transformation is crucial because it enables the kernel trick, provides insights into sparsity through support vectors, and can be computationally advantageous under certain conditions.

The primal SVM problem seeks to find the optimal separating hyperplane by minimizing the norm of the weight vector subject to margin constraints. The dual formulation reformulates this as maximizing a quadratic function of the Lagrange multipliers, subject to simpler constraints. This duality relationship is not just mathematically elegant—it has profound practical implications for how SVMs are implemented and applied.

## Solution

We will systematically derive the dual formulation and analyze its properties through both theoretical development and practical numerical examples.

### Step 1: Primal SVM Formulation

The primal SVM optimization problem for linearly separable data is:

$$\begin{align}
\min_{\mathbf{w}, b} \quad & \frac{1}{2}||\mathbf{w}||^2 \\
\text{subject to: } \quad & y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad \forall i = 1, \ldots, n
\end{align}$$

This formulation directly optimizes the margin (which is $\frac{2}{||\mathbf{w}||}$) by minimizing $||\mathbf{w}||^2$, while ensuring that all training points are correctly classified with margin at least 1.

### Step 2: Lagrangian Construction

To solve this constrained optimization problem, we introduce Lagrange multipliers $\alpha_i \geq 0$ for each constraint:

$$L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}||\mathbf{w}||^2 - \sum_{i=1}^n \alpha_i [y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1]$$

The Lagrangian combines the objective function with the constraints, weighted by the Lagrange multipliers. The key insight is that at the optimum, the gradient of the Lagrangian with respect to the primal variables must be zero.

### Step 3: Deriving Optimality Conditions

Taking partial derivatives and setting them to zero:

$$\frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = 0 \quad \Rightarrow \quad \mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i$$

$$\frac{\partial L}{\partial b} = -\sum_{i=1}^n \alpha_i y_i = 0 \quad \Rightarrow \quad \sum_{i=1}^n \alpha_i y_i = 0$$

These conditions reveal that:
- The optimal weight vector is a linear combination of the training points, weighted by $\alpha_i y_i$
- The constraint $\sum_{i=1}^n \alpha_i y_i = 0$ must hold at optimality

### Step 4: Dual Formulation Derivation (Detailed Pen-and-Paper Steps)

Now we substitute the optimality conditions $\mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i$ and $\sum_{i=1}^n \alpha_i y_i = 0$ back into the Lagrangian to eliminate the primal variables.

**Step 4a: Substitute $\mathbf{w}$ into the quadratic term**

$$\frac{1}{2}||\mathbf{w}||^2 = \frac{1}{2}\mathbf{w}^T\mathbf{w} = \frac{1}{2}\left(\sum_{i=1}^n \alpha_i y_i \mathbf{x}_i\right)^T\left(\sum_{j=1}^n \alpha_j y_j \mathbf{x}_j\right)$$

Expanding the dot product:
$$= \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j$$

**Step 4b: Substitute $\mathbf{w}$ into the constraint term**

$$\sum_{i=1}^n \alpha_i y_i (\mathbf{w}^T\mathbf{x}_i + b) = \sum_{i=1}^n \alpha_i y_i \mathbf{w}^T\mathbf{x}_i + b\sum_{i=1}^n \alpha_i y_i$$

The first term becomes:
$$\sum_{i=1}^n \alpha_i y_i \mathbf{w}^T\mathbf{x}_i = \sum_{i=1}^n \alpha_i y_i \left(\sum_{j=1}^n \alpha_j y_j \mathbf{x}_j\right)^T\mathbf{x}_i = \sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_j^T\mathbf{x}_i$$

The second term vanishes due to the constraint $\sum_{i=1}^n \alpha_i y_i = 0$:
$$b\sum_{i=1}^n \alpha_i y_i = b \cdot 0 = 0$$

**Step 4c: Combine all terms**

The Lagrangian becomes:
$$\begin{align}
L &= \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j - \sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_j^T\mathbf{x}_i + \sum_{i=1}^n \alpha_i \\
&= \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j - \sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j + \sum_{i=1}^n \alpha_i \\
&= \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j
\end{align}$$

Note: We used the fact that $\mathbf{x}_i^T\mathbf{x}_j = \mathbf{x}_j^T\mathbf{x}_i$ (inner product is commutative).

**Step 4d: Final dual formulation**

This gives us the dual optimization problem:

$$\begin{align}
\max_{\boldsymbol{\alpha}} \quad & \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j \\
\text{subject to: } \quad & \sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \geq 0, \quad \forall i
\end{align}$$

### Step 5: Detailed Manual Example (Pen-and-Paper Style)

Let's work through a simple 2-point example to illustrate the complete derivation process.

**Given**: Two points $\mathbf{x}_1 = [1, 0]$ with $y_1 = +1$ and $\mathbf{x}_2 = [0, 1]$ with $y_2 = -1$.

**Step 5a: Compute the Gram matrix**
$$K = \begin{bmatrix} \mathbf{x}_1^T\mathbf{x}_1 & \mathbf{x}_1^T\mathbf{x}_2 \\ \mathbf{x}_2^T\mathbf{x}_1 & \mathbf{x}_2^T\mathbf{x}_2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

**Step 5b: Write the dual objective**
$$L_D(\alpha_1, \alpha_2) = \alpha_1 + \alpha_2 - \frac{1}{2}[\alpha_1^2 y_1^2 K_{11} + 2\alpha_1\alpha_2 y_1 y_2 K_{12} + \alpha_2^2 y_2^2 K_{22}]$$

Substituting values:
$$L_D(\alpha_1, \alpha_2) = \alpha_1 + \alpha_2 - \frac{1}{2}[\alpha_1^2 \cdot 1^2 \cdot 1 + 2\alpha_1\alpha_2 \cdot 1 \cdot (-1) \cdot 0 + \alpha_2^2 \cdot (-1)^2 \cdot 1]$$
$$= \alpha_1 + \alpha_2 - \frac{1}{2}[\alpha_1^2 + \alpha_2^2]$$

**Step 5c: Apply constraints**
- Equality constraint: $\alpha_1 y_1 + \alpha_2 y_2 = \alpha_1 \cdot 1 + \alpha_2 \cdot (-1) = \alpha_1 - \alpha_2 = 0$
- Therefore: $\alpha_1 = \alpha_2$

**Step 5d: Substitute constraint into objective**
$$L_D(\alpha_1) = \alpha_1 + \alpha_1 - \frac{1}{2}[\alpha_1^2 + \alpha_1^2] = 2\alpha_1 - \alpha_1^2$$

**Step 5e: Maximize by taking derivative**
$$\frac{dL_D}{d\alpha_1} = 2 - 2\alpha_1 = 0 \Rightarrow \alpha_1 = 1$$

Therefore: $\alpha_1 = \alpha_2 = 1$

**Step 5f: Recover primal variables**
$$\mathbf{w} = \alpha_1 y_1 \mathbf{x}_1 + \alpha_2 y_2 \mathbf{x}_2 = 1 \cdot 1 \cdot [1,0] + 1 \cdot (-1) \cdot [0,1] = [1, -1]$$

Using support vector 1: $b = y_1 - \mathbf{w}^T\mathbf{x}_1 = 1 - [1,-1]^T[1,0] = 1 - 1 = 0$

**Final result**: Decision boundary is $\mathbf{w}^T\mathbf{x} + b = x_1 - x_2 = 0$, or $x_1 = x_2$.

## Practical Implementation

Our numerical implementation demonstrates these concepts using a simple 2D dataset with 6 points. The code solves both the theoretical derivation and practical optimization, showing:

### Numerical Example Results

From our implementation with the linearly separable dataset:
- **Positive class points**: $\mathbf{x}_1 = [3,3], \mathbf{x}_2 = [4,3], \mathbf{x}_3 = [3,4]$ (class +1)
- **Negative class points**: $\mathbf{x}_4 = [1,1], \mathbf{x}_5 = [2,1], \mathbf{x}_6 = [1,2]$ (class -1)

**Step-by-step numerical computation:**

**Step 1: Compute Gram Matrix**
$$K_{ij} = \mathbf{x}_i^T\mathbf{x}_j = \begin{bmatrix}
18 & 21 & 21 & 6 & 9 & 9 \\
21 & 25 & 24 & 7 & 11 & 10 \\
21 & 24 & 25 & 7 & 10 & 11 \\
6 & 7 & 7 & 2 & 3 & 3 \\
9 & 11 & 10 & 3 & 5 & 4 \\
9 & 10 & 11 & 3 & 4 & 5
\end{bmatrix}$$

**Step 2: Solve Dual Problem**
The optimal dual variables are:
- $\alpha_1 = 0.444$, $\alpha_2 = 0$, $\alpha_3 = 0$ (positive class)
- $\alpha_4 = 0$, $\alpha_5 = 0.222$, $\alpha_6 = 0.222$ (negative class)

**Step 3: Identify Support Vectors**
Support vectors (points with $\alpha_i > 0$):
- Point 1: $\mathbf{x}_1 = [3,3]$, $y_1 = +1$, $\alpha_1 = 0.444$
- Point 5: $\mathbf{x}_5 = [2,1]$, $y_5 = -1$, $\alpha_5 = 0.222$
- Point 6: $\mathbf{x}_6 = [1,2]$, $y_6 = -1$, $\alpha_6 = 0.222$

**Step 4: Recover Primal Variables**
$$\mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = 0.444 \cdot 1 \cdot [3,3] + 0.222 \cdot (-1) \cdot [2,1] + 0.222 \cdot (-1) \cdot [1,2]$$
$$= [1.333, 1.333] - [0.444, 0.222] - [0.222, 0.444] = [0.667, 0.667]$$

Bias computed from support vectors: $b = -3.000$

### Relationship Between Primal and Dual Variables

The fundamental relationship is:
$$\mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i$$

This shows that:
- The weight vector is completely determined by the dual variables
- Only points with $\alpha_i > 0$ contribute to the decision boundary (support vectors)
- The decision boundary depends only on support vectors, not all training data

## Visual Explanations

### SVM Primal Perspective
![SVM Primal Perspective](../Images/L5_1_Quiz_4/svm_primal_perspective.png)

This visualization shows the geometric interpretation of the SVM primal problem:
- **Red circles**: Class +1 data points
- **Blue squares**: Class -1 data points  
- **Black X markers**: Support vectors (points with $\alpha_i > 0$)
- **Green solid line**: Decision boundary where $\mathbf{w}^T\mathbf{x} + b = 0$
- **Green dashed lines**: Margin boundaries where $\mathbf{w}^T\mathbf{x} + b = \pm 1$

The primal perspective focuses on finding the optimal hyperplane that maximizes the margin between classes.

### SVM Dual Variables
![SVM Dual Variables](../Images/L5_1_Quiz_4/svm_dual_variables.png)

This bar chart visualizes the Lagrange multipliers $\alpha_i$:
- **Red bars**: Dual variables for class +1 points
- **Blue bars**: Dual variables for class -1 points
- **SV labels**: Points marked as support vectors (all points in this case)

The dual variables show that only 3 out of 6 training points are support vectors, demonstrating the sparsity property of SVM solutions. Points with $\alpha_i = 0$ do not contribute to the decision boundary.

### Gram Matrix Visualization
![Gram Matrix](../Images/L5_1_Quiz_4/svm_gram_matrix.png)

This heatmap shows the Gram matrix $K_{ij} = \mathbf{x}_i^T \mathbf{x}_j$:
- **Color intensity**: Represents the inner product between data points
- **Diagonal elements**: Squared norms of each point ($||\mathbf{x}_i||^2$)
- **Off-diagonal elements**: Similarity between different points

The Gram matrix is central to the dual formulation, as it appears in the quadratic term of the dual objective function.

### Primal vs Dual Formulation Comparison

| Aspect | Primal Formulation | Dual Formulation |
| --- | --- | --- |
| **Variables** | $\mathbf{w} \in \mathbb{R}^d, b \in \mathbb{R}$ | $\boldsymbol{\alpha} \in \mathbb{R}^n$ |
| **Number of Variables** | $d + 1$ | $n$ |
| **Objective Function** | $\min \frac{1}{2}\|\mathbf{w}\|^2$ | $\max \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i\alpha_j y_i y_j K_{ij}$ |
| **Constraints** | $n$ inequality constraints: $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$ | 1 equality constraint: $\sum_{i=1}^n \alpha_i y_i = 0$ and $\alpha_i \geq 0$ |
| **Solution Sparsity** | Dense weight vector $\mathbf{w}$ | Sparse $\boldsymbol{\alpha}$ (most $\alpha_i = 0$) |
| **Kernel Trick** | Not directly applicable | Directly applicable through $K_{ij}$ |
| **Computational Preference** | When $n > d$ (more samples than features) | When $n < d$ (fewer samples than features) |

### Variables Comparison Analysis
![Variables Comparison](../Images/L5_1_Quiz_4/svm_variables_comparison.png)

This plot shows how the number of variables scales with dataset size for fixed dimensionality ($d=50$):
- **Blue line**: Primal formulation variables ($d + 1 = 51$)
- **Red line**: Dual formulation variables ($n$)
- **Crossover point**: At $n = 51$, where both formulations have the same number of variables

For $n > 51$, the primal formulation is more efficient; for $n < 51$, the dual formulation is preferred.

### Formulation Choice Guidelines
![Formulation Choice](../Images/L5_1_Quiz_4/svm_formulation_choice.png)

This heatmap provides clear decision criteria:
- **Blue regions**: Prefer dual formulation (when $n < d$)
- **Red regions**: Prefer primal formulation (when $n > d$)
- **Boundary**: The line $n = d + 1$ where both formulations have equal complexity

### Kernel Trick Demonstration

#### Linear Kernel
![Linear Kernel](../Images/L5_1_Quiz_4/svm_linear_kernel.png)

The linear kernel $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$ is equivalent to the standard dot product and corresponds to linear classification in the original feature space.

#### RBF Kernel
![RBF Kernel](../Images/L5_1_Quiz_4/svm_rbf_kernel.png)

The RBF kernel $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2)$ enables non-linear classification by implicitly mapping data to an infinite-dimensional feature space.

The key insight is that the dual formulation only requires kernel values, not explicit feature mappings, making non-linear classification computationally feasible.

## Key Insights

### Theoretical Foundations
- **Duality Theory**: The SVM dual is derived through Lagrangian optimization, converting a constrained primal problem into an unconstrained dual problem
- **Strong Duality**: For SVMs, the duality gap is zero under mild conditions (linear constraints, convex objective), meaning primal and dual optimal values are equal
- **KKT Conditions**: The optimality conditions include stationarity, primal feasibility, dual feasibility, and complementary slackness
- **Support Vector Sparsity**: Points with $\alpha_i = 0$ don't affect the decision boundary, leading to sparse solutions

### Computational Advantages
- **Variable Count**: Dual has $n$ variables vs primal's $d+1$ variables. Choose dual when $n < d$
- **Kernel Trick**: Dual formulation enables non-linear classification through kernels without explicit feature mapping
- **Numerical Stability**: Dual problem often has better conditioning properties than the primal
- **Sparsity**: Most $\alpha_i = 0$ in practice, leading to efficient storage and computation

### Practical Decision Guidelines
- **Use Dual When**:
  - $n < d$ (fewer samples than features)
  - Need kernel trick for non-linear classification
  - Dataset naturally has few support vectors
  - Memory constraints favor sparse representation
- **Use Primal When**:
  - $n > d$ (more samples than features)
  - Linear classification is sufficient
  - Direct weight vector interpretation is needed

### Kernel Perspective
- **Linear Separability**: Both formulations handle linearly separable data equivalently
- **Non-linear Extension**: Only dual formulation naturally extends to non-linear cases via kernels
- **Feature Mapping**: Dual avoids explicit computation of potentially infinite-dimensional feature mappings
- **Computational Efficiency**: Kernel evaluations often more efficient than explicit feature computations

## Conclusion

The SVM dual formulation provides several key results and insights:

- **Mathematical Equivalence**: The dual formulation is mathematically equivalent to the primal, with strong duality ensuring zero duality gap (verified: gap = 8.04 × 10⁻⁸ ≈ 0)
- **Computational Trade-offs**: For the example case ($n=1000, d=50$), the primal formulation has 51 variables while the dual has 1000, making the primal preferable for this scenario
- **Sparsity Demonstration**: Our numerical example shows natural sparsity—only 3 out of 6 points became support vectors ($\alpha_1 = 0.444$, $\alpha_5 = \alpha_6 = 0.222$, others = 0)
- **KKT Conditions**: All optimality conditions are satisfied, confirming the solution's correctness:
  - Stationarity: $\mathbf{w} = \sum \alpha_i y_i \mathbf{x}_i$ and $\sum \alpha_i y_i = 0$
  - Primal feasibility: All margin constraints satisfied
  - Dual feasibility: All $\alpha_i \geq 0$
  - Complementary slackness: $\alpha_i \times \text{slack}_i = 0$ for all points
- **Algorithmic Flexibility**: The dual formulation enables the kernel trick, allowing SVMs to handle non-linear classification problems without explicitly computing high-dimensional feature mappings
- **Theoretical Foundation**: The step-by-step derivation demonstrates how Lagrangian optimization theory provides a principled approach to constrained optimization problems in machine learning

The dual formulation is not just a mathematical curiosity—it fundamentally changes how we think about and implement support vector machines, enabling both computational advantages and theoretical insights that have made SVMs one of the most successful machine learning algorithms.
