# SVM Cheatsheet and "How To" Guide for Pen & Paper Exams

## 📋 Quick Reference Cheatsheet

### Core SVM Formulas

**Primal Formulation (Hard Margin):**
$$\min_{\mathbf{w}, b} \frac{1}{2}||\mathbf{w}||^2$$
$$\text{subject to: } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n$$

**Primal Formulation (Soft Margin):**
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^n \xi_i$$
$$\text{subject to: } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**Dual Formulation:**
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$
$$\text{subject to: } \sum_{i=1}^n \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C$$

**Decision Function:**
$$f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$$

### Key Concepts

**Functional Margin:** $\hat{\gamma}_i = y_i(\mathbf{w}^T\mathbf{x}_i + b)$
**Geometric Margin:** $\gamma_i = \frac{y_i(\mathbf{w}^T\mathbf{x}_i + b)}{||\mathbf{w}||}$
**Margin Width:** $\frac{2}{||\mathbf{w}||}$

**Hinge Loss:** $L_h(y, f(x)) = \max(0, 1 - y \cdot f(x))$
**ε-insensitive Loss:** $L_ε(y, f(x)) = \max(0, |y - f(x)| - ε)$

### Common Kernels

**Linear:** $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T\mathbf{z}$
**Polynomial:** $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + c)^d$
**RBF/Gaussian:** $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma ||\mathbf{x} - \mathbf{z}||^2)$
**Sigmoid:** $K(\mathbf{x}, \mathbf{z}) = \tanh(\kappa \mathbf{x}^T\mathbf{z} + \theta)$

---

## 🎯 Question Type 1: Linear Separability & Hyperplane Analysis

### How to Approach:

**Step 1: Visualize the Data**
- Draw coordinate system
- Plot all points with different symbols for each class
- Look for patterns and potential separating lines

**Step 2: Test Linear Separability**
- Try to draw a line that separates all points
- If impossible → not linearly separable
- If possible → verify with calculations

**Step 3: Verify Given Hyperplane**
- Calculate $f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b$ for each point
- Check if positive class has $f(\mathbf{x}) > 0$
- Check if negative class has $f(\mathbf{x}) < 0$

**Step 4: Calculate Margins**
- **Functional margin:** $\hat{\gamma}_i = y_i \times f(\mathbf{x}_i)$
- **Geometric margin:** $\gamma_i = \frac{\hat{\gamma}_i}{||\mathbf{w}||}$
- **Margin width:** $\frac{2}{||\mathbf{w}||}$

**Step 5: Find Optimal Hyperplane**
- Identify support vectors (points with minimum margin)
- Use KKT conditions: $\alpha_i > 0$ only for support vectors
- Calculate $\mathbf{w} = \sum_{i} \alpha_i y_i \mathbf{x}_i$

### Example Template:
```
Given: Points (x1, y1), (x2, y2), ... with labels
1. Plot points → [sketch]
2. Test separability → [yes/no with reasoning]
3. Verify hyperplane w₁x₁ + w₂x₂ + b = 0:
   - Point (a,b): f(a,b) = w₁a + w₂b + b = [value] [>0/<0] ✓
4. Functional margins:
   - Point (a,b): γ̂ = y × f(a,b) = [value]
5. Geometric margin = γ̂/||w|| = [value]
```

---

## 🎯 Question Type 2: Primal vs Dual Formulation

### How to Approach:

**Step 1: Write Primal Formulation**
- Objective: minimize $\frac{1}{2}||\mathbf{w}||^2$
- Constraints: $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$ (hard margin)
- Add slack variables $\xi_i$ for soft margin

**Step 2: Write Lagrangian**
$$L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}||\mathbf{w}||^2 - \sum_{i=1}^n \alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1]$$

**Step 3: Derive KKT Conditions**
- $\frac{\partial L}{\partial \mathbf{w}} = 0$ → $\mathbf{w} = \sum_{i} \alpha_i y_i \mathbf{x}_i$
- $\frac{\partial L}{\partial b} = 0$ → $\sum_{i} \alpha_i y_i = 0$
- $\alpha_i \geq 0$ (dual feasibility)
- $\alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1] = 0$ (complementary slackness)

**Step 4: Substitute into Lagrangian**
- Replace $\mathbf{w}$ with $\sum_{i} \alpha_i y_i \mathbf{x}_i$
- Simplify to get dual formulation

**Step 5: Compare Complexity**
- Primal: $d+1$ variables (w, b)
- Dual: $n$ variables ($\alpha_i$)
- Choose based on $n$ vs $d$ relationship

### Example Template:
```
Primal: min ½||w||² s.t. y_i(w^T x_i + b) ≥ 1
Lagrangian: L = ½||w||² - Σα_i[y_i(w^T x_i + b) - 1]
KKT: ∂L/∂w = 0 → w = Σα_i y_i x_i
     ∂L/∂b = 0 → Σα_i y_i = 0
Dual: max Σα_i - ½ΣΣα_i α_j y_i y_j x_i^T x_j
      s.t. Σα_i y_i = 0, α_i ≥ 0
```

---

## 🎯 Question Type 3: Support Vector Identification

### How to Approach:

**Step 1: Understand Support Vector Properties**
- Support vectors have $\alpha_i > 0$
- They satisfy $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$ (on margin)
- They determine the decision boundary

**Step 2: Analyze Given Data**
- Calculate distances from potential hyperplanes
- Points closest to decision boundary are likely support vectors
- Points with minimum functional margin are candidates

**Step 3: Verify KKT Conditions**
- Check if $\alpha_i > 0$ for identified support vectors
- Verify $\sum_{i} \alpha_i y_i = 0$
- Confirm complementary slackness

**Step 4: Calculate Weight Vector**
- $\mathbf{w} = \sum_{i} \alpha_i y_i \mathbf{x}_i$ (sum only over support vectors)
- Verify with given hyperplane equation

**Step 5: Find Bias Term**
- Use support vector condition: $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$
- Solve for $b$ using any support vector

### Example Template:
```
Given hyperplane: w₁x₁ + w₂x₂ + b = 0
1. Calculate distances to hyperplane:
   - Point (a,b): distance = |w₁a + w₂b + b|/√(w₁² + w₂²)
2. Support vectors: points with minimum distance
3. Verify KKT: α_i > 0 only for support vectors
4. Weight vector: w = Σα_i y_i x_i (sum over SVs only)
5. Bias: b = y_i - w^T x_i (using any SV)
```

---

## 🎯 Question Type 4: Soft Margin & Slack Variables

### How to Approach:

**Step 1: Identify the Problem**
- Check if data is linearly separable
- If not → need soft margin SVM
- Identify potential outliers or noisy points

**Step 2: Understand Slack Variables**
- $\xi_i = \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))$
- $\xi_i > 0$ means constraint violation
- $\xi_i = 0$ means point is correctly classified with margin ≥ 1

**Step 3: Analyze C Parameter**
- Small C → more tolerance for violations (larger margin)
- Large C → less tolerance (smaller margin, closer to hard margin)
- $C \to \infty$ → hard margin SVM

**Step 4: Calculate Slack Values**
- For given hyperplane, calculate $f(\mathbf{x}_i) = \mathbf{w}^T\mathbf{x}_i + b$
- $\xi_i = \max(0, 1 - y_i f(\mathbf{x}_i))$
- Interpret: $\xi_i = 0$ (correct), $0 < \xi_i < 1$ (within margin), $\xi_i > 1$ (misclassified)

**Step 5: Compare with Hinge Loss**
- Hinge loss = slack variable value
- Total penalty = $\sum_{i} \xi_i$

### Example Template:
```
Given: Soft margin SVM with C = [value]
1. Slack variables: ξ_i = max(0, 1 - y_i f(x_i))
2. For point (a,b): f(a,b) = [value]
   ξ = max(0, 1 - y × [value]) = [value]
3. Interpretation:
   - ξ = 0: correctly classified with margin ≥ 1
   - 0 < ξ < 1: correctly classified but within margin
   - ξ > 1: misclassified
4. Total penalty = Σξ_i = [value]
```

---

## 🎯 Question Type 5: Kernel Trick & Feature Transformation

### How to Approach:

**Step 1: Prove Non-Linear Separability**
- Show that no linear hyperplane can separate the data
- Use contradiction method
- Set up system of inequalities and show inconsistency

**Step 2: Apply Feature Transformation**
- Given $\phi(\mathbf{x})$, calculate transformed points
- Verify linear separability in feature space
- Find separating hyperplane in feature space

**Step 3: Calculate Kernel Function**
- $K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^T\phi(\mathbf{z})$
- Expand and simplify
- Verify it matches given kernel formula

**Step 4: Express Decision Boundary**
- Transform hyperplane back to original space
- Decision rule: $f(\mathbf{x}) = \text{sign}(\sum_{i} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b)$

**Step 5: Analyze Computational Complexity**
- Explicit mapping: $O(d_{feature})$ per evaluation
- Kernel trick: $O(d_{input})$ per evaluation
- Compare for given $n$ and $d$

### Example Template:
```
Given: Non-linearly separable data in ℝ²
1. Prove non-separability:
   - Assume linear separator: w₁x₁ + w₂x₂ + b = 0
   - Set up constraints for all points
   - Show contradiction
2. Feature transformation: φ(x₁,x₂) = [transformation]
   - Transformed points: [list]
   - Linear separator in feature space: [equation]
3. Kernel: K(x,z) = φ(x)^T φ(z) = [simplified form]
4. Decision boundary in original space: [equation]
5. Complexity: explicit vs kernel trick
```

---

## 🎯 Question Type 6: Multi-Class SVM

### How to Approach:

**Step 1: One-vs-Rest (OvR)**
- Train $K$ binary classifiers: class $k$ vs all others
- Decision rule: $\hat{y} = \arg\max_k f_k(\mathbf{x})$
- Handle ties and confidence issues

**Step 2: One-vs-One (OvO)**
- Train $\binom{K}{2}$ pairwise classifiers
- Voting scheme: count wins for each class
- Handle ties with confidence scores

**Step 3: Analyze Class Imbalance**
- Calculate imbalance ratio for each binary problem
- OvR suffers more from imbalance than OvO
- Consider cost-sensitive modifications

**Step 4: Compare Computational Complexity**
- OvR: $K$ classifiers, $O(K)$ prediction time
- OvO: $\binom{K}{2}$ classifiers, $O(K^2)$ training, $O(K^2)$ prediction
- Choose based on $K$ and dataset size

**Step 5: Handle Ambiguities**
- Multiple positive predictions in OvR
- Voting ties in OvO
- Use confidence scores or probability estimates

### Example Template:
```
Given: K-class problem with n samples
1. OvR: K binary classifiers
   - Class k vs Rest: [n_k vs n-n_k samples]
   - Decision: ŷ = argmax_k f_k(x)
2. OvO: (K choose 2) = [number] classifiers
   - Pairwise problems: [list]
   - Voting: count wins for each class
3. Class imbalance:
   - OvR imbalance ratio: max(n_k)/(n-n_k)
   - OvO: more balanced
4. Complexity: OvR O(K) vs OvO O(K²) prediction
5. Ambiguity handling: [strategy]
```

---

## 🎯 Question Type 7: Support Vector Regression (SVR)

### How to Approach:

**Step 1: Understand ε-tube Concept**
- Points inside ε-tube have zero loss
- Points outside ε-tube have loss = $|y - f(x)| - ε$
- ε controls tolerance for prediction errors

**Step 2: Formulate SVR Problem**
- Primal: minimize $\frac{1}{2}||\mathbf{w}||^2 + C\sum_{i}(\xi_i + \xi_i^*)$
- Constraints: $y_i - f(\mathbf{x}_i) \leq ε + \xi_i$, $f(\mathbf{x}_i) - y_i \leq ε + \xi_i^*$
- Need both upper and lower slack variables

**Step 3: Calculate ε-insensitive Loss**
- $L_ε(y, f(x)) = \max(0, |y - f(x)| - ε)$
- Compare with squared loss for robustness analysis

**Step 4: Identify Support Vectors**
- Points outside ε-tube are support vectors
- Points on ε-tube boundary have $\alpha_i > 0$ or $\alpha_i^* > 0$
- Points inside ε-tube have $\alpha_i = \alpha_i^* = 0$

**Step 5: Analyze ε Parameter Effect**
- Small ε → more support vectors, less tolerance
- Large ε → fewer support vectors, more tolerance
- Choose based on noise level and accuracy requirements

### Example Template:
```
Given: SVR with ε = [value], C = [value]
1. ε-insensitive loss: L_ε(y,f(x)) = max(0, |y-f(x)| - ε)
2. For point (x_i, y_i):
   - Prediction: f(x_i) = [value]
   - Loss: max(0, |y_i - f(x_i)| - ε) = [value]
3. Support vectors: points outside ε-tube
4. ε effect:
   - Small ε: [more/fewer] support vectors, [tighter/looser] fit
   - Large ε: [more/fewer] support vectors, [tighter/looser] fit
5. Robustness: ε-insensitive vs squared loss
```

---

## 🎯 Question Type 8: Computational Considerations

### How to Approach:

**Step 1: Analyze Kernel Matrix Complexity**
- Size: $n \times n$ matrix
- Memory: $O(n^2)$ storage
- Computation: $O(n^2d)$ for linear kernel, $O(n^2)$ for RBF

**Step 2: Understand SMO Algorithm**
- Optimize exactly 2 variables at each iteration
- Constraint: $\sum_{i} \alpha_i y_i = 0$
- If $\alpha_i$ changes, $\alpha_j$ must change to maintain constraint

**Step 3: Working Set Selection**
- Find pair that maximally violates KKT conditions
- Violation measure: $|y_i f(\mathbf{x}_i) - 1|$ for misclassified points
- Convergence when no violating pairs exist

**Step 4: Memory Management**
- Kernel caching: store recently computed values
- Chunking: optimize subset of variables at a time
- Approximate methods for large datasets

**Step 5: Scaling Analysis**
- Linear kernel: $O(nd)$ memory vs $O(n^2)$ for kernel matrix
- RBF kernel: always $O(n^2)$ memory
- Choose based on $n$ and $d$ relationship

### Example Template:
```
Given: n samples, d features
1. Kernel matrix:
   - Size: n × n = [number] elements
   - Memory: [number] MB (8 bytes per float)
   - Computation: O(n²d) for linear, O(n²) for RBF
2. SMO algorithm:
   - Variables per iteration: 2
   - Constraint: Σα_i y_i = 0
   - Working set selection: maximal KKT violation
3. Memory strategies:
   - Linear kernel: store data matrix O(nd)
   - RBF kernel: cache kernel values
   - Chunking: optimize subset of variables
4. Scaling: when n > [threshold], use approximations
```

---

## 🎯 Question Type 9: Loss Function Analysis

### How to Approach:

**Step 1: Calculate Different Loss Values**
- 0-1 Loss: $L_{01}(y, f(x)) = \mathbb{I}[y \cdot f(x) \leq 0]$
- Hinge Loss: $L_h(y, f(x)) = \max(0, 1 - y \cdot f(x))$
- Logistic Loss: $L_{\ell}(y, f(x)) = \log(1 + e^{-y \cdot f(x)})$
- Squared Loss: $L_s(y, f(x)) = (y - f(x))^2$

**Step 2: Compare Properties**
- **0-1 Loss**: Non-differentiable, not convex
- **Hinge Loss**: Convex, piecewise linear, upper bounds 0-1 loss
- **Logistic Loss**: Smooth, convex, differentiable
- **Squared Loss**: Smooth, convex, but not suitable for classification

**Step 3: Analyze Derivatives**
- Hinge loss: discontinuous derivative at $y \cdot f(x) = 1$
- Logistic loss: smooth derivative everywhere
- Implications for optimization algorithms

**Step 4: Robustness Analysis**
- Hinge loss: robust to outliers (bounded loss)
- Squared loss: sensitive to outliers (unbounded)
- ε-insensitive loss: robust in regression

**Step 5: Practical Considerations**
- Hinge loss: good for SVM optimization
- Logistic loss: good for probability estimation
- Choose based on problem requirements

### Example Template:
```
Given: predictions f(x) and true labels y
1. Calculate losses for each point:
   - Point (y=1, f(x)=0.8): 
     * 0-1: [value], Hinge: [value], Logistic: [value]
2. Compare properties:
   - Hinge: convex, piecewise linear, upper bounds 0-1
   - Logistic: smooth, differentiable, probability interpretation
   - Squared: smooth but not suitable for classification
3. Derivatives:
   - Hinge: discontinuous at y·f(x) = 1
   - Logistic: smooth everywhere
4. Robustness: Hinge more robust to outliers than squared
5. Practical choice: [recommendation with reasoning]
```

---

## 🎯 Question Type 10: Parameter Tuning & Model Selection

### How to Approach:

**Step 1: Understand Parameter Effects**
- **C (regularization)**: controls margin vs error trade-off
- **γ (RBF kernel)**: controls influence radius of support vectors
- **ε (SVR)**: controls tolerance for prediction errors
- **d (polynomial kernel)**: controls polynomial degree

**Step 2: Grid Search Strategy**
- Define parameter ranges: $C \in [C_{min}, C_{max}]$, $\gamma \in [\gamma_{min}, \gamma_{max}]$
- Use logarithmic scale for better coverage
- Cross-validation to avoid overfitting

**Step 3: Validation Curves**
- Plot performance vs parameter value
- Identify underfitting/overfitting regions
- Choose parameter in sweet spot

**Step 4: Bias-Variance Trade-off**
- Small C/γ → high bias, low variance (underfitting)
- Large C/γ → low bias, high variance (overfitting)
- Optimal: balanced bias and variance

**Step 5: Computational Considerations**
- Kernel matrix computation dominates training time
- Parameter search multiplies training time
- Use efficient search strategies (Bayesian optimization)

### Example Template:
```
Given: SVM with parameters C and γ
1. Parameter effects:
   - Small C: [larger/smaller] margin, [more/fewer] support vectors
   - Large γ: [tighter/looser] decision boundary, [more/fewer] support vectors
2. Grid search:
   - C: [range], γ: [range]
   - Total combinations: [number]
   - Cross-validation: k-fold
3. Validation curves:
   - Underfitting: [parameter range]
   - Overfitting: [parameter range]
   - Optimal: [parameter range]
4. Bias-variance:
   - Small C/γ: high bias, low variance
   - Large C/γ: low bias, high variance
5. Computational cost: [estimation]
```

---

## 📝 General Exam Strategy

### Before Starting:
1. **Read the entire question** - identify all parts
2. **Identify question type** - use appropriate approach
3. **Plan your time** - allocate based on question complexity
4. **Gather formulas** - write down relevant equations

### During Solution:
1. **Show all steps** - even if you can do mental math
2. **Use clear notation** - define variables explicitly
3. **Draw diagrams** - visualization helps understanding
4. **Check units** - ensure dimensional consistency
5. **Verify results** - plug back into original equations

### Common Mistakes to Avoid:
- **Forgetting constraints** in optimization problems
- **Mixing up signs** in margin calculations
- **Ignoring KKT conditions** in dual formulation
- **Not checking linear separability** before applying hard margin
- **Forgetting slack variables** in soft margin problems
- **Miscalculating kernel functions** - expand carefully
- **Not considering computational complexity** in algorithm choice

### Time Management:
- **Simple calculations**: 2-3 minutes
- **Medium complexity**: 5-8 minutes  
- **Complex derivations**: 10-15 minutes
- **Multi-part problems**: 15-20 minutes

### Final Checklist:
- [ ] All parts of question addressed
- [ ] Units and signs correct
- [ ] Results make intuitive sense
- [ ] Diagrams labeled clearly
- [ ] Key steps explained

---

## 🎯 Quick Reference: Decision Trees

### Which SVM to Use?
```
Is data linearly separable?
├─ Yes → Hard Margin SVM
└─ No → Soft Margin SVM
    ├─ Small C → More tolerance
    └─ Large C → Less tolerance
```

### Which Kernel to Choose?
```
What's the data structure?
├─ Linear → Linear kernel
├─ Polynomial patterns → Polynomial kernel
├─ Radial patterns → RBF kernel
└─ Unknown → Try RBF (default choice)
```

### Which Multi-class Method?
```
How many classes (K)?
├─ K = 2 → Binary SVM
├─ K small (3-5) → One-vs-One
└─ K large (>5) → One-vs-Rest
```

### Which Loss Function?
```
What's the task?
├─ Classification → Hinge loss
├─ Regression → ε-insensitive loss
└─ Probability estimation → Logistic loss
```

---

*This guide covers the most common SVM question types. Practice with each approach and adapt based on specific problem requirements. Remember: understanding the concepts is more important than memorizing formulas!*
