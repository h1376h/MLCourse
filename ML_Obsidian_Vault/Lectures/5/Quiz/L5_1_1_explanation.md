# Question 1: Separating Hyperplane Analysis

## Problem Statement
Consider a linearly separable dataset in $\mathbb{R}^2$ with the following points:
- Class $+1$: $(2, 3)$, $(3, 4)$, $(4, 2)$
- Class $-1$: $(0, 1)$, $(1, 0)$, $(0, 0)$

### Task
1. Draw these points on a coordinate system and sketch a possible separating hyperplane
2. For the hyperplane $w_1 x_1 + w_2 x_2 + b = 0$ with $w_1 = 1$, $w_2 = 1$, $b = -2$, verify that this hyperplane separates the two classes
3. Calculate the functional margin for each training point using this hyperplane
4. Calculate the geometric margin for the point $(2, 3)$
5. You are a city planner designing a neighborhood with two residential zones. Zone A houses are at $(2, 3)$, $(3, 4)$, $(4, 2)$ and Zone B houses are at $(0, 1)$, $(1, 0)$, $(0, 0)$. Design a straight road boundary that maximizes the minimum distance from any house to the road. If a new house is placed at $(2.5, 2.5)$, which zone should it belong to?

## Understanding the Problem
This problem explores the fundamental concepts of linear classification and maximum margin theory in Support Vector Machines (SVMs). We have a binary classification problem in 2D space where we need to:
- Visualize the data and understand linear separability
- Verify that a given hyperplane correctly separates the classes
- Calculate functional and geometric margins
- Find the optimal separating hyperplane that maximizes the margin
- Apply these concepts to a practical city planning scenario

The key concepts involved are:
- **Linear separability**: The ability to separate two classes with a straight line (in 2D) or hyperplane (in higher dimensions)
- **Functional margin**: A measure of how well a point is classified, given by $y_i \times (\mathbf{w}^T \mathbf{x}_i + b)$
- **Geometric margin**: The actual distance from a point to the hyperplane, given by $\frac{y_i \times (\mathbf{w}^T \mathbf{x}_i + b)}{||\mathbf{w}||}$
- **Maximum margin classifier**: The optimal hyperplane that maximizes the minimum distance from any point to the decision boundary

## Solution

### Step 1: Dataset Visualization and Hyperplane Sketch

**Given Data:**
- Class +1 points: $(2, 3)$, $(3, 4)$, $(4, 2)$
- Class -1 points: $(0, 1)$, $(1, 0)$, $(0, 0)$
- Hyperplane parameters: $w_1 = 1$, $w_2 = 1$, $b = -2$

**Hyperplane Equation:**
The given hyperplane is:
$$w_1 x_1 + w_2 x_2 + b = 0$$
$$1 \cdot x_1 + 1 \cdot x_2 + (-2) = 0$$
$$x_1 + x_2 - 2 = 0$$
$$x_1 + x_2 = 2$$

This can be rewritten in slope-intercept form as:
$$x_2 = -x_1 + 2$$

This represents a line with slope $-1$ and y-intercept $2$.

![Dataset with Separating Hyperplane](../Images/L5_1_Quiz_1/dataset_and_hyperplane.png)

**Visualization Components:**
- **Red circles**: Class +1 points at $(2, 3)$, $(3, 4)$, $(4, 2)$
- **Blue squares**: Class -1 points at $(0, 1)$, $(1, 0)$, $(0, 0)$
- **Green solid line**: The separating hyperplane $x_1 + x_2 = 2$
- **Green dashed lines**: Margin boundaries at $x_1 + x_2 = 1$ and $x_1 + x_2 = 3$
- **Shaded regions**: Red for Class +1 region (above line), Blue for Class -1 region (below line)

### Step 2: Verification of Class Separation

**Pen-and-Paper Calculation:**

To verify that the hyperplane separates the classes, we calculate the activation function $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + b$ for each point.

**Decision Rule:**
- If $f(\mathbf{x}) > 0$, the point belongs to Class +1
- If $f(\mathbf{x}) < 0$, the point belongs to Class -1

**Class +1 points (should have positive activation):**

1. **Point $(2, 3)$:**
   $$f(2, 3) = 1 \times 2 + 1 \times 3 + (-2) = 2 + 3 - 2 = 3 > 0$$ ✓

2. **Point $(3, 4)$:**
   $$f(3, 4) = 1 \times 3 + 1 \times 4 + (-2) = 3 + 4 - 2 = 5 > 0$$ ✓

3. **Point $(4, 2)$:**
   $$f(4, 2) = 1 \times 4 + 1 \times 2 + (-2) = 4 + 2 - 2 = 4 > 0$$ ✓

**Class -1 points (should have negative activation):**

1. **Point $(0, 1)$:**
   $$f(0, 1) = 1 \times 0 + 1 \times 1 + (-2) = 0 + 1 - 2 = -1 < 0$$ ✓

2. **Point $(1, 0)$:**
   $$f(1, 0) = 1 \times 1 + 1 \times 0 + (-2) = 1 + 0 - 2 = -1 < 0$$ ✓

3. **Point $(0, 0)$:**
   $$f(0, 0) = 1 \times 0 + 1 \times 0 + (-2) = 0 + 0 - 2 = -2 < 0$$ ✓

**Verification Result:**
- All Class +1 points have positive activation: ✓
- All Class -1 points have negative activation: ✓
- **Conclusion**: The hyperplane $x_1 + x_2 - 2 = 0$ successfully separates the two classes.

### Step 3: Functional Margin Calculations

**Pen-and-Paper Calculation:**

The functional margin for a point $(\mathbf{x}_i, y_i)$ is defined as:
$$\hat{\gamma}_i = y_i \times (\mathbf{w}^T \mathbf{x}_i + b)$$

This measures how confident we are in the classification. A larger positive functional margin indicates the point is farther from the decision boundary on the correct side.

**Step-by-step calculations for all points:**

**Class +1 points (label $y_i = +1$):**

1. **Point $(2, 3)$:**
   - Activation: $f(2, 3) = 1 \times 2 + 1 \times 3 + (-2) = 3$
   - Functional margin: $\hat{\gamma}_1 = (+1) \times 3 = 3$

2. **Point $(3, 4)$:**
   - Activation: $f(3, 4) = 1 \times 3 + 1 \times 4 + (-2) = 5$
   - Functional margin: $\hat{\gamma}_2 = (+1) \times 5 = 5$

3. **Point $(4, 2)$:**
   - Activation: $f(4, 2) = 1 \times 4 + 1 \times 2 + (-2) = 4$
   - Functional margin: $\hat{\gamma}_3 = (+1) \times 4 = 4$

**Class -1 points (label $y_i = -1$):**

4. **Point $(0, 1)$:**
   - Activation: $f(0, 1) = 1 \times 0 + 1 \times 1 + (-2) = -1$
   - Functional margin: $\hat{\gamma}_4 = (-1) \times (-1) = 1$

5. **Point $(1, 0)$:**
   - Activation: $f(1, 0) = 1 \times 1 + 1 \times 0 + (-2) = -1$
   - Functional margin: $\hat{\gamma}_5 = (-1) \times (-1) = 1$

6. **Point $(0, 0)$:**
   - Activation: $f(0, 0) = 1 \times 0 + 1 \times 0 + (-2) = -2$
   - Functional margin: $\hat{\gamma}_6 = (-1) \times (-2) = 2$

**Summary Table:**

| Point | Coordinates | Label | Activation | Functional Margin |
|-------|-------------|-------|------------|-------------------|
| 1 | $(2, 3)$ | $+1$ | $3$ | $3$ |
| 2 | $(3, 4)$ | $+1$ | $5$ | $5$ |
| 3 | $(4, 2)$ | $+1$ | $4$ | $4$ |
| 4 | $(0, 1)$ | $-1$ | $-1$ | $1$ |
| 5 | $(1, 0)$ | $-1$ | $-1$ | $1$ |
| 6 | $(0, 0)$ | $-1$ | $-2$ | $2$ |

**Key Observations:**
- All functional margins are positive, confirming correct classification
- **Minimum functional margin**: $\min(\hat{\gamma}_i) = 1$
- **Points with minimum margin**: $(0, 1)$ and $(1, 0)$ (points 4 and 5)
- These minimum margin points are closest to the decision boundary

### Step 4: Geometric Margin Calculation

**Pen-and-Paper Calculation:**

The geometric margin for a point $(\mathbf{x}_i, y_i)$ is the actual distance from the point to the hyperplane:
$$\gamma_i = \frac{y_i \times (\mathbf{w}^T \mathbf{x}_i + b)}{||\mathbf{w}||}$$

This represents the perpendicular distance from the point to the decision boundary, taking into account the correct classification.

**For point $(2, 3)$ with label $y = +1$:**

**Step 1: Calculate the weight vector norm**
$$||\mathbf{w}|| = \sqrt{w_1^2 + w_2^2} = \sqrt{1^2 + 1^2} = \sqrt{2} = 1.4142135...$$

**Step 2: Calculate the activation (already done in Step 2)**
$$f(2, 3) = 1 \times 2 + 1 \times 3 + (-2) = 3$$

**Step 3: Calculate functional margin (already done in Step 3)**
$$\hat{\gamma} = y \times f(\mathbf{x}) = (+1) \times 3 = 3$$

**Step 4: Calculate geometric margin**
$$\gamma = \frac{\hat{\gamma}}{||\mathbf{w}||} = \frac{3}{\sqrt{2}} = \frac{3\sqrt{2}}{2} = \frac{3 \times 1.4142135...}{1} = 2.1213203...$$

**Verification using direct distance formula:**
The distance from point $(x_0, y_0)$ to line $ax + by + c = 0$ is:
$$d = \frac{|ax_0 + by_0 + c|}{\sqrt{a^2 + b^2}}$$

For our hyperplane $1 \cdot x_1 + 1 \cdot x_2 + (-2) = 0$ and point $(2, 3)$:
$$d = \frac{|1 \times 2 + 1 \times 3 + (-2)|}{\sqrt{1^2 + 1^2}} = \frac{|3|}{\sqrt{2}} = \frac{3}{\sqrt{2}} = 2.1213203...$$

Since the point is correctly classified (positive activation for Class +1), the geometric margin is:
$$\gamma = (+1) \times 2.1213203... = 2.1213203...$$

**Result**: The geometric margin for point $(2, 3)$ is exactly $\frac{3}{\sqrt{2}} = \frac{3\sqrt{2}}{2} \approx 2.1213$ units.

### Step 5: City Planning Problem - Optimal Road Boundary

**Two Approaches: Geometric Method and SVM Optimization**

This is a practical application of maximum margin classification. We need to find the optimal straight road boundary that maximizes the minimum distance from any house to the road.

**Problem Setup:**
- Zone A houses: $(2, 3)$, $(3, 4)$, $(4, 2)$ (treat as Class +1)
- Zone B houses: $(0, 1)$, $(1, 0)$, $(0, 0)$ (treat as Class -1)
- Goal: Find hyperplane that maximizes the minimum distance to any house

---

## **APPROACH 1: SIMPLE GEOMETRIC METHOD (Pen-and-Paper)**

**Step 1: Identify the Convex Hulls**

First, let's find the convex hull (smallest convex polygon) containing each set of points:

**Zone A convex hull:** The points $(2,3)$, $(3,4)$, $(4,2)$ form a triangle.
**Zone B convex hull:** The points $(0,1)$, $(1,0)$, $(0,0)$ form a triangle.

**Step 2: Find the Closest Points Between Convex Hulls**

The optimal separating line will be equidistant from the closest points of the two convex hulls. We need to find which points from each zone are closest to each other.

**Distance calculations between all pairs:**
- Distance from $(2,3)$ to $(0,1)$: $\sqrt{(2-0)^2 + (3-1)^2} = \sqrt{4+4} = 2\sqrt{2} = 2.828$
- Distance from $(2,3)$ to $(1,0)$: $\sqrt{(2-1)^2 + (3-0)^2} = \sqrt{1+9} = \sqrt{10} = 3.162$
- Distance from $(2,3)$ to $(0,0)$: $\sqrt{(2-0)^2 + (3-0)^2} = \sqrt{4+9} = \sqrt{13} = 3.606$

- Distance from $(3,4)$ to $(0,1)$: $\sqrt{(3-0)^2 + (4-1)^2} = \sqrt{9+9} = 3\sqrt{2} = 4.243$
- Distance from $(3,4)$ to $(1,0)$: $\sqrt{(3-1)^2 + (4-0)^2} = \sqrt{4+16} = 2\sqrt{5} = 4.472$
- Distance from $(3,4)$ to $(0,0)$: $\sqrt{(3-0)^2 + (4-0)^2} = \sqrt{9+16} = 5$

- Distance from $(4,2)$ to $(0,1)$: $\sqrt{(4-0)^2 + (2-1)^2} = \sqrt{16+1} = \sqrt{17} = 4.123$
- Distance from $(4,2)$ to $(1,0)$: $\sqrt{(4-1)^2 + (2-0)^2} = \sqrt{9+4} = \sqrt{13} = 3.606$
- Distance from $(4,2)$ to $(0,0)$: $\sqrt{(4-0)^2 + (2-0)^2} = \sqrt{16+4} = 2\sqrt{5} = 4.472$

**Minimum distance:** $2\sqrt{2} = 2.828$ between points $(2,3)$ and $(0,1)$.

**Step 3: Find the Perpendicular Bisector**

The optimal separating line is the perpendicular bisector of the line segment connecting the closest points $(2,3)$ and $(0,1)$.

**Midpoint calculation:**
$$\text{Midpoint} = \left(\frac{2+0}{2}, \frac{3+1}{2}\right) = (1, 2)$$

**Slope of line segment $(2,3)$ to $(0,1)$:**
$$m_{\text{segment}} = \frac{1-3}{0-2} = \frac{-2}{-2} = 1$$

**Slope of perpendicular bisector:**
$$m_{\perp} = -\frac{1}{m_{\text{segment}}} = -\frac{1}{1} = -1$$

**Equation of perpendicular bisector:**
Using point-slope form with point $(1,2)$ and slope $-1$:
$$y - 2 = -1(x - 1)$$
$$y - 2 = -x + 1$$
$$y = -x + 3$$
$$x + y = 3$$

**Step 4: Verify This is the Optimal Solution**

The line $x + y = 3$ should be equidistant from $(2,3)$ and $(0,1)$.

**Distance from $(2,3)$ to line $x + y - 3 = 0$:**
$$d_1 = \frac{|2 + 3 - 3|}{\sqrt{1^2 + 1^2}} = \frac{|2|}{\sqrt{2}} = \frac{2}{\sqrt{2}} = \sqrt{2} = 1.414$$

**Distance from $(0,1)$ to line $x + y - 3 = 0$:**
$$d_2 = \frac{|0 + 1 - 3|}{\sqrt{1^2 + 1^2}} = \frac{|-2|}{\sqrt{2}} = \frac{2}{\sqrt{2}} = \sqrt{2} = 1.414$$

✓ **Verified:** Both distances are equal to $\sqrt{2} = 1.414$ units.

**Step 5: Check All Other Points**

Let's verify that all other points are at least this distance from the line:

- $(3,4)$: $d = \frac{|3 + 4 - 3|}{\sqrt{2}} = \frac{4}{\sqrt{2}} = 2\sqrt{2} = 2.828$ ✓
- $(4,2)$: $d = \frac{|4 + 2 - 3|}{\sqrt{2}} = \frac{3}{\sqrt{2}} = \frac{3\sqrt{2}}{2} = 2.121$ ✓
- $(1,0)$: $d = \frac{|1 + 0 - 3|}{\sqrt{2}} = \frac{2}{\sqrt{2}} = \sqrt{2} = 1.414$ ✓
- $(0,0)$: $d = \frac{|0 + 0 - 3|}{\sqrt{2}} = \frac{3}{\sqrt{2}} = \frac{3\sqrt{2}}{2} = 2.121$ ✓

**Result:** The minimum distance is $\sqrt{2} = 1.414$ units, achieved by points $(2,3)$, $(0,1)$, and $(1,0)$.

**Step 6: Classification of New House (Geometric Method)**

For the new house at $(2.5, 2.5)$:

**Method 1: Direct substitution**
$$2.5 + 2.5 = 5.0$$
Since $5.0 > 3.0$, the house is on the Zone A side of the boundary.

**Method 2: Distance calculation**
Distance from $(2.5, 2.5)$ to line $x + y - 3 = 0$:
$$d = \frac{|2.5 + 2.5 - 3|}{\sqrt{1^2 + 1^2}} = \frac{|2.0|}{\sqrt{2}} = \frac{2}{\sqrt{2}} = \sqrt{2} = 1.414 \text{ units}$$

Since the activation $2.5 + 2.5 - 3 = 2.0 > 0$, the house belongs to Zone A.

**Special observation:** The distance $\sqrt{2} = 1.414$ units is exactly equal to the margin distance, meaning the new house lies on the positive margin boundary!

**Geometric Solution Summary:**
- **Optimal road boundary:** $x + y = 3$
- **Primary support vectors:** $(2,3)$ and $(0,1)$ (closest points between zones)
- **Points on margin boundary:** $(2,3)$, $(0,1)$, and $(1,0)$
- **True support vectors (with $\alpha > 0$):** Only $(2,3)$ and $(0,1)$
- **Maximum minimum distance:** $\sqrt{2} = 1.414$ units
- **Margin width:** $2\sqrt{2} = 2.828$ units
- **New house classification:** Zone A (on positive margin boundary)

**Key insight:** While three points lie on the margin boundary, only two are needed to define the optimal hyperplane - the closest points between the convex hulls of each class.

---

## **APPROACH 2: SVM OPTIMIZATION METHOD**

**SVM Solution - Finding the Optimal Road Boundary:**

The Support Vector Machine finds the hyperplane that maximizes the margin. We need to solve the optimization problem mathematically.

**Step 1: SVM Primal Optimization Problem**

We want to find the hyperplane that maximizes the margin. The primal optimization problem is:

$$\min_{\mathbf{w}, b} \frac{1}{2}||\mathbf{w}||^2$$

Subject to constraints:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i = 1, 2, ..., 6$$

**Step 2: Setting up the Lagrangian**

The Lagrangian is:
$$L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}||\mathbf{w}||^2 - \sum_{i=1}^{6} \alpha_i [y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1]$$

where $\alpha_i \geq 0$ are the Lagrange multipliers.

**Step 3: KKT Conditions**

Taking partial derivatives and setting them to zero:

$$\frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^{6} \alpha_i y_i \mathbf{x}_i = 0$$

$$\frac{\partial L}{\partial b} = -\sum_{i=1}^{6} \alpha_i y_i = 0$$

From the first condition:
$$\mathbf{w}^* = \sum_{i=1}^{6} \alpha_i y_i \mathbf{x}_i$$

From the second condition:
$$\sum_{i=1}^{6} \alpha_i y_i = 0$$

**Step 4: Dual Problem and Solution**

The dual optimization problem becomes:
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{6} \alpha_i - \frac{1}{2} \sum_{i=1}^{6} \sum_{j=1}^{6} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

Subject to:
$$\sum_{i=1}^{6} \alpha_i y_i = 0, \quad \alpha_i \geq 0$$

**Step 5: Deriving Optimal Hyperplane from First Principles**

We need to solve the SVM optimization problem without using any solver. Let's derive the optimal hyperplane parameters mathematically.

**SVM Dual Problem Setup:**
The dual problem is:
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{6} \alpha_i - \frac{1}{2} \sum_{i=1}^{6} \sum_{j=1}^{6} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

Subject to:
$$\sum_{i=1}^{6} \alpha_i y_i = 0, \quad \alpha_i \geq 0$$

**Key Insight - Geometric Intuition:**
The optimal hyperplane will be determined by the points that are closest between the two classes. From geometric analysis, we expect the closest points to be the support vectors.

**Step 5a: Identifying Potential Support Vectors**
Let's assume that the optimal hyperplane has the form $w_1 x_1 + w_2 x_2 + b = 0$. Due to the symmetry of our problem (both classes are roughly aligned along the $x_1 + x_2$ direction), we can hypothesize that $w_1 = w_2$.

Let $w_1 = w_2 = w$ (to be determined). Then our hyperplane is:
$$wx_1 + wx_2 + b = 0$$
$$w(x_1 + x_2) + b = 0$$

**Step 5b: Using Support Vector Conditions**
Support vectors satisfy $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$. Let's check which points could be support vectors by examining their positions relative to potential hyperplanes.

**Step 5c: Systematic Analysis of Potential Support Vectors**

For our hyperplane $w(x_1 + x_2) + b = 0$, let's examine which points could be support vectors.

**Examining Class +1 points:**
- $(2,3)$: $x_1 + x_2 = 5$
- $(3,4)$: $x_1 + x_2 = 7$
- $(4,2)$: $x_1 + x_2 = 6$

**Examining Class -1 points:**
- $(0,1)$: $x_1 + x_2 = 1$
- $(1,0)$: $x_1 + x_2 = 1$
- $(0,0)$: $x_1 + x_2 = 0$

**Key observation:** The points with the smallest $x_1 + x_2$ values in Class +1 and the largest $x_1 + x_2$ values in Class -1 are most likely to be support vectors, as they are closest to the decision boundary.

**Most likely support vector candidates:**
- From Class +1: $(2,3)$ with $x_1 + x_2 = 5$ (smallest sum)
- From Class -1: $(0,1)$ and $(1,0)$ both with $x_1 + x_2 = 1$ (largest sum)

**Step 6: Setting Up the Support Vector System**

Let's assume $(2,3)$, $(0,1)$, and $(1,0)$ are support vectors. Then:

For support vector $(2,3)$ with $y_1 = +1$:
$$y_1(w \cdot 5 + b) = 1 \Rightarrow 5w + b = 1 \quad \text{...(1)}$$

For support vector $(0,1)$ with $y_4 = -1$:
$$y_4(w \cdot 1 + b) = 1 \Rightarrow (-1)(w + b) = 1 \Rightarrow w + b = -1 \quad \text{...(2)}$$

For support vector $(1,0)$ with $y_5 = -1$:
$$y_5(w \cdot 1 + b) = 1 \Rightarrow (-1)(w + b) = 1 \Rightarrow w + b = -1 \quad \text{...(3)}$$

**Notice:** Equations (2) and (3) are identical! This suggests that both $(0,1)$ and $(1,0)$ lie on the same margin boundary.

**Step 7: Solving for Optimal Parameters**

From the two independent equations:
1. $5w + b = 1$ (from support vector $(2,3)$)
2. $w + b = -1$ (from support vectors $(0,1)$ and $(1,0)$)

**Solving the system:**
Subtracting equation (2) from equation (1):
$$5w + b - (w + b) = 1 - (-1)$$
$$4w = 2$$
$$w = 0.5$$

Substituting back into equation (2):
$$0.5 + b = -1$$
$$b = -1.5$$

**Therefore, the optimal hyperplane parameters are:**
$$w_1 = w_2 = 0.5, \quad b = -1.5$$

**Optimal hyperplane equation:**
$$0.5x_1 + 0.5x_2 - 1.5 = 0$$
$$x_1 + x_2 = 3$$

**Step 8: Verification of All Constraints**

Let's verify that our solution satisfies all constraints $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$:

1. **Point $(2,3)$:** $y_1 = +1$, constraint = $(+1)(0.5 \cdot 2 + 0.5 \cdot 3 - 1.5) = 1$ ✓
2. **Point $(3,4)$:** $y_2 = +1$, constraint = $(+1)(0.5 \cdot 3 + 0.5 \cdot 4 - 1.5) = 2 > 1$ ✓
3. **Point $(4,2)$:** $y_3 = +1$, constraint = $(+1)(0.5 \cdot 4 + 0.5 \cdot 2 - 1.5) = 1.5 > 1$ ✓
4. **Point $(0,1)$:** $y_4 = -1$, constraint = $(-1)(0.5 \cdot 0 + 0.5 \cdot 1 - 1.5) = 1$ ✓
5. **Point $(1,0)$:** $y_5 = -1$, constraint = $(-1)(0.5 \cdot 1 + 0.5 \cdot 0 - 1.5) = 1$ ✓
6. **Point $(0,0)$:** $y_6 = -1$, constraint = $(-1)(0.5 \cdot 0 + 0.5 \cdot 0 - 1.5) = 1.5 > 1$ ✓

All constraints are satisfied!

**Step 9: Determining True Support Vectors via Lagrange Multipliers**

Now we need to determine which points are true support vectors by solving for the Lagrange multipliers.

From the KKT condition $\mathbf{w}^* = \sum_{i} \alpha_i y_i \mathbf{x}_i$, using our derived parameters $\mathbf{w}^* = [0.5, 0.5]$:

$$\begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix} = \alpha_1(+1)\begin{bmatrix} 2 \\ 3 \end{bmatrix} + \alpha_4(-1)\begin{bmatrix} 0 \\ 1 \end{bmatrix} + \alpha_5(-1)\begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

This gives us:
- $0.5 = 2\alpha_1 - \alpha_5$ (first component)
- $0.5 = 3\alpha_1 - \alpha_4$ (second component)

From the dual constraint: $\alpha_1 - \alpha_4 - \alpha_5 = 0$

**Solving the 3×3 system:**
From the dual constraint: $\alpha_1 = \alpha_4 + \alpha_5$

Substituting into the first component equation:
$$0.5 = 2(\alpha_4 + \alpha_5) - \alpha_5 = 2\alpha_4 + \alpha_5$$

Substituting into the second component equation:
$$0.5 = 3(\alpha_4 + \alpha_5) - \alpha_4 = 2\alpha_4 + 3\alpha_5$$

From these two equations:
$$2\alpha_4 + \alpha_5 = 2\alpha_4 + 3\alpha_5$$
$$\alpha_5 = 3\alpha_5$$
$$2\alpha_5 = 0$$
$$\alpha_5 = 0$$

Therefore:
- $2\alpha_4 + 0 = 0.5 \Rightarrow \alpha_4 = 0.25$
- $\alpha_1 = 0.25 + 0 = 0.25$

**Final Lagrange multipliers:**
$$\alpha_1 = 0.25, \quad \alpha_4 = 0.25, \quad \alpha_5 = 0$$

**True support vectors:** Only $(2,3)$ and $(0,1)$ with $\alpha_1 = \alpha_4 = 0.25 > 0$.
**Point $(1,0)$:** Has $\alpha_5 = 0$, so it's NOT a support vector despite lying on the margin.

**Verification:**
- $\mathbf{w}^* = 0.25 \cdot (+1) \cdot [2,3] + 0.25 \cdot (-1) \cdot [0,1] + 0 \cdot (-1) \cdot [1,0]$
- $= [0.5, 0.75] + [0, -0.25] + [0, 0] = [0.5, 0.5]$ ✓
- Dual constraint: $0.25 \cdot (+1) + 0.25 \cdot (-1) + 0 \cdot (-1) = 0$ ✓

**Step 7: Independent Derivation of Optimal Hyperplane**

Now let's verify our solution by deriving the optimal hyperplane parameters using only the true support vectors (those with $\alpha_i > 0$).

**From the Lagrange multiplier solution:**
- $\alpha_1 = 0.25$ for point $(2,3)$
- $\alpha_4 = 0.25$ for point $(0,1)$
- $\alpha_5 = 0$ for point $(1,0)$

**Deriving $\mathbf{w}^*$ from true support vectors:**
$$\mathbf{w}^* = \sum_{i} \alpha_i y_i \mathbf{x}_i = \alpha_1 y_1 \mathbf{x}_1 + \alpha_4 y_4 \mathbf{x}_4$$

$$= 0.25 \times (+1) \times \begin{bmatrix} 2 \\ 3 \end{bmatrix} + 0.25 \times (-1) \times \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

$$= \begin{bmatrix} 0.5 \\ 0.75 \end{bmatrix} + \begin{bmatrix} 0 \\ -0.25 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}$$

✓ **Verification:** This matches the SVM solution $\mathbf{w}^* = [0.5, 0.5]$.

**Deriving $b^*$ using any true support vector:**
Using support vector $(2,3)$ with $y_1 = +1$:
$$b^* = y_1 - \mathbf{w}^{*T}\mathbf{x}_1 = 1 - [0.5, 0.5] \cdot [2, 3] = 1 - 2.5 = -1.5$$

✓ **Verification:** This matches the SVM solution $b^* = -1.5$.

**Critical Insight:** Point $(1,0)$ is NOT actually a support vector!

Even though $(1,0)$ has functional margin = 1 (lies on the margin boundary), its Lagrange multiplier $\alpha_5 = 0$. This means it doesn't contribute to the optimal hyperplane construction.

**True support vectors:** Only $(2,3)$ and $(0,1)$ with $\alpha_1 = \alpha_4 = 0.25 > 0$.

**Mathematical explanation:** The optimal hyperplane is uniquely determined by the minimal set of support vectors needed to satisfy the KKT conditions. While three points lie on the margin boundary, only two are linearly independent and sufficient to define the optimal solution.

**Step 7: Computing Optimal Weight Vector**

From the SVM solution, we identify the support vectors (points with functional margin = 1):
- Point $(2,3)$ with $y_1 = +1$ and $\alpha_1 > 0$
- Point $(0,1)$ with $y_4 = -1$ and $\alpha_4 > 0$
- Point $(1,0)$ with $y_5 = -1$ and $\alpha_5 > 0$

The optimal weight vector is:
$$\mathbf{w}^* = \sum_{i=1}^{6} \alpha_i y_i \mathbf{x}_i = \alpha_1 y_1 \mathbf{x}_1 + \alpha_4 y_4 \mathbf{x}_4 + \alpha_5 y_5 \mathbf{x}_5$$

From the SVM solver, we get:
$$\mathbf{w}^* = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}$$

**Verification of the calculation:**
We can verify this by checking that the support vectors satisfy the constraint $y_i(\mathbf{w}^{*T} \mathbf{x}_i + b^*) = 1$.

**Step 8: Computing Optimal Bias**

Using support vector $(2,3)$ with the constraint $y_1(\mathbf{w}^{*T} \mathbf{x}_1 + b^*) = 1$:
$$(+1)(0.5 \cdot 2 + 0.5 \cdot 3 + b^*) = 1$$
$$0.5 \cdot 2 + 0.5 \cdot 3 + b^* = 1$$
$$1.0 + 1.5 + b^* = 1$$
$$2.5 + b^* = 1$$
$$b^* = -1.5$$

**Verification with other support vectors:**
- For $(0,1)$: $(-1)(0.5 \cdot 0 + 0.5 \cdot 1 - 1.5) = (-1)(-1.0) = 1$ ✓
- For $(1,0)$: $(-1)(0.5 \cdot 1 + 0.5 \cdot 0 - 1.5) = (-1)(-1.0) = 1$ ✓

**Final Optimal Parameters:**
$$w_1^* = 0.5, \quad w_2^* = 0.5, \quad b^* = -1.5$$

**Optimal hyperplane equation:**
$$0.5x_1 + 0.5x_2 + (-1.5) = 0$$
$$0.5x_1 + 0.5x_2 = 1.5$$

Multiplying by 2 to simplify:
$$x_1 + x_2 = 3$$

**This means the optimal road boundary is the line $x_1 + x_2 = 3$**

**Step 9: Detailed Margin Analysis**

The SVM creates three parallel lines that define the margin:

1. **Positive margin boundary**: $0.5x_1 + 0.5x_2 - 1.5 = +1$
   $$0.5x_1 + 0.5x_2 = 2.5 \Rightarrow x_1 + x_2 = 5$$

2. **Decision boundary (road)**: $0.5x_1 + 0.5x_2 - 1.5 = 0$
   $$0.5x_1 + 0.5x_2 = 1.5 \Rightarrow x_1 + x_2 = 3$$

3. **Negative margin boundary**: $0.5x_1 + 0.5x_2 - 1.5 = -1$
   $$0.5x_1 + 0.5x_2 = 0.5 \Rightarrow x_1 + x_2 = 1$$

**Margin Width Calculation:**
$$\text{Margin width} = \frac{2}{||\mathbf{w}^*||} = \frac{2}{\sqrt{0.5^2 + 0.5^2}} = \frac{2}{\sqrt{0.5}} = \frac{2}{\frac{\sqrt{2}}{2}} = \frac{4}{\sqrt{2}} = 2\sqrt{2} = 2.8284 \text{ units}$$

**Half-margin (minimum distance):**
$$\text{Half-margin} = \frac{1}{||\mathbf{w}^*||} = \frac{1}{\frac{\sqrt{2}}{2}} = \frac{2}{\sqrt{2}} = \sqrt{2} = 1.4142 \text{ units}$$

**Support Vector Verification:**
The support vectors lie exactly on the margin boundaries:
- $(2,3)$: $2 + 3 = 5$ (on positive margin boundary)
- $(0,1)$: $0 + 1 = 1$ (on negative margin boundary)
- $(1,0)$: $1 + 0 = 1$ (on negative margin boundary)

**Geometric Interpretation:**
- All Zone A houses satisfy $x_1 + x_2 \geq 3$ (above or on the road)
- All Zone B houses satisfy $x_1 + x_2 \leq 3$ (below or on the road)
- The margin band extends from $x_1 + x_2 = 1$ to $x_1 + x_2 = 5$
- **Total margin width**: $2.8284$ units
- **Minimum distance from any house to road**: $1.4142$ units

**Comparison with Given Hyperplane:**
- Given hyperplane: $x_1 + x_2 = 2$
- Optimal hyperplane: $x_1 + x_2 = 3$
- The optimal line is parallel but shifted upward by 1 unit
- The optimal solution maximizes the minimum distance to any point

**Step 10: Step-by-step Distance Calculations**

**Weight vector norm for optimal hyperplane:**
$$||\mathbf{w}^*|| = \sqrt{(0.5)^2 + (0.5)^2} = \sqrt{0.25 + 0.25} = \sqrt{0.5} = \frac{\sqrt{2}}{2} \approx 0.707107$$

**Distance formula:** For any point $\mathbf{x}$, the distance to hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$ is:
$$d = \frac{|\mathbf{w}^T\mathbf{x} + b|}{||\mathbf{w}||}$$

**Distance calculations for each house:**

1. **House $(2, 3)$ in Zone A:**
   - Activation: $f(2, 3) = 0.5 \times 2 + 0.5 \times 3 - 1.5 = 1.0 + 1.5 - 1.5 = 1.0$
   - Distance: $\frac{|1.0|}{0.707107} = 1.414214$ units
   - **Support Vector** (functional margin = 1)

2. **House $(3, 4)$ in Zone A:**
   - Activation: $f(3, 4) = 0.5 \times 3 + 0.5 \times 4 - 1.5 = 1.5 + 2.0 - 1.5 = 2.0$
   - Distance: $\frac{|2.0|}{0.707107} = 2.828427$ units

3. **House $(4, 2)$ in Zone A:**
   - Activation: $f(4, 2) = 0.5 \times 4 + 0.5 \times 2 - 1.5 = 2.0 + 1.0 - 1.5 = 1.5$
   - Distance: $\frac{|1.5|}{0.707107} = 2.121320$ units

4. **House $(0, 1)$ in Zone B:**
   - Activation: $f(0, 1) = 0.5 \times 0 + 0.5 \times 1 - 1.5 = 0 + 0.5 - 1.5 = -1.0$
   - Distance: $\frac{|-1.0|}{0.707107} = 1.414214$ units
   - **Support Vector** (functional margin = 1)

5. **House $(1, 0)$ in Zone B:**
   - Activation: $f(1, 0) = 0.5 \times 1 + 0.5 \times 0 - 1.5 = 0.5 + 0 - 1.5 = -1.0$
   - Distance: $\frac{|-1.0|}{0.707107} = 1.414214$ units
   - **Support Vector** (functional margin = 1)

6. **House $(0, 0)$ in Zone B:**
   - Activation: $f(0, 0) = 0.5 \times 0 + 0.5 \times 0 - 1.5 = 0 + 0 - 1.5 = -1.5$
   - Distance: $\frac{|-1.5|}{0.707107} = 2.121320$ units

**Summary of distances:**

| House | Coordinates | Zone | Activation | Distance to Road | On Margin | Support Vector ($\alpha > 0$) |
|-------|-------------|------|------------|------------------|-----------|-------------------------------|
| 1 | $(2, 3)$ | A | $+1.0$ | $1.414214$ units | ✓ | ✓ ($\alpha_1 = 0.25$) |
| 2 | $(3, 4)$ | A | $+2.0$ | $2.828427$ units | ✗ | ✗ |
| 3 | $(4, 2)$ | A | $+1.5$ | $2.121320$ units | ✗ | ✗ |
| 4 | $(0, 1)$ | B | $-1.0$ | $1.414214$ units | ✓ | ✓ ($\alpha_4 = 0.25$) |
| 5 | $(1, 0)$ | B | $-1.0$ | $1.414214$ units | ✓ | ✗ ($\alpha_5 = 0$) |
| 6 | $(0, 0)$ | B | $-1.5$ | $2.121320$ units | ✗ | ✗ |

**Key Insight - The Margin:**
The **margin** is the perpendicular distance between the decision boundary and the closest points from either class.

**Margin Calculation:**
- **Support vectors**: $(2,3)$, $(0,1)$, and $(1,0)$ (points with functional margin = 1)
- All support vectors are exactly $\sqrt{2} = 1.414214$ units from the boundary
- **Half-margin** = $1.414214$ units
- **Full margin width** = $2 \times 1.414214 = 2.828427$ units

**Mathematical Verification:**
For our optimal hyperplane with $\mathbf{w}^* = [0.5, 0.5]$:
- $||\mathbf{w}^*|| = \sqrt{0.5^2 + 0.5^2} = \frac{\sqrt{2}}{2} \approx 0.707107$
- **Margin width** = $\frac{2}{||\mathbf{w}^*||} = \frac{2}{0.707107} = 2.828427$ units ✓

**Support Vector Verification:**
Points that lie exactly on the margin boundaries ($\mathbf{w}^T\mathbf{x} + b = \pm 1$):
- $(2,3)$: $0.5(2) + 0.5(3) - 1.5 = 2.5 - 1.5 = 1.0$ (positive margin)
- $(0,1)$: $0.5(0) + 0.5(1) - 1.5 = 0.5 - 1.5 = -1.0$ (negative margin)
- $(1,0)$: $0.5(1) + 0.5(0) - 1.5 = 0.5 - 1.5 = -1.0$ (negative margin)

**Step 11: Classification of New House at $(2.5, 2.5)$**

**Detailed calculation:**
- Activation: $f(2.5, 2.5) = 0.5 \times 2.5 + 0.5 \times 2.5 - 1.5$
- $= 1.25 + 1.25 - 1.5 = 2.5 - 1.5 = 1.0$
- Distance to road: $\frac{|1.0|}{0.707107} = 1.414214$ units
- Since activation = $1.0 > 0$, the house belongs to Zone A

**Special observation**:
- The new house has activation = $1.0$, which means it lies **exactly on the positive margin boundary**!
- This makes it equidistant from the decision boundary as the support vectors
- **Result**: The new house should be assigned to **Zone A**

**Geometric verification:**
- New house coordinates: $(2.5, 2.5)$
- Sum: $2.5 + 2.5 = 5.0$
- Since $5.0 > 3.0$ (decision boundary), it's in Zone A
- Since $5.0 = 5.0$ (positive margin boundary), it's on the margin!

![Optimal Solution Comparison](../Images/L5_1_Quiz_1/optimal_solution_comparison.png)

## Visual Explanations

### Comparison of Given vs Optimal Hyperplane

The visualization shows two key insights:

1. **Given Hyperplane (Green)**: $x_1 + x_2 = 2$
   - Successfully separates the classes
   - Has a smaller margin than the optimal solution
   - Points $(0, 1)$ and $(1, 0)$ are closest to the boundary

2. **Optimal Hyperplane (Red)**: $0.5x_1 + 0.5x_2 = 1.5$
   - Maximizes the minimum distance from any point to the boundary
   - Provides better generalization potential
   - Creates equal minimum distances for multiple points

### City Planning Application

The city planning visualization demonstrates:
- **Optimal road boundary** (black line) that maximizes safety margins
- **Distance annotations** showing how far each house is from the road
- **Zone assignments** based on which side of the road each house falls
- **New house placement** with clear zone assignment (Zone A)

## Key Insights

### Geometric Interpretation
- The weight vector $\mathbf{w} = [w_1, w_2]^T$ is perpendicular to the decision boundary
- The bias term $b$ determines the offset of the hyperplane from the origin
- The margin width is inversely proportional to $||\mathbf{w}||$
- Points with minimum functional margin lie closest to the decision boundary

### Maximum Margin Principle
- The SVM finds the hyperplane that maximizes the minimum distance from any point to the boundary
- This provides better generalization compared to other separating hyperplanes
- The optimal solution is unique (up to scaling) for linearly separable data
- Support vectors are the points that lie exactly on the margin boundaries

### Practical Applications
- **City planning**: Maximizing safety margins between residential zones
- **Medical diagnosis**: Finding optimal decision boundaries for disease classification
- **Quality control**: Separating defective from non-defective products
- **Financial risk assessment**: Classifying high-risk vs low-risk investments

### Mathematical Properties
- **Functional margin**: Scale-dependent measure of classification confidence
- **Geometric margin**: Scale-invariant measure of actual distance
- **Normalization**: The constraint $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1$ normalizes the functional margin
- **Margin width**: Given by $\frac{2}{||\mathbf{w}||}$ for normalized hyperplanes

## Conclusion

**Summary of Pen-and-Paper Calculations:**

1. **Dataset Visualization**: Successfully plotted the linearly separable dataset with 3 points in each class and sketched the given hyperplane $x_1 + x_2 = 2$.

2. **Class Separation Verification**: Through detailed activation calculations, we verified that:
   - All Class +1 points have positive activations: $(2,3) \rightarrow 3$, $(3,4) \rightarrow 5$, $(4,2) \rightarrow 4$
   - All Class -1 points have negative activations: $(0,1) \rightarrow -1$, $(1,0) \rightarrow -1$, $(0,0) \rightarrow -2$

3. **Functional Margin Analysis**: Calculated functional margins for all 6 points:
   - Class +1: margins of 3, 5, and 4
   - Class -1: margins of 1, 1, and 2
   - **Minimum functional margin**: 1 (achieved by points $(0,1)$ and $(1,0)$)

4. **Geometric Margin Calculation**: For point $(2,3)$:
   - Exact value: $\frac{3}{\sqrt{2}} = \frac{3\sqrt{2}}{2}$
   - Decimal approximation: $2.1213$ units

5. **Optimal City Planning Solution**: Using both geometric and SVM approaches:
   - **Optimal road boundary**: $x_1 + x_2 = 3$ (both methods agree)
   - **Maximum minimum distance**: $\sqrt{2} \approx 1.4142$ units
   - **Support vectors**: $(2,3)$, $(0,1)$, $(1,0)$ (closest points between zones)
   - **New house classification**: $(2.5, 2.5)$ belongs to **Zone A** (on positive margin boundary)

**Key Mathematical Insights:**

- **Functional vs Geometric Margins**: Functional margin depends on the scale of $\mathbf{w}$, while geometric margin is scale-invariant
- **Support Vectors**: Points $(2,3)$, $(0,1)$, and $(1,0)$ are equidistant from the optimal boundary
- **Maximum Margin Principle**: The optimal hyperplane maximizes the minimum distance, providing better generalization
- **Practical Application**: The city planning problem demonstrates how SVM optimization ensures maximum safety margins

**Comparison of Independent Methods:**

| Aspect | Geometric Method | SVM Optimization |
|--------|------------------|------------------|
| **Approach** | Find perpendicular bisector of closest points | Solve constrained optimization via dual problem |
| **Complexity** | Simple pen-and-paper calculation | Requires Lagrangian/KKT conditions |
| **Starting Point** | Distance calculations between all point pairs | Primal optimization problem formulation |
| **Key Insight** | Closest points determine optimal boundary | Active constraints determine support vectors |
| **Derivation** | Perpendicular bisector of segment $(2,3)$-$(0,1)$ | Solve linear system for Lagrange multipliers |
| **Result** | $x_1 + x_2 = 3$ | $x_1 + x_2 = 3$ (identical result!) |
| **Support Vectors** | $(2,3)$ and $(0,1)$ (closest points) | $(2,3)$ and $(0,1)$ (points with $\alpha_i > 0$) |
| **Verification** | Distance calculations | KKT conditions and constraint verification |

**Real-World Significance:**
Both methods demonstrate that the maximum margin approach provides not only correct classification but also optimal generalization properties. The geometric method offers intuitive understanding, while the SVM optimization provides the mathematical framework for more complex, high-dimensional problems where geometric visualization is impossible.
