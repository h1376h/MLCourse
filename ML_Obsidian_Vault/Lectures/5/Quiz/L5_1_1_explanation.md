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

The dataset consists of 6 points in 2D space, with 3 points in each class. The given hyperplane is:
$$w_1 x_1 + w_2 x_2 + b = 0$$
$$1 \cdot x_1 + 1 \cdot x_2 - 2 = 0$$
$$x_1 + x_2 = 2$$

This can be rewritten as:
$$x_2 = -x_1 + 2$$

![Dataset with Separating Hyperplane](../Images/L5_1_Quiz_1/dataset_and_hyperplane.png)

The visualization shows:
- **Red circles**: Class +1 points at $(2, 3)$, $(3, 4)$, $(4, 2)$
- **Blue squares**: Class -1 points at $(0, 1)$, $(1, 0)$, $(0, 0)$
- **Green solid line**: The separating hyperplane $x_1 + x_2 = 2$
- **Green dashed lines**: Margin boundaries at $x_1 + x_2 = 1$ and $x_1 + x_2 = 3$
- **Shaded regions**: Red for Class +1 region, Blue for Class -1 region

### Step 2: Verification of Class Separation

To verify that the hyperplane separates the classes, we check that all Class +1 points have positive activation and all Class -1 points have negative activation.

**Class +1 points:**
- Point $(2, 3)$: Activation = $1 \times 2 + 1 \times 3 - 2 = 3 > 0$ ✓
- Point $(3, 4)$: Activation = $1 \times 3 + 1 \times 4 - 2 = 5 > 0$ ✓
- Point $(4, 2)$: Activation = $1 \times 4 + 1 \times 2 - 2 = 4 > 0$ ✓

**Class -1 points:**
- Point $(0, 1)$: Activation = $1 \times 0 + 1 \times 1 - 2 = -1 < 0$ ✓
- Point $(1, 0)$: Activation = $1 \times 1 + 1 \times 0 - 2 = -1 < 0$ ✓
- Point $(0, 0)$: Activation = $1 \times 0 + 1 \times 0 - 2 = -2 < 0$ ✓

**Result**: All Class +1 points have positive activation and all Class -1 points have negative activation. The hyperplane successfully separates the two classes.

### Step 3: Functional Margin Calculations

The functional margin for a point $(\mathbf{x}_i, y_i)$ is defined as:
$$\hat{\gamma}_i = y_i \times (\mathbf{w}^T \mathbf{x}_i + b)$$

**Functional margins for all points:**

| Point | Coordinates | Label | Activation | Functional Margin |
|-------|-------------|-------|------------|-------------------|
| 1 | $(2, 3)$ | $+1$ | $1 \times 2 + 1 \times 3 - 2 = 3$ | $+1 \times 3 = 3$ |
| 2 | $(3, 4)$ | $+1$ | $1 \times 3 + 1 \times 4 - 2 = 5$ | $+1 \times 5 = 5$ |
| 3 | $(4, 2)$ | $+1$ | $1 \times 4 + 1 \times 2 - 2 = 4$ | $+1 \times 4 = 4$ |
| 4 | $(0, 1)$ | $-1$ | $1 \times 0 + 1 \times 1 - 2 = -1$ | $-1 \times (-1) = 1$ |
| 5 | $(1, 0)$ | $-1$ | $1 \times 1 + 1 \times 0 - 2 = -1$ | $-1 \times (-1) = 1$ |
| 6 | $(0, 0)$ | $-1$ | $1 \times 0 + 1 \times 0 - 2 = -2$ | $-1 \times (-2) = 2$ |

**Key observations:**
- All functional margins are positive, confirming correct classification
- The minimum functional margin is 1, achieved by points $(0, 1)$ and $(1, 0)$
- These minimum margin points are closest to the decision boundary

### Step 4: Geometric Margin Calculation

The geometric margin for a point $(\mathbf{x}_i, y_i)$ is the actual distance from the point to the hyperplane:
$$\gamma_i = \frac{y_i \times (\mathbf{w}^T \mathbf{x}_i + b)}{||\mathbf{w}||}$$

For point $(2, 3)$ with label $y = +1$:

**Method 1: Using functional margin**
- Functional margin = $+1 \times 3 = 3$
- $||\mathbf{w}|| = \sqrt{1^2 + 1^2} = \sqrt{2} \approx 1.4142$
- Geometric margin = $\frac{3}{\sqrt{2}} \approx 2.1213$

**Method 2: Direct distance calculation**
- Distance = $\frac{|1 \times 2 + 1 \times 3 - 2|}{\sqrt{2}} = \frac{|3|}{\sqrt{2}} \approx 2.1213$
- Geometric margin = $+1 \times 2.1213 = 2.1213$

**Result**: The geometric margin for point $(2, 3)$ is approximately $2.1213$ units.

### Step 5: City Planning Problem - Optimal Road Boundary

This is a practical application of maximum margin classification. We need to find the optimal straight road boundary that maximizes the minimum distance from any house to the road.

**Using SVM to find the optimal boundary:**
- Zone A houses: $(2, 3)$, $(3, 4)$, $(4, 2)$ (Class +1)
- Zone B houses: $(0, 1)$, $(1, 0)$, $(0, 0)$ (Class -1)

The SVM algorithm finds the optimal hyperplane:
$$0.5x_1 + 0.5x_2 - 1.5 = 0$$

**Distances from houses to the optimal road:**

| House | Coordinates | Zone | Distance to Road |
|-------|-------------|------|------------------|
| 1 | $(2, 3)$ | A | 1.4142 units |
| 2 | $(3, 4)$ | A | 2.8284 units |
| 3 | $(4, 2)$ | A | 2.1213 units |
| 4 | $(0, 1)$ | B | 1.4142 units |
| 5 | $(1, 0)$ | B | 1.4142 units |
| 6 | $(0, 0)$ | B | 2.1213 units |

**Minimum distance**: 1.4142 units (achieved by houses at $(2, 3)$, $(0, 1)$, and $(1, 0)$)

**Classification of new house at $(2.5, 2.5)$:**
- Activation = $0.5 \times 2.5 + 0.5 \times 2.5 - 1.5 = 1.0 > 0$
- Distance to road = $\frac{|1.0|}{\sqrt{0.5^2 + 0.5^2}} = 1.4142$ units
- **Result**: The new house should be assigned to Zone A

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
- We successfully visualized the linearly separable dataset and verified the given hyperplane separates the classes
- We calculated functional margins for all points, finding a minimum margin of 1
- We computed the geometric margin for point $(2, 3)$ as approximately 2.1213 units
- We solved the city planning problem using SVM to find the optimal road boundary
- The new house at $(2.5, 2.5)$ should be assigned to Zone A
- The optimal solution maximizes the minimum distance from any house to the road, ensuring maximum safety margins

The maximum margin approach provides not only correct classification but also optimal generalization properties, making it superior to arbitrary separating hyperplanes for real-world applications.
