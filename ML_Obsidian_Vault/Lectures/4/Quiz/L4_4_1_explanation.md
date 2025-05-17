# Question 1: Linear Separability in 2D

## Problem Statement
Consider a 2D feature space with the following four data points:
- Class A: $(1, 1)$, $(2, 3)$
- Class B: $(-1, 0)$, $(0, -2)$

### Task
1. Sketch these points in a 2D coordinate system
2. Draw a linear decision boundary that separates these two classes
3. Write the equation of this decision boundary in the form $$w_1x_1 + w_2x_2 + b = 0$$
4. Is this dataset linearly separable? Explain why or why not in one sentence

## Understanding the Problem
Linear separability is a fundamental concept in machine learning that determines whether two classes can be completely separated by a linear boundary (a line in 2D, a plane in 3D, or a hyperplane in higher dimensions). 

In this problem, we have four points in a 2D feature space, with two points belonging to Class A and two belonging to Class B. We need to determine if there exists a linear decision boundary that can perfectly separate these classes. If such a boundary exists, the dataset is linearly separable.

## Solution

### Step 1: Plot the data points
First, let's visualize the data points in a 2D coordinate system to better understand their spatial distribution:
- Class A: $(1, 1)$, $(2, 3)$ (marked as blue circles)
- Class B: $(-1, 0)$, $(0, -2)$ (marked as red crosses)

![Data Points](../Images/L4_4_Quiz_1/data_points.png)

Looking at the plot, we can see that the points from Class A are located in the top-right quadrant of the coordinate system, while the points from Class B are positioned in the left and bottom parts. This visual inspection suggests that a linear boundary might be able to separate these classes.

### Step 2: Find a linear decision boundary
To find a linear decision boundary, we need to determine parameters $w_1$, $w_2$, and $b$ for the equation:
$$w_1x_1 + w_2x_2 + b = 0$$

We can use basic geometry to find this boundary:

1. First, calculate the center (centroid) of each class:
   - Center of Class A: $(1.5, 2.0)$
   - Center of Class B: $(-0.5, -1.0)$

2. Find the slope of the line connecting these centroids:
   $$m_{centroids} = \frac{y_B - y_A}{x_B - x_A} = \frac{-1.0 - 2.0}{-0.5 - 1.5} = \frac{-3.0}{-2.0} = 1.5$$

3. The decision boundary should be perpendicular to this line, so its slope is:
   $$m_{boundary} = -\frac{1}{m_{centroids}} = -\frac{1}{1.5} = -0.6667$$

4. Find the midpoint between the centroids:
   - Midpoint = $\left(\frac{1.5 + (-0.5)}{2}, \frac{2.0 + (-1.0)}{2}\right) = (0.5, 0.5)$

5. Use the point-slope form to get the equation of the decision boundary:
   $$\begin{align}
   y - y_1 &= m(x - x_1) \\
   y - 0.5 &= -0.6667(x - 0.5) \\
   y - 0.5 &= -0.6667x + 0.3333 \\
   y &= -0.6667x + 0.8333
   \end{align}$$

6. Convert to the standard form $w_1x_1 + w_2x_2 + b = 0$:
   $$\begin{align}
   -0.6667x + y - 0.8333 &= 0 \\
   0.6667x + 1.0y - 0.8333 &= 0
   \end{align}$$

Therefore, the parameters are:
$$w_1 = 0.6667, \quad w_2 = 1.0000, \quad b = -0.8333$$

This gives us the decision boundary equation:
$$0.6667x_1 + 1.0000x_2 - 0.8333 = 0$$

### Step 3: Verify the classification
Let's check if this decision boundary correctly classifies all points:

![Decision Boundary](../Images/L4_4_Quiz_1/decision_boundary.png)

For linear classifiers, a point is classified as:
- Class A (positive class) if $w_1x_1 + w_2x_2 + b > 0$
- Class B (negative class) if $w_1x_1 + w_2x_2 + b < 0$

Checking each point:
1. Point $(1, 1)$ from Class A: 
   $$0.6667 \times 1 + 1.0 \times 1 - 0.8333 = 0.8333 > 0 \implies \text{Correctly classified as Class A}$$
   
2. Point $(2, 3)$ from Class A: 
   $$0.6667 \times 2 + 1.0 \times 3 - 0.8333 = 3.5000 > 0 \implies \text{Correctly classified as Class A}$$
   
3. Point $(-1, 0)$ from Class B: 
   $$0.6667 \times (-1) + 1.0 \times 0 - 0.8333 = -1.5000 < 0 \implies \text{Correctly classified as Class B}$$
   
4. Point $(0, -2)$ from Class B: 
   $$0.6667 \times 0 + 1.0 \times (-2) - 0.8333 = -2.8333 < 0 \implies \text{Correctly classified as Class B}$$

All points are correctly classified by our decision boundary.

### Step 4: Demonstrate alternative boundaries
To further illustrate that the dataset is linearly separable, we can show that different linear boundaries can also separate the classes. Let's try an alternative boundary:
$$x_1 + x_2 - 1.5 = 0$$

![Multiple Boundaries](../Images/L4_4_Quiz_1/multiple_boundaries.png)

This alternative boundary also correctly classifies all points:
1. Point $(1, 1)$ from Class A: 
   $$1 \times 1 + 1 \times 1 - 1.5 = 0.5 > 0 \implies \text{Correctly classified as Class A}$$
   
2. Point $(2, 3)$ from Class A: 
   $$1 \times 2 + 1 \times 3 - 1.5 = 3.5 > 0 \implies \text{Correctly classified as Class A}$$
   
3. Point $(-1, 0)$ from Class B: 
   $$1 \times (-1) + 1 \times 0 - 1.5 = -2.5 < 0 \implies \text{Correctly classified as Class B}$$
   
4. Point $(0, -2)$ from Class B: 
   $$1 \times 0 + 1 \times (-2) - 1.5 = -3.5 < 0 \implies \text{Correctly classified as Class B}$$

The fact that we can find multiple linear boundaries that perfectly separate the classes confirms that the dataset is linearly separable.

## Key Insights

### Geometric Interpretation
- In 2D space, a linear decision boundary is a straight line
- Classes are linearly separable if there exists at least one straight line that can separate them
- For this dataset, the two classes are located in different quadrants of the coordinate system, making it visually apparent that a linear boundary can separate them
- The boundary can be found using the perpendicular bisector of the line connecting the class centroids

### Mathematical Formulation
- A linear decision boundary in 2D has the form $w_1x_1 + w_2x_2 + b = 0$
- The parameters $w_1$ and $w_2$ define the orientation of the boundary (normal vector)
- The parameter $b$ controls the offset from the origin
- A point $(x_1, x_2)$ is classified based on the sign of $w_1x_1 + w_2x_2 + b$

### Practical Implications
- Linearly separable datasets can be classified perfectly using simple linear models
- The existence of multiple valid decision boundaries suggests that the classes are well-separated
- Linear separability is a useful property that simplifies the learning problem

## Conclusion
The dataset is linearly separable because we can find a linear decision boundary (in the form $w_1x_1 + w_2x_2 + b = 0$) that perfectly separates the two classes, with all points from Class A on one side of the boundary and all points from Class B on the other side.

Specifically, we found two different linear boundaries that achieve perfect separation:
1. Geometric boundary: $$0.6667x_1 + 1.0000x_2 - 0.8333 = 0$$
2. Alternative boundary: $$x_1 + x_2 - 1.5 = 0$$

This confirms that the dataset is linearly separable, meaning a linear classifier can achieve perfect classification on this dataset. 