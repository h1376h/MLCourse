# Contour Plot Examples

This document provides practical, pen-and-paper examples of contour plots for various scenarios, illustrating their utility in visualizing multivariate distributions and functions.

## Key Concepts and Formulas

Contour plots visualize 3D surfaces on a 2D plane by connecting points of equal value. For a function $f(x,y)$, contour lines represent the sets:
$$\{(x,y) : f(x,y) = c\}$$
where $c$ is a constant value.

## Examples

The following examples demonstrate contour plots in different contexts:

- **Example 1**: Simple Quadratic Function
- **Example 2**: Bivariate Normal Distribution
- **Example 3**: Manhattan Distance Function
- **Example 4**: Product Function
- **Example 5**: Circle Plus Linear Term
- **Example 6**: Saddle Function
- **Example 7**: Finding Local Extrema

---

### Example 1: Simple Quadratic Function

#### Problem Statement
Sketch the contour plot for the function $f(x,y) = x^2 + y^2$ for the contour levels $c = 1, 4, 9$.

#### Solution

This function represents the squared distance from the origin.

##### Step 1: Identify the shape of each contour
For any constant value $c$:
$$x^2 + y^2 = c$$

This is the equation of a circle centered at the origin with radius $\sqrt{c}$.

##### Step 2: Draw the contours
- For $c = 1$: Circle with radius 1
- For $c = 4$: Circle with radius 2
- For $c = 9$: Circle with radius 3

The contour plot consists of concentric circles around the origin, with larger circles corresponding to higher values of the function.

![Quadratic Function Contour Plot](../Images/Contour_Plots/example1_quadratic.png)

---

### Example 2: Linear Function

#### Problem Statement
Sketch the contour plot for the function $f(x,y) = 2x + 3y$ for the contour levels $c = -3, 0, 3, 6$.

#### Solution

##### Step 1: Identify the equation of each contour
For a constant value $c$:
$$2x + 3y = c$$

Rearranging for $y$:
$$y = \frac{c - 2x}{3}$$

This is the equation of a straight line with slope $-\frac{2}{3}$ and y-intercept $\frac{c}{3}$.

##### Step 2: Find key points for each contour line
For each value of $c$, we can find where the line crosses the axes:
- x-axis (y = 0): $x = \frac{c}{2}$
- y-axis (x = 0): $y = \frac{c}{3}$

##### Step 3: Draw the contours
- For $c = -3$: Line through $(-1.5, 0)$ and $(0, -1)$
- For $c = 0$: Line through $(0, 0)$
- For $c = 3$: Line through $(1.5, 0)$ and $(0, 1)$
- For $c = 6$: Line through $(3, 0)$ and $(0, 2)$

The contour plot consists of parallel lines with slope $-\frac{2}{3}$. Higher contours appear toward the top-right of the plot.

![Linear Function Contour Plot](../Images/Contour_Plots/example2_linear.png)

---

### Example 3: Manhattan Distance Function

#### Problem Statement
Sketch the contour plot for the function $f(x,y) = |x| + |y|$ for the contour levels $c = 1, 2, 3$.

#### Solution

##### Step 1: Analyze the function
This function represents the Manhattan (or L1) distance from the origin.

##### Step 2: For each constant $c$, find points where $|x| + |y| = c$
When $x,y \geq 0$: $x + y = c$
When $x \geq 0, y \leq 0$: $x - y = c$
When $x \leq 0, y \geq 0$: $-x + y = c$
When $x,y \leq 0$: $-x - y = c$

##### Step 3: Draw the contours
Each contour forms a diamond (or square rotated 45°) centered at the origin:
- For $c = 1$: Diamond with vertices at $(0,1)$, $(1,0)$, $(0,-1)$, $(-1,0)$
- For $c = 2$: Diamond with vertices at $(0,2)$, $(2,0)$, $(0,-2)$, $(-2,0)$
- For $c = 3$: Diamond with vertices at $(0,3)$, $(3,0)$, $(0,-3)$, $(-3,0)$

![Manhattan Distance Contour Plot](../Images/Contour_Plots/example3_manhattan.png)

---

### Example 4: Product Function

#### Problem Statement
Sketch the contour plot for $f(x,y) = xy$ for the contour levels $c = -2, -1, 0, 1, 2$.

#### Solution

##### Step 1: Set up the contour equation
For each constant $c$:
$$xy = c$$

##### Step 2: Rewrite in a form easier to plot
$$y = \frac{c}{x}$$

This is a hyperbola for any non-zero value of $c$.

##### Step 3: Draw the contours
- For $c = 0$: The contour consists of the two axes ($x = 0$ and $y = 0$)
- For $c = 1$: A hyperbola in the first and third quadrants
- For $c = 2$: Another hyperbola in the first and third quadrants, further from the origin
- For $c = -1$: A hyperbola in the second and fourth quadrants
- For $c = -2$: Another hyperbola in the second and fourth quadrants, further from the origin

The contours form a family of hyperbolas, with the axes serving as asymptotes.

![Product Function Contour Plot](../Images/Contour_Plots/example4_product.png)

---

### Example 5: Circle Plus Linear Term

#### Problem Statement
Sketch the contour plot for the function $f(x,y) = x^2 + y^2 + y$ for the contour levels $c = 0, 1, 4, 9$.

#### Solution

##### Step 1: Complete the square for the y term
$$f(x,y) = x^2 + y^2 + y$$
$$f(x,y) = x^2 + (y^2 + y)$$
$$f(x,y) = x^2 + (y^2 + y + \frac{1}{4} - \frac{1}{4})$$
$$f(x,y) = x^2 + (y + \frac{1}{2})^2 - \frac{1}{4}$$

##### Step 2: Rearrange for contour lines
For a constant value $c$:
$$x^2 + (y + \frac{1}{2})^2 = c + \frac{1}{4}$$

##### Step 3: Draw the contours
Each contour is a circle centered at $(0, -\frac{1}{2})$ with radius $\sqrt{c + \frac{1}{4}}$:
- For $c = 0$: Circle with radius $\frac{1}{2}$ centered at $(0, -\frac{1}{2})$
- For $c = 1$: Circle with radius $\sqrt{1.25} \approx 1.12$ centered at $(0, -\frac{1}{2})$
- For $c = 4$: Circle with radius $\sqrt{4.25} \approx 2.06$ centered at $(0, -\frac{1}{2})$
- For $c = 9$: Circle with radius $\sqrt{9.25} \approx 3.04$ centered at $(0, -\frac{1}{2})$

The contour plot consists of concentric circles, but the center is shifted to $(0, -\frac{1}{2})$ rather than being at the origin.

![Circle Plus Line Contour Plot](../Images/Contour_Plots/example5_circle_plus_line.png)

---

### Example 6: Saddle Function

#### Problem Statement
Sketch the contour plot for the function $f(x,y) = x^2 - y^2$ for the contour levels $c = -4, -1, 0, 1, 4$.

#### Solution

##### Step 1: Analyze the function
This is a saddle function, with the shape of a horse saddle near the origin.

##### Step 2: Rearrange for contour lines
For a constant value $c$:
$$x^2 - y^2 = c$$

This can be rewritten in different ways depending on the value of $c$:

If $c = 0$:
$$x^2 = y^2$$
$$x = \pm y$$

These are the lines $y = x$ and $y = -x$, which form an "X" shape through the origin.

If $c \neq 0$:
$$\frac{x^2}{c} - \frac{y^2}{c} = 1$$ (when $c > 0$)
$$\frac{y^2}{-c} - \frac{x^2}{-c} = 1$$ (when $c < 0$)

These are hyperbolas. When $c > 0$, the hyperbolas open along the x-axis. When $c < 0$, they open along the y-axis.

##### Step 3: Draw the contours
- For $c = 0$: Lines $y = x$ and $y = -x$
- For $c = 1$: Hyperbola with transverse axis along the x-axis
- For $c = 4$: Wider hyperbola with transverse axis along the x-axis
- For $c = -1$: Hyperbola with transverse axis along the y-axis
- For $c = -4$: Wider hyperbola with transverse axis along the y-axis

The contour plot shows a saddle point at the origin, where the function changes from increasing to decreasing in different directions.

![Saddle Function Contour Plot](../Images/Contour_Plots/example6_saddle.png)

---

### Example 7: Finding Local Extrema

#### Problem Statement
Use contour plot analysis to identify the critical points of the function $f(x,y) = x^2 + y^2 - 4x - 6y + 13$ and determine their nature.

#### Solution

##### Step 1: Complete the square to rewrite the function
$$f(x,y) = x^2 - 4x + y^2 - 6y + 13$$
$$f(x,y) = (x^2 - 4x + 4) + (y^2 - 6y + 9) + 13 - 4 - 9$$
$$f(x,y) = (x-2)^2 + (y-3)^2 + 0$$

##### Step 2: Analyze the contours
The contours are circles centered at the point $(2,3)$.

##### Step 3: Identify the critical point
Since the function is in the form of squared terms plus a constant, it has a unique critical point at $(2,3)$.

##### Step 4: Determine the nature of the critical point
Because the contours are circles centered at the critical point and the function increases as we move away from this point, $(2,3)$ is a local minimum with value $f(2,3) = 0$.

![Local Extrema Contour Plot](../Images/Contour_Plots/example7_local_extrema.png)

## Key Insights from Contour Plots

### Shape Identification
- **Circles**: Indicate functions like $x^2 + y^2 + C$ (possibly shifted)
- **Straight Lines**: Indicate linear functions like $ax + by + c$
- **Diamonds**: Indicate functions with Manhattan distance ($|x| + |y|$)
- **Hyperbolas**: Indicate functions with products ($xy$) or differences of squares ($x^2 - y^2$)

### Critical Points
- **Concentric circles**: Indicate a local minimum or maximum
- **Hyperbolic patterns**: Indicate a saddle point
- **Parallel lines**: Indicate no critical points (constantly sloped plane)

### Practical Applications
- Identifying the nature of critical points without calculus
- Understanding the behavior of multivariate functions at a glance
- Visualizing probability densities in statistics
- Analyzing optimization landscapes in machine learning

## Related Topics

- [[L2_1_Contour_Plots|Contour Plots]]: Core principles and interpretations
- [[L2_1_Multivariate_Distributions|Multivariate Distributions]]: Theoretical foundation
- [[L2_1_Covariance_Correlation|Covariance and Correlation]]: Related concepts for understanding distribution shapes
