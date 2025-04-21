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
- **Example 3**: Distance Function
- **Example 4**: Temperature Distribution

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

---

### Example 2: Bivariate Normal Distribution

#### Problem Statement
Sketch the contour lines for the probability density function of a bivariate normal distribution with mean $\mu = (0,0)$ and covariance matrix $\Sigma = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$.

#### Solution

##### Step 1: Identify the PDF formula
For the standard bivariate normal distribution:
$$f(x,y) = \frac{1}{2\pi} e^{-\frac{x^2 + y^2}{2}}$$

##### Step 2: Find the contour equation
Taking the natural logarithm of both sides:
$$\ln(f(x,y)) = -\ln(2\pi) - \frac{x^2 + y^2}{2}$$

Contours of equal probability correspond to contours of equal $x^2 + y^2$.

##### Step 3: Draw the contours
The contours are circles centered at the origin. Each circle represents points of equal probability density.

For example:
- 68% of the probability mass lies within the circle of radius 1.15
- 95% of the probability mass lies within the circle of radius 2.45
- 99% of the probability mass lies within the circle of radius 3.03

---

### Example 3: Distance Function

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

---

### Example 5: Finding Local Extrema

#### Problem Statement
Use contour plot analysis to identify the critical points of the function $f(x,y) = x^2 + y^2 - 2x - 4y + 5$ and determine their nature.

#### Solution

##### Step 1: Complete the square to rewrite the function
$$f(x,y) = x^2 - 2x + y^2 - 4y + 5$$
$$f(x,y) = (x^2 - 2x + 1) + (y^2 - 4y + 4) + 5 - 1 - 4$$
$$f(x,y) = (x-1)^2 + (y-2)^2 + 0$$

##### Step 2: Analyze the contours
The contours are circles centered at the point $(1,2)$.

##### Step 3: Identify the critical point
Since the function is in the form of squared terms plus a constant, it has a unique critical point at $(1,2)$.

##### Step 4: Determine the nature of the critical point
Because the contours are circles centered at the critical point and the function increases as we move away from this point, $(1,2)$ is a local minimum with value $f(1,2) = 0$.

## Related Topics

- [[L2_1_Contour_Plots|Contour Plots]]: Core principles and interpretations
- [[L2_1_Multivariate_Distributions|Multivariate Distributions]]: Theoretical foundation
- [[L2_1_Covariance_Correlation|Covariance and Correlation]]: Related concepts for understanding distribution shapes
