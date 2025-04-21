import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def explain_example(name, function, formula, explanation_steps, explanation_math=None, contour_values=None):
    """Print a detailed explanation for a contour plot example."""
    print(f"\n{'='*80}")
    print(f"Example: {name}")
    print(f"{'='*80}")
    
    print(f"Function: f(x,y) = {formula}")
    
    print("\nStep-by-Step Solution:")
    for i, step in enumerate(explanation_steps, 1):
        print(f"\nStep {i}: {step['title']}")
        print(f"{step['description']}")
        
        # Print mathematical derivation if provided
        if explanation_math and i <= len(explanation_math):
            for line in explanation_math[i-1].split('\n'):
                print(f"  {line}")
    
    if contour_values:
        print("\nContour Values:")
        for c in contour_values:
            print(f"c = {c}")
    
    print("\nKey Insights:")
    print("- The contour lines connect all points (x,y) where f(x,y) equals the contour value c.")
    print("- Each contour shape provides insight into the function's behavior in that region.")
    print(f"- This function produces {function} contours, which indicate {function.lower()} behavior.")
    print(f"- Understanding these shapes allows quick identification of {function.lower()}-type functions.")
    
    print(f"{'='*80}\n")

def generate_explanations():
    """Generate detailed explanations for all contour plot examples."""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*50)
    print("CONTOUR PLOT EXAMPLES: STEP-BY-STEP EXPLANATIONS")
    print("="*50)
    print("\nThese explanations show how to sketch contour plots by hand using simple mathematical analysis.")
    print("For each function, we'll work through the steps of identifying contour shapes and drawing them.")
    
    # Example 1: Quadratic Function
    explain_example(
        name="Simple Quadratic Function",
        function="Circular",
        formula="x² + y²",
        contour_values=[1, 4, 9],
        explanation_steps=[
            {
                "title": "Identify the shape of each contour",
                "description": "For any constant value c, we have the equation x² + y² = c, which is the equation of a circle centered at the origin with radius √c."
            },
            {
                "title": "Calculate the radius for each contour level",
                "description": "We need to find the radius for each contour value by taking the square root."
            },
            {
                "title": "Draw the circles",
                "description": "With the radius for each contour level, we draw circles centered at the origin."
            }
        ],
        explanation_math=[
            "For c = 1: x² + y² = 1   =>   radius = √1 = 1\nFor c = 4: x² + y² = 4   =>   radius = √4 = 2\nFor c = 9: x² + y² = 9   =>   radius = √9 = 3",
            "Center: (0,0)\nRadius for c = 1: r = 1\nRadius for c = 4: r = 2\nRadius for c = 9: r = 3",
            "Draw circle with radius 1 centered at origin\nDraw circle with radius 2 centered at origin\nDraw circle with radius 3 centered at origin"
        ]
    )
    
    # Example 2: Linear Function
    explain_example(
        name="Linear Function",
        function="Linear",
        formula="2x + 3y",
        contour_values=[-3, 0, 3, 6],
        explanation_steps=[
            {
                "title": "Identify the equation of each contour",
                "description": "For any constant value c, we have the equation 2x + 3y = c, which represents a straight line."
            },
            {
                "title": "Rewrite in slope-intercept form",
                "description": "We can rewrite the equation as y = (c - 2x)/3, which is a line with slope -2/3 and y-intercept c/3."
            },
            {
                "title": "Find key points for each contour",
                "description": "For each value of c, we can find where the line crosses the x and y axes to help with plotting."
            }
        ],
        explanation_math=[
            "2x + 3y = c\ny = (c - 2x)/3\nSlope = -2/3, y-intercept = c/3",
            "x-axis intercept (y = 0): x = c/2\ny-axis intercept (x = 0): y = c/3",
            "For c = -3:\n  x-intercept: x = -3/2 = -1.5\n  y-intercept: y = -3/3 = -1\nFor c = 0:\n  x-intercept: x = 0\n  y-intercept: y = 0\nFor c = 3:\n  x-intercept: x = 3/2 = 1.5\n  y-intercept: y = 3/3 = 1\nFor c = 6:\n  x-intercept: x = 6/2 = 3\n  y-intercept: y = 6/3 = 2"
        ]
    )
    
    # Example 3: Manhattan Distance
    explain_example(
        name="Manhattan Distance Function",
        function="Diamond",
        formula="|x| + |y|",
        contour_values=[1, 2, 3],
        explanation_steps=[
            {
                "title": "Analyze the function",
                "description": "This function represents the Manhattan (or L1) distance from the origin, which gives the sum of the absolute distances along the x and y axes."
            },
            {
                "title": "Identify the equations in different quadrants",
                "description": "The absolute value function behaves differently in each quadrant, so we need to analyze each case separately."
            },
            {
                "title": "Draw the contours",
                "description": "Each contour forms a diamond (or rotated square) centered at the origin."
            }
        ],
        explanation_math=[
            "The function is |x| + |y| = c, which represents the Manhattan distance from the origin.",
            "In quadrant 1 (x ≥ 0, y ≥ 0): x + y = c\nIn quadrant 2 (x < 0, y ≥ 0): -x + y = c → y = c + x\nIn quadrant 3 (x < 0, y < 0): -x - y = c → y = -c - x\nIn quadrant 4 (x ≥ 0, y < 0): x - y = c → y = x - c",
            "For c = 1, vertices at: (0,1), (1,0), (0,-1), (-1,0)\nFor c = 2, vertices at: (0,2), (2,0), (0,-2), (-2,0)\nFor c = 3, vertices at: (0,3), (3,0), (0,-3), (-3,0)"
        ]
    )
    
    # Example 4: Product Function
    explain_example(
        name="Product Function",
        function="Hyperbolic",
        formula="xy",
        contour_values=[-2, -1, 0, 1, 2],
        explanation_steps=[
            {
                "title": "Set up the contour equation",
                "description": "For each constant value c, we have the equation xy = c."
            },
            {
                "title": "Rewrite in a form easier to plot",
                "description": "We can rewrite the equation as y = c/x, which is a hyperbola for any non-zero value of c."
            },
            {
                "title": "Analyze special cases",
                "description": "The case when c = 0 is special, giving the two coordinate axes."
            }
        ],
        explanation_math=[
            "xy = c",
            "Solving for y: y = c/x\nThis is a hyperbola with the coordinate axes as asymptotes.",
            "For c = 0: xy = 0, which means either x = 0 or y = 0, giving the two coordinate axes.\nFor c > 0: Hyperbolas in the first and third quadrants.\nFor c < 0: Hyperbolas in the second and fourth quadrants."
        ]
    )
    
    # Example 8: Maximum Function
    explain_example(
        name="Maximum Function",
        function="L-Shaped",
        formula="max(x, y)",
        contour_values=[-1, 0, 1, 2],
        explanation_steps=[
            {
                "title": "Understand the function",
                "description": "The function f(x,y) = max(x, y) returns the larger of the two values x and y."
            },
            {
                "title": "Identify regions where each variable dominates",
                "description": "We need to determine where x is larger than y and where y is larger than x."
            },
            {
                "title": "Draw the contours",
                "description": "For each contour level, we'll draw an L-shaped curve consisting of a horizontal and a vertical line."
            }
        ],
        explanation_math=[
            "f(x,y) = max(x, y) means:\n  When x > y: f(x,y) = x\n  When y > x: f(x,y) = y\n  When x = y: f(x,y) = x = y",
            "The line y = x divides the plane into two regions:\n  Above the line (y > x): f(x,y) = y\n  Below the line (x > y): f(x,y) = x",
            "For contour level c:\n  When x > y: The contour is the vertical line x = c\n  When y > x: The contour is the horizontal line y = c\n  These lines meet at the point (c,c) on the line y = x\n\nFor c = -1: L-shape through (-1,-1)\nFor c = 0: L-shape through (0,0)\nFor c = 1: L-shape through (1,1)\nFor c = 2: L-shape through (2,2)"
        ]
    )
    
    # Example 9: Circular Crater Function
    explain_example(
        name="Circular Crater Function",
        function="Nested Circular",
        formula="(x² + y² - 4)²",
        contour_values=[0, 1, 4, 9],
        explanation_steps=[
            {
                "title": "Analyze the function structure",
                "description": "This function squares the expression (x² + y² - 4), which represents the squared distance from a circle of radius 2."
            },
            {
                "title": "Find where the function equals each contour value",
                "description": "We need to solve the equation (x² + y² - 4)² = c for each contour value c."
            },
            {
                "title": "Calculate the radius for each contour",
                "description": "For each contour value, we may get two different circles based on the square root."
            }
        ],
        explanation_math=[
            "(x² + y² - 4)² = c",
            "Taking the square root of both sides:\nx² + y² - 4 = ±√c\nRearranging:\nx² + y² = 4 ± √c\n\nThis gives two circles for each positive c value:",
            "For c = 0:\n  x² + y² = 4 (single circle with radius 2)\n\nFor c = 1:\n  x² + y² = 4 + 1 = 5 (outer circle with radius √5 ≈ 2.24)\n  x² + y² = 4 - 1 = 3 (inner circle with radius √3 ≈ 1.73)\n\nFor c = 4:\n  x² + y² = 4 + 2 = 6 (outer circle with radius √6 ≈ 2.45)\n  x² + y² = 4 - 2 = 2 (inner circle with radius √2 ≈ 1.41)\n\nFor c = 9:\n  x² + y² = 4 + 3 = 7 (outer circle with radius √7 ≈ 2.65)\n  x² + y² = 4 - 3 = 1 (inner circle with radius 1)"
        ]
    )
    
    # Example 10: Simple Rotation Function
    explain_example(
        name="Simple Rotation Function",
        function="Modified Hyperbolic",
        formula="xy + x - y",
        contour_values=[-2, -1, 0, 1, 2],
        explanation_steps=[
            {
                "title": "Rearrange the function for easier analysis",
                "description": "We can factor the expression to better understand its structure."
            },
            {
                "title": "For each contour value, find the equation",
                "description": "We need to solve the equation xy + x - y = c for each contour value c."
            },
            {
                "title": "Identify key features of each contour",
                "description": "By analyzing the equation, we can determine the shape and asymptotes."
            }
        ],
        explanation_math=[
            "f(x,y) = xy + x - y\n  = x(y + 1) - y\n  = x(y + 1) - 1·y",
            "x(y + 1) - y = c\nx(y + 1) = c + y\nx = (c + y)/(y + 1)",
            "For c = 0:\n  x = y/(y + 1)\n  As y → ∞, x → 1\n  As y → 0, x → 0\n  Vertical asymptote at y = -1\n\nFor c = 1:\n  x = (1 + y)/(y + 1) = 1\n  This is the vertical line x = 1\n\nFor c = -1:\n  x = (-1 + y)/(y + 1) = 1 - 2/(y + 1)\n  As y → ∞, x → 1\n  Vertical asymptote at y = -1\n\nFor c = 2:\n  x = (2 + y)/(y + 1) = 1 + 1/(y + 1)\n  As y → ∞, x → 1\n  Vertical asymptote at y = -1\n\nFor c = -2:\n  x = (-2 + y)/(y + 1) = 1 - 3/(y + 1)\n  As y → ∞, x → 1\n  Vertical asymptote at y = -1"
        ]
    )
    
    # Example 11: Absolute Value Difference
    explain_example(
        name="Absolute Value Difference Function",
        function="Bowtie",
        formula="|x| - |y|",
        contour_values=[-2, -1, 0, 1, 2],
        explanation_steps=[
            {
                "title": "Analyze the function in different quadrants",
                "description": "The function |x| - |y| behaves differently in each quadrant due to the absolute value operations."
            },
            {
                "title": "Find the contour equations in each quadrant",
                "description": "We need to solve the equation |x| - |y| = c for each quadrant and contour value c."
            },
            {
                "title": "Draw the contours",
                "description": "The contours form distinctive shapes based on the quadrant and contour value."
            }
        ],
        explanation_math=[
            "The function is |x| - |y| = c\n\nIn quadrant 1 (x ≥ 0, y ≥ 0): x - y = c\nIn quadrant 2 (x < 0, y ≥ 0): -x - y = c\nIn quadrant 3 (x < 0, y < 0): -x + y = c\nIn quadrant 4 (x ≥ 0, y < 0): x + y = c",
            "Solving for y in each quadrant:\n\nQuadrant 1: y = x - c\nQuadrant 2: y = -x - c\nQuadrant 3: y = c + x\nQuadrant 4: y = c - x",
            "For c = 0:\n  Q1: y = x (line with slope 1)\n  Q2: y = -x (line with slope -1)\n  Q3: y = x (line with slope 1)\n  Q4: y = -x (line with slope -1)\n  This forms an \"X\" or \"bowtie\" pattern\n\nFor c = 1:\n  Q1: y = x - 1 (shifted down from y = x)\n  Q2: y = -x - 1 (shifted down from y = -x)\n  Q3: y = 1 + x (shifted up from y = x)\n  Q4: y = 1 - x (shifted up from y = -x)\n  This forms a shifted bowtie pattern\n\nSimilar patterns for other values of c."
        ]
    )
    
    # Example 12: Simple Sum Function - NEW SIMPLE EXAMPLE
    explain_example(
        name="Simple Sum Function",
        function="Diagonal Line",
        formula="x + y",
        contour_values=[0, 1, 2, 3],
        explanation_steps=[
            {
                "title": "Set up the contour equation",
                "description": "For each constant value c, we have the equation x + y = c."
            },
            {
                "title": "Identify the type of curve this represents",
                "description": "This is a straight line with slope -1 for any value of c."
            },
            {
                "title": "Find intercepts for easy plotting",
                "description": "For each contour value, find where the line crosses the axes to make plotting easier."
            },
            {
                "title": "Draw the contours",
                "description": "Plot the lines for different values of c to create the contour plot."
            }
        ],
        explanation_math=[
            "x + y = c",
            "Rewriting as y = c - x, we get a line with slope -1 and y-intercept c.",
            "For each contour level:\n  x-intercept (y = 0): x = c\n  y-intercept (x = 0): y = c",
            "For c = 0: Line through (0,0)\nFor c = 1: Line through (0,1) and (1,0)\nFor c = 2: Line through (0,2) and (2,0)\nFor c = 3: Line through (0,3) and (3,0)\n\nThe contour plot consists of parallel lines with slope -1, equally spaced and moving up/right as c increases."
        ]
    )
    
    print("\n" + "="*50)
    print("SUMMARY OF COMMON CONTOUR SHAPES")
    print("="*50)
    
    print("\nCircular Contours:")
    print("- Come from functions like f(x,y) = x² + y²")
    print("- Indicate rotational symmetry")
    print("- Often appear in distance-based functions")
    
    print("\nLinear Contours:")
    print("- Come from functions like f(x,y) = ax + by + c")
    print("- Indicate a planar surface")
    print("- Equally spaced parallel lines indicate constant gradient")
    
    print("\nHyperbolic Contours:")
    print("- Come from functions like f(x,y) = xy")
    print("- Indicate saddle-like behavior or product relationships")
    print("- Asymptotes often align with coordinate axes")
    
    print("\nDiamond Contours:")
    print("- Come from functions like f(x,y) = |x| + |y|")
    print("- Indicate Manhattan/L1 distance metric")
    print("- Corners always point along the coordinate axes")
    
    print("\nL-Shaped Contours:")
    print("- Come from functions like f(x,y) = max(x,y) or min(x,y)")
    print("- Indicate non-differentiable behavior along the diagonal")
    print("- Common in optimization and robust statistics")
    
    print("\nElliptical Contours:")
    print("- Come from multivariate Gaussian distributions or quadratic functions")
    print("- Shape determined by covariance matrix or coefficients of quadratic terms")
    print("- Orientation indicates correlation between variables")
    print("- Axis lengths indicate variance in principal directions")
    
    return "Explanations generated successfully!"

if __name__ == "__main__":
    generate_explanations() 