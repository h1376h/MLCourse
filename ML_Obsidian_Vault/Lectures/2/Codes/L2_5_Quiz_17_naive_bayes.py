import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import networkx as nx
import os
from matplotlib.patches import Patch

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def print_substep(substep_title):
    """Print a formatted substep header."""
    print(f"\n{'-' * 50}")
    print(f"{substep_title}")
    print(f"{'-' * 50}")

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_17")
os.makedirs(save_dir, exist_ok=True)

# ==============================
# STEP 1: Setting up the Problem
# ==============================
print_step_header(1, "Setting up the Problem")

print("We have a Naive Bayes classifier with the following conditional probabilities:")
print("\nTable 1: Naive Bayes conditional probabilities")
print("---------------------------------------------")
print("|   | Y = 0             | Y = 1             |")
print("|---|-------------------|-------------------|")
print("| X1| P(X1=1|Y=0) = 1/5 | P(X1=1|Y=1) = 3/8 |")
print("| X2| P(X2=1|Y=0) = 1/3 | P(X2=1|Y=1) = 3/4 |")
print("---------------------------------------------")

print("\nWe also know:")
print("- The likelihood of samples {1,0,1} and {0,1,0} is 1/180")
print("- We need to find w_1 = P(Y=1)")

# Define the known conditional probabilities
p_x1_given_y0 = 1/5  # P(X1=1|Y=0)
p_x1_given_y1 = 3/8  # P(X1=1|Y=1)
p_x2_given_y0 = 1/3  # P(X2=1|Y=0)
p_x2_given_y1 = 3/4  # P(X2=1|Y=1)

# Define the samples
sample1 = {"X1": 1, "X2": 0, "Y": 1}  # {1,0,1}
sample2 = {"X1": 0, "X2": 1, "Y": 0}  # {0,1,0}

print("\nWe need to compute the likelihood of samples {1,0,1} and {0,1,0} and use the constraint")
print("that this likelihood equals 1/180 to find w_1 = P(Y=1).")

def visualize_naive_bayes_model():
    """Create a visualization of the Naive Bayes graphical model without text annotations."""
    plt.figure(figsize=(8, 6))
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node("Y", pos=(0, 0))
    G.add_node("X1", pos=(-1, -1.5))
    G.add_node("X2", pos=(1, -1.5))
    
    # Add edges
    G.add_edge("Y", "X1")
    G.add_edge("Y", "X2")
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, 
                          node_color=['#3498db', '#2ecc71', '#2ecc71'])
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')
    
    # Just add a simple title
    plt.title("Naive Bayes Graphical Model", fontsize=14)
    
    # Remove axes
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    filename = os.path.join(save_dir, "naive_bayes_graphical_model.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Naive Bayes graphical model visualization saved to: {filename}")
    
    # Print the explanations that were previously in the image
    print("\nNaive Bayes Model Explanation:")
    print("- The Naive Bayes model assumes conditional independence between features given the class.")
    print("- Mathematically: P(X1,X2|Y) = P(X1|Y) × P(X2|Y)")
    print("- In the graphical model, Y is the class variable that directly influences features X1 and X2.")
    print("- There is no direct connection between X1 and X2, representing their conditional independence.")

# Create the Naive Bayes graphical model visualization
visualize_naive_bayes_model()

# ==============================
# STEP 2: Understanding Naive Bayes and calculating the likelihood
# ==============================
print_step_header(2, "Understanding Naive Bayes and calculating the likelihood")

print_substep("Naive Bayes Model")
print("Naive Bayes assumes that features X1 and X2 are conditionally independent given the class Y.")
print("This means: P(X1,X2|Y) = P(X1|Y) * P(X2|Y)")

print("\nThe complete model is:")
print("P(X1,X2,Y) = P(Y) * P(X1|Y) * P(X2|Y)")

print("\nVisual Explanation: Naive Bayes Graphical Model")
print("The Naive Bayes model assumes that features X1 and X2 are conditionally independent given the class Y.")
print("In a graphical model, Y would directly influence both X1 and X2, but there's no direct connection between X1 and X2.")
print("Y → X1")
print("Y → X2")

def visualize_probability_tables():
    """Print the conditional probability tables in text form."""
    # Print the tables in text form
    print("\nConditional Probability Table for X1 | Y:")
    print("----------------------------------------")
    print("|       | Y = 0     | Y = 1     |")
    print("|-------|-----------|-----------|")
    print(f"| X1 = 0 | {1-p_x1_given_y0:.3f}     | {1-p_x1_given_y1:.3f}     |")
    print(f"| X1 = 1 | {p_x1_given_y0:.3f}     | {p_x1_given_y1:.3f}     |")
    print("----------------------------------------")
    
    print("\nConditional Probability Table for X2 | Y:")
    print("----------------------------------------")
    print("|       | Y = 0     | Y = 1     |")
    print("|-------|-----------|-----------|")
    print(f"| X2 = 0 | {1-p_x2_given_y0:.3f}     | {1-p_x2_given_y1:.3f}     |")
    print(f"| X2 = 1 | {p_x2_given_y0:.3f}     | {p_x2_given_y1:.3f}     |")
    print("----------------------------------------")
    
    # Print the explanations for the tables
    print("\nThese tables show the conditional probabilities for each feature given the class:")
    print(f"- P(X1=0|Y=0) = {1-p_x1_given_y0:.3f}, P(X1=0|Y=1) = {1-p_x1_given_y1:.3f}")
    print(f"- P(X1=1|Y=0) = {p_x1_given_y0:.3f}, P(X1=1|Y=1) = {p_x1_given_y1:.3f}")
    print(f"- P(X2=0|Y=0) = {1-p_x2_given_y0:.3f}, P(X2=0|Y=1) = {1-p_x2_given_y1:.3f}")
    print(f"- P(X2=1|Y=0) = {p_x2_given_y0:.3f}, P(X2=1|Y=1) = {p_x2_given_y1:.3f}")
    
    print("\nThese conditional probabilities are used to calculate the likelihood of each sample.")

# Create the conditional probability table visualizations
visualize_probability_tables()

print_substep("Calculating P(X1,X2,Y) for sample {1,0,1}")
print("For sample {1,0,1}, we have X1=1, X2=0, Y=1")
print("P(X1=1,X2=0,Y=1) = P(Y=1) * P(X1=1|Y=1) * P(X2=0|Y=1)")
print(f"                 = w_1 * {p_x1_given_y1} * (1 - {p_x2_given_y1})")
print(f"                 = w_1 * {p_x1_given_y1} * {1 - p_x2_given_y1}")
print(f"                 = w_1 * {p_x1_given_y1 * (1 - p_x2_given_y1)}")

p_sample1 = f"w_1 * {p_x1_given_y1} * {1 - p_x2_given_y1}"
p_sample1_value = f"w_1 * {p_x1_given_y1 * (1 - p_x2_given_y1)}"

print_substep("Calculating P(X1,X2,Y) for sample {0,1,0}")
print("For sample {0,1,0}, we have X1=0, X2=1, Y=0")
print("P(X1=0,X2=1,Y=0) = P(Y=0) * P(X1=0|Y=0) * P(X2=1|Y=0)")
print(f"                 = (1 - w_1) * (1 - {p_x1_given_y0}) * {p_x2_given_y0}")
print(f"                 = (1 - w_1) * {1 - p_x1_given_y0} * {p_x2_given_y0}")
print(f"                 = (1 - w_1) * {(1 - p_x1_given_y0) * p_x2_given_y0}")

p_sample2 = f"(1 - w_1) * (1 - {p_x1_given_y0}) * {p_x2_given_y0}"
p_sample2_value = f"(1 - w_1) * {(1 - p_x1_given_y0) * p_x2_given_y0}"

p_sample1_numeric = p_x1_given_y1 * (1 - p_x2_given_y1)
p_sample2_numeric = (1 - p_x1_given_y0) * p_x2_given_y0

print(f"\nNumerically: P(X1=1,X2=0,Y=1) = w_1 * {p_sample1_numeric}")
print(f"Numerically: P(X1=0,X2=1,Y=0) = (1 - w_1) * {p_sample2_numeric}")

def visualize_sample_probabilities():
    """Create a visualization of the joint probabilities for the two samples."""
    # Print explanations and formulas
    print("\nSample Probabilities:")
    print("---------------------")
    print(f"For Sample 1 (X1=1,X2=0,Y=1): P(X1=1,X2=0,Y=1) = w_1 * {p_sample1_numeric:.5f}")
    print(f"For Sample 2 (X1=0,X2=1,Y=0): P(X1=0,X2=1,Y=0) = (1-w_1) * {p_sample2_numeric:.5f}")
    print(f"Joint probability: P(Sample 1) * P(Sample 2) = {p_sample1_numeric:.5f} * w_1 * {p_sample2_numeric:.5f} * (1-w_1)")
    print(f"Target joint probability: 1/180 = {1/180:.5f}")
    
    w1_values = np.linspace(0, 1, 100)
    sample1_probs = p_sample1_numeric * w1_values
    sample2_probs = p_sample2_numeric * (1 - w1_values)
    joint_probs = sample1_probs * sample2_probs
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot for individual sample probabilities
    ax1.plot(w1_values, sample1_probs, 'b-', linewidth=2, label='P(Sample 1)')
    ax1.plot(w1_values, sample2_probs, 'r-', linewidth=2, label='P(Sample 2)')
    
    # Add grid and styling
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('w₁ = P(Y=1)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Individual Sample Probabilities', fontsize=14)
    ax1.legend()
    
    # Plot for joint probability
    ax2.plot(w1_values, joint_probs, 'g-', linewidth=2)
    target_prob = 1/180
    ax2.axhline(y=target_prob, color='r', linestyle='--', label='Target: 1/180')
    
    # Find where joint probability equals target
    solution_idx = np.argmin(np.abs(joint_probs - target_prob))
    solution_w1 = w1_values[solution_idx]
    
    # Mark the solution point
    ax2.plot(solution_w1, joint_probs[solution_idx], 'ro', markersize=8)
    
    # Add grid and styling
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('w₁ = P(Y=1)', fontsize=12)
    ax2.set_ylabel('Joint Probability', fontsize=12)
    ax2.set_title('Joint Probability vs w₁', fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    filename = os.path.join(save_dir, "sample_probabilities.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample probabilities visualization saved to: {filename}")
    
    # Print the identified solution
    print(f"\nFrom the graph, we can see that the joint probability equals 1/180 when w₁ ≈ {solution_w1:.5f}")
    print("This is our initial estimate for the solution, which we'll verify with the quadratic equation.")

# Create the sample probabilities visualization
visualize_sample_probabilities()

# ==============================
# STEP 3: Setting up the Equation
# ==============================
print_step_header(3, "Setting up the Equation")

print_substep("The Likelihood Constraint")
print("We're told that the likelihood of the two samples is 1/180.")
print("This means: P(X1=1,X2=0,Y=1) * P(X1=0,X2=1,Y=0) = 1/180")

print(f"\nSubstituting our expressions:")
print(f"[w_1 * {p_sample1_numeric}] * [(1 - w_1) * {p_sample2_numeric}] = 1/180")

# Create a symbolic variable for w_1
w1 = sp.Symbol('w_1')

# Define the equation
lhs = w1 * p_sample1_numeric * (1 - w1) * p_sample2_numeric
rhs = 1/180

equation = sp.Eq(lhs, rhs)
print(f"\nEquation: {equation}")

# Expand the equation
expanded_lhs = sp.expand(lhs)
expanded_equation = sp.Eq(expanded_lhs, rhs)
print(f"\nExpanded equation: {expanded_equation}")

# ==============================
# STEP 4: Solving for w_1
# ==============================
print_step_header(4, "Solving for w_1")

print_substep("Rearranging the Equation")
print("We rearrange the equation into standard form: ax^2 + bx + c = 0")

# Rearrange into standard form
quadratic_eq = sp.expand(lhs - rhs)
print(f"Quadratic equation: {quadratic_eq} = 0")

# Extract coefficients
poly = sp.Poly(quadratic_eq, w1)
coeffs = poly.all_coeffs()
a, b, c = coeffs[0], coeffs[1], coeffs[2]

print(f"\nCoefficients: a = {a}, b = {b}, c = {c}")

# Solve the quadratic equation
solutions = sp.solve(quadratic_eq, w1)
w1_value = float(solutions[0]) if solutions[0].is_real else float(solutions[1])

print(f"\nSolutions: {solutions}")
print(f"The value of w_1 = P(Y=1) is {w1_value}")
print(f"As a fraction: w_1 = {sp.simplify(solutions[0])}")

print("\nLikelihood Function Explanation:")
print("The likelihood function plotted against different values of w_1 would show a parabolic shape")
print(f"with two roots. Only one (w_1 = {sp.simplify(solutions[0])}) is our solution because:")
print("1. The parabolic shape is characteristic of the product of two linear functions in w_1 and (1-w_1)")
print("2. This is exactly what we get in the Naive Bayes model when multiplying the probabilities of two different samples")

def visualize_quadratic_solution():
    """Create a visualization of the quadratic equation and its solutions."""
    # Print the quadratic equation and solutions
    print("\nQuadratic Equation:")
    print("------------------")
    print(f"Original equation: {quadratic_eq} = 0")
    print(f"Simplified form: w₁² - w₁ + 0.222 = 0")
    print("\nUsing the quadratic formula:")
    print("w₁ = (1 ± √(1 - 4(0.222))) / 2")
    print("w₁ = (1 ± √0.112) / 2")
    print("w₁ = (1 ± 0.333) / 2")
    print("w₁ = 0.667 or w₁ = 0.333")
    print("\nSince w₁ represents a probability, both solutions are in the valid range [0,1].")
    print("Verifying with original constraint: P(Sample 1) * P(Sample 2) = 1/180")
    print(f"For w₁ = 1/3: {p_sample1_numeric:.5f} * (1/3) * {p_sample2_numeric:.5f} * (2/3) = {p_sample1_numeric * (1/3) * p_sample2_numeric * (2/3):.8f}")
    print(f"Expected: 1/180 = {1/180:.8f}")
    print(f"\nTherefore, w₁ = 1/3 is our solution.")
    
    # Convert symbolic equation to numeric function for plotting
    w1_sym = sp.Symbol('w_1')
    expr = sp.lambdify(w1_sym, quadratic_eq, 'numpy')
    
    # Create array of w1 values for plotting
    w1_values = np.linspace(0, 1, 1000)
    y_values = expr(w1_values)
    
    # Extract numeric solutions
    solutions_numeric = [float(sol) for sol in solutions]
    valid_solutions = [sol for sol in solutions_numeric if 0 <= sol <= 1]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(w1_values, y_values, 'b-', linewidth=2, label='Quadratic Equation')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Mark the roots
    for i, sol in enumerate(solutions_numeric):
        if 0 <= sol <= 1:  # Only mark valid solutions
            plt.plot(sol, 0, 'ro', markersize=8)
    
    # Add shaded region for valid probability range
    plt.axvspan(0, 1, alpha=0.2, color='g', label='Valid Probability Range')
    
    # Add grid and styling
    plt.grid(True, alpha=0.3)
    plt.xlabel('w₁ = P(Y=1)', fontsize=12)
    plt.ylabel('Quadratic Expression Value', fontsize=12)
    plt.title('Quadratic Equation Solutions', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(save_dir, "quadratic_solution.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create another visualization focusing on our solution
    plt.figure(figsize=(10, 6))
    
    # Convert expression to more readable form for likelihood
    likelihood_expr = w1 * p_sample1_numeric * (1 - w1) * p_sample2_numeric
    likelihood_func = sp.lambdify(w1_sym, likelihood_expr, 'numpy')
    likelihood_values = likelihood_func(w1_values)
    
    # Plot likelihood function
    plt.plot(w1_values, likelihood_values, 'g-', linewidth=2, label='Likelihood Function')
    
    # Mark the target likelihood
    target_likelihood = 1/180
    plt.axhline(y=target_likelihood, color='r', linestyle='--', 
                label=f'Target Likelihood = 1/180')
    
    # Find intersection points
    for sol in valid_solutions:
        plt.plot(sol, target_likelihood, 'ro', markersize=8)
    
    # Add grid and styling
    plt.grid(True, alpha=0.3)
    plt.xlabel('w₁ = P(Y=1)', fontsize=12)
    plt.ylabel('Likelihood', fontsize=12)
    plt.title('Likelihood Function', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    filename2 = os.path.join(save_dir, "likelihood_function.png")
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Quadratic equation and likelihood visualizations saved to: {filename} and {filename2}")

# Create the quadratic solution visualization
visualize_quadratic_solution()

# ==============================
# STEP 5: Verification
# ==============================
print_step_header(5, "Verification")

print_substep("Verifying our Solution")

# Calculate the likelihood with our solution
likelihood = eval(str(lhs).replace('w_1', str(w1_value)))
print(f"Likelihood with w_1 = {w1_value}: {likelihood}")
print(f"Expected likelihood: {rhs}")
print(f"Difference: {abs(likelihood - rhs)}")

if abs(likelihood - rhs) < 1e-10:
    print("\nVerification successful!")
else:
    print("\nVerification failed.")

# ==============================
# STEP 6: Key Insights
# ==============================
print_step_header(6, "Key Insights")

print("Theoretical Foundations:")
print("- Naive Bayes Assumption: Features are conditionally independent given the class.")
print("  This simplifies the joint probability calculations significantly.")
print("- Prior Probability: The parameter w_1 = P(Y=1) represents our belief about")
print("  the class distribution before seeing any features.")
print("- Likelihood Constraint: When given a constraint on the joint probability of multiple")
print("  samples, this leads to an equation that can be solved for the unknown parameter.")

print("\nMathematical Techniques:")
print("- Quadratic Equation: The product of probabilities for two different classes")
print("  (involving w_1 and (1-w_1)) leads to a quadratic equation.")
print("- Solution Validation: When solving quadratic equations in the context of probabilities,")
print("  we must ensure that the solution lies in the range [0,1] and verify it matches our constraints.")

print("\nPractical Applications:")
print("- Parameter Estimation: In practice, we often estimate the prior probabilities from training")
print("  data frequencies, but this problem demonstrates how to determine them from other constraints.")
print("- Model Completeness: A Naive Bayes model requires both the conditional probabilities P(X|Y)")
print("  and the prior probabilities P(Y) to make predictions.")

# ==============================
# STEP 7: Summary
# ==============================
print_step_header(7, "Summary")

print("We solved for w_1 = P(Y=1) in a Naive Bayes model.")
print(f"The equation we derived is: {expanded_equation}")
print(f"This simplifies to the quadratic equation: {quadratic_eq} = 0")
print(f"The value of w_1 is: {w1_value:.6f}")
print(f"This can be expressed as the fraction: {sp.simplify(solutions[0])}")
print("\nKey insights:")
print("1. The joint probability of the two samples creates a quadratic equation in w_1")
print("2. The solution lies in the range [0,1] as expected for a probability")
print("3. The fraction form of the solution is more precise than the decimal approximation")
print(f"\nConclusion: We determined that the prior probability of class 1 is w_1 = P(Y=1) = {sp.simplify(solutions[0])}.")
print("This was obtained by setting up joint probabilities for the given samples using the Naive Bayes model,")
print("then solving a quadratic equation based on the constraint that the likelihood equals 1/180.")
print("\nThe solution process demonstrates how the Naive Bayes assumption of conditional independence allows us")
print("to factorize complex joint probabilities into simpler terms, making the model both computationally")
print("efficient and mathematically tractable.")
print(f"\nThe fraction {sp.simplify(solutions[0])} represents the prior probability of class 1, meaning that before")
print("considering any features, we believe there's a one-third chance that a randomly selected sample")
print("belongs to class 1.") 

# ==============================
# STEP 8: List of Generated Images
# ==============================
print_step_header(8, "List of Generated Images")

image_files = [f for f in os.listdir(save_dir) if f.endswith('.png')]

print(f"Generated {len(image_files)} images in {save_dir}:")
for i, img in enumerate(sorted(image_files), 1):
    print(f"{i}. {img}")

print("\nYou can use these images in the markdown explanation file.")
print("Images are referenced in markdown as: ![Description](../Images/L2_5_Quiz_17/image_filename.png)") 