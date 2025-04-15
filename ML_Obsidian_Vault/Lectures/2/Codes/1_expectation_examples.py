import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

print("\n=== EXPECTATION EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Discrete Random Variable Expectation
print("Example 1: Discrete Random Variable")
x = np.array([1, 2, 3, 4])
probs = np.array([0.2, 0.3, 0.4, 0.1])
print(f"Values (x): {x}")
print(f"Probabilities P(X=x): {probs}")

# Calculation steps
print("\nStep-by-step calculation:")
expected_value = 0
for i in range(len(x)):
    term = x[i] * probs[i]
    expected_value += term
    print(f"  {x[i]} × {probs[i]:.1f} = {term:.1f}")

print(f"Sum = {expected_value}")
print(f"Therefore, E[X] = {expected_value}")

# Create bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(x, probs, color='skyblue', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('Value of X', fontsize=12)
plt.ylabel('Probability P(X=x)', fontsize=12)
plt.title('Discrete Random Variable Distribution', fontsize=14)
plt.xticks(x)
plt.ylim(0, 0.5)

# Add probability values on top of each bar
for i, prob in enumerate(probs):
    plt.text(x[i], prob + 0.02, f'{prob:.1f}', ha='center', fontsize=10)

# Add expectation line
plt.axvline(x=expected_value, color='red', linestyle='--', label=f'E[X] = {expected_value:.1f}')
plt.legend()

# Add annotation for the calculation
calculation_text = '$E[X] = '
for i in range(len(x)):
    if i > 0:
        calculation_text += " + "
    calculation_text += f'{x[i]} \\times {probs[i]:.1f}'
calculation_text += f' = {expected_value:.1f}$'

plt.annotate(calculation_text, 
            xy=(2.5, 0.45),
            xytext=(2.5, 0.45),
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'discrete_expectation.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Dice Roll Expected Value
print("\n\nExample 2: Dice Roll Expected Value")
x = np.arange(1, 7)  # Possible outcomes (1 to 6)
probs = np.ones(6) / 6  # Equal probability for each outcome
print(f"Values (x): {x}")
print(f"Probabilities P(X=x): {[f'{p:.3f}' for p in probs]}")

# Calculation steps
print("\nStep-by-step calculation for E[X]:")
expected_value = 0
for i in range(len(x)):
    term = x[i] * probs[i]
    expected_value += term
    print(f"  {x[i]} × {probs[i]:.3f} = {term:.3f}")

print(f"Sum = {expected_value} = 21/6")
print(f"Therefore, E[X] = {expected_value}")

# Calculate E[X^2]
print("\nStep-by-step calculation for E[X²]:")
expected_square = 0
for i in range(len(x)):
    term = (x[i]**2) * probs[i]
    expected_square += term
    print(f"  {x[i]}² × {probs[i]:.3f} = {x[i]**2} × {probs[i]:.3f} = {term:.3f}")

print(f"Sum = {expected_square:.3f} = 91/6")
print(f"Therefore, E[X²] = {expected_square:.3f}")

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(x, probs, color='skyblue', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('Dice Face', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Dice Roll Distribution with Expected Value', fontsize=14)
plt.xticks(x)
plt.ylim(0, 0.2)

# Add probability values on top of each bar
for i, prob in enumerate(probs):
    plt.text(i + 1, prob + 0.01, f'{prob:.3f}', ha='center', fontsize=9)

# Add expectation line
plt.axvline(x=expected_value, color='red', linestyle='--', label=f'E[X] = {expected_value:.1f}')
plt.legend()

# Add annotations for the calculations
plt.annotate(f'$E[X] = \\frac{{1+2+3+4+5+6}}{{6}} = \\frac{{21}}{{6}} = {expected_value:.1f}$', 
            xy=(3.5, 0.18),
            xytext=(3.5, 0.18),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.annotate(f'$E[X^2] = \\frac{{1^2+2^2+3^2+4^2+5^2+6^2}}{{6}} = \\frac{{91}}{{6}} \\approx {expected_square:.2f}$', 
            xy=(3.5, 0.15),
            xytext=(3.5, 0.15),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'dice_expectation.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Binomial Expected Value
print("\n\nExample 3: Binomial Expected Value")
n = 10  # Number of trials
p = 0.7  # Probability of success
x = np.arange(0, n+1)  # Possible outcomes (0 to 10)
probs = stats.binom.pmf(x, n, p)
print(f"Number of trials (n): {n}")
print(f"Probability of success (p): {p}")

# Calculation steps
print("\nFor a binomial distribution, E[X] = n × p")
expected_value = n * p
print(f"E[X] = {n} × {p} = {expected_value}")

# Print the probability mass function
print("\nProbability mass function values:")
for i, prob in enumerate(probs):
    if prob > 0.01:
        print(f"  P(X={i}) = {prob:.4f}")

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(x, probs, color='skyblue', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('Number of Successful Free Throws', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Binomial Distribution: n=10, p=0.7', fontsize=14)
plt.xticks(x)
plt.ylim(0, 0.3)

# Add probability values on top of each bar
for i, prob in enumerate(probs):
    if prob > 0.02:  # Only add text for larger bars
        plt.text(i, prob + 0.01, f'{prob:.3f}', ha='center', fontsize=8)

# Add expectation line
plt.axvline(x=expected_value, color='red', linestyle='--', label=f'E[X] = np = {expected_value:.1f}')
plt.legend()

# Add annotation for the expectation formula
plt.annotate(f'$E[X] = n \\times p = 10 \\times 0.7 = 7$', 
            xy=(5, 0.28),
            xytext=(5, 0.28),
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'binomial_expectation.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Investment Return Expected Value
print("\n\nExample 4: Investment Return")
x = np.array([-1000, 0, 2000, 5000])
probs = np.array([0.2, 0.3, 0.4, 0.1])
labels = ['Loss', 'Break-even', 'Profit', 'High Profit']
print(f"Possible returns: {x}")
print(f"Probabilities: {probs}")

# Calculation steps
print("\nStep-by-step calculation:")
expected_value = 0
for i in range(len(x)):
    term = x[i] * probs[i]
    expected_value += term
    print(f"  {x[i]} × {probs[i]:.1f} = {term}")

print(f"Sum = {expected_value}")
print(f"Therefore, the expected return is ${expected_value}")

# Create bar plot
plt.figure(figsize=(10, 6))
colors = ['red', 'yellow', 'green', 'darkgreen']
plt.bar(labels, probs, color=colors, alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('Investment Outcome', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Investment Return Distribution', fontsize=14)
plt.ylim(0, 0.5)

# Add probability values on top of each bar
for i, prob in enumerate(probs):
    plt.text(i, prob + 0.02, f'{prob:.1f}', ha='center', fontsize=10)

# Add expected value annotation
calculation_text = '$E[X] = '
for i in range(len(x)):
    if i > 0:
        calculation_text += " + "
    if x[i] < 0:
        calculation_text += f'({x[i]}) \\times {probs[i]:.1f}'
    else:
        calculation_text += f'{x[i]} \\times {probs[i]:.1f}'
calculation_text += f' = {expected_value:.0f}$'

plt.annotate(calculation_text, 
            xy=(1.5, 0.45),
            xytext=(1.5, 0.45),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Add "Expected Return" callout
plt.annotate(f'Expected Return = ${expected_value:.0f}', 
            xy=(1.5, 0.35),
            xytext=(1.5, 0.35),
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'investment_expectation.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Marketing Strategy Comparison
print("\n\nExample 5: Marketing Strategy Comparison")
strategies = ['Strategy A', 'Strategy B']
outcomes_A = ['Profit', 'Loss']
outcomes_B = ['Profit', 'Loss']
probs_A = [0.6, 0.4]
probs_B = [0.8, 0.2]
values_A = [10000, -2000]
values_B = [5000, -1000]
print("Strategy A:")
print(f"  Profit: ${values_A[0]} with probability {probs_A[0]}")
print(f"  Loss: ${values_A[1]} with probability {probs_A[1]}")
print("Strategy B:")
print(f"  Profit: ${values_B[0]} with probability {probs_B[0]}")
print(f"  Loss: ${values_B[1]} with probability {probs_B[1]}")

# Calculation steps
print("\nStep-by-step calculation for Strategy A:")
expected_A = 0
for i in range(len(values_A)):
    term = values_A[i] * probs_A[i]
    expected_A += term
    print(f"  {values_A[i]} × {probs_A[i]} = {term}")

print(f"Sum = {expected_A}")
print(f"Therefore, E[A] = ${expected_A}")

print("\nStep-by-step calculation for Strategy B:")
expected_B = 0
for i in range(len(values_B)):
    term = values_B[i] * probs_B[i]
    expected_B += term
    print(f"  {values_B[i]} × {probs_B[i]} = {term}")

print(f"Sum = {expected_B}")
print(f"Therefore, E[B] = ${expected_B}")

if expected_A > expected_B:
    print(f"\nStrategy A has a higher expected value by ${expected_A - expected_B}")
else:
    print(f"\nStrategy B has a higher expected value by ${expected_B - expected_A}")

# Create subfigures
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
colors = ['green', 'red']
plt.bar(outcomes_A, probs_A, color=colors, alpha=0.7)
plt.grid(True, alpha=0.3)
plt.title('Strategy A', fontsize=14)
plt.ylabel('Probability', fontsize=12)
plt.ylim(0, 1.0)

# Add value and probability on each bar
for i, (outcome, prob, value) in enumerate(zip(outcomes_A, probs_A, values_A)):
    plt.text(i, prob + 0.05, f'{prob:.1f}', ha='center', fontsize=10)
    if value > 0:
        plt.text(i, prob/2, f'${value:,}', ha='center', color='white', fontsize=10)
    else:
        plt.text(i, prob/2, f'-${abs(value):,}', ha='center', color='white', fontsize=10)

plt.subplot(1, 2, 2)
plt.bar(outcomes_B, probs_B, color=colors, alpha=0.7)
plt.grid(True, alpha=0.3)
plt.title('Strategy B', fontsize=14)
plt.ylim(0, 1.0)

# Add value and probability on each bar
for i, (outcome, prob, value) in enumerate(zip(outcomes_B, probs_B, values_B)):
    plt.text(i, prob + 0.05, f'{prob:.1f}', ha='center', fontsize=10)
    if value > 0:
        plt.text(i, prob/2, f'${value:,}', ha='center', color='white', fontsize=10)
    else:
        plt.text(i, prob/2, f'-${abs(value):,}', ha='center', color='white', fontsize=10)

# Add expected values
plt.figtext(0.25, 0.01, f'Expected Value: ${expected_A:,.0f}', ha='center', fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
plt.figtext(0.75, 0.01, f'Expected Value: ${expected_B:,.0f}', ha='center', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

# Add comparison
if expected_A > expected_B:
    plt.figtext(0.5, 0.95, f'Strategy A has higher expected value by ${expected_A - expected_B:,.0f}', 
               ha='center', fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
else:
    plt.figtext(0.5, 0.95, f'Strategy B has higher expected value by ${expected_B - expected_A:,.0f}', 
               ha='center', fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(os.path.join(images_dir, 'strategy_comparison.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 6: Data Scientist Bonus - Expected Value
print("\n\nExample 6: Data Scientist Bonus")
bonus_values = [500, 1000, 2000, 5000]
probs = [0.15, 0.25, 0.40, 0.20]
labels = ['<85%', '85-90%', '90-95%', '≥95%']
print("Bonus structure:")
for i, (bonus, prob, label) in enumerate(zip(bonus_values, probs, labels)):
    print(f"  ${bonus} for accuracy {label} with probability {prob}")

# Calculation steps
print("\nStep-by-step calculation:")
expected_value = 0
for i in range(len(bonus_values)):
    term = bonus_values[i] * probs[i]
    expected_value += term
    print(f"  ${bonus_values[i]} × {probs[i]:.2f} = ${term:.2f}")

print(f"Sum = ${expected_value:.2f}")
print(f"Therefore, the expected bonus is ${expected_value:.2f}")

# Create bar plot
plt.figure(figsize=(10, 6))
color_gradient = plt.cm.YlGn(np.linspace(0.3, 0.9, len(bonus_values)))
plt.bar(labels, bonus_values, color=color_gradient, alpha=0.7, width=0.6)
plt.grid(True, alpha=0.3)
plt.xlabel('Model Accuracy', fontsize=12)
plt.ylabel('Bonus Amount ($)', fontsize=12)
plt.title('Data Scientist Bonus Structure', fontsize=14)

# Add probability values on top of each bar
for i, (value, prob) in enumerate(zip(bonus_values, probs)):
    plt.text(i, value + 200, f'p = {prob:.2f}', ha='center', fontsize=10)
    plt.text(i, value/2, f'${value:,}', ha='center', color='white', fontsize=11, weight='bold')

# Add expected value calculation
calculation_text = '$E[X] = '
for i in range(len(bonus_values)):
    if i > 0:
        calculation_text += " + "
    calculation_text += f'{bonus_values[i]} \\times {probs[i]:.2f}'
calculation_text += f' = {expected_value:.0f}$'

plt.annotate(calculation_text, 
            xy=(1.5, 5500),
            xytext=(1.5, 5500),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Add "Expected Bonus" callout
plt.annotate(f'Expected Bonus = ${expected_value:.0f}', 
            xy=(1.5, 6000),
            xytext=(1.5, 6000),
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'bonus_expectation.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll expectation example images created successfully.") 