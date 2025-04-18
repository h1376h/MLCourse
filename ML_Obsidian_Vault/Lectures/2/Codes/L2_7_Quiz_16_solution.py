import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_16")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- Three classifiers (C₁, C₂, C₃) with posterior probabilities:")
print("  P(C₁|D) = 0.4, P(C₂|D) = 0.35, P(C₃|D) = 0.25")
print("- For a new patient with symptoms x, the models predict:")
print("  P(disease|x, C₁) = 0.75")
print("  P(disease|x, C₂) = 0.65")
print("  P(disease|x, C₃) = 0.85")
print("- Treatment threshold is 0.7 (treat if probability > 0.7)")
print("\nTask:")
print("1. Calculate the model-averaged probability using Bayesian Model Averaging (BMA)")
print("2. Determine the treatment decision using BMA")
print("3. Select the model using MAP and determine the treatment decision")
print("4. Explain advantage and disadvantage of BMA in this context")

# Step 2: Calculating the Model-Averaged Probability
print_step_header(2, "Calculating the Model-Averaged Probability")

# Model posterior probabilities
p_m1 = 0.4   # P(C₁|D)
p_m2 = 0.35  # P(C₂|D)
p_m3 = 0.25  # P(C₃|D)

# Check that probabilities sum to 1
assert abs(p_m1 + p_m2 + p_m3 - 1.0) < 1e-10, "Model probabilities must sum to 1"

# Disease probabilities for each model
p_disease_m1 = 0.75  # P(disease|x, C₁)
p_disease_m2 = 0.65  # P(disease|x, C₂)
p_disease_m3 = 0.85  # P(disease|x, C₃)

# Calculate model-averaged probability using Bayesian Model Averaging
p_disease_bma = p_m1 * p_disease_m1 + p_m2 * p_disease_m2 + p_m3 * p_disease_m3

print("Using Bayesian Model Averaging (BMA), we calculate the weighted average of predictions:")
print("P(disease|x) = ∑ P(disease|x, Cₖ) × P(Cₖ|D)")
print(f"P(disease|x) = {p_m1} × {p_disease_m1} + {p_m2} × {p_disease_m2} + {p_m3} × {p_disease_m3}")
print(f"P(disease|x) = {p_m1 * p_disease_m1:.4f} + {p_m2 * p_disease_m2:.4f} + {p_m3 * p_disease_m3:.4f}")
print(f"P(disease|x) = {p_disease_bma:.4f}")

# Create a visualization of the model-averaged probability
fig, ax = plt.subplots(figsize=(10, 6))

models = ['C₁', 'C₂', 'C₃', 'BMA']
probs = [p_disease_m1, p_disease_m2, p_disease_m3, p_disease_bma]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

alphas = [p_m1, p_m2, p_m3, 1.0]  # Alpha based on model posterior

# Plot individual model predictions with alpha based on posterior
for i in range(3):
    ax.bar(i, probs[i], color=colors[i], alpha=alphas[i], 
           label=f'{models[i]}: {probs[i]:.2f} (weight: {alphas[i]:.2f})')

# Plot BMA prediction
ax.bar(3, probs[3], color=colors[3], 
       label=f'BMA: {probs[3]:.2f}')

# Add horizontal line for treatment threshold
ax.axhline(y=0.7, color='black', linestyle='--', 
           label='Treatment Threshold = 0.7')

# Fill the area to show which models are above threshold
for i, prob in enumerate(probs):
    if prob > 0.7:
        ax.bar(i, 0.7, color='lightgray', alpha=0.5, width=0.8)

ax.set_ylabel('Probability of Disease', fontsize=12)
ax.set_title('Disease Probability Predictions by Different Models', fontsize=14)
ax.set_ylim(0, 1)
ax.set_xticks(range(4))
ax.set_xticklabels(models)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "model_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 3: Determining the Treatment Decision Using BMA
print_step_header(3, "Determining the Treatment Decision Using BMA")

treatment_threshold = 0.7

bma_decision = "Treat" if p_disease_bma > treatment_threshold else "Don't treat"

print(f"Using Bayesian Model Averaging:")
print(f"P(disease|x) = {p_disease_bma:.4f}")
print(f"Treatment threshold = {treatment_threshold}")
print(f"Decision: {bma_decision}")

if p_disease_bma > treatment_threshold:
    print("Since the BMA probability is above the threshold, the patient should be treated.")
else:
    print("Since the BMA probability is below the threshold, the patient should not be treated.")

# Create a visualization of the treatment decision
fig, ax = plt.subplots(figsize=(8, 6))

# Create a horizontal bar chart of the BMA probability
ax.barh("BMA", p_disease_bma, color='#9b59b6', alpha=0.7)

# Add vertical line for treatment threshold
ax.axvline(x=0.7, color='red', linestyle='--', 
           label=f'Treatment Threshold = {treatment_threshold}')

# Highlight the decision
if p_disease_bma > treatment_threshold:
    decision_color = 'green'
    x_text = min(p_disease_bma + 0.05, 0.95)
else:
    decision_color = 'red'
    x_text = max(p_disease_bma - 0.05, 0.05)

ax.text(x_text, 0, f"Decision: {bma_decision}", 
        fontsize=12, color=decision_color, 
        verticalalignment='center')

# Fill the region based on the decision
if p_disease_bma > treatment_threshold:
    ax.axvspan(0.7, 1.0, alpha=0.2, color='green', label='Treat Region')
else:
    ax.axvspan(0, 0.7, alpha=0.2, color='red', label='Don\'t Treat Region')

ax.set_xlim(0, 1)
ax.set_xlabel('Probability of Disease', fontsize=12)
ax.set_title('Treatment Decision Based on Bayesian Model Averaging', fontsize=14)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bma_decision.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 4: Selecting a Single Model Using MAP
print_step_header(4, "Selecting a Single Model Using MAP")

# Find the model with the highest posterior probability
model_posteriors = [p_m1, p_m2, p_m3]
model_names = ['C₁', 'C₂', 'C₃']
disease_probs = [p_disease_m1, p_disease_m2, p_disease_m3]

map_model_index = np.argmax(model_posteriors)
map_model = model_names[map_model_index]
map_prob = disease_probs[map_model_index]

map_decision = "Treat" if map_prob > treatment_threshold else "Don't treat"

print(f"Using Maximum A Posteriori (MAP) model selection:")
print(f"The model with the highest posterior probability is {map_model} with P({map_model}|D) = {model_posteriors[map_model_index]:.4f}")
print(f"P(disease|x, {map_model}) = {map_prob:.4f}")
print(f"Treatment threshold = {treatment_threshold}")
print(f"Decision: {map_decision}")

if map_prob > treatment_threshold:
    print(f"Since the {map_model} probability is above the threshold, the patient should be treated.")
else:
    print(f"Since the {map_model} probability is below the threshold, the patient should not be treated.")

# Create a visualization of the MAP model selection
fig, ax = plt.subplots(figsize=(10, 6))

# Create a bar chart of the model posteriors
bars = ax.bar(model_names, model_posteriors, alpha=0.7, 
             color=['#3498db', '#2ecc71', '#e74c3c'])

# Highlight the MAP model
bars[map_model_index].set_color('gold')
bars[map_model_index].set_alpha(1.0)

# Add a line for each model's disease probability
for i, (name, prob) in enumerate(zip(model_names, disease_probs)):
    ax.plot([i-0.4, i+0.4], [prob, prob], 'k-', linewidth=2)
    ax.text(i, prob + 0.03, f"P(disease) = {prob:.2f}", 
           ha='center', va='bottom', fontsize=10)

# Add horizontal line for treatment threshold
ax.axhline(y=0.7, color='red', linestyle='--', 
           label='Treatment Threshold = 0.7')

ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Model Posterior Probabilities and Disease Predictions', fontsize=14)
ax.set_ylim(0, 1)
ax.legend([f'MAP Model: {map_model}', 'Treatment Threshold'], 
         loc='upper center', bbox_to_anchor=(0.5, -0.15))

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "map_selection.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 5: Comparing BMA and MAP Decisions
print_step_header(5, "Comparing BMA and MAP Decisions")

print(f"BMA probability: {p_disease_bma:.4f} → Decision: {bma_decision}")
print(f"MAP model ({map_model}) probability: {map_prob:.4f} → Decision: {map_decision}")

if bma_decision == map_decision:
    print("\nBoth approaches lead to the same treatment decision.")
else:
    print("\nThe approaches lead to different treatment decisions!")
    print("This highlights how model selection can impact clinical decisions.")

# Create a visualization comparing both approaches
fig, ax = plt.subplots(figsize=(10, 6))

# Create bar chart comparing the probabilities
approaches = ['MAP Model\n(C₁)', 'BMA\n(Weighted Average)']
probs = [map_prob, p_disease_bma]
colors = ['gold', '#9b59b6']

bars = ax.bar(approaches, probs, alpha=0.7, color=colors)

# Add horizontal line for treatment threshold
ax.axhline(y=0.7, color='red', linestyle='--', 
           label='Treatment Threshold = 0.7')

# Fill the regions based on decisions
for i, prob in enumerate(probs):
    if prob > 0.7:
        ax.text(i, prob + 0.03, "Treat", ha='center', color='green', fontweight='bold')
        ax.bar(i, 0.7, color='lightgreen', alpha=0.3, width=0.8)
    else:
        ax.text(i, prob + 0.03, "Don't Treat", ha='center', color='red', fontweight='bold')
        
ax.set_ylabel('Probability of Disease', fontsize=12)
ax.set_title('Comparison of MAP and BMA Approaches to Treatment Decision', fontsize=14)
ax.set_ylim(0, 1)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))

# Add model weight information
model_weights_text = (f"Model Weights:\n"
                      f"C₁: {p_m1:.2f} (P(disease) = {p_disease_m1:.2f})\n"
                      f"C₂: {p_m2:.2f} (P(disease) = {p_disease_m2:.2f})\n"
                      f"C₃: {p_m3:.2f} (P(disease) = {p_disease_m3:.2f})")
                      
plt.figtext(0.02, 0.02, model_weights_text, fontsize=9)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bma_vs_map.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 6: Advantages and Disadvantages of BMA in Medical Context
print_step_header(6, "Advantages and Disadvantages of BMA in Medical Context")

print("Advantages of Bayesian Model Averaging in a medical context:")
print("1. Accounts for model uncertainty: Incorporates all models weighted by their posterior probabilities")
print("2. Reduces the risk of selecting a single incorrect model")
print("3. Can provide more robust predictions, especially when no single model is clearly superior")
print("4. Reflects the full uncertainty in the diagnosis, which is important for medical decision-making")
print("5. Allows incorporation of predictions from multiple diagnostic tools or expert opinions")

print("\nDisadvantages of Bayesian Model Averaging in a medical context:")
print("1. More complex to implement and explain to medical practitioners")
print("2. May lead to 'middle ground' predictions that aren't actionable (e.g., close to threshold)")
print("3. If one model is clearly correct, averaging can dilute its accurate prediction")
print("4. Computational complexity increases with the number of models")
print("5. Requires reliable estimates of model posterior probabilities, which may be difficult to obtain")

print("\nIn this specific case:")
if bma_decision == map_decision:
    print(f"Both BMA and MAP approaches lead to the same decision: {bma_decision}")
    print("However, the probability estimates differ, which could matter in borderline cases")
else:
    print("The BMA and MAP approaches lead to different decisions")
    print(f"BMA: {bma_decision} (P(disease) = {p_disease_bma:.4f})")
    print(f"MAP: {map_decision} (P(disease) = {map_prob:.4f})")
    print("This demonstrates how model selection can significantly impact clinical decisions")

# Create a final summary visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Setup for radar chart
attributes = ['Accounts for\nmodel uncertainty', 
              'Simplicity', 
              'Clinical\ninterpretability', 
              'Computational\nefficiency',
              'Predictive\nperformance']
num_attrs = len(attributes)

# Angle of each axis
angles = np.linspace(0, 2*np.pi, num_attrs, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Scores (subjective ratings from 0-10)
# [uncertainty, simplicity, interpretability, efficiency, performance]
bma_scores = [9, 5, 6, 6, 8]  
map_scores = [3, 9, 8, 9, 6]
bma_scores += bma_scores[:1]  # Close the loop
map_scores += map_scores[:1]  # Close the loop

# Plot radar chart
ax = plt.subplot(111, polar=True)
ax.fill(angles, bma_scores, color='#9b59b6', alpha=0.25)
ax.plot(angles, bma_scores, 'o-', linewidth=2, color='#9b59b6', label='BMA')
ax.fill(angles, map_scores, color='gold', alpha=0.25)
ax.plot(angles, map_scores, 'o-', linewidth=2, color='gold', label='MAP')

# Fix axis to go in the right order and start at 12 o'clock
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Set labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(attributes)

# Y axis limits
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(['2', '4', '6', '8', '10'])

ax.set_title('Comparison of BMA and MAP Approaches', fontsize=15)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bma_vs_map_radar.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}") 