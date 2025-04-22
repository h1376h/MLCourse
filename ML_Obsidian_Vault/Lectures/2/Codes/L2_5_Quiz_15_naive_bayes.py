import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Load and prepare the data
print_step_header(1, "Data Preparation")

# Create the dataset
data = pd.DataFrame({
    'Diabetes': ['Y', 'Y', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y'],
    'Smoke': ['N', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'N'],
    'Heart_Disease': ['Y', 'N', 'N', 'N', 'N', 'Y', 'N', 'Y', 'N', 'N']
})

print("Dataset:")
print(data)
print("\nShape:", data.shape)

# Step 2: Calculate prior probabilities
print_step_header(2, "Prior Probabilities")

p_h_yes = (data['Heart_Disease'] == 'Y').mean()
p_h_no = 1 - p_h_yes

print(f"P(H = Yes) = {p_h_yes:.3f}")
print(f"P(H = No) = {p_h_no:.3f}")

# Step 3: Calculate conditional probabilities
print_step_header(3, "Conditional Probabilities")

# For Diabetes given Heart Disease
p_d_yes_h_yes = len(data[(data['Diabetes'] == 'Y') & (data['Heart_Disease'] == 'Y')]) / len(data[data['Heart_Disease'] == 'Y'])
p_d_yes_h_no = len(data[(data['Diabetes'] == 'Y') & (data['Heart_Disease'] == 'N')]) / len(data[data['Heart_Disease'] == 'N'])

# For Smoking given Heart Disease
p_s_yes_h_yes = len(data[(data['Smoke'] == 'Y') & (data['Heart_Disease'] == 'Y')]) / len(data[data['Heart_Disease'] == 'Y'])
p_s_yes_h_no = len(data[(data['Smoke'] == 'Y') & (data['Heart_Disease'] == 'N')]) / len(data[data['Heart_Disease'] == 'N'])

print("Conditional Probabilities:")
print(f"P(D = Yes | H = Yes) = {p_d_yes_h_yes:.3f}")
print(f"P(D = Yes | H = No) = {p_d_yes_h_no:.3f}")
print(f"P(S = Yes | H = Yes) = {p_s_yes_h_yes:.3f}")
print(f"P(S = Yes | H = No) = {p_s_yes_h_no:.3f}")

# Step 4: Visualize the conditional probabilities
print_step_header(4, "Visualizing Conditional Probabilities")

# Set style for better visualization
plt.style.use('default')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot for Diabetes
diabetes_probs = pd.DataFrame({
    'Heart Disease': ['Yes', 'No'],
    'Has Diabetes': [p_d_yes_h_yes, p_d_yes_h_no],
    'No Diabetes': [1-p_d_yes_h_yes, 1-p_d_yes_h_no]
})

diabetes_probs.set_index('Heart Disease').plot(kind='bar', stacked=True, ax=ax1, 
                                             color=['#ff9999', '#66b3ff'])
ax1.set_title('Conditional Probabilities of Diabetes\nGiven Heart Disease', pad=20)
ax1.set_ylabel('Probability')
ax1.legend(title='Diabetes Status', bbox_to_anchor=(0.5, -0.15), loc='upper center')

# Add value labels on the bars
for c in ax1.containers:
    ax1.bar_label(c, fmt='%.3f', label_type='center')

# Plot for Smoking
smoking_probs = pd.DataFrame({
    'Heart Disease': ['Yes', 'No'],
    'Smokes': [p_s_yes_h_yes, p_s_yes_h_no],
    'Does Not Smoke': [1-p_s_yes_h_yes, 1-p_s_yes_h_no]
})

smoking_probs.set_index('Heart Disease').plot(kind='bar', stacked=True, ax=ax2,
                                            color=['#ff9999', '#66b3ff'])
ax2.set_title('Conditional Probabilities of Smoking\nGiven Heart Disease', pad=20)
ax2.set_ylabel('Probability')
ax2.legend(title='Smoking Status', bbox_to_anchor=(0.5, -0.15), loc='upper center')

# Add value labels on the bars
for c in ax2.containers:
    ax2.bar_label(c, fmt='%.3f', label_type='center')

plt.tight_layout()
file_path = os.path.join(save_dir, "conditional_probabilities.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()

# Step 5: Calculate posterior probabilities for new patient
print_step_header(5, "Posterior Probabilities for New Patient")

# For H = Yes
p_h_yes_given_evidence = p_h_yes * p_d_yes_h_yes * p_s_yes_h_yes

# For H = No
p_h_no_given_evidence = p_h_no * p_d_yes_h_no * p_s_yes_h_no

# Normalize the probabilities
total = p_h_yes_given_evidence + p_h_no_given_evidence
p_h_yes_given_evidence_normalized = p_h_yes_given_evidence / total
p_h_no_given_evidence_normalized = p_h_no_given_evidence / total

print("Unnormalized Posterior Probabilities:")
print(f"P(H = Yes | D = Yes, S = Yes) ∝ {p_h_yes_given_evidence:.6f}")
print(f"P(H = No | D = Yes, S = Yes) ∝ {p_h_no_given_evidence:.6f}")
print("\nNormalized Posterior Probabilities:")
print(f"P(H = Yes | D = Yes, S = Yes) = {p_h_yes_given_evidence_normalized:.6f}")
print(f"P(H = No | D = Yes, S = Yes) = {p_h_no_given_evidence_normalized:.6f}")

# Step 6: Visualize the posterior probabilities
print_step_header(6, "Visualizing Posterior Probabilities")

# Create bar plot of posterior probabilities
plt.figure(figsize=(10, 6))
posterior_probs = [p_h_yes_given_evidence_normalized, p_h_no_given_evidence_normalized]
colors = ['#ff9999', '#66b3ff']
bars = plt.bar(['Heart Disease', 'No Heart Disease'], posterior_probs, color=colors)
plt.title('Posterior Probabilities for New Patient\n(Diabetes = Yes, Smoke = Yes)', pad=20)
plt.ylabel('Probability')
plt.ylim(0, 1)

# Add probability values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')

plt.grid(True, alpha=0.3)
file_path = os.path.join(save_dir, "posterior_probabilities.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()

# Step 7: Visualize the independence assumption
print_step_header(7, "Visualizing Feature Independence Assumption")

# Calculate observed joint probabilities
joint_probs = pd.crosstab(data['Diabetes'], data['Smoke'], normalize='all')
print("\nObserved Joint Probabilities P(D, S):")
print(joint_probs)

# Calculate marginal probabilities
p_d = (data['Diabetes'] == 'Y').mean()
p_s = (data['Smoke'] == 'Y').mean()

# Calculate expected joint probabilities under independence
expected_joint_probs = pd.DataFrame(
    [[p_d * p_s, p_d * (1-p_s)],
     [(1-p_d) * p_s, (1-p_d) * (1-p_s)]],
    index=['Y', 'N'],
    columns=['Y', 'N']
)

print("\nExpected Joint Probabilities under Independence P(D)P(S):")
print(expected_joint_probs)

# Visualize the difference
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Custom colormap
cmap = sns.color_palette("YlOrRd", as_cmap=True)

sns.heatmap(joint_probs, annot=True, fmt='.3f', cmap=cmap, ax=ax1, 
            cbar_kws={'label': 'Probability'})
ax1.set_title('Observed Joint Probabilities\nP(D, S)', pad=20)

sns.heatmap(expected_joint_probs, annot=True, fmt='.3f', cmap=cmap, ax=ax2,
            cbar_kws={'label': 'Probability'})
ax2.set_title('Expected Joint Probabilities\nunder Independence P(D)P(S)', pad=20)

plt.tight_layout()
file_path = os.path.join(save_dir, "independence_assumption.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()

# Step 8: Conclusion
print_step_header(8, "Conclusion")

print("Based on the Naive Bayes analysis:")
print(f"1. For a new patient with diabetes who smokes:")
print(f"   - Probability of heart disease: {p_h_yes_given_evidence_normalized:.3f}")
print(f"   - Probability of no heart disease: {p_h_no_given_evidence_normalized:.3f}")
print("\n2. The predicted class would be:", 
      "Heart Disease" if p_h_yes_given_evidence_normalized > p_h_no_given_evidence_normalized 
      else "No Heart Disease")
print("\n3. The independence assumption between features (Diabetes and Smoking) can be assessed by")
print("   comparing the observed joint probabilities with expected probabilities under independence.")
print("   The difference between these values indicates the degree of feature dependence.") 