import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import pandas as pd
from matplotlib.patches import Patch

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_22")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 22: LDA Classification for Gender Prediction based on Height")
print("====================================================================")

# Step 1: Load the data
print("\nStep 1: Load and explore the data")
print("--------------------------------")

# Given data
gender = np.array(['F', 'M', 'F', 'M', 'M', 'M'])
height = np.array([160, 160, 170, 170, 170, 180])

# Create a table to display the data
print("Original data:")
print("| Index | Gender | Height |")
print("|-------|--------|--------|")
for i in range(len(gender)):
    print(f"| {i+1}    | {gender[i]}      | {height[i]}    |")

# Separate data by class
female_indices = np.where(gender == 'F')[0]
male_indices = np.where(gender == 'M')[0]

female_heights = height[female_indices]
male_heights = height[male_indices]

print("\nData separated by gender:")
print(f"Female heights: {female_heights}")
print(f"Male heights: {male_heights}")

# Step 2: Calculate class statistics with detailed steps
print("\nStep 2: Calculate class statistics (with detailed steps)")
print("------------------------------------------------------")

# Calculate means
female_mean = np.mean(female_heights)
male_mean = np.mean(male_heights)
overall_mean = np.mean(height)

print("\nDetailed Mean Calculations:")
print(f"Female mean = ({' + '.join(map(str, female_heights))}) / {len(female_heights)} = {female_mean:.2f}")
print(f"Male mean = ({' + '.join(map(str, male_heights))}) / {len(male_heights)} = {male_mean:.2f}")
print(f"Overall mean = ({' + '.join(map(str, height))}) / {len(height)} = {overall_mean:.2f}")

# Calculate count and prior probabilities
n_female = len(female_heights)
n_male = len(male_heights)
n_total = len(height)

prior_female = n_female / n_total
prior_male = n_male / n_total

print("\nPrior Probability Calculations:")
print(f"P(Female) = Number of females / Total = {n_female} / {n_total} = {prior_female:.2f}")
print(f"P(Male) = Number of males / Total = {n_male} / {n_total} = {prior_male:.2f}")

# Calculate within-class variances with detailed steps
if len(female_heights) > 1:
    female_variance_calc = sum([(x - female_mean)**2 for x in female_heights]) / (len(female_heights) - 1)
    print("\nFemale Variance Calculation:")
    terms = [f"({x} - {female_mean:.2f})² = {(x - female_mean)**2:.2f}" for x in female_heights]
    print(f"Var(Female) = ({' + '.join(terms)}) / {len(female_heights) - 1} = {female_variance_calc:.2f}")
    female_variance = female_variance_calc
else:
    female_variance = 0

if len(male_heights) > 1:
    male_variance_calc = sum([(x - male_mean)**2 for x in male_heights]) / (len(male_heights) - 1)
    print("\nMale Variance Calculation:")
    terms = [f"({x} - {male_mean:.2f})² = {(x - male_mean)**2:.2f}" for x in male_heights]
    print(f"Var(Male) = ({' + '.join(terms)}) / {len(male_heights) - 1} = {male_variance_calc:.2f}")
    male_variance = male_variance_calc
else:
    male_variance = 0

# Calculate pooled variance (weighted average of class variances)
pooled_variance = ((n_female - 1) * female_variance + (n_male - 1) * male_variance) / (n_total - 2)

print("\nPooled Variance Calculation:")
print(f"Pooled Variance = [(nₙ - 1)Var(Female) + (nₘ - 1)Var(Male)] / (nₙ + nₘ - 2)")
print(f"                = [({n_female} - 1) × {female_variance:.2f} + ({n_male} - 1) × {male_variance:.2f}] / ({n_total} - 2)")
print(f"                = [{n_female - 1} × {female_variance:.2f} + {n_male - 1} × {male_variance:.2f}] / {n_total - 2}")
print(f"                = [{(n_female - 1) * female_variance:.2f} + {(n_male - 1) * male_variance:.2f}] / {n_total - 2}")
print(f"                = {(n_female - 1) * female_variance + (n_male - 1) * male_variance:.2f} / {n_total - 2}")
print(f"                = {pooled_variance:.2f}")

print("\nSummary of Class Statistics:")
print(f"Female mean height: {female_mean:.2f}")
print(f"Male mean height: {male_mean:.2f}")
print(f"Overall mean height: {overall_mean:.2f}")
print(f"Female variance: {female_variance:.2f}")
print(f"Male variance: {male_variance:.2f}")
print(f"Pooled variance: {pooled_variance:.2f}")
print(f"Prior probability (Female): {prior_female:.2f}")
print(f"Prior probability (Male): {prior_male:.2f}")

# Step 3: Visualize the distributions
print("\nStep 3: Visualize the distributions")
print("----------------------------------")

# Create a figure
plt.figure(figsize=(12, 6))

# Histogram of heights by gender
plt.subplot(1, 2, 1)
bins = np.arange(155, 185, 5)
plt.hist([female_heights, male_heights], bins=bins, alpha=0.6, label=['Female', 'Male'])
plt.axvline(female_mean, color='tab:blue', linestyle='dashed', linewidth=2, label=f'Female mean: {female_mean:.1f}')
plt.axvline(male_mean, color='tab:orange', linestyle='dashed', linewidth=2, label=f'Male mean: {male_mean:.1f}')
plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Height Distribution by Gender', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Distribution curves
plt.subplot(1, 2, 2)
x = np.linspace(150, 190, 1000)
female_pdf = stats.norm.pdf(x, female_mean, np.sqrt(pooled_variance))
male_pdf = stats.norm.pdf(x, male_mean, np.sqrt(pooled_variance))

plt.plot(x, female_pdf, 'b-', linewidth=2, label='Female (Normal Distribution)')
plt.plot(x, male_pdf, 'orange', linewidth=2, label='Male (Normal Distribution)')
plt.axvline(female_mean, color='blue', linestyle='dashed', linewidth=2, label=f'Female mean: {female_mean:.1f}')
plt.axvline(male_mean, color='orange', linestyle='dashed', linewidth=2, label=f'Male mean: {male_mean:.1f}')

# Find decision boundary (where likelihood ratio = prior ratio)
# For LDA with equal variances, the decision boundary is at the midpoint between the means
decision_boundary = (female_mean + male_mean) / 2
plt.axvline(decision_boundary, color='red', linestyle='--', linewidth=2, 
            label=f'Decision boundary: {decision_boundary:.1f}')

plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Normal Distributions and Decision Boundary', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "distributions.png"), dpi=300, bbox_inches='tight')
print(f"The distribution plot has been saved to 'distributions.png'")

# Step 4: Implement LDA classifier with detailed calculations
print("\nStep 4: Implement LDA classifier with detailed calculations")
print("--------------------------------------------------------")

# LDA discriminant function: d_k(x) = x * (mean_k / pooled_variance) - (mean_k^2 / (2 * pooled_variance)) + ln(prior_k)
# We can simplify this for the binary case using the discriminant function:
# d(x) = x * ((mean_1 - mean_0) / pooled_variance) - ((mean_1^2 - mean_0^2) / (2 * pooled_variance)) + ln(prior_1 / prior_0)

print("\nLDA Discriminant Function Derivation:")
print("Starting with the quadratic discriminant function for class k:")
print("δₖ(x) = -1/2 (x - μₖ)ᵀ Σ⁻¹ (x - μₖ) + ln(πₖ)")

print("\nFor a 1D case with equal variances (σ²), this simplifies to:")
print("δₖ(x) = -1/(2σ²) (x - μₖ)² + ln(πₖ)")
print("      = -1/(2σ²) (x² - 2xμₖ + μₖ²) + ln(πₖ)")

print("\nFor binary classification (Female vs. Male), we can calculate:")
print("δᵐₐₗₑ(x) - δfₑₘₐₗₑ(x) = [xμₘ/σ² - μₘ²/(2σ²) + ln(πₘ)] - [xμf/σ² - μf²/(2σ²) + ln(πf)]")
print("                      = x(μₘ - μf)/σ² - (μₘ² - μf²)/(2σ²) + ln(πₘ/πf)")

print("\nWe classify as Male if this difference is > 0, Female otherwise.")

print("\nSubstituting our values:")
print(f"μf = {female_mean:.2f}, μₘ = {male_mean:.2f}, σ² = {pooled_variance:.2f}, πf = {prior_female:.2f}, πₘ = {prior_male:.2f}")

coef = (male_mean - female_mean) / pooled_variance
term2 = (male_mean**2 - female_mean**2) / (2 * pooled_variance)
term3 = np.log(prior_male / prior_female)
intercept = -term2 + term3

print(f"\nCalculation of coefficient term: (μₘ - μf)/σ² = ({male_mean:.2f} - {female_mean:.2f})/{pooled_variance:.2f} = {coef:.4f}")
print(f"Calculation of squared term: (μₘ² - μf²)/(2σ²) = ({male_mean:.2f}² - {female_mean:.2f}²)/(2×{pooled_variance:.2f})")
print(f"                                                = ({male_mean**2:.2f} - {female_mean**2:.2f})/{2*pooled_variance:.2f}")
print(f"                                                = {male_mean**2 - female_mean**2:.2f}/{2*pooled_variance:.2f}")
print(f"                                                = {term2:.4f}")
print(f"Calculation of prior term: ln(πₘ/πf) = ln({prior_male:.2f}/{prior_female:.2f}) = ln({prior_male/prior_female:.4f}) = {term3:.4f}")

print(f"\nFinal discriminant function: d(x) = {coef:.4f}x - {term2:.4f} + {term3:.4f} = {coef:.4f}x + {intercept:.4f}")

# Define the discriminant function
def lda_discriminant(x, mean_0, mean_1, pooled_var, prior_0, prior_1):
    """Calculate the LDA discriminant value for binary classification"""
    # Using the formula: d(x) = x * ((mean_1 - mean_0) / pooled_var) - ((mean_1^2 - mean_0^2) / (2 * pooled_var)) + ln(prior_1 / prior_0)
    # d(x) > 0 implies class 1, d(x) < 0 implies class 0
    # In our case, class 0 = Female, class 1 = Male
    term1 = x * ((mean_1 - mean_0) / pooled_var)
    term2 = ((mean_1**2 - mean_0**2) / (2 * pooled_var))
    term3 = np.log(prior_1 / prior_0) if prior_0 > 0 else 0
    return term1 - term2 + term3

def lda_classify(x, mean_0, mean_1, pooled_var, prior_0, prior_1):
    """Classify using LDA discriminant function - 1 if male, 0 if female"""
    discriminant = lda_discriminant(x, mean_0, mean_1, pooled_var, prior_0, prior_1)
    return 1 if discriminant > 0 else 0  # 1 for Male, 0 for Female

# Step 5: Make predictions with detailed calculations
print("\nStep 5: Make predictions with detailed calculations")
print("------------------------------------------------")

# For each unique height, calculate discriminant and prediction with steps
unique_heights = np.unique(height)
discriminant_values = []
predictions = []
prediction_labels = []

print("Detailed discriminant calculations for each height:")
for h in unique_heights:
    print(f"\nHeight = {h} cm:")
    
    # Calculate each term of the discriminant function
    term1 = h * ((male_mean - female_mean) / pooled_variance)
    term2 = ((male_mean**2 - female_mean**2) / (2 * pooled_variance))
    term3 = np.log(prior_male / prior_female)
    
    print(f"  Term 1: x(μₘ - μf)/σ² = {h} × ({male_mean:.2f} - {female_mean:.2f})/{pooled_variance:.2f} = {term1:.4f}")
    print(f"  Term 2: (μₘ² - μf²)/(2σ²) = ({male_mean**2:.2f} - {female_mean**2:.2f})/(2×{pooled_variance:.2f}) = {term2:.4f}")
    print(f"  Term 3: ln(πₘ/πf) = ln({prior_male:.2f}/{prior_female:.2f}) = {term3:.4f}")
    
    disc_value = term1 - term2 + term3
    discriminant_values.append(disc_value)
    
    pred = 1 if disc_value > 0 else 0
    predictions.append(pred)
    
    pred_label = 'M' if pred == 1 else 'F'
    prediction_labels.append(pred_label)
    
    print(f"  Discriminant value: d({h}) = {term1:.4f} - {term2:.4f} + {term3:.4f} = {disc_value:.4f}")
    print(f"  Since discriminant is {'positive' if disc_value > 0 else 'negative'}, predict: {pred_label}")

# Summary table of discriminant values and predictions
print("\nSummary of discriminant values and predictions:")
print("| Height | Discriminant | Prediction |")
print("|--------|--------------|------------|")
for h, disc, pred in zip(unique_heights, discriminant_values, prediction_labels):
    print(f"| {h}    | {disc:.4f}     | {pred}        |")

# Create an improved visualization of the classifier (without the table)
plt.figure(figsize=(10, 6))

# Plot the data points
for i, h in enumerate(height):
    if gender[i] == 'F':
        marker_color = 'blue'
        marker = 'o'
    else:
        marker_color = 'orange'
        marker = 'x'
    plt.scatter(h, 0, color=marker_color, s=100, marker=marker, 
                label=f"{gender[i]}" if i == 0 or i == 2 else "")

# Plot the decision boundary
plt.axvline(x=decision_boundary, color='red', linestyle='--', linewidth=2, 
            label=f'Decision Boundary: {decision_boundary:.1f}')

# Add discriminant function line
x_range = np.linspace(155, 185, 1000)
disc_values = [lda_discriminant(x, female_mean, male_mean, pooled_variance, prior_female, prior_male) for x in x_range]
scaled_disc = np.array(disc_values) / max(abs(min(disc_values)), abs(max(disc_values))) * 1.5  # Scale for visualization

plt.plot(x_range, scaled_disc, 'g-', linewidth=2, label='Discriminant Function (Scaled)')

# Add regions
plt.fill_betweenx([-2, 2], 150, decision_boundary, color='blue', alpha=0.1, label='Predicted Female')
plt.fill_betweenx([-2, 2], decision_boundary, 190, color='orange', alpha=0.1, label='Predicted Male')

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.text(female_mean, 0.8, f'Female Mean\n{female_mean:.1f}', ha='center', fontsize=10)
plt.text(male_mean, 0.8, f'Male Mean\n{male_mean:.1f}', ha='center', fontsize=10)

# Correct the legend for Female and Male data points
handles, labels = plt.gca().get_legend_handles_labels()
female_patch = Patch(color='blue', label='Female')
male_patch = Patch(color='orange', label='Male')
handles = [female_patch, male_patch] + handles[2:]
labels = ['Female', 'Male'] + labels[2:]
plt.legend(handles=handles, labels=labels)

plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Discriminant Value (Scaled)', fontsize=12)
plt.title('LDA Classification of Gender based on Height', fontsize=14)
plt.xlim(150, 190)
plt.ylim(-2, 2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lda_classification.png"), dpi=300, bbox_inches='tight')
print(f"The improved LDA classification plot has been saved to 'lda_classification.png'")

# Step 6: Create an additional visualization showing the Bayes' Decision Rule
plt.figure(figsize=(12, 6))

# Prepare data for visualization
x = np.linspace(150, 190, 1000)
p_x_female = stats.norm.pdf(x, female_mean, np.sqrt(pooled_variance))
p_x_male = stats.norm.pdf(x, male_mean, np.sqrt(pooled_variance))

posterior_female = prior_female * p_x_female
posterior_male = prior_male * p_x_male

# Normalize posteriors to get proper probability
posterior_sum = posterior_female + posterior_male
posterior_female_norm = posterior_female / posterior_sum
posterior_male_norm = posterior_male / posterior_sum

# Left plot: Class-conditional probabilities
plt.subplot(1, 2, 1)
plt.plot(x, p_x_female, 'b-', linewidth=2, label=f'P(X|Female)')
plt.plot(x, p_x_male, 'orange', linewidth=2, label=f'P(X|Male)')
plt.axvline(female_mean, color='blue', linestyle=':', linewidth=1)
plt.axvline(male_mean, color='orange', linestyle=':', linewidth=1)
plt.axvline(decision_boundary, color='red', linestyle='--', linewidth=2, 
            label=f'Decision Boundary: {decision_boundary:.1f}')

# Highlight regions
plt.fill_between(x, p_x_female, where=(x <= decision_boundary), color='blue', alpha=0.1)
plt.fill_between(x, p_x_male, where=(x >= decision_boundary), color='orange', alpha=0.1)

# Mark the points from the dataset
for h in unique_heights:
    female_height_prob = stats.norm.pdf(h, female_mean, np.sqrt(pooled_variance))
    male_height_prob = stats.norm.pdf(h, male_mean, np.sqrt(pooled_variance))
    
    plt.plot(h, female_height_prob, 'bo', markersize=8)
    plt.plot(h, male_height_prob, 'o', color='orange', markersize=8)
    
    plt.text(h, female_height_prob, f" P(X={h}|Female)={female_height_prob:.4f}", va='bottom')
    plt.text(h, male_height_prob, f" P(X={h}|Male)={male_height_prob:.4f}", va='bottom')

plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Class-Conditional Probabilities', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Right plot: Posterior probabilities
plt.subplot(1, 2, 2)
plt.plot(x, posterior_female_norm, 'b-', linewidth=2, label=f'P(Female|X)')
plt.plot(x, posterior_male_norm, 'orange', linewidth=2, label=f'P(Male|X)')
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold: 0.5')
plt.axvline(decision_boundary, color='red', linestyle='--', linewidth=2)

# Mark decision regions
plt.fill_between(x, posterior_female_norm, where=(posterior_female_norm > 0.5), color='blue', alpha=0.1)
plt.fill_between(x, posterior_male_norm, where=(posterior_male_norm > 0.5), color='orange', alpha=0.1)

# Mark the points from the dataset
for h in unique_heights:
    female_posterior = prior_female * stats.norm.pdf(h, female_mean, np.sqrt(pooled_variance))
    male_posterior = prior_male * stats.norm.pdf(h, male_mean, np.sqrt(pooled_variance))
    sum_posterior = female_posterior + male_posterior
    female_posterior_norm = female_posterior / sum_posterior
    male_posterior_norm = male_posterior / sum_posterior
    
    plt.plot(h, female_posterior_norm, 'bo', markersize=8)
    plt.plot(h, male_posterior_norm, 'o', color='orange', markersize=8)
    
    plt.text(h, female_posterior_norm, f" P(Female|X={h})={female_posterior_norm:.4f}", va='bottom')
    plt.text(h, male_posterior_norm, f" P(Male|X={h})={male_posterior_norm:.4f}", va='bottom')

plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Posterior Probability', fontsize=12)
plt.title('Posterior Probabilities: P(Gender|Height)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "bayes_decision.png"), dpi=300, bbox_inches='tight')
print(f"The Bayes' decision rule visualization has been saved to 'bayes_decision.png'")

# Step 7: Create classification table with detailed Bayesian calculations
print("\nStep 7: Create complete classification table with detailed Bayesian calculations")
print("----------------------------------------------------------------------------")

# Create a detailed table with Bayesian calculations
print("| Gender | Height | P(X|Female) | P(X|Male) | P(Female)×P(X|Female) | P(Male)×P(X|Male) | P(Female|X) | P(Male|X) | Prediction |")
print("|--------|--------|-------------|-----------|------------------------|-------------------|-------------|-----------|------------|")

unique_data = []
for g, h in zip(gender, height):
    if (g, h) not in unique_data:
        unique_data.append((g, h))

for g, h in unique_data:
    # Calculate class conditional probabilities
    p_x_female = stats.norm.pdf(h, female_mean, np.sqrt(pooled_variance))
    p_x_male = stats.norm.pdf(h, male_mean, np.sqrt(pooled_variance))
    
    # Calculate un-normalized posterior probabilities
    posterior_female = prior_female * p_x_female
    posterior_male = prior_male * p_x_male
    
    # Normalize to get proper probabilities
    sum_posterior = posterior_female + posterior_male
    post_female_norm = posterior_female / sum_posterior
    post_male_norm = posterior_male / sum_posterior
    
    # Make prediction
    prediction = 'M' if posterior_male > posterior_female else 'F'
    
    print(f"| {g}      | {h}    | {p_x_female:.6f} | {p_x_male:.6f} | {posterior_female:.6f} | {posterior_male:.6f} | {post_female_norm:.6f} | {post_male_norm:.6f} | {prediction} |")

print("\nDetailed Bayesian Calculations for height = 160 cm:")
h = 160
p_x_female = stats.norm.pdf(h, female_mean, np.sqrt(pooled_variance))
p_x_male = stats.norm.pdf(h, male_mean, np.sqrt(pooled_variance))
posterior_female = prior_female * p_x_female
posterior_male = prior_male * p_x_male
sum_posterior = posterior_female + posterior_male
post_female_norm = posterior_female / sum_posterior
post_male_norm = posterior_male / sum_posterior

print(f"P(X=160|Female) = {p_x_female:.6f}")
print(f"P(X=160|Male) = {p_x_male:.6f}")
print(f"P(Female)×P(X=160|Female) = {prior_female:.2f} × {p_x_female:.6f} = {posterior_female:.6f}")
print(f"P(Male)×P(X=160|Male) = {prior_male:.2f} × {p_x_male:.6f} = {posterior_male:.6f}")
print(f"P(Female|X=160) = {posterior_female:.6f} / ({posterior_female:.6f} + {posterior_male:.6f}) = {post_female_norm:.6f}")
print(f"P(Male|X=160) = {posterior_male:.6f} / ({posterior_female:.6f} + {posterior_male:.6f}) = {post_male_norm:.6f}")
print(f"Since P(Male|X=160) > P(Female|X=160), predict Male")

# Step 8: Calculate final classification accuracy
print("\nStep 8: Calculate final classification accuracy")
print("--------------------------------------------")

# Calculate predicted labels for all data points
all_predictions = []
for h in height:
    idx = np.where(unique_heights == h)[0][0]
    all_predictions.append(prediction_labels[idx])

# Convert to numpy arrays for easier comparison
all_predictions = np.array(all_predictions)
correct = (all_predictions == gender)
accuracy = np.mean(correct) * 100

print("Predicted vs. Actual:")
for i, (actual, pred) in enumerate(zip(gender, all_predictions)):
    match = "✓" if actual == pred else "✗"
    print(f"Person {i+1}: Actual = {actual}, Predicted = {pred} {match}")

print(f"\nOverall Accuracy: {accuracy:.2f}%")

# Step 9: Create a confusion matrix visualization
plt.figure(figsize=(8, 6))

# Calculate confusion matrix
cm = np.zeros((2, 2), dtype=int)
for i, (true, pred) in enumerate(zip(gender, all_predictions)):
    true_idx = 0 if true == 'F' else 1  # 0 for female, 1 for male
    pred_idx = 0 if pred == 'F' else 1
    cm[true_idx, pred_idx] += 1

# Display confusion matrix as a heatmap
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=14)
plt.colorbar()
classes = ['Female', 'Male']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, fontsize=12)
plt.yticks(tick_marks, classes, fontsize=12)

# Add text annotations to the cells
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

plt.ylabel('True Gender', fontsize=12)
plt.xlabel('Predicted Gender', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
print(f"The confusion matrix has been saved to 'confusion_matrix.png'")

# Step 10: Explore LDA assumptions
print("\nStep 10: Explore LDA assumptions")
print("-----------------------------")

print("LDA makes the following assumptions:")
print("1. The classes have equal covariance matrices (in 1D case, equal variances)")
print("2. The features follow a Gaussian distribution within each class")
print("3. The features are statistically independent (not relevant for 1D case)")

print("\nLet's check these assumptions on our data:")

# Check for equal variance using a visual comparison
female_std = np.sqrt(female_variance)
male_std = np.sqrt(male_variance)
pooled_std = np.sqrt(pooled_variance)

print(f"\nStandard deviations:")
print(f"Female: {female_std:.2f}")
print(f"Male: {male_std:.2f}")
print(f"Pooled: {pooled_std:.2f}")

# Create a visualization of the variance assumption
plt.figure(figsize=(10, 6))

x = np.linspace(150, 190, 1000)
female_pdf = stats.norm.pdf(x, female_mean, female_std)
male_pdf = stats.norm.pdf(x, male_mean, male_std)
female_pdf_pooled = stats.norm.pdf(x, female_mean, pooled_std)
male_pdf_pooled = stats.norm.pdf(x, male_mean, pooled_std)

plt.plot(x, female_pdf, 'b--', linewidth=2, label='Female (Actual Variance)')
plt.plot(x, male_pdf, 'orange', linestyle='--', linewidth=2, label='Male (Actual Variance)')
plt.plot(x, female_pdf_pooled, 'b-', linewidth=2, label='Female (Pooled Variance)')
plt.plot(x, male_pdf_pooled, 'orange', linewidth=2, label='Male (Pooled Variance)')
plt.axvline(female_mean, color='blue', linestyle=':', linewidth=1)
plt.axvline(male_mean, color='orange', linestyle=':', linewidth=1)

plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Comparison of Class-specific vs. Pooled Variances', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "variance_comparison.png"), dpi=300, bbox_inches='tight')
print(f"The variance comparison plot has been saved to 'variance_comparison.png'")

# Step 11: Create a visualization for equal prior analysis
print("\nStep 11: Additional analysis with equal priors")
print("-------------------------------------------")

# Calculate LDA with equal priors
equal_prior = 0.5
equal_term3 = np.log(equal_prior / equal_prior)  # Should be 0
equal_intercept = -term2 + equal_term3

print("Analysis with equal priors (0.5 for each class):")
print(f"Discriminant function with equal priors: d(x) = {coef:.4f}x - {term2:.4f} + 0 = {coef:.4f}x - {term2:.4f}")

# Calculate the equal-prior decision boundary
# When d(x) = 0: coef * x - term2 = 0 => x = term2 / coef
equal_decision_boundary = term2 / coef
print(f"Decision boundary with equal priors: x = {term2:.4f} / {coef:.4f} = {equal_decision_boundary:.2f}")

# Visualize the impact of priors on the decision boundary
plt.figure(figsize=(10, 6))

# Plot distribution curves
plt.plot(x, female_pdf_pooled, 'b-', linewidth=2, label='Female (Normal Distribution)')
plt.plot(x, male_pdf_pooled, 'orange', linewidth=2, label='Male (Normal Distribution)')

# Plot actual decision boundary (with original priors)
plt.axvline(decision_boundary, color='red', linestyle='--', linewidth=2, 
            label=f'Decision boundary (actual priors): {decision_boundary:.1f}')

# Plot the equal prior decision boundary
plt.axvline(equal_decision_boundary, color='green', linestyle='--', linewidth=2, 
            label=f'Decision boundary (equal priors): {equal_decision_boundary:.1f}')

# Add shading for decision regions with actual priors
plt.fill_betweenx([0, 0.06], 150, decision_boundary, color='blue', alpha=0.1)
plt.fill_betweenx([0, 0.06], decision_boundary, 190, color='orange', alpha=0.1)

plt.axvline(female_mean, color='blue', linestyle=':', linewidth=1)
plt.axvline(male_mean, color='orange', linestyle=':', linewidth=1)

plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Impact of Prior Probabilities on Decision Boundary', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "equal_prior_analysis.png"), dpi=300, bbox_inches='tight')
print(f"The equal prior analysis plot has been saved to 'equal_prior_analysis.png'")

print("\nConclusion:")
print("-----------")
print(f"1. The LDA model classifies all individuals as Male with an accuracy of {accuracy:.2f}%")
print(f"2. The decision boundary with actual priors is at height = {decision_boundary:.1f} cm")
print(f"3. The decision boundary with equal priors would be at height = {equal_decision_boundary:.1f} cm")
print("4. The actual prior probabilities (2/3 Male, 1/3 Female) shift the boundary toward lower heights")
print("5. This causes the misclassification of both females as males")
print("6. The LDA model assumes equal variance for both genders, which we enforced by using pooled variance")
print("7. Due to the small sample size, it's difficult to fully verify the normality assumption") 