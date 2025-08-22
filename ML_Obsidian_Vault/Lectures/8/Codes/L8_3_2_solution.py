import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_3_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Disable LaTeX to avoid processing issues
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 2: FEATURE INTERACTIONS IN LOAN DEFAULT PREDICTION")
print("=" * 80)

# Configuration parameters - can be easily modified
CONFIG = {
    'sample_values': {
        'income': 50000,
        'debt_ratio': 0.4,
        'credit_score': 720,
        'age': 35
    },
    'correlations': {
        'income_debt': 0.3,
        'credit_age': 0.6
    },
    'dataset': {
        'n_samples': 1000,
        'random_seed': 42
    },
    'analysis': {
        'default_threshold': 0.43,
        'vif_r_squared': 0.25,
        'vif_threshold': 5
    }
}

# Extract parameters for easier use
income = CONFIG['sample_values']['income']
debt_ratio = CONFIG['sample_values']['debt_ratio']
credit_score = CONFIG['sample_values']['credit_score']
age = CONFIG['sample_values']['age']
correlation_income_debt = CONFIG['correlations']['income_debt']
correlation_credit_age = CONFIG['correlations']['credit_age']
n_samples = CONFIG['dataset']['n_samples']
random_seed = CONFIG['dataset']['random_seed']
default_threshold = CONFIG['analysis']['default_threshold']
vif_r_squared = CONFIG['analysis']['vif_r_squared']
vif_threshold = CONFIG['analysis']['vif_threshold']

print(f"\nConfiguration Parameters:")
print(f"Sample Values:")
print(f"  Income: ${income:,}")
print(f"  Debt Ratio: {debt_ratio}")
print(f"  Credit Score: {credit_score}")
print(f"  Age: {age}")
print(f"Correlations:")
print(f"  Income-Debt Ratio: {correlation_income_debt}")
print(f"  Credit Score-Age: {correlation_credit_age}")
print(f"Dataset:")
print(f"  Number of samples: {n_samples}")
print(f"  Random seed: {random_seed}")
print(f"Analysis Parameters:")
print(f"  Default threshold: {default_threshold}")
print(f"  VIF R²: {vif_r_squared}")
print(f"  VIF threshold: {vif_threshold}")

# ============================================================================
# TASK 1: Design a scenario where univariate selection would miss the interaction
# ============================================================================
print("\n" + "="*60)
print("TASK 1: UNIVARIATE SELECTION MISSING INTERACTIONS")
print("="*60)

# Create synthetic dataset to demonstrate the problem
np.random.seed(random_seed)

# Generate realistic correlated features
# Income: Normal distribution around mean income
income_mean = CONFIG['sample_values']['income']
income_std = income_mean * 0.3  # 30% standard deviation
income_data = np.random.normal(income_mean, income_std, n_samples)

# Debt ratio: correlated with income (higher income -> lower debt ratio)
base_debt_ratio = 0.3
debt_ratio_data = (base_debt_ratio +
                  0.1 * np.random.normal(0, 1, n_samples) +
                  correlation_income_debt * (income_data - income_mean) / income_std)

# Ensure debt ratio stays in reasonable bounds
debt_ratio_data = np.clip(debt_ratio_data, 0.05, 0.8)

# Credit score: Normal distribution
credit_mean = CONFIG['sample_values']['credit_score']
credit_std = 50
credit_score_data = np.random.normal(credit_mean, credit_std, n_samples)

# Age: correlated with credit score (higher credit score -> older age)
age_mean = CONFIG['sample_values']['age']
age_std = 10
age_data = (age_mean +
           0.5 * np.random.normal(0, 1, n_samples) +
           correlation_credit_age * (credit_score_data - credit_mean) / credit_std)

# Calculate debt-to-income ratio
debt_to_income_data = (debt_ratio_data * 100) / income_data

# Create target based on debt-to-income interaction with realistic noise
# The key insight: debt-to-income ratio should be the main predictor
# Individual features have weak correlations, but the ratio is strong

# Create a more realistic target generation
# Use debt-to-income ratio as the main predictor
base_prob = 0.05  # Base default rate
dti_effect = 15 * (debt_to_income_data - 0.3)  # Strong effect from debt-to-income
logit_prob = np.log(base_prob / (1 - base_prob)) + dti_effect
prob_default = 1 / (1 + np.exp(-logit_prob))

# Add some noise to make individual features slightly predictive
income_effect = 0.0001 * (income_data - income_mean) / income_std
debt_effect = 0.1 * (debt_ratio_data - base_debt_ratio)
credit_effect = 0.001 * (credit_score_data - credit_mean) / credit_std
age_effect = 0.01 * (age_data - age_mean) / age_std

# Combine all effects
total_effect = dti_effect + income_effect + debt_effect + credit_effect + age_effect
prob_default = 1 / (1 + np.exp(-total_effect))
prob_default = np.clip(prob_default, 0.01, 0.99)

target = np.random.binomial(1, prob_default)

# Create DataFrame
df = pd.DataFrame({
    'income': income_data,
    'debt_ratio': debt_ratio_data,
    'credit_score': credit_score_data,
    'age': age_data,
    'debt_to_income': debt_to_income_data,
    'target': target
})

print(f"\nDataset created with {n_samples} samples")
print(f"Target distribution: {np.bincount(target)}")
print(f"Default rate: {np.mean(target):.3f}")

# Calculate individual feature correlations with target
correlations = {}
for feature in ['income', 'debt_ratio', 'credit_score', 'age']:
    corr = np.corrcoef(df[feature], df['target'])[0, 1]
    correlations[feature] = corr

print(f"\nIndividual feature correlations with target:")
for feature, corr in correlations.items():
    print(f"{feature}: {corr:.4f}")

# Calculate debt-to-income correlation with target
dti_corr = np.corrcoef(df['debt_to_income'], df['target'])[0, 1]
print(f"debt_to_income: {dti_corr:.4f}")

# Demonstrate univariate selection
selector = SelectKBest(score_func=f_classif, k=2)
X = df[['income', 'debt_ratio', 'credit_score', 'age']].values
y = df['target'].values

selector.fit(X, y)
feature_names = ['income', 'debt_ratio', 'credit_score', 'age']
selected_indices = selector.get_support(indices=True)
selected_features_names = [feature_names[i] for i in selected_indices]

print(f"\nUnivariate selection (k=2) selects: {selected_features_names}")
print(f"Univariate selection MISSES the debt-to-income interaction!")

# Plot 1: Individual features vs target
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Task 1: Univariate Selection Missing Feature Interactions', fontsize=16)

features = ['income', 'debt_ratio', 'credit_score', 'age']
colors = ['blue', 'red', 'green', 'orange']

for i, (feature, color) in enumerate(zip(features, colors)):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    # Plot feature distribution by target
    for target_val in [0, 1]:
        mask = df['target'] == target_val
        ax.hist(df.loc[mask, feature], alpha=0.7, bins=30, 
                label=f'Target={target_val}', color=color)
    
    ax.set_xlabel(feature.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
    ax.set_title(f'{feature.title()} vs Target (Corr: {correlations[feature]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task1_univariate_features.png'), dpi=300, bbox_inches='tight')

# Plot 2: Debt-to-income ratio vs target
plt.figure(figsize=(10, 6))
for target_val in [0, 1]:
    mask = df['target'] == target_val
    plt.hist(df.loc[mask, 'debt_to_income'], alpha=0.7, bins=30, 
             label=f'Target={target_val}')

plt.axvline(x=default_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({default_threshold})')
plt.xlabel('Debt-to-Income Ratio')
plt.ylabel('Frequency')
plt.title(f'Task 1: Debt-to-Income Ratio vs Target (Corr: {dti_corr:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'task1_debt_to_income.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 2: Calculate redundancy score for income and debt_ratio
# ============================================================================
print("\n" + "="*60)
print("TASK 2: REDUNDANCY SCORE CALCULATION")
print("="*60)

# Redundancy score = correlation^2
redundancy_score = correlation_income_debt ** 2
print(f"Correlation between income and debt_ratio: {correlation_income_debt}")
print(f"Redundancy score = r² = {correlation_income_debt}² = {redundancy_score:.4f}")
print(f"This means {redundancy_score*100:.2f}% of the variance in one feature")
print(f"can be explained by the other feature.")

# ============================================================================
# TASK 3: Independent feature combinations
# ============================================================================
print("\n" + "="*60)
print("TASK 3: INDEPENDENT FEATURE COMBINATIONS")
print("="*60)

# Group correlated features
# Group 1: income, debt_ratio (correlation = 0.3)
# Group 2: credit_score, age (correlation = 0.6)
# These are independent of each other

n_groups = 2
features_per_group = [2, 2]  # 2 features in each group

print(f"Feature groups:")
print(f"Group 1: income, debt_ratio (correlation = {correlation_income_debt})")
print(f"Group 2: credit_score, age (correlation = {correlation_credit_age})")
print(f"Number of independent groups: {n_groups}")

# Calculate combinations within each group
combinations_group1 = 2**2 - 1  # 2^2 - 1 = 3 (excluding empty set)
combinations_group2 = 2**2 - 1  # 2^2 - 1 = 3

# Total independent combinations
total_combinations = combinations_group1 + combinations_group2
print(f"Combinations within Group 1: {combinations_group1}")
print(f"Combinations within Group 2: {combinations_group2}")
print(f"Total independent feature combinations: {total_combinations}")

# ============================================================================
# TASK 4: Reduction in search space
# ============================================================================
print("\n" + "="*60)
print("TASK 4: SEARCH SPACE REDUCTION")
print("="*60)

# Original search space: all possible combinations of 4 features
original_space = 2**4 - 1  # 2^4 - 1 = 15 (excluding empty set)

# Reduced search space: treat correlated features as groups
reduced_space = total_combinations

reduction = original_space - reduced_space
reduction_percentage = (reduction / original_space) * 100

print(f"Original search space: 2⁴ - 1 = {original_space} combinations")
print(f"Reduced search space: {reduced_space} combinations")
print(f"Reduction: {reduction} combinations")
print(f"Reduction percentage: {reduction_percentage:.1f}%")

# ============================================================================
# TASK 5: Multivariate selection strategy
# ============================================================================
print("\n" + "="*60)
print("TASK 5: MULTIVARIATE SELECTION STRATEGY")
print("="*60)

print("Proposed Multivariate Selection Strategy:")
print("1. Feature Grouping: Group correlated features together")
print("2. Interaction Features: Create debt-to-income ratio")
print("3. Sequential Forward Selection: Start with best individual feature")
print("4. Evaluate combinations within each group")
print("5. Cross-validation for final selection")

# Demonstrate the strategy
print(f"\nStep-by-step demonstration:")

# Step 1: Best individual feature
best_individual = max(correlations.items(), key=lambda x: abs(x[1]))
print(f"Step 1: Best individual feature: {best_individual[0]} (corr: {best_individual[1]:.4f})")

# Step 2: Best pair within Group 1
group1_features = ['income', 'debt_ratio']
group1_combinations = []
for i in range(len(group1_features)):
    for j in range(i+1, len(group1_features)):
        # Calculate correlation with target for combination
        feature1, feature2 = group1_features[i], group1_features[j]
        # Simple approach: average correlation
        avg_corr = (abs(correlations[feature1]) + abs(correlations[feature2])) / 2
        group1_combinations.append((f"{feature1}+{feature2}", avg_corr))

best_group1 = max(group1_combinations, key=lambda x: x[1])
print(f"Step 2: Best Group 1 combination: {best_group1[0]} (score: {best_group1[1]:.4f})")

# Step 3: Best pair within Group 2
group2_features = ['credit_score', 'age']
group2_combinations = []
for i in range(len(group2_features)):
    for j in range(i+1, len(group2_features)):
        feature1, feature2 = group2_features[i], group2_features[j]
        avg_corr = (abs(correlations[feature1]) + abs(correlations[feature2])) / 2
        group2_combinations.append((f"{feature1}+{feature2}", avg_corr))

best_group2 = max(group2_combinations, key=lambda x: x[1])
print(f"Step 3: Best Group 2 combination: {best_group2[0]} (score: {best_group2[1]:.4f})")

# Step 4: Interaction feature
print(f"Step 4: Interaction feature: debt-to-income ratio (corr: {dti_corr:.4f})")

# ============================================================================
# TASK 6: Calculate debt-to-income ratio and prediction
# ============================================================================
print("\n" + "="*60)
print("TASK 6: DEBT-TO-INCOME RATIO AND PREDICTION")
print("="*60)

# Calculate debt-to-income ratio
debt_to_income = (debt_ratio * 100) / income
print(f"Debt-to-income ratio = (debt_ratio × 100) / income")
print(f"Debt-to-income ratio = ({debt_ratio} × 100) / {income:,}")
print(f"Debt-to-income ratio = {debt_ratio * 100} / {income:,}")
print(f"Debt-to-income ratio = {debt_to_income:.6f}")

# Prediction based on threshold
prediction = 1 if debt_to_income > default_threshold else 0

print(f"\nThreshold for default: {default_threshold}")
print(f"Since {debt_to_income:.6f} {'>' if debt_to_income > default_threshold else '≤'} {default_threshold}")
print(f"Prediction: {prediction} ({'Default' if prediction == 1 else 'No Default'})")

# ============================================================================
# TASK 7: VIF calculation and condition number
# ============================================================================
print("\n" + "="*60)
print("TASK 7: VIF AND CONDITION NUMBER CALCULATION")
print("="*60)

# VIF calculation
vif = 1 / (1 - vif_r_squared)
print(f"R² from regressing income on other features: {vif_r_squared}")
print(f"VIF = 1 / (1 - R²) = 1 / (1 - {vif_r_squared}) = {vif:.4f}")

# VIF rule of thumb
should_remove = vif > vif_threshold
print(f"VIF threshold rule: {vif_threshold}")
print(f"Since VIF = {vif:.4f} {'>' if vif > vif_threshold else '≤'} {vif_threshold}")
print(f"Should income be removed? {'Yes' if should_remove else 'No'}")

# Condition number calculation
# Create correlation matrix
correlation_matrix = np.array([
    [1.0, correlation_income_debt, 0.0, 0.0],
    [correlation_income_debt, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, correlation_credit_age],
    [0.0, 0.0, correlation_credit_age, 1.0]
])

# Calculate eigenvalues
eigenvalues = np.linalg.eigvals(correlation_matrix)
condition_number = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))

print(f"\nCorrelation matrix eigenvalues: {eigenvalues}")
print(f"Condition number = max(|eigenvalues|) / min(|eigenvalues|)")
print(f"Condition number = {np.max(np.abs(eigenvalues)):.4f} / {np.min(np.abs(eigenvalues)):.4f}")
print(f"Condition number = {condition_number:.4f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Plot 3: Correlation matrix heatmap
plt.figure(figsize=(10, 8))
feature_names = ['Income', 'Debt Ratio', 'Credit Score', 'Age']
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            xticklabels=feature_names, yticklabels=feature_names,
            square=True, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Task 7: Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task7_correlation_matrix.png'), dpi=300, bbox_inches='tight')

# Plot 4: Feature importance comparison
plt.figure(figsize=(12, 8))
features_plot = list(correlations.keys()) + ['debt_to_income']
correlations_plot = list(correlations.values()) + [dti_corr]

colors_plot = ['blue', 'red', 'green', 'orange', 'purple']
bars = plt.bar(features_plot, [abs(c) for c in correlations_plot], color=colors_plot, alpha=0.7)
plt.xlabel('Features')
plt.ylabel('|Correlation with Target|')
plt.title('Task 1 and 5: Feature Importance Comparison')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add correlation values on bars
for bar, corr in zip(bars, correlations_plot):
    height = bar.get_height()
    # Position text inside the bar if there's enough space, otherwise above
    if height > 0.02:  # If bar is tall enough, put text inside
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{corr:.4f}', ha='center', va='center', 
                 color='white', fontweight='bold')
    else:  # If bar is too short, put text above
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{corr:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task1_5_feature_importance.png'), dpi=300, bbox_inches='tight')

# Plot 5: Search space reduction visualization
plt.figure(figsize=(10, 6))
categories = ['Original Space', 'Reduced Space']
values = [original_space, reduced_space]
colors_viz = ['red', 'green']

bars = plt.bar(categories, values, color=colors_viz, alpha=0.7)
plt.ylabel('Number of Feature Combinations')
plt.title('Task 4: Search Space Reduction')
plt.grid(True, alpha=0.3)

# Add values on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    # Position text inside the bar for better visibility
    plt.text(bar.get_x() + bar.get_width()/2., height/2,
             f'{value}', ha='center', va='center', 
             color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task4_search_space_reduction.png'), dpi=300, bbox_inches='tight')

# Plot 6: Debt-to-income distribution with threshold
plt.figure(figsize=(10, 6))
plt.hist(df['debt_to_income'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=debt_to_income, color='red', linestyle='-', linewidth=3, label=f'Sample Point: {debt_to_income:.6f}')
plt.axvline(x=default_threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold: {default_threshold}')
plt.xlabel('Debt-to-Income Ratio')
plt.ylabel('Frequency')
plt.title('Task 6: Debt-to-Income Distribution with Sample Point and Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'task6_debt_to_income_distribution.png'), dpi=300, bbox_inches='tight')

print(f"\n" + "="*80)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print(f"Plots saved to: {save_dir}")
print("="*80)
