import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_4_Quiz_11")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

print("Question 11: Decision Tree Overfitting and Pruning Analysis")
print("=" * 60)

# 1. Methods to detect overfitting in decision trees
print("\n1. Methods to Detect Overfitting in Decision Trees:")
print("-" * 50)

overfitting_methods = [
    "Training vs Validation Accuracy Gap: Large difference indicates overfitting",
    "Cross-validation Performance: Declining performance with increasing complexity",
    "Learning Curves: Training accuracy increases while validation decreases",
    "Tree Depth Analysis: Performance plateaus or decreases with more depth",
    "Feature Importance Stability: Unstable feature rankings across folds",
    "Residual Analysis: Overly complex patterns in residuals"
]

for i, method in enumerate(overfitting_methods, 1):
    print(f"{i}. {method}")

# 2. Create synthetic data based on the tree structure
print("\n2. Creating Synthetic Data Based on Tree Structure:")
print("-" * 50)

np.random.seed(42)
n_samples = 1000

# Generate data based on the tree structure
data = []
for _ in range(n_samples):
    # Purchase_Frequency: 60% High, 40% Low
    purchase_freq = np.random.choice(['High', 'Low'], p=[0.6, 0.4])
    
    if purchase_freq == 'High':
        # Customer_Service_Rating: 70% Excellent, 30% Good
        service_rating = np.random.choice(['Excellent', 'Good'], p=[0.7, 0.3])
        
        if service_rating == 'Excellent':
            # Churn: 98% Leave, 2% Stay
            churn = np.random.choice(['Leave', 'Stay'], p=[0.98, 0.02])
        else:  # Good
            # Purchase_Amount: 60% >$100, 40% <=$100
            purchase_amount = np.random.choice(['>$100', '<=$100'], p=[0.6, 0.4])
            
            if purchase_amount == '>$100':
                # Stay: 95% Stay, 5% Leave
                churn = np.random.choice(['Stay', 'Leave'], p=[0.95, 0.05])
            else:  # <=$100
                # Churn: 97% Leave, 3% Stay
                churn = np.random.choice(['Leave', 'Stay'], p=[0.97, 0.03])
    else:  # Low
        # Account_Age: 70% >2 years, 30% <=2 years
        account_age = np.random.choice(['>2 years', '<=2 years'], p=[0.7, 0.3])
        
        if account_age == '>2 years':
            # Stay: 88% Stay, 12% Leave
            churn = np.random.choice(['Stay', 'Leave'], p=[0.88, 0.12])
        else:  # <=2 years
            # Churn: 85% Leave, 15% Stay
            churn = np.random.choice(['Leave', 'Stay'], p=[0.85, 0.15])
    
    data.append({
        'Purchase_Frequency': purchase_freq,
        'Customer_Service_Rating': service_rating if purchase_freq == 'High' else 'N/A',
        'Purchase_Amount': purchase_amount if (purchase_freq == 'High' and service_rating == 'Good') else 'N/A',
        'Account_Age': account_age if purchase_freq == 'Low' else 'N/A',
        'Churn': churn
    })

df = pd.DataFrame(data)
print(f"Generated {len(df)} samples")
print(f"Churn distribution: {df['Churn'].value_counts().to_dict()}")

# 3. Create and train the overfitted tree
print("\n3. Creating and Training the Overfitted Tree:")
print("-" * 50)

# Convert categorical variables to numerical
df_encoded = df.copy()
df_encoded['Purchase_Frequency'] = df_encoded['Purchase_Frequency'].map({'Low': 0, 'High': 1})
df_encoded['Customer_Service_Rating'] = df_encoded['Customer_Service_Rating'].map({'N/A': -1, 'Good': 0, 'Excellent': 1})
df_encoded['Purchase_Amount'] = df_encoded['Purchase_Amount'].map({'N/A': -1, '<=$100': 0, '>$100': 1})
df_encoded['Account_Age'] = df_encoded['Account_Age'].map({'N/A': -1, '<=2 years': 0, '>2 years': 1})
df_encoded['Churn'] = df_encoded['Churn'].map({'Stay': 0, 'Leave': 1})

# Features and target
X = df_encoded[['Purchase_Frequency', 'Customer_Service_Rating', 'Purchase_Amount', 'Account_Age']]
y = df_encoded['Churn']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Create overfitted tree (very deep)
overfitted_tree = DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)
overfitted_tree.fit(X_train, y_train)

# Calculate accuracies
train_acc = accuracy_score(y_train, overfitted_tree.predict(X_train))
val_acc = accuracy_score(y_val, overfitted_tree.predict(X_val))

print(f"Overfitted Tree:")
print(f"Training Accuracy: {train_acc:.3f}")
print(f"Validation Accuracy: {val_acc:.3f}")
print(f"Overfitting Gap: {train_acc - val_acc:.3f}")

# 4. Plot tree complexity vs performance
print("\n4. Plotting Tree Complexity vs Performance:")
print("-=" * 50)

max_depths = range(1, 21)
train_scores = []
val_scores = []

for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    train_scores.append(accuracy_score(y_train, tree.predict(X_train)))
    val_scores.append(accuracy_score(y_val, tree.predict(X_val)))

plt.figure(figsize=(12, 8))
plt.plot(max_depths, train_scores, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
plt.plot(max_depths, val_scores, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
plt.axhline(y=0.72, color='g', linestyle='--', alpha=0.7, label='Target Validation Acc (72%)')
plt.axhline(y=0.98, color='orange', linestyle='--', alpha=0.7, label='Target Training Acc (98%)')

plt.xlabel('Tree Depth (Complexity)')
plt.ylabel('Accuracy')
plt.title('Decision Tree Complexity vs Performance')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(max_depths[::2])

# Highlight overfitting region
plt.axvspan(8, 20, alpha=0.2, color='red', label='Overfitting Region')
plt.axvspan(1, 7, alpha=0.2, color='green', label='Good Generalization')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'complexity_vs_performance.png'), dpi=300, bbox_inches='tight')

# 5. Apply pruning techniques
print("\n5. Applying Pruning Techniques:")
print("-" * 50)

# Technique 1: Pre-pruning (max_depth limitation)
print("Technique 1: Pre-pruning with max_depth=4")
pre_pruned_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
pre_pruned_tree.fit(X_train, y_train)

pre_train_acc = accuracy_score(y_train, pre_pruned_tree.predict(X_train))
pre_val_acc = accuracy_score(y_val, pre_pruned_tree.predict(X_val))

print(f"Pre-pruned Tree (max_depth=4):")
print(f"Training Accuracy: {pre_train_acc:.3f}")
print(f"Validation Accuracy: {pre_val_acc:.3f}")
print(f"Overfitting Gap: {pre_train_acc - pre_val_acc:.3f}")

# Technique 2: Post-pruning (cost complexity pruning)
print("\nTechnique 2: Post-pruning with cost complexity")
post_pruned_tree = DecisionTreeClassifier(random_state=42)
path = post_pruned_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Find optimal alpha
optimal_alpha = ccp_alphas[np.argmax(val_scores)]
print(f"Optimal alpha for post-pruning: {optimal_alpha:.6f}")

post_pruned_tree = DecisionTreeClassifier(ccp_alpha=optimal_alpha, random_state=42)
post_pruned_tree.fit(X_train, y_train)

post_train_acc = accuracy_score(y_train, post_pruned_tree.predict(X_train))
post_val_acc = accuracy_score(y_val, post_pruned_tree.predict(X_val))

print(f"Post-pruned Tree (alpha={optimal_alpha:.6f}):")
print(f"Training Accuracy: {post_train_acc:.3f}")
print(f"Validation Accuracy: {post_val_acc:.3f}")
print(f"Overfitting Gap: {post_train_acc - post_val_acc:.3f}")

# 6. Plot pruning comparison - three separate images
# Original overfitted tree
plt.figure(figsize=(10, 8))
plot_tree(overfitted_tree, feature_names=['Purchase_Freq', 'Service_Rating', 'Purchase_Amount', 'Account_Age'], 
          class_names=['Stay', 'Leave'], filled=True, rounded=True, fontsize=10)
plt.title(f'Overfitted Tree\nTrain: {train_acc:.3f}, Val: {val_acc:.3f}')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'overfitted_tree.png'), dpi=300, bbox_inches='tight')

# Pre-pruned tree
plt.figure(figsize=(10, 8))
plot_tree(pre_pruned_tree, feature_names=['Purchase_Freq', 'Service_Rating', 'Purchase_Amount', 'Account_Age'], 
          class_names=['Stay', 'Leave'], filled=True, rounded=True, fontsize=10)
plt.title(f'Pre-pruned Tree (max_depth=4)\nTrain: {pre_train_acc:.3f}, Val: {pre_val_acc:.3f}')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pre_pruned_tree.png'), dpi=300, bbox_inches='tight')

# Post-pruned tree
plt.figure(figsize=(10, 8))
plot_tree(post_pruned_tree, feature_names=['Purchase_Freq', 'Service_Rating', 'Purchase_Amount', 'Account_Age'], 
          class_names=['Stay', 'Leave'], filled=True, rounded=True, fontsize=10)
plt.title(f'Post-pruned Tree (alpha={optimal_alpha:.6f})\nTrain: {post_train_acc:.3f}, Val: {post_val_acc:.3f}')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'post_pruned_tree.png'), dpi=300, bbox_inches='tight')

# 7. Information gain calculation
print("\n7. Information Gain Calculation:")
print("-" * 50)

def entropy(y):
    """Calculate entropy of a binary classification"""
    if len(y) == 0:
        return 0
    p = np.mean(y)
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def information_gain(X, y, feature_idx):
    """Calculate information gain for a feature"""
    parent_entropy = entropy(y)
    
    # Get unique values for the feature
    unique_values = np.unique(X[:, feature_idx])
    
    # Calculate weighted entropy for each value
    weighted_entropy = 0
    for value in unique_values:
        mask = X[:, feature_idx] == value
        if np.sum(mask) > 0:
            weight = np.sum(mask) / len(y)
            weighted_entropy += weight * entropy(y[mask])
    
    return parent_entropy - weighted_entropy

# Calculate information gain for each feature
feature_names = ['Purchase_Frequency', 'Customer_Service_Rating', 'Purchase_Amount', 'Account_Age']
X_array = X_train.values

print("Information Gain for each feature:")
for i, feature in enumerate(feature_names):
    ig = information_gain(X_array, y_train.values, i)
    print(f"{feature}: {ig:.4f}")

# 8. Business costs analysis
print("\n8. Business Costs of Overfitting:")
print("-" * 50)

business_costs = [
    "False Positives: Unnecessary retention campaigns for customers who won't churn",
    "False Negatives: Missing high-risk customers who will actually churn",
    "Resource Misallocation: Spending on wrong customer segments",
    "Reduced Customer Trust: Irrelevant marketing messages",
    "Operational Inefficiency: Poor decision-making based on unreliable predictions",
    "Revenue Loss: Ineffective churn prevention strategies"
]

for i, cost in enumerate(business_costs, 1):
    print(f"{i}. {cost}")

# 9. Validation of pruning decisions
print("\n9. Validation of Pruning Decisions:")
print("-" * 50)

validation_methods = [
    "Cross-validation: Use k-fold CV to ensure pruning stability",
    "Holdout Set: Reserve a third dataset for final validation",
    "Business Metrics: Align with business KPIs and constraints",
    "Model Interpretability: Ensure business analysts can understand the tree",
    "Performance Stability: Check consistency across different time periods"
]

for i, method in enumerate(validation_methods, 1):
    print(f"{i}. {method}")

# 10. Recommendation for ≤4 nodes constraint
print("\n10. Recommendation for ≤4 Nodes Constraint:")
print("-" * 50)

print("Given the constraint of <=4 nodes for business analyst understanding:")
print("1. Use max_depth=2 (maximum 4 nodes: 1 root + 2 internal + 1 leaf)")
print("2. Focus on the most important features: Purchase_Frequency and Customer_Service_Rating")
print("3. Accept slightly lower accuracy for better interpretability")
print("4. Validate with business stakeholders on interpretability")

# Create final simplified tree
final_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
final_tree.fit(X_train, y_train)

final_train_acc = accuracy_score(y_train, final_tree.predict(X_train))
final_val_acc = accuracy_score(y_val, final_tree.predict(X_val))

print(f"\nFinal Simplified Tree (<=4 nodes):")
print(f"Training Accuracy: {final_train_acc:.3f}")
print(f"Validation Accuracy: {final_val_acc:.3f}")
print(f"Overfitting Gap: {final_train_acc - final_val_acc:.3f}")

# Plot final simplified tree
plt.figure(figsize=(10, 8))
plot_tree(final_tree, feature_names=['Purchase_Freq', 'Service_Rating', 'Purchase_Amount', 'Account_Age'], 
          class_names=['Stay', 'Leave'], filled=True, rounded=True, fontsize=12)
plt.title(f'Final Simplified Tree (≤4 nodes)\nTrain: {final_train_acc:.3f}, Val: {final_val_acc:.3f}')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'final_simplified_tree.png'), dpi=300, bbox_inches='tight')

# 11. Summary comparison
print("\n11. Summary Comparison of All Approaches:")
print("-" * 50)

comparison_data = {
    'Approach': ['Overfitted', 'Pre-pruned (depth=4)', 'Post-pruned', 'Simplified (<=4 nodes)'],
    'Training Acc': [train_acc, pre_train_acc, post_train_acc, final_train_acc],
    'Validation Acc': [val_acc, pre_val_acc, post_val_acc, final_val_acc],
    'Overfitting Gap': [train_acc - val_acc, pre_train_acc - pre_val_acc, 
                       post_train_acc - post_val_acc, final_train_acc - final_val_acc],
    'Complexity': ['Very High', 'Medium', 'Low', 'Very Low']
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False, float_format='%.3f'))

# Save comparison plot
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(comparison_data['Approach']))
width = 0.35

plt.bar(x_pos - width/2, comparison_data['Training Acc'], width, label='Training Accuracy', alpha=0.8)
plt.bar(x_pos + width/2, comparison_data['Validation Acc'], width, label='Validation Accuracy', alpha=0.8)

plt.xlabel('Pruning Approach')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy Comparison')
plt.xticks(x_pos, comparison_data['Approach'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')

print(f"\nAll plots saved to: {save_dir}")
print("\nAnalysis complete!")
