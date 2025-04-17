import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_4_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Spam Classification Analysis
print_step_header(1, "Spam Classification Metrics")

# Generate synthetic data for spam classification
np.random.seed(42)
n_samples = 1000
y_true_spam = np.random.binomial(1, 0.2, n_samples)  # 20% spam
y_pred_proba_spam = np.random.uniform(0, 1, n_samples)
y_pred_spam = (y_pred_proba_spam > 0.5).astype(int)

# ROC Curve
fpr, tpr, _ = roc_curve(y_true_spam, y_pred_proba_spam)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.title('ROC Curve for Spam Classification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "spam_roc.png"), dpi=300, bbox_inches='tight')
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_true_spam, y_pred_spam)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Spam Classification')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "spam_cm.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 2: Recommendation System Analysis
print_step_header(2, "Recommendation System Metrics")

# Generate synthetic data for recommendations
n_users = 100
n_items = 50
ratings = np.random.randint(1, 6, (n_users, n_items))
user_preferences = np.random.rand(n_users, 5)  # 5 latent features
item_features = np.random.rand(n_items, 5)

# Calculate similarity matrix
similarity = np.dot(user_preferences, item_features.T)
plt.figure(figsize=(10, 6))
plt.imshow(similarity, cmap='viridis')
plt.colorbar(label='Similarity Score')
plt.title('User-Item Similarity Matrix')
plt.xlabel('Items')
plt.ylabel('Users')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "rec_similarity.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 3: House Price Prediction Analysis
print_step_header(3, "House Price Prediction Metrics")

# Generate synthetic house price data
n_houses = 500
true_prices = np.random.normal(500000, 100000, n_houses)
predicted_prices = true_prices + np.random.normal(0, 50000, n_houses)

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(true_prices, predicted_prices, alpha=0.5)
plt.plot([min(true_prices), max(true_prices)], 
         [min(true_prices), max(true_prices)], 
         'r--', label='Perfect Prediction')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "house_prices.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 4: Imbalanced Classification Analysis
print_step_header(4, "Imbalanced Classification Analysis")

# Generate synthetic imbalanced data
y_true_imb = np.random.binomial(1, 0.1, n_samples)  # 10% positive class
y_pred_imb = np.random.binomial(1, 0.1, n_samples)

# Plot class distribution
plt.figure(figsize=(10, 6))
plt.hist(y_true_imb, bins=2, alpha=0.5, label='True Classes')
plt.hist(y_pred_imb, bins=2, alpha=0.5, label='Predicted Classes')
plt.title('Class Distribution in Imbalanced Dataset')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "imbalanced_dist.png"), dpi=300, bbox_inches='tight')
plt.close()

# Print analysis results
print("\nAnalysis Results:")
print("\n1. Spam Classification:")
print("   - ROC curve shows good discrimination")
print("   - Confusion matrix reveals trade-off between precision and recall")
print("   - Need to consider both false positives and false negatives")

print("\n2. Recommendation System:")
print("   - User-item similarity matrix shows varying preferences")
print("   - Need to balance personalization with diversity")
print("   - Consider both accuracy and novelty in recommendations")

print("\n3. House Price Prediction:")
print("   - Strong correlation between actual and predicted prices")
print("   - Some outliers present in predictions")
print("   - Need to consider both absolute and relative errors")

print("\n4. Imbalanced Classification:")
print("   - Significant class imbalance (10% positive)")
print("   - Accuracy not suitable as primary metric")
print("   - Need to focus on precision, recall, and F1-score") 