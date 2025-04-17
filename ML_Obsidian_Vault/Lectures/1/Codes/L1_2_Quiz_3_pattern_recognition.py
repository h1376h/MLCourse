import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_2_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Introduction to Pattern Recognition
print_step_header(1, "Introduction to Pattern Recognition")

print("Pattern recognition is a fundamental aspect of machine learning that involves")
print("identifying and categorizing patterns or regularities in data.")
print("\nWe'll explore four different pattern recognition scenarios:")
print("1. Identifying handwritten digits (0-9)")
print("2. Detecting fraudulent credit card transactions")
print("3. Recognizing emotions in facial expressions")
print("4. Identifying seasonal patterns in retail sales data")
print("\nFor each scenario, we'll discuss the appropriate approach and techniques.")

# Step 2: Handwritten Digit Recognition
print_step_header(2, "Handwritten Digit Recognition")

print("Approach for handwritten digit recognition:")
print("\n1. Problem Type: Image Classification (Multi-class)")
print("2. Key Challenges:")
print("   - Variations in handwriting styles")
print("   - Different stroke widths and orientations")
print("   - Noise and image quality issues")
print("\n3. Suitable Algorithms:")
print("   - Convolutional Neural Networks (CNNs)")
print("   - Support Vector Machines (SVMs)")
print("   - K-Nearest Neighbors (KNN)")
print("\n4. Feature Engineering:")
print("   - Raw pixel values")
print("   - Edge detection features")
print("   - HOG (Histogram of Oriented Gradients)")
print("\n5. Preprocessing Steps:")
print("   - Normalization and centering")
print("   - Noise reduction")
print("   - Image size standardization")

# Visualize handwritten digit recognition
# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Create separate figures for digit recognition

# Figure 1: Sample digits
plt.figure(figsize=(8, 8))
fig_indices = np.random.choice(len(X), 25, replace=False)
for i, idx in enumerate(fig_indices[:25]):
    plt.subplot(5, 5, i + 1)
    plt.imshow(digits.images[idx], cmap='gray')
    plt.title(f"Digit: {y[idx]}")
    plt.axis('off')
plt.suptitle('Sample Handwritten Digits', fontsize=14)
plt.tight_layout()
file_path = os.path.join(save_dir, "digit_samples.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Figure 2: Feature visualization using PCA
plt.figure(figsize=(10, 6))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for i in range(10):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=colors[i], 
                alpha=0.6, label=f"Digit {i}")
plt.title('PCA: Digits Projected to 2D', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
file_path = os.path.join(save_dir, "digit_pca.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Figure 3: CNN architecture visualization
plt.figure(figsize=(12, 6))
ax = plt.gca()
ax.axis('off')

# Draw CNN architecture
def draw_layer(ax, x, y, width, height, color, text):
    rect = plt.Rectangle((x, y), width, height, facecolor=color, alpha=0.7, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', color='black')

# Simplified CNN architecture
# Input layer
draw_layer(ax, 0.05, 0.2, 0.1, 0.6, '#a1dab4', 'Input\n28x28\nImage')

# Convolutional layers
draw_layer(ax, 0.2, 0.15, 0.1, 0.7, '#41b6c4', 'Conv\n3x3\nFilters')
draw_layer(ax, 0.35, 0.1, 0.1, 0.8, '#41b6c4', 'Conv\n3x3\nFilters')

# Pooling layer
draw_layer(ax, 0.5, 0.2, 0.1, 0.6, '#2c7fb8', 'Max\nPooling\n2x2')

# Fully connected layers
draw_layer(ax, 0.65, 0.3, 0.1, 0.4, '#253494', 'Fully\nConnected\n128')
draw_layer(ax, 0.8, 0.35, 0.1, 0.3, '#253494', 'Output\n10\nClasses')

# Add arrows
arrows = [
    (0.15, 0.5, 0.2, 0.5),
    (0.3, 0.5, 0.35, 0.5),
    (0.45, 0.5, 0.5, 0.5),
    (0.6, 0.5, 0.65, 0.5),
    (0.75, 0.5, 0.8, 0.5)
]

for start_x, start_y, end_x, end_y in arrows:
    ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

plt.title('Convolutional Neural Network Architecture for Digit Recognition', fontsize=14)
plt.tight_layout()
file_path = os.path.join(save_dir, "digit_cnn_architecture.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Fraud Detection
print_step_header(3, "Fraud Detection in Credit Card Transactions")

print("Approach for fraud detection in credit card transactions:")
print("\n1. Problem Type: Binary Classification with Class Imbalance")
print("2. Key Challenges:")
print("   - Highly imbalanced data (most transactions are legitimate)")
print("   - Need for real-time detection")
print("   - Evolving fraud patterns")
print("   - High cost of false positives and false negatives")
print("\n3. Suitable Algorithms:")
print("   - Gradient Boosting (XGBoost, LightGBM)")
print("   - Random Forests")
print("   - Anomaly detection methods")
print("   - Deep learning with attention mechanisms")
print("\n4. Feature Engineering:")
print("   - Transaction amount, time, location")
print("   - Customer historical behavior")
print("   - Velocity features (rate of transactions)")
print("   - Derived features (e.g., deviation from normal spending)")
print("\n5. Handling Class Imbalance:")
print("   - SMOTE (Synthetic Minority Over-sampling Technique)")
print("   - Class weighting")
print("   - Anomaly detection approaches")
print("   - Ensemble methods")

# Visualize fraud detection
# Create a synthetic dataset for fraud detection
np.random.seed(42)
n_samples = 1000
n_fraud = 50  # 5% fraud rate

# Create features
transaction_amounts = np.random.exponential(scale=50, size=n_samples)
transaction_amounts[n_samples-n_fraud:] *= 2  # Fraudulent transactions tend to be larger

# Time of day (normalized to 0-1)
time_of_day = np.random.beta(2, 2, size=n_samples)
time_of_day[n_samples-n_fraud:] = np.random.beta(1, 3, size=n_fraud)  # Fraudulent more likely at night

# Distance from home
distance = np.random.gamma(1, scale=10, size=n_samples)
distance[n_samples-n_fraud:] *= 3  # Fraudulent more likely to be far from home

# Create labels (0 = legitimate, 1 = fraud)
labels = np.zeros(n_samples)
labels[n_samples-n_fraud:] = 1

# Create the dataset
X = np.column_stack((transaction_amounts, time_of_day * 24, distance))  # Convert time to hours
y = labels.astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Figure 1: Transaction amount vs. time of day
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 1], X[:, 0], c=y, alpha=0.6, 
                   cmap=ListedColormap(['#1f77b4', '#d62728']),
                   s=np.log1p(X[:, 0])*5)
plt.title('Transaction Amount vs. Time of Day', fontsize=12)
plt.xlabel('Time of Day (hour)')
plt.ylabel('Transaction Amount ($)')
plt.xlim(0, 24)
legend1 = plt.legend(*scatter.legend_elements(), title="Transaction Type")
plt.gca().add_artist(legend1)
plt.tight_layout()
file_path = os.path.join(save_dir, "fraud_amount_time.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Figure 2: Transaction amount vs. distance from home
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 2], X[:, 0], c=y, alpha=0.6, 
                   cmap=ListedColormap(['#1f77b4', '#d62728']),
                   s=np.log1p(X[:, 0])*5)
plt.title('Transaction Amount vs. Distance from Home', fontsize=12)
plt.xlabel('Distance from Home (miles)')
plt.ylabel('Transaction Amount ($)')
legend2 = plt.legend(*scatter.legend_elements(), title="Transaction Type")
plt.gca().add_artist(legend2)
plt.tight_layout()
file_path = os.path.join(save_dir, "fraud_amount_distance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Figure 3: Decision boundary
# Train a simple classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train[:, [1, 2]], y_train)  # Using only time and distance

# Create a mesh grid for visualization
h = 0.5
x_min, x_max = 0, 24
y_min, y_max = 0, max(X[:, 2]) + 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict probabilities on the mesh grid
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
contour = plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r)
plt.scatter(X[:, 1], X[:, 2], c=y, alpha=0.8, 
           cmap=ListedColormap(['#1f77b4', '#d62728']))
plt.title('Decision Boundary: Time vs. Distance', fontsize=12)
plt.xlabel('Time of Day (hour)')
plt.ylabel('Distance from Home (miles)')
plt.xlim(0, 24)
plt.colorbar(contour, label='Fraud Probability')
plt.tight_layout()
file_path = os.path.join(save_dir, "fraud_decision_boundary.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Figure 4: Feature importance
# Train a classifier on all features
full_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
full_clf.fit(X_train, y_train)
importances = full_clf.feature_importances_
feature_names = ['Transaction Amount', 'Time of Day', 'Distance from Home']

plt.figure(figsize=(8, 6))
indices = np.argsort(importances)
plt.barh(range(len(indices)), importances[indices], color='#2ca02c')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title('Feature Importance for Fraud Detection', fontsize=12)
plt.xlabel('Relative Importance')
plt.tight_layout()
file_path = os.path.join(save_dir, "fraud_feature_importance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")
print(f"Figure saved to: {file_path}")

# Create a partial file, to be continued in part 2