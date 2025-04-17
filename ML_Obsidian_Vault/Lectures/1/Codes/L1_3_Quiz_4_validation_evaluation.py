import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score, 
                           roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

# Attempt to import imblearn - optional dependency for conceptual plot
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. Skipping conceptual imbalance correction plot.")
    print("To install: pip install -U imbalanced-learn")

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_3_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def plot_data_splits(total_size=100):
    """Visualize train/validation/test splits."""
    fig, ax = plt.subplots(figsize=(10, 1.5))
    train_end = 0.6 * total_size
    val_end = 0.8 * total_size
    
    # Draw rectangles
    ax.add_patch(patches.Rectangle((0, 0), train_end, 1, color='skyblue', label='Train (e.g., 60%)'))
    ax.add_patch(patches.Rectangle((train_end, 0), val_end - train_end, 1, color='lightgreen', label='Validation (e.g., 20%)'))
    ax.add_patch(patches.Rectangle((val_end, 0), total_size - val_end, 1, color='salmon', label='Test (e.g., 20%)'))
    
    # Add text
    ax.text(train_end / 2, 0.5, 'Train Set', ha='center', va='center', color='black', fontsize=12)
    ax.text(train_end + (val_end - train_end) / 2, 0.5, 'Validation Set', ha='center', va='center', color='black', fontsize=12)
    ax.text(val_end + (total_size - val_end) / 2, 0.5, 'Test Set', ha='center', va='center', color='black', fontsize=12)
    
    ax.set_xlim(0, total_size)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Typical Data Splitting', fontsize=14)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "data_splits.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Data split visualization saved to: {file_path}")

def plot_kfold(n_splits=5, total_size=100):
    """Visualize K-Fold Cross-Validation."""
    fig, ax = plt.subplots(n_splits, 1, figsize=(10, 4), sharex=True)
    indices = np.arange(total_size)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (train_idx, val_idx) in enumerate(kf.split(indices)):
        fold_data = np.zeros(total_size)
        fold_data[train_idx] = 1 # Mark training data
        fold_data[val_idx] = 2 # Mark validation data

        cmap = plt.get_cmap('viridis', 3)
        ax[i].pcolormesh(fold_data.reshape(1, -1), cmap=cmap, edgecolors='k', linewidth=0.5)
        ax[i].set_yticks([])
        ax[i].set_ylabel(f'Fold {i+1}', rotation=0, ha='right', va='center', labelpad=20)
        if i == n_splits - 1:
            ax[i].set_xlabel('Data Index')

    # Create custom legend
    train_patch = patches.Patch(color=cmap(1.0/3.0), label='Training Data')
    val_patch = patches.Patch(color=cmap(2.0/3.0), label='Validation Data')
    fig.legend(handles=[train_patch, val_patch], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f'{n_splits}-Fold Cross-Validation', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    file_path = os.path.join(save_dir, "kfold_visualization.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"K-Fold visualization saved to: {file_path}")

def plot_roc_curve(y_true, y_prob, filename):
    """Plot ROC curve and calculate AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve plot saved to: {file_path}")
    return roc_auc

def plot_pr_curve(y_true, y_prob, filename):
    """Plot Precision-Recall curve and calculate Average Precision (AP)."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='blue', alpha=0.8, where='post',
             label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    # Calculate baseline (proportion of positive class)
    baseline = np.sum(y_true) / len(y_true)
    plt.plot([0, 1], [baseline, baseline], linestyle='--', color='red', label=f'No Skill Baseline ({baseline:.2f})')

    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best") # Changed loc
    plt.grid(True)
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve plot saved to: {file_path}")
    return average_precision

def plot_imbalance_correction_conceptual(filename="imbalance_correction_conceptual.png"):
    """Conceptual plot showing how imbalance techniques affect decision boundary."""
    if not IMBLEARN_AVAILABLE:
        print("Skipping conceptual imbalance plot as imbalanced-learn is not available.")
        return
        
    # Generate conceptual imbalanced data (needs only 2 features for plotting)
    X_imb, y_imb = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1,
                               weights=[0.95, 0.05], flip_y=0, random_state=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    
    # Plot original data and boundary using SVM for visualization
    ax = axes[0]
    model_orig = SVC(kernel='linear', probability=True, random_state=42).fit(X_imb, y_imb)
    plot_decision_boundary(X_imb, y_imb, model_orig, ax, title="Original Imbalanced Data")

    # Plot with SMOTE (oversampling)
    ax = axes[1]
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_imb, y_imb)
    model_smote = SVC(kernel='linear', probability=True, random_state=42).fit(X_smote, y_smote)
    plot_decision_boundary(X_smote, y_smote, model_smote, ax, title="After SMOTE Oversampling")

    # Plot with Class Weighting
    ax = axes[2]
    model_weighted = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42).fit(X_imb, y_imb)
    plot_decision_boundary(X_imb, y_imb, model_weighted, ax, title="Using Class Weights")

    plt.tight_layout()
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Conceptual imbalance correction plot saved to: {file_path}")

def plot_decision_boundary(X, y, model, ax, title):
    """Helper function to plot data points and decision boundary for 2D features."""
    # Scatter plot
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='k', s=30, zorder=2)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    
    # Create legend handles manually if needed (check number of classes)
    classes = np.unique(y)
    if len(classes) == 2:
      legend_handles = scatter.legend_elements()[0]
      ax.legend(handles=legend_handles, labels=[f'Class {c}' for c in classes], loc='upper right')

    # Create mesh grid
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    
    # Predict on grid
    try:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    except ValueError:
        print(f"Warning: Could not predict on meshgrid for {title}. Skipping boundary.")
        return 
        
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins (if SVC)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2, zorder=1)
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-', linewidths=1, zorder=3)
    
    ax.grid(True, linestyle='--', alpha=0.6)


# --- Step 1: Data Splitting ---
print_step_header(1, "Data Splitting: Train, Validation, Test Sets")

print("Purpose: To reliably estimate the model's performance on unseen data and tune hyperparameters.")
print("- Training Set:")
print("  - Used to train the model (learn parameters like weights).")
print("  - Typically the largest portion (e.g., 60-80%).")
print("- Validation Set (or Development Set):")
print("  - Used to tune model hyperparameters (e.g., learning rate, regularization strength, model architecture) and perform model selection.")
print("  - Provides an unbiased estimate of performance during development.")
print("  - Typically smaller (e.g., 10-20%).")
print("- Test Set:")
print("  - Used *only once* at the very end to get a final, unbiased estimate of the chosen model\'s performance on completely unseen data.")
print("  - Should not be used for any training or tuning.")
print("  - Typically similar size to the validation set (e.g., 10-20%).")

plot_data_splits()

# --- Step 2: K-Fold Cross-Validation ---
print_step_header(2, "K-Fold Cross-Validation")

print("K-Fold Cross-Validation is a technique to evaluate model performance and tune hyperparameters, especially when data is limited.")
print("How it works:")
print("1. Split the *training data* (excluding the final test set) into K equal-sized folds (e.g., K=5 or 10).")
print("2. For each fold k from 1 to K:")
print("   - Use fold k as the validation set.")
print("   - Use the remaining K-1 folds as the training set.")
print("   - Train the model on the K-1 folds and evaluate it on fold k.")
print("3. Calculate the average performance metric across the K folds. This average is a more robust estimate of the model\'s performance than a single train-validation split.")
print()
print("Why it\'s preferred over a simple train-test split (for tuning/evaluation):")
print("- More Robust Evaluation: Performance estimate is averaged over K different validation sets, reducing variability.")
print("- Better Data Utilization: Every data point gets used for both training and validation across the K iterations.")
print("- Less Sensitive to Specific Split: Reduces the chance that a lucky or unlucky single validation split gives a misleading performance estimate.")
print("(Note: A final, held-out test set is STILL needed for the ultimate performance evaluation after tuning with CV).")

plot_kfold(n_splits=5)

# --- Step 3: Diagnosing Overfitting ---
print_step_header(3, "Diagnosing Overfitting (High Train Acc, Low Test Acc)")

print("Scenario: Model achieves 95% accuracy on training data, but only 75% on test data.")
print("Diagnosis: This is a classic sign of **Overfitting**.")
print("- The large gap between training performance (95%) and test performance (75%) indicates the model has learned the training data too well, including its noise and specific patterns, but fails to generalize to new, unseen data.")
print("- The model has high variance.")
print()
print("Strategies to Address Overfitting:")
print("- Get More Training Data: Often the most effective way to improve generalization.")
print("- Simplify the Model: Reduce complexity (e.g., fewer layers/neurons in NN, shallower trees, lower polynomial degree).")
print("- Apply Regularization: Use L1 (Lasso) or L2 (Ridge) regularization to penalize large weights.")
print("- Use Dropout (for Neural Networks): Randomly deactivate neurons during training.")
print("- Early Stopping: Monitor validation performance during training and stop when it starts to degrade.")
print("- Feature Selection/Engineering: Remove irrelevant features or create more robust ones.")
print("- Data Augmentation: Create more training examples by transforming existing ones (e.g., image rotations, flips).")

# --- Step 4: Handling Imbalanced Datasets ---
print_step_header(4, "Handling Imbalanced Datasets (e.g., 5% Churn)")

print("Problem: Classification dataset where one class (e.g., churners) is much rarer than the other (e.g., non-churners).")
print("Impact on Evaluation Metrics:")
print("- Accuracy becomes misleading: A model predicting \"no churn\" for everyone would achieve 95% accuracy but be useless for identifying churners.")
print("- Standard metrics (like accuracy) don\'t reflect performance on the minority class, which is often the class of interest.")
print()
print("Better Evaluation Metrics for Imbalance:")
print("- Confusion Matrix: Visualizes True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN).")
print("- Precision (Positive Predictive Value): TP / (TP + FP). Of those predicted positive, how many actually are? (Focuses on avoiding false alarms).")
print("- Recall (Sensitivity, True Positive Rate): TP / (TP + FN). Of all actual positives, how many were correctly identified? (Focuses on finding all positives).")
print("- F1-Score: Harmonic mean of Precision and Recall (2 * Precision * Recall) / (Precision + Recall). Balances precision and recall.")
print("- Area Under the ROC Curve (AUC-ROC): Measures the model\'s ability to distinguish between classes across different thresholds.")
print("- Precision-Recall Curve (AUC-PR): Often more informative than ROC for highly imbalanced datasets.")
print()
print("Techniques to Address Imbalance:")
print("- Resampling Techniques:")
print("  - Undersampling: Remove samples from the majority class.")
print("  - Oversampling: Create copies or synthetic samples (e.g., SMOTE - Synthetic Minority Over-sampling Technique) of the minority class.")
print("- Algorithmic Approaches:")
print("  - Cost-Sensitive Learning: Assign higher misclassification costs to the minority class during training.")
print("  - Class Weights: Adjust model algorithms to give more weight to the minority class (e.g., `class_weight='balanced'` in scikit-learn).")
print("- Use Different Models: Some models like tree-based ensembles (Random Forest, Gradient Boosting) can sometimes handle imbalance better.")
print("- Anomaly Detection Approach: Treat the minority class detection as an anomaly detection problem.")

# Generate imbalanced data and show confusion matrix
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
                           n_redundant=10, n_clusters_per_class=1,
                           weights=[0.95, 0.05], flip_y=0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # Stratify is important!

# Train a simple model and a dummy model
model = LogisticRegression(solver='liblinear', random_state=42) # Added random state
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)
y_pred_dummy = dummy_model.predict(X_test)

# Get probabilities for ROC/PR curves (need probability of the positive class, class 1)
try:
    y_prob = model.predict_proba(X_test)[:, 1]
except AttributeError:
    print("Warning: Model does not support predict_proba. ROC/PR curves skipped.")
    y_prob = None

# Calculate metrics
acc_model = accuracy_score(y_test, y_pred)
acc_dummy = accuracy_score(y_test, y_pred_dummy)
prec_model = precision_score(y_test, y_pred, zero_division=0)
rec_model = recall_score(y_test, y_pred, zero_division=0)
f1_model = f1_score(y_test, y_pred, zero_division=0)
roc_auc_model = None
ap_model = None
if y_prob is not None:
    roc_auc_model = roc_auc_score(y_test, y_prob)
    ap_model = average_precision_score(y_test, y_prob)

print(f"\nExample Metrics on Imbalanced Data (95% vs 5%):")
print(f"- Dummy Classifier (Predicts Majority Class) Accuracy: {acc_dummy:.4f}")
print(f"- Logistic Regression Accuracy: {acc_model:.4f}")
print(f"- Logistic Regression Precision: {prec_model:.4f}")
print(f"- Logistic Regression Recall: {rec_model:.4f}")
print(f"- Logistic Regression F1-Score: {f1_model:.4f}")
if roc_auc_model is not None:
    print(f"- Logistic Regression AUC-ROC: {roc_auc_model:.4f}")
if ap_model is not None:
     print(f"- Logistic Regression Average Precision (AUC-PR): {ap_model:.4f}")
print("-> Note how accuracy is high for both, but other metrics reveal struggles with the minority class.")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Churn (0)', 'Churn (1)'])

fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Confusion Matrix (Imbalanced Data Example)', fontsize=14)
plt.tight_layout()

file_path = os.path.join(save_dir, "confusion_matrix_imbalanced.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nConfusion matrix plot saved to: {file_path}")
print("- The matrix clearly shows the model correctly identifies many non-churners (TN) but misses many actual churners (FN). The number of FP might also be significant relative to TP.") # Updated description slightly

# Plot ROC and PR curves if probabilities are available
if y_prob is not None:
    plot_roc_curve(y_test, y_prob, "roc_curve_imbalanced.png")
    plot_pr_curve(y_test, y_prob, "pr_curve_imbalanced.png")

# Add conceptual plot for imbalance correction
print("\nGenerating conceptual plot for imbalance correction techniques...")
try:
    plot_imbalance_correction_conceptual()
except NameError as e:
    print(f"Skipping conceptual plot due to missing dependency: {e}") # Handle potential imblearn/svm missing
except Exception as e:
    print(f"Skipping conceptual plot due to error: {e}")

print("\nScript finished. Plots saved in:", save_dir) 