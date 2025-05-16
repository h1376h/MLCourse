import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.model_selection import KFold, LeaveOneOut
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_6_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('ggplot')
sns.set_style("whitegrid")
np.random.seed(42)  # For reproducibility

def statement1_kfold_vs_loo():
    """
    Statement 1: K-fold cross-validation with K=n (where n is the number of samples) 
    is equivalent to leave-one-out cross-validation.
    """
    print("\n==== Statement 1: K-fold with K=n vs. Leave-One-Out ====")
    
    # Generate a small dataset for demonstration
    n_samples = 10
    X = np.arange(n_samples).reshape(-1, 1)
    y = 2 * X.squeeze() + np.random.normal(0, 1, n_samples)
    
    # K-fold with K=n
    kf = KFold(n_splits=n_samples, shuffle=False)
    kfold_indices = [(train_idx, test_idx) for train_idx, test_idx in kf.split(X)]
    
    # Leave-One-Out
    loo = LeaveOneOut()
    loo_indices = [(train_idx, test_idx) for train_idx, test_idx in loo.split(X)]
    
    # Compare the indices
    are_equal = all(
        np.array_equal(kf_train, loo_train) and np.array_equal(kf_test, loo_test)
        for (kf_train, kf_test), (loo_train, loo_test) in zip(kfold_indices, loo_indices)
    )
    
    print(f"K-fold with K=n is equivalent to Leave-One-Out: {are_equal}")
    
    # Print the first few splits to demonstrate equivalence textually
    print("\nComparing the first 3 splits:")
    for i in range(min(3, n_samples)):
        print(f"Split {i+1}:")
        print(f"  KFold train indices: {kfold_indices[i][0]}")
        print(f"  LOO train indices:   {loo_indices[i][0]}")
        print(f"  KFold test indices:  {kfold_indices[i][1]}")
        print(f"  LOO test indices:    {loo_indices[i][1]}")

    # Print mathematical explanation
    print("\nMathematical explanation:")
    print("In K-fold cross-validation with K=n (where n is the number of samples):")
    print("- The data is split into n folds, each containing exactly 1 sample")
    print("- For each fold i, we train on (n-1) samples and test on the remaining 1 sample")
    print("- Every sample is used exactly once as a test sample")
    print("- This is mathematically identical to Leave-One-Out Cross-Validation (LOOCV)")
    print("- In both methods, the training sets are exactly (n-1) samples and test sets are 1 sample")
    
    # Visualize the comparison (simplified)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plt.subplots_adjust(hspace=0.4)
    
    # Function to plot the fold splits (simplified)
    def plot_cv_indices(cv, X, ax, title):
        # Create a matrix to visualize folds
        n = len(X)
        fold_matrix = np.zeros((n, n))
        
        # For each fold, set the test sample
        for i, (_, test_idx) in enumerate(cv.split(X)):
            fold_matrix[i, test_idx[0]] = 1
        
        # Plot the matrix with grayscale
        im = ax.imshow(fold_matrix, cmap='gray', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Fold Number')
        ax.set_yticks(range(n))
        ax.set_xticks(range(n))
    
    # Plot K-fold with K=n
    plot_cv_indices(kf, X, axes[0], f'K-fold Cross-Validation with K={n_samples}')
    
    # Plot Leave-One-Out
    plot_cv_indices(loo, X, axes[1], 'Leave-One-Out Cross-Validation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement1_kfold_vs_loo.png'), dpi=300, bbox_inches='tight')
    
    # Add a clearer visualization to demonstrate what happens in folds
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Explanation text
    print("\nIn both K-fold CV with K=n and LOOCV:")
    print("- We perform n different train-test splits")
    print("- In each iteration, we leave out exactly one sample for testing")
    print("- We train the model on the remaining n-1 samples")
    print("- This process is repeated n times so that each sample is used once for testing")
    print("- The final performance is the average of the n test results")
    
    # Create visual representation of the first 3 folds
    n_folds_to_show = 3
    n_samples_to_show = n_samples
    
    # Create a matrix to show which samples are in train/test for each fold
    fold_matrix = np.zeros((n_folds_to_show, n_samples_to_show))
    
    # Fill the matrix: 1 for test, 0.5 for train
    for i in range(n_folds_to_show):
        fold_matrix[i, :] = 0.3  # All samples start as training (light gray)
        fold_matrix[i, i] = 0.9  # The ith sample becomes test for fold i (dark gray)
    
    # Create a simpler grayscale heatmap
    im = ax.imshow(fold_matrix, cmap='gray', aspect='auto', vmin=0, vmax=1)
    
    # Annotate cells
    for i in range(n_folds_to_show):
        for j in range(n_samples_to_show):
            if fold_matrix[i, j] > 0.5:
                text = 'Test'
            else:
                text = 'Train'
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    # Add labels
    ax.set_title('First 3 Folds: Sample Usage in K-fold (K=n) and LOOCV')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Fold Number')
    ax.set_yticks(range(n_folds_to_show))
    ax.set_xticks(range(n_samples_to_show))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement1_fold_visualization.png'), dpi=300, bbox_inches='tight')
    
    # NEW VISUALIZATION: Show concrete examples of model fits for different folds
    # Generate a more visually intuitive dataset
    np.random.seed(42)
    X_visual = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y_visual = 2 * X_visual.squeeze() + np.random.normal(0, 1, n_samples)
    
    # Create a figure showing how models are fit in different folds
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Show full dataset in first subplot
    axes[0].scatter(X_visual, y_visual, color='black', alpha=0.7)
    axes[0].set_title('Full Dataset')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    
    # Show 3 different folds and the resulting models
    for i, ax in enumerate(axes[1:]):
        if i < 3:  # Show first 3 folds
            train_idx = kfold_indices[i][0]
            test_idx = kfold_indices[i][1]
            
            # Plot training and test points
            ax.scatter(X_visual[train_idx], y_visual[train_idx], color='blue', alpha=0.7, label='Training')
            ax.scatter(X_visual[test_idx], y_visual[test_idx], color='red', s=100, label='Test')
            
            # Fit a simple linear model to the training data
            coeffs = np.polyfit(X_visual[train_idx].squeeze(), y_visual[train_idx], 1)
            x_line = np.linspace(0, 10, 100)
            y_line = coeffs[0] * x_line + coeffs[1]
            
            # Plot the model fit
            ax.plot(x_line, y_line, 'g-', linewidth=2, label='Model Fit')
            
            ax.set_title(f'Fold {i+1}: Test on Sample {test_idx[0]}')
            ax.set_xlabel('X')
            ax.set_ylabel('y')
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement1_kfold_vs_loo_examples.png'), dpi=300, bbox_inches='tight')
    
    result = {
        'statement': "K-fold cross-validation with K=n (where n is the number of samples) is equivalent to leave-one-out cross-validation.",
        'is_true': True,
        'explanation': "When K equals the number of samples (n), K-fold cross-validation creates n folds, each with n-1 training samples and 1 test sample. This is identical to leave-one-out cross-validation, as demonstrated by the identical fold assignments in the visualization.",
        'image_path': ['statement1_kfold_vs_loo.png', 'statement1_fold_visualization.png', 'statement1_kfold_vs_loo_examples.png']
    }
    
    return result

if __name__ == "__main__":
    result = statement1_kfold_vs_loo()
    print(f"\nStatement: {result['statement']}")
    print(f"True or False: {'True' if result['is_true'] else 'False'}")
    print(f"Explanation: {result['explanation']}")
    print(f"Images saved: {', '.join(result['image_path'])}") 