import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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

def statement6_cross_validation():
    """
    Statement 6: Cross-validation is a resampling technique used to assess the 
    performance of a model on unseen data.
    """
    print("\n==== Statement 6: Cross-Validation for Model Assessment ====")
    
    # Print explanation of cross-validation
    print("\nCross-Validation Explained:")
    print("- Cross-validation (CV) is a resampling technique for model evaluation")
    print("- It repeatedly divides data into training and validation sets")
    print("- Each data point is used for both training and validation")
    print("- Performance metrics are averaged across all iterations")
    print("- This provides a more reliable estimate of model performance on unseen data")
    
    # Generate data
    np.random.seed(42)
    n_samples = 100
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = 2 * X.squeeze() + np.random.normal(0, 5, n_samples)
    
    # Perform cross-validation
    cv_folds = 5
    cv_scores = cross_val_score(LinearRegression(), X, y, cv=cv_folds, scoring='r2')
    
    print("\nCross-Validation Results:")
    print(f"Cross-validation R-squared scores (5-fold): {np.round(cv_scores, 3)}")
    print(f"Mean R-squared: {np.mean(cv_scores):.2f}")
    print(f"Standard deviation of R-squared: {np.std(cv_scores):.2f}")
    
    # Compare with train-test split
    print("\nComparing CV with Traditional Train-Test Split:")
    train_test_scores = []
    n_splits = 5
    
    for i in range(n_splits):
        # Different random split each time
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        model = LinearRegression().fit(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test))
        train_test_scores.append(score)
    
    print(f"Train-test split R-squared scores (5 random splits): {np.round(train_test_scores, 3)}")
    print(f"Mean R-squared: {np.mean(train_test_scores):.2f}")
    print(f"Standard deviation of R-squared: {np.std(train_test_scores):.2f}")
    
    print("\nComparison Analysis:")
    print(f"CV standard deviation: {np.std(cv_scores):.3f} vs. Train-test standard deviation: {np.std(train_test_scores):.3f}")
    print(f"CV range: {np.max(cv_scores) - np.min(cv_scores):.3f} vs. Train-test range: {np.max(train_test_scores) - np.min(train_test_scores):.3f}")
    print(f"Conclusion: {'CV generally provides more stable metrics' if np.std(cv_scores) < np.std(train_test_scores) else 'In this case, train-test split was more stable'}")
    
    # Generate a smaller dataset for visualization of CV folds
    n_viz_samples = 20
    X_viz = np.linspace(0, 10, n_viz_samples).reshape(-1, 1)
    y_viz = 2 * X_viz.squeeze() + np.random.normal(0, 5, n_viz_samples)
    
    # Plot 1: Simplified visualization of cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    plt.figure(figsize=(10, 8))
    
    # Create a matrix to visualize fold assignments
    # Rows = folds, Columns = samples
    fold_matrix = np.zeros((cv_folds, n_viz_samples))
    
    # Fill the matrix: 1 for test, 0.5 for train
    for i, (train_idx, test_idx) in enumerate(kf.split(X_viz)):
        fold_matrix[i, :] = 0.5  # All samples start as training
        fold_matrix[i, test_idx] = 1  # Validation samples
    
    # Create a heatmap
    plt.imshow(fold_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Sample Usage (Training/Validation)')
    
    # Add annotations to cells
    for i in range(cv_folds):
        for j in range(n_viz_samples):
            if fold_matrix[i, j] == 1:
                text = 'Val'
            else:
                text = 'Train'
            plt.text(j, i, text, ha="center", va="center", color="black", fontsize=7)
    
    plt.xlabel('Sample Index')
    plt.ylabel('Fold Number')
    plt.title('5-Fold Cross-Validation: Data Usage Pattern')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'statement6_cv_pattern.png'), dpi=300, bbox_inches='tight')
    
    # Print explanation of the k-fold cross-validation technique
    print("\nHow K-Fold Cross-Validation Works:")
    print("1. Data is divided into K equally sized folds")
    print("2. For each fold i from 1 to K:")
    print("   - Use fold i as the validation set")
    print("   - Use all other K-1 folds as the training set")
    print("   - Train the model and evaluate performance on the validation set")
    print("3. Average the K performance measurements")
    
    # Plot 2: Score comparison between CV and train-test
    plt.figure(figsize=(10, 6))
    
    # Prepare data for the plot
    all_scores = np.array([cv_scores, train_test_scores])
    labels = ['Cross-Validation', 'Train-Test Split']
    
    # Create boxplot
    box = plt.boxplot([cv_scores, train_test_scores], labels=labels, patch_artist=True)
    
    # Color boxes
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add individual points
    for i, scores in enumerate([cv_scores, train_test_scores]):
        # Add jitter to x-position
        x = np.random.normal(i+1, 0.05, size=len(scores))
        plt.scatter(x, scores, alpha=0.7, color='blue', marker='o')
    
    # Add mean lines
    for i, scores in enumerate([cv_scores, train_test_scores]):
        plt.axhline(y=np.mean(scores), xmin=(i)/2, xmax=(i+1)/2, 
                   color='red', linestyle='--', linewidth=2)
    
    plt.title('Comparison of Model Evaluation Methods', fontsize=14)
    plt.ylabel('R-squared Score', fontsize=12)
    plt.grid(True, axis='y')
    
    # Add annotation with LaTeX formula
    plt.figtext(0.2, 0.01, r'$R^2_{CV} = \frac{1}{K}\sum_{i=1}^{K} R^2_i$', fontsize=12)
    
    plt.savefig(os.path.join(save_dir, 'statement6_method_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Plot 3: Visualization comparing stability of the methods
    plt.figure(figsize=(10, 6))
    
    # Generate data for multiple runs
    n_runs = 30
    cv_run_scores = []
    tt_run_scores = []
    
    for run in range(n_runs):
        # Generate slightly different data each time
        X_run = np.linspace(0, 10, n_samples).reshape(-1, 1)
        y_run = 2 * X_run.squeeze() + np.random.normal(0, 5, n_samples)
        
        # Cross-validation score
        cv_run_score = np.mean(cross_val_score(LinearRegression(), X_run, y_run, cv=cv_folds, scoring='r2'))
        cv_run_scores.append(cv_run_score)
        
        # Train-test split score
        X_tr, X_te, y_tr, y_te = train_test_split(X_run, y_run, test_size=0.2, random_state=run)
        tt_model = LinearRegression().fit(X_tr, y_tr)
        tt_run_score = r2_score(y_te, tt_model.predict(X_te))
        tt_run_scores.append(tt_run_score)
    
    # Create plot
    plt.plot(range(1, n_runs+1), cv_run_scores, 'o-', color='blue', label='Cross-Validation')
    plt.plot(range(1, n_runs+1), tt_run_scores, 'o-', color='green', label='Train-Test Split')
    
    # Add mean lines
    plt.axhline(y=np.mean(cv_run_scores), color='blue', linestyle='--', 
               label=f'CV Mean: {np.mean(cv_run_scores):.2f}')
    plt.axhline(y=np.mean(tt_run_scores), color='green', linestyle='--',
               label=f'TT Mean: {np.mean(tt_run_scores):.2f}')
    
    plt.xlabel('Simulation Run', fontsize=12)
    plt.ylabel('R-squared Score', fontsize=12)
    plt.title('Stability Comparison: Cross-Validation vs. Train-Test Split', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    # Add text with standard deviations
    plt.annotate(f'CV Std: {np.std(cv_run_scores):.3f}', xy=(0.05, 0.05), xycoords='axes fraction', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.annotate(f'TT Std: {np.std(tt_run_scores):.3f}', xy=(0.05, 0.12), xycoords='axes fraction', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.savefig(os.path.join(save_dir, 'statement6_stability_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Print the main advantages of cross-validation
    print("\nKey Advantages of Cross-Validation:")
    print("1. EFFICIENCY: Uses all data for both training and validation")
    print("2. RELIABILITY: Provides a more robust performance estimate")
    print("3. GENERALIZATION: Better reflects how model will perform on unseen data")
    print("4. VARIANCE REDUCTION: Averaging multiple evaluations reduces estimate variance")
    print("5. OVERFITTING DETECTION: Helps identify if model is memorizing instead of learning")
    
    result = {
        'statement': "Cross-validation is a resampling technique used to assess the performance of a model on unseen data.",
        'is_true': True,
        'explanation': "This statement is TRUE. Cross-validation is a resampling method that provides a more reliable estimate of model performance on unseen data compared to a single train-test split. It works by dividing the dataset into multiple subsets or 'folds,' training the model on some folds, and validating it on the remaining folds. This process is repeated multiple times with different fold combinations, and the performance metrics are averaged. Cross-validation helps detect overfitting and provides a more robust assessment of how well a model will generalize to new, unseen data.",
        'image_path': ['statement6_cv_pattern.png', 'statement6_method_comparison.png', 'statement6_stability_comparison.png']
    }
    
    return result

if __name__ == "__main__":
    result = statement6_cross_validation()
    print(f"\nStatement: {result['statement']}")
    print(f"True or False: {'True' if result['is_true'] else 'False'}")
    print(f"Explanation: {result['explanation']}")
    print(f"Images saved: {', '.join(result['image_path'])}") 