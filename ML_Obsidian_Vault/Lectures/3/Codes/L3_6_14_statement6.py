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
    
    plt.figure(figsize=(10, 6))
    
    # Create a matrix to visualize fold assignments
    # Rows = folds, Columns = samples
    fold_matrix = np.zeros((cv_folds, n_viz_samples))
    
    # Fill the matrix: 0.8 for test (dark gray), 0.3 for train (light gray)
    for i, (train_idx, test_idx) in enumerate(kf.split(X_viz)):
        fold_matrix[i, :] = 0.3  # All samples start as training
        fold_matrix[i, test_idx] = 0.8  # Validation samples
    
    # Create a simplified grayscale heatmap
    plt.imshow(fold_matrix, cmap='gray', aspect='auto', vmin=0, vmax=1)
    
    # Add annotations to cells with bold text for better visibility
    for i in range(cv_folds):
        for j in range(n_viz_samples):
            if fold_matrix[i, j] > 0.5:
                text = 'Val'
            else:
                text = 'Train'
            plt.text(j, i, text, ha="center", va="center", color="black", 
                     fontsize=9, fontweight='bold')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Fold Number')
    plt.title('5-Fold Cross-Validation: Data Usage Pattern')
    plt.grid(False)  # Remove grid for cleaner look
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
    
    # NEW VISUALIZATION: Cross-validation for preventing overfitting across models of different complexity
    plt.figure(figsize=(14, 10))
    
    # Generate a dataset with a known nonlinear pattern
    np.random.seed(42)
    n_data = 100
    X_full = np.linspace(-3, 3, n_data).reshape(-1, 1)
    # True function: y = 1 + 2x + 0.5x^2
    y_true = 1 + 2 * X_full.squeeze() + 0.5 * X_full.squeeze()**2
    y_full = y_true + np.random.normal(0, 3, n_data)
    
    # Create a held-out test set (not used in training or cross-validation)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    
    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Polynomial degrees to try
    degrees = [1, 2, 5, 15]
    
    # Setup for the training, validation, and test scores
    train_scores = []
    cv_scores = []
    test_scores = []
    
    # For the plot of all scores vs. complexity
    for degree in degrees:
        # Use polynomial features
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        
        # Create polynomial features and a model pipeline
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        # Get cross-validation score
        cv_score = np.mean(cross_val_score(model, X_train_val, y_train_val, cv=5, scoring='r2'))
        cv_scores.append(cv_score)
        
        # Train on all training data
        model.fit(X_train_val, y_train_val)
        
        # Get training score
        train_score = model.score(X_train_val, y_train_val)
        train_scores.append(train_score)
        
        # Get test score
        test_score = model.score(X_test, y_test)
        test_scores.append(test_score)
    
    # 1. Top-left: Model fits to data
    ax = axes[0, 0]
    
    # Plot the training data
    ax.scatter(X_train_val, y_train_val, alpha=0.5, label='Training data')
    
    # Plot the test data
    ax.scatter(X_test, y_test, color='red', alpha=0.5, label='Test data')
    
    # Plot the true function
    x_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_true_plot = 1 + 2 * x_plot.squeeze() + 0.5 * x_plot.squeeze()**2
    ax.plot(x_plot, y_true_plot, 'k--', label='True function')
    
    # Create and plot models with different complexities
    colors = ['blue', 'green', 'orange', 'red']
    for i, degree in enumerate(degrees):
        # Create and fit the model
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        model.fit(X_train_val, y_train_val)
        
        # Make predictions on a grid
        y_pred = model.predict(x_plot)
        
        # Plot the model prediction
        ax.plot(x_plot, y_pred, color=colors[i], lw=2, 
                label=f'Degree {degree}')
    
    ax.set_title('Model Fits of Different Complexity', fontsize=12)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True)
    
    # 2. Top-right: Examination of a specific model (degree 15) with CV
    ax = axes[0, 1]
    
    # Pick the highest degree model for demonstration
    high_degree = 15
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=high_degree)),
        ('linear', LinearRegression())
    ])
    
    # Setup K-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Plot the data
    ax.scatter(X_train_val, y_train_val, alpha=0.3, color='gray', label='All data')
    
    # Plot the true function
    ax.plot(x_plot, y_true_plot, 'k--', label='True function')
    
    # Plot each fold's model
    cv_predictions = []
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
        # Get train and validation data for this fold
        X_fold_train, X_fold_val = X_train_val[train_idx], X_train_val[val_idx]
        y_fold_train, y_fold_val = y_train_val[train_idx], y_train_val[val_idx]
        
        # Fit the model
        model.fit(X_fold_train, y_fold_train)
        
        # Make predictions
        y_fold_pred = model.predict(x_plot)
        
        # Store predictions
        cv_predictions.append(y_fold_pred)
        
        # Plot the fold's model
        ax.plot(x_plot, y_fold_pred, alpha=0.5, lw=1, color='red')
        
        # Plot training and validation data for first fold only to avoid clutter
        if i == 0:
            ax.scatter(X_fold_train, y_fold_train, alpha=0.7, color='blue', s=20, label='Fold 1: Train')
            ax.scatter(X_fold_val, y_fold_val, alpha=0.7, color='red', s=40, label='Fold 1: Validation')
    
    # Calculate and plot the average model across folds
    cv_predictions = np.array(cv_predictions)
    mean_cv_prediction = np.mean(cv_predictions, axis=0)
    ax.plot(x_plot, mean_cv_prediction, color='purple', lw=2, label='CV average')
    
    # Final model trained on all data
    model.fit(X_train_val, y_train_val)
    y_final_pred = model.predict(x_plot)
    ax.plot(x_plot, y_final_pred, color='red', lw=2, label=f'Degree {high_degree} (all data)')
    
    ax.set_title(f'Cross-Validation of Complex Model (Degree {high_degree})', fontsize=12)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True)
    
    # Add explanation of overfitting
    ax.text(0.05, 0.95, "Complex model overfits each fold differently\nCV helps identify overfitting",
           transform=ax.transAxes, fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 3. Bottom-left: Performance metrics on train, validation, and test sets
    ax = axes[1, 0]
    
    width = 0.25
    x = np.arange(len(degrees))
    
    # Plot grouped bars
    ax.bar(x - width, train_scores, width, label='Training R²', color='blue', alpha=0.7)
    ax.bar(x, cv_scores, width, label='CV R²', color='green', alpha=0.7)
    ax.bar(x + width, test_scores, width, label='Test R²', color='red', alpha=0.7)
    
    # Label the x-axis with degrees
    ax.set_xticks(x)
    ax.set_xticklabels([f'Degree {d}' for d in degrees])
    
    # Add score values
    for i, (tr, cv, te) in enumerate(zip(train_scores, cv_scores, test_scores)):
        ax.text(i - width, tr + 0.02, f'{tr:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i, cv + 0.02, f'{cv:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width, te + 0.02, f'{te:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Performance Comparison Across Model Complexity', fontsize=12)
    ax.set_xlabel('Model Complexity', fontsize=10)
    ax.set_ylabel('R-squared Score', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y')
    
    # Add explanation text about overfitting detection
    ax.text(0.05, 0.05, 
           "Signs of overfitting:\n- Training R² increases with complexity\n- Test R² decreases for complex models\n- CV R² helps detect optimal complexity",
           transform=ax.transAxes, fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 4. Bottom-right: Decision flowchart with CV
    ax = axes[1, 1]
    ax.axis('off')  # Turn off the axes
    
    # Draw a flowchart explaining the role of CV in model selection
    # Create a conceptual illustration
    from matplotlib.patches import Rectangle, Arrow, FancyArrowPatch
    
    # Define some positions
    start_x, start_y = 0.1, 0.9
    box_width, box_height = 0.3, 0.1
    h_space, v_space = 0.4, 0.2
    
    # Function to draw a box with text
    def draw_box(x, y, text, color='lightblue'):
        box = Rectangle((x, y), box_width, box_height, facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(box)
        ax.text(x + box_width/2, y + box_height/2, text, ha='center', va='center', fontsize=10)
        return x, y
    
    # Function to draw an arrow between boxes
    def draw_arrow(start, end, label=None):
        arrow = FancyArrowPatch(start, end, arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                               color='gray', linewidth=1.5)
        ax.add_patch(arrow)
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Create the flowchart
    box1 = draw_box(start_x, start_y, "Collect Data")
    box2 = draw_box(start_x + h_space, start_y, "Split into K Folds")
    box3 = draw_box(start_x + h_space, start_y - v_space, "Train Multiple Models\nof Different Complexity")
    box4 = draw_box(start_x, start_y - v_space, "Evaluate Each Model\nwith Cross-Validation")
    box5 = draw_box(start_x, start_y - 2*v_space, "Select Best Model\nBased on CV Score")
    box6 = draw_box(start_x + h_space, start_y - 2*v_space, "Evaluate Final Model\non Test Set")
    box7 = draw_box(start_x + h_space/2, start_y - 3*v_space, "Deploy Model")
    
    # Draw arrows
    draw_arrow((box1[0] + box_width, box1[1] + box_height/2), 
              (box2[0], box2[1] + box_height/2))
    
    draw_arrow((box2[0] + box_width/2, box2[1]), 
              (box3[0] + box_width/2, box3[1] + box_height))
    
    draw_arrow((box3[0], box3[1] + box_height/2), 
              (box4[0] + box_width, box4[1] + box_height/2))
    
    draw_arrow((box4[0] + box_width/2, box4[1]), 
              (box5[0] + box_width/2, box5[1] + box_height))
    
    draw_arrow((box5[0] + box_width, box5[1] + box_height/2), 
              (box6[0], box6[1] + box_height/2))
    
    draw_arrow((box6[0] + box_width/2, box6[1]), 
              (box7[0] + box_width/2, box7[1] + box_height))
    
    # Add cycle for model tuning
    draw_arrow((box5[0], box5[1] + box_height/2), 
              (box3[0], box3[1] + box_height/2), "Tune hyperparameters")
    
    # Add a title to the flowchart
    ax.set_title('Cross-Validation in Machine Learning Workflow', fontsize=12)
    
    # Add explanation of CV benefits
    ax.text(0.02, 0.02, 
           "Benefits of Cross-Validation:\n1. More reliable performance estimate\n2. Helps detect overfitting\n3. Works well with limited data\n4. Guides model selection\n5. Reduces selection bias",
           transform=ax.transAxes, fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Add a title for the entire figure
    plt.suptitle('Cross-Validation: A Resampling Technique for Model Assessment', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, 'statement6_cv_overfitting.png'), dpi=300, bbox_inches='tight')
    
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
        'image_path': ['statement6_cv_pattern.png', 'statement6_method_comparison.png', 'statement6_stability_comparison.png', 'statement6_cv_overfitting.png']
    }
    
    return result

if __name__ == "__main__":
    result = statement6_cross_validation()
    print(f"\nStatement: {result['statement']}")
    print(f"True or False: {'True' if result['is_true'] else 'False'}")
    print(f"Explanation: {result['explanation']}")
    print(f"Images saved: {', '.join(result['image_path'])}") 