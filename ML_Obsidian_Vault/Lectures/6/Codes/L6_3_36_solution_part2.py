import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_36")
os.makedirs(save_dir, exist_ok=True)

# Set style for plots
plt.style.use('ggplot')
sns.set_style("whitegrid")
np.random.seed(42)

def statement6_multiway_vs_binary_splits():
    """
    Statement 6: Multi-way splits in ID3 always create more interpretable decision boundaries than binary splits
    """
    print("\n==== Statement 6: Multi-way vs Binary Splits ====")
    
    # Create synthetic data with categorical features
    np.random.seed(42)
    n_samples = 300
    
    # Create categorical feature with 4 categories
    categories = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    
    # Create target based on categories (some groups together)
    # A and B -> class 0, C and D -> class 1
    y = np.where(np.isin(categories, ['A', 'B']), 0, 1)
    
    # Add some noise
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    # Convert categories to numerical for visualization
    cat_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    X_num = np.array([cat_to_num[cat] for cat in categories])
    X = np.column_stack([X_num, feature2])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create trees with different strategies
    # Binary splits (sklearn default)
    dt_binary = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_binary.fit(X_train, y_train)
    
    # Simulate multi-way split decision by creating a more complex visualization
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Data distribution by category
    categories_train = [list(cat_to_num.keys())[int(x)] for x in X_train[:, 0]]
    for i, cat in enumerate(['A', 'B', 'C', 'D']):
        mask = np.array(categories_train) == cat
        if np.any(mask):
            axes[0, 0].scatter(X_train[mask, 0], X_train[mask, 1], 
                             c=y_train[mask], alpha=0.7, s=50, 
                             marker='o' if cat in ['A', 'B'] else 's',
                             label=f'Category {cat}')
    
    axes[0, 0].set_xlabel('Category (numerical)')
    axes[0, 0].set_ylabel('Feature 2')
    axes[0, 0].set_title('Data Distribution by Category')
    axes[0, 0].legend()
    axes[0, 0].set_xticks([0, 1, 2, 3])
    axes[0, 0].set_xticklabels(['A', 'B', 'C', 'D'])
    
    # Plot 2: Binary split decision boundary
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = dt_binary.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[0, 1].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = axes[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
    axes[0, 1].set_xlabel('Category (numerical)')
    axes[0, 1].set_ylabel('Feature 2')
    axes[0, 1].set_title('Binary Split Decision Boundary')
    axes[0, 1].set_xticks([0, 1, 2, 3])
    axes[0, 1].set_xticklabels(['A', 'B', 'C', 'D'])
    
    # Plot 3: Interpretability comparison
    interpretability_aspects = ['Rule Complexity', 'Number of Rules', 'Feature Interactions']
    binary_scores = [3, 4, 2]  # Hypothetical scores
    multiway_scores = [2, 2, 4]  # Hypothetical scores
    
    x = np.arange(len(interpretability_aspects))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, binary_scores, width, label='Binary Splits', alpha=0.7)
    axes[1, 0].bar(x + width/2, multiway_scores, width, label='Multi-way Splits', alpha=0.7)
    axes[1, 0].set_xlabel('Interpretability Aspects')
    axes[1, 0].set_ylabel('Complexity Score (lower = more interpretable)')
    axes[1, 0].set_title('Interpretability Comparison (Conceptual)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(interpretability_aspects, rotation=45)
    axes[1, 0].legend()
    
    # Plot 4: Class distribution by category
    categories_list = ['A', 'B', 'C', 'D']
    class_0_counts = []
    class_1_counts = []
    
    for cat in categories_list:
        mask = np.array(categories_train) == cat
        if np.any(mask):
            class_0_count = np.sum(y_train[mask] == 0)
            class_1_count = np.sum(y_train[mask] == 1)
        else:
            class_0_count = class_1_count = 0
        class_0_counts.append(class_0_count)
        class_1_counts.append(class_1_count)
    
    x = np.arange(len(categories_list))
    axes[1, 1].bar(x, class_0_counts, label='Class 0', alpha=0.7)
    axes[1, 1].bar(x, class_1_counts, bottom=class_0_counts, label='Class 1', alpha=0.7)
    axes[1, 1].set_xlabel('Category')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Class Distribution by Category')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories_list)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement6_multiway_vs_binary.png'), dpi=300, bbox_inches='tight')
    
    # Calculate accuracies
    train_acc = dt_binary.score(X_train, y_train)
    test_acc = dt_binary.score(X_test, y_test)
    
    print(f"Binary split tree accuracy: Train={train_acc:.4f}, Test={test_acc:.4f}")
    print(f"Tree depth: {dt_binary.tree_.max_depth}")
    print(f"Number of leaves: {dt_binary.tree_.n_leaves}")
    
    # Analyze interpretability
    print("\nInterpretability Analysis:")
    print("Multi-way splits:")
    print("  + Can capture natural categorical relationships in one split")
    print("  + Fewer decision nodes for categorical variables")
    print("  - Can create overly complex rules for high-cardinality features")
    print("  - May lead to data fragmentation")
    print("\nBinary splits:")
    print("  + Consistent structure regardless of feature type")
    print("  + Better handling of overfitting with many categories")
    print("  - May require multiple splits for simple categorical relationships")
    print("  + More robust to noise in categorical features")
    
    result = {
        'statement': "Multi-way splits in ID3 always create more interpretable decision boundaries than binary splits",
        'is_true': False,
        'explanation': "Multi-way splits do not always create more interpretable decision boundaries. While they can be more natural for categorical variables, they can also lead to overly complex rules and data fragmentation, especially with high-cardinality features. Binary splits provide more consistent and often more robust decision boundaries.",
        'train_acc': train_acc,
        'test_acc': test_acc
    }
    
    return result

def statement7_gain_ratio_vs_information_gain():
    """
    Statement 7: C4.5's gain ratio is always less than or equal to the corresponding information gain value
    """
    print("\n==== Statement 7: Gain Ratio vs Information Gain ====")
    
    def calculate_entropy(y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-8))
    
    def calculate_information_gain(y_parent, splits):
        n = len(y_parent)
        entropy_parent = calculate_entropy(y_parent)
        
        weighted_entropy = 0
        for split in splits:
            if len(split) > 0:
                weighted_entropy += (len(split) / n) * calculate_entropy(split)
        
        return entropy_parent - weighted_entropy
    
    def calculate_split_info(splits, n_total):
        if n_total == 0:
            return 0
        split_info = 0
        for split in splits:
            if len(split) > 0:
                p = len(split) / n_total
                split_info -= p * np.log2(p + 1e-8)
        return split_info
    
    def calculate_gain_ratio(y_parent, splits):
        ig = calculate_information_gain(y_parent, splits)
        split_info = calculate_split_info(splits, len(y_parent))
        
        if split_info == 0:
            return 0  # Avoid division by zero
        
        return ig / split_info
    
    # Test different splitting scenarios
    scenarios = []
    
    # Scenario 1: Balanced binary split
    y1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    splits1 = [np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])]
    ig1 = calculate_information_gain(y1, splits1)
    gr1 = calculate_gain_ratio(y1, splits1)
    scenarios.append(('Balanced Binary', ig1, gr1))
    
    # Scenario 2: Unbalanced binary split
    y2 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    splits2 = [np.array([0, 0, 0, 0, 1, 1, 1]), np.array([1])]
    ig2 = calculate_information_gain(y2, splits2)
    gr2 = calculate_gain_ratio(y2, splits2)
    scenarios.append(('Unbalanced Binary', ig2, gr2))
    
    # Scenario 3: Multi-way split (3 branches)
    y3 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0])
    splits3 = [np.array([0, 0, 0]), np.array([0, 1, 1]), np.array([1, 1, 1])]
    ig3 = calculate_information_gain(y3, splits3)
    gr3 = calculate_gain_ratio(y3, splits3)
    scenarios.append(('3-way Split', ig3, gr3))
    
    # Scenario 4: Highly fragmented split (many small branches)
    y4 = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    splits4 = [np.array([0]), np.array([0]), np.array([1]), np.array([1]), 
               np.array([0]), np.array([1]), np.array([0]), np.array([1])]
    ig4 = calculate_information_gain(y4, splits4)
    gr4 = calculate_gain_ratio(y4, splits4)
    scenarios.append(('Highly Fragmented', ig4, gr4))
    
    # Generate more scenarios with random data
    np.random.seed(42)
    random_scenarios = []
    
    for i in range(20):
        # Random parent node
        n_samples = np.random.randint(10, 50)
        y_parent = np.random.choice([0, 1], size=n_samples)
        
        # Random split (2-5 branches)
        n_branches = np.random.randint(2, 6)
        indices = np.random.permutation(n_samples)
        branch_sizes = np.random.multinomial(n_samples, np.ones(n_branches)/n_branches)
        
        splits = []
        start_idx = 0
        for size in branch_sizes:
            if size > 0:
                end_idx = start_idx + size
                splits.append(y_parent[indices[start_idx:end_idx]])
                start_idx = end_idx
        
        ig = calculate_information_gain(y_parent, splits)
        gr = calculate_gain_ratio(y_parent, splits)
        random_scenarios.append((ig, gr))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Specific scenarios comparison
    scenario_names = [s[0] for s in scenarios]
    information_gains = [s[1] for s in scenarios]
    gain_ratios = [s[2] for s in scenarios]
    
    x = np.arange(len(scenario_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, information_gains, width, label='Information Gain', alpha=0.7)
    axes[0, 0].bar(x + width/2, gain_ratios, width, label='Gain Ratio', alpha=0.7)
    axes[0, 0].set_xlabel('Split Scenario')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Information Gain vs Gain Ratio')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(scenario_names, rotation=45)
    axes[0, 0].legend()
    
    # Add value labels
    for i, (ig, gr) in enumerate(zip(information_gains, gain_ratios)):
        axes[0, 0].text(i - width/2, ig + 0.01, f'{ig:.3f}', ha='center', va='bottom')
        axes[0, 0].text(i + width/2, gr + 0.01, f'{gr:.3f}', ha='center', va='bottom')
    
    # Plot 2: Scatter plot of random scenarios
    random_igs = [s[0] for s in random_scenarios]
    random_grs = [s[1] for s in random_scenarios]
    
    axes[0, 1].scatter(random_igs, random_grs, alpha=0.7)
    axes[0, 1].plot([0, max(random_igs)], [0, max(random_igs)], 'r--', label='IG = GR line')
    axes[0, 1].set_xlabel('Information Gain')
    axes[0, 1].set_ylabel('Gain Ratio')
    axes[0, 1].set_title('Information Gain vs Gain Ratio (Random Scenarios)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Ratio distribution
    ratios = [gr/ig if ig > 0 else 0 for ig, gr in random_scenarios]
    axes[1, 0].hist(ratios, bins=15, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=1, color='red', linestyle='--', label='GR = IG')
    axes[1, 0].set_xlabel('Gain Ratio / Information Gain')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of GR/IG Ratios')
    axes[1, 0].legend()
    
    # Plot 4: Specific examples with values
    examples_data = {
        'Scenario': scenario_names,
        'Information Gain': information_gains,
        'Gain Ratio': gain_ratios,
        'GR/IG Ratio': [gr/ig if ig > 0 else 0 for ig, gr in zip(information_gains, gain_ratios)]
    }
    
    y_pos = np.arange(len(scenario_names))
    axes[1, 1].barh(y_pos, examples_data['GR/IG Ratio'], alpha=0.7)
    axes[1, 1].axvline(x=1, color='red', linestyle='--', label='GR = IG')
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(scenario_names)
    axes[1, 1].set_xlabel('Gain Ratio / Information Gain')
    axes[1, 1].set_title('GR/IG Ratio by Scenario')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement7_gain_ratio.png'), dpi=300, bbox_inches='tight')
    
    # Statistical analysis
    greater_than_one = sum(1 for gr, ig in zip(random_grs, random_igs) if gr > ig)
    equal_count = sum(1 for gr, ig in zip(random_grs, random_igs) if abs(gr - ig) < 1e-6)
    less_than_count = sum(1 for gr, ig in zip(random_grs, random_igs) if gr < ig)
    
    print(f"\nAnalysis of {len(random_scenarios)} random scenarios:")
    print(f"Gain Ratio > Information Gain: {greater_than_one} cases")
    print(f"Gain Ratio = Information Gain: {equal_count} cases")
    print(f"Gain Ratio < Information Gain: {less_than_count} cases")
    
    print(f"\nSpecific scenario results:")
    for name, ig, gr in scenarios:
        print(f"{name}: IG={ig:.4f}, GR={gr:.4f}, GR/IG={gr/ig if ig > 0 else 0:.4f}")
    
    # Check if gain ratio can exceed information gain
    can_exceed = any(gr > ig for ig, gr in random_scenarios if ig > 0)
    
    result = {
        'statement': "C4.5's gain ratio is always less than or equal to the corresponding information gain value",
        'is_true': True,
        'explanation': "Gain ratio is defined as Information Gain divided by Split Information. Since Split Information is always positive (except for trivial cases), and both IG and Split Info are non-negative, the gain ratio is typically less than or equal to the information gain. The gain ratio equals IG only when Split Information equals 1.",
        'greater_than_one_count': greater_than_one,
        'scenarios': scenarios,
        'can_exceed': can_exceed
    }
    
    return result

def statement8_regression_tree_mse():
    """
    Statement 8: Regression trees use mean squared error reduction as their default splitting criterion
    """
    print("\n==== Statement 8: Regression Tree MSE Criterion ====")
    
    # Generate regression data
    X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create regression trees with different criteria
    tree_mse = DecisionTreeRegressor(criterion='squared_error', max_depth=5, random_state=42)
    tree_mae = DecisionTreeRegressor(criterion='absolute_error', max_depth=5, random_state=42)
    
    # Fit trees
    tree_mse.fit(X_train, y_train)
    tree_mae.fit(X_train, y_train)
    
    # Make predictions
    y_pred_mse = tree_mse.predict(X_test)
    y_pred_mae = tree_mae.predict(X_test)
    
    # Calculate errors
    mse_mse = mean_squared_error(y_test, y_pred_mse)
    mse_mae = mean_squared_error(y_test, y_pred_mae)
    mae_mse = np.mean(np.abs(y_test - y_pred_mse))
    mae_mae = np.mean(np.abs(y_test - y_pred_mae))
    
    # Demonstrate MSE calculation manually
    def calculate_mse_reduction(y_parent, y_left, y_right):
        def mse(y):
            if len(y) == 0:
                return 0
            return np.mean((y - np.mean(y)) ** 2)
        
        n = len(y_parent)
        n_left, n_right = len(y_left), len(y_right)
        
        mse_parent = mse(y_parent)
        mse_left = mse(y_left)
        mse_right = mse(y_right)
        
        weighted_mse = (n_left/n) * mse_left + (n_right/n) * mse_right
        mse_reduction = mse_parent - weighted_mse
        
        return mse_reduction, mse_parent, mse_left, mse_right
    
    # Example calculation
    y_example = np.array([1, 2, 3, 4, 10, 11, 12, 13])
    y_left_ex = y_example[:4]
    y_right_ex = y_example[4:]
    
    mse_red, mse_par, mse_left, mse_right = calculate_mse_reduction(y_example, y_left_ex, y_right_ex)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Predictions comparison
    axes[0, 0].scatter(y_test, y_pred_mse, alpha=0.6, label='MSE Criterion')
    axes[0, 0].scatter(y_test, y_pred_mae, alpha=0.6, label='MAE Criterion')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Predictions: MSE vs MAE Criterion')
    axes[0, 0].legend()
    
    # Plot 2: Error comparison
    criteria = ['MSE Criterion', 'MAE Criterion']
    mse_errors = [mse_mse, mse_mae]
    mae_errors = [mae_mse, mae_mae]
    
    x = np.arange(len(criteria))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, mse_errors, width, label='MSE', alpha=0.7)
    axes[0, 1].bar(x + width/2, mae_errors, width, label='MAE', alpha=0.7)
    axes[0, 1].set_xlabel('Tree Criterion')
    axes[0, 1].set_ylabel('Error Value')
    axes[0, 1].set_title('Error Comparison by Criterion')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(criteria)
    axes[0, 1].legend()
    
    # Plot 3: MSE calculation example
    groups = ['Parent', 'Left Child', 'Right Child']
    mse_values = [mse_par, mse_left, mse_right]
    colors = ['purple', 'green', 'blue']
    
    bars = axes[1, 0].bar(groups, mse_values, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('MSE Value')
    axes[1, 0].set_title(f'MSE Calculation Example (Reduction: {mse_red:.3f})')
    
    for bar, mse_val in zip(bars, mse_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{mse_val:.3f}', ha='center', va='bottom')
    
    # Plot 4: Tree performance metrics
    metrics = ['Tree Depth', 'Number of Leaves', 'MSE Error', 'MAE Error']
    mse_tree_metrics = [tree_mse.tree_.max_depth, tree_mse.tree_.n_leaves, mse_mse/10, mae_mse/10]  # Scaled for visualization
    mae_tree_metrics = [tree_mae.tree_.max_depth, tree_mae.tree_.n_leaves, mse_mae/10, mae_mae/10]  # Scaled for visualization
    
    x = np.arange(len(metrics))
    axes[1, 1].bar(x - width/2, mse_tree_metrics, width, label='MSE Criterion', alpha=0.7)
    axes[1, 1].bar(x + width/2, mae_tree_metrics, width, label='MAE Criterion', alpha=0.7)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Value (errors scaled by 1/10)')
    axes[1, 1].set_title('Tree Characteristics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement8_regression_mse.png'), dpi=300, bbox_inches='tight')
    
    print(f"MSE Criterion Tree:")
    print(f"  Test MSE: {mse_mse:.4f}")
    print(f"  Test MAE: {mae_mse:.4f}")
    print(f"  Tree depth: {tree_mse.tree_.max_depth}")
    print(f"  Number of leaves: {tree_mse.tree_.n_leaves}")
    
    print(f"\nMAE Criterion Tree:")
    print(f"  Test MSE: {mse_mae:.4f}")
    print(f"  Test MAE: {mae_mae:.4f}")
    print(f"  Tree depth: {tree_mae.tree_.max_depth}")
    print(f"  Number of leaves: {tree_mae.tree_.n_leaves}")
    
    print(f"\nMSE Reduction Example:")
    print(f"  Parent MSE: {mse_par:.4f}")
    print(f"  Left child MSE: {mse_left:.4f}")
    print(f"  Right child MSE: {mse_right:.4f}")
    print(f"  MSE reduction: {mse_red:.4f}")
    
    result = {
        'statement': "Regression trees use mean squared error reduction as their default splitting criterion",
        'is_true': True,
        'explanation': "Yes, regression trees typically use mean squared error (MSE) reduction as their default splitting criterion. This measures how much the split reduces the variance in the target variable. Some implementations also offer alternative criteria like mean absolute error (MAE).",
        'mse_criterion_error': mse_mse,
        'mae_criterion_error': mse_mae,
        'example_reduction': mse_red
    }
    
    return result

def statement9_feature_bagging_correlation():
    """
    Statement 9: Feature bagging in Random Forest reduces correlation between individual trees in the ensemble
    """
    print("\n==== Statement 9: Feature Bagging and Tree Correlation ====")
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10, 
                              n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create different forest configurations
    # Regular Random Forest (with feature bagging)
    rf_normal = RandomForestClassifier(n_estimators=50, max_features='sqrt', 
                                     bootstrap=True, random_state=42)
    rf_normal.fit(X_train, y_train)
    
    # Random Forest without feature bagging (use all features)
    rf_no_bagging = RandomForestClassifier(n_estimators=50, max_features=None, 
                                         bootstrap=True, random_state=42)
    rf_no_bagging.fit(X_train, y_train)
    
    # Calculate correlations between tree predictions
    def calculate_tree_correlations(forest, X_test):
        tree_predictions = []
        for tree in forest.estimators_:
            pred = tree.predict_proba(X_test)[:, 1]  # Probability of class 1
            tree_predictions.append(pred)
        
        # Calculate correlation matrix
        tree_predictions = np.array(tree_predictions)
        correlation_matrix = np.corrcoef(tree_predictions)
        
        # Get upper triangular correlations (excluding diagonal)
        upper_tri_indices = np.triu_indices_from(correlation_matrix, k=1)
        correlations = correlation_matrix[upper_tri_indices]
        
        return correlations, correlation_matrix
    
    # Calculate correlations for both forests
    corr_normal, corr_matrix_normal = calculate_tree_correlations(rf_normal, X_test)
    corr_no_bagging, corr_matrix_no_bagging = calculate_tree_correlations(rf_no_bagging, X_test)
    
    # Calculate diversity metrics
    def calculate_diversity_metrics(forest, X_test, y_test):
        # Get individual tree predictions
        tree_predictions = []
        for tree in forest.estimators_:
            pred = tree.predict(X_test)
            tree_predictions.append(pred)
        
        tree_predictions = np.array(tree_predictions)
        
        # Calculate disagreement rate
        n_trees = len(tree_predictions)
        n_samples = len(X_test)
        
        disagreements = 0
        total_pairs = 0
        
        for i in range(n_trees):
            for j in range(i+1, n_trees):
                disagreements += np.sum(tree_predictions[i] != tree_predictions[j])
                total_pairs += n_samples
        
        disagreement_rate = disagreements / total_pairs if total_pairs > 0 else 0
        
        # Calculate accuracy
        ensemble_pred = np.round(np.mean(tree_predictions, axis=0))
        accuracy = np.mean(ensemble_pred == y_test)
        
        return disagreement_rate, accuracy
    
    disagree_normal, acc_normal = calculate_diversity_metrics(rf_normal, X_test, y_test)
    disagree_no_bagging, acc_no_bagging = calculate_diversity_metrics(rf_no_bagging, X_test, y_test)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Correlation distribution comparison
    axes[0, 0].hist(corr_normal, bins=20, alpha=0.7, label='With Feature Bagging', density=True)
    axes[0, 0].hist(corr_no_bagging, bins=20, alpha=0.7, label='Without Feature Bagging', density=True)
    axes[0, 0].set_xlabel('Tree Correlation')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Distribution of Tree Correlations')
    axes[0, 0].legend()
    axes[0, 0].axvline(np.mean(corr_normal), color='blue', linestyle='--', alpha=0.7)
    axes[0, 0].axvline(np.mean(corr_no_bagging), color='orange', linestyle='--', alpha=0.7)
    
    # Plot 2: Correlation matrix heatmap (with feature bagging)
    im1 = axes[0, 1].imshow(corr_matrix_normal, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 1].set_title('Tree Correlations (With Feature Bagging)')
    axes[0, 1].set_xlabel('Tree Index')
    axes[0, 1].set_ylabel('Tree Index')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot 3: Correlation matrix heatmap (without feature bagging)
    im2 = axes[1, 0].imshow(corr_matrix_no_bagging, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 0].set_title('Tree Correlations (Without Feature Bagging)')
    axes[1, 0].set_xlabel('Tree Index')
    axes[1, 0].set_ylabel('Tree Index')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot 4: Summary metrics comparison
    metrics = ['Mean Correlation', 'Disagreement Rate', 'Accuracy']
    with_bagging = [np.mean(corr_normal), disagree_normal, acc_normal]
    without_bagging = [np.mean(corr_no_bagging), disagree_no_bagging, acc_no_bagging]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, with_bagging, width, label='With Feature Bagging', alpha=0.7)
    axes[1, 1].bar(x + width/2, without_bagging, width, label='Without Feature Bagging', alpha=0.7)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Ensemble Diversity and Performance Metrics')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    
    # Add value labels
    for i, (wb, wob) in enumerate(zip(with_bagging, without_bagging)):
        axes[1, 1].text(i - width/2, wb + 0.01, f'{wb:.3f}', ha='center', va='bottom')
        axes[1, 1].text(i + width/2, wob + 0.01, f'{wob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement9_feature_bagging.png'), dpi=300, bbox_inches='tight')
    
    print(f"Random Forest with Feature Bagging:")
    print(f"  Mean tree correlation: {np.mean(corr_normal):.4f}")
    print(f"  Std tree correlation: {np.std(corr_normal):.4f}")
    print(f"  Disagreement rate: {disagree_normal:.4f}")
    print(f"  Accuracy: {acc_normal:.4f}")
    
    print(f"\nRandom Forest without Feature Bagging:")
    print(f"  Mean tree correlation: {np.mean(corr_no_bagging):.4f}")
    print(f"  Std tree correlation: {np.std(corr_no_bagging):.4f}")
    print(f"  Disagreement rate: {disagree_no_bagging:.4f}")
    print(f"  Accuracy: {acc_no_bagging:.4f}")
    
    correlation_reduction = np.mean(corr_no_bagging) - np.mean(corr_normal)
    print(f"\nCorrelation reduction due to feature bagging: {correlation_reduction:.4f}")
    
    result = {
        'statement': "Feature bagging in Random Forest reduces correlation between individual trees in the ensemble",
        'is_true': True,
        'explanation': "Feature bagging (selecting a random subset of features at each split) reduces correlation between trees by forcing them to consider different feature combinations. This increases diversity in the ensemble, which generally improves generalization performance.",
        'mean_corr_with_bagging': np.mean(corr_normal),
        'mean_corr_without_bagging': np.mean(corr_no_bagging),
        'correlation_reduction': correlation_reduction,
        'accuracy_with_bagging': acc_normal,
        'accuracy_without_bagging': acc_no_bagging
    }
    
    return result

def statement10_global_optimal_tree():
    """
    Statement 10: Decision tree algorithms guarantee finding the globally optimal tree structure
    """
    print("\n==== Statement 10: Global Optimality in Decision Trees ====")
    
    # Create a simple example where greedy approach might not find global optimum
    # XOR-like problem where early splits are not obviously beneficial
    np.random.seed(42)
    n_samples = 200
    
    # Create XOR pattern
    X1 = np.random.uniform(-1, 1, n_samples)
    X2 = np.random.uniform(-1, 1, n_samples)
    
    # XOR target: positive if signs are different
    y = ((X1 > 0) != (X2 > 0)).astype(int)
    
    # Add noise
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    X = np.column_stack([X1, X2])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train trees with different strategies
    trees = {
        'Greedy (depth 1)': DecisionTreeClassifier(max_depth=1, random_state=42),
        'Greedy (depth 2)': DecisionTreeClassifier(max_depth=2, random_state=42),
        'Greedy (depth 3)': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Greedy (depth 5)': DecisionTreeClassifier(max_depth=5, random_state=42)
    }
    
    results = {}
    for name, tree in trees.items():
        tree.fit(X_train, y_train)
        train_acc = tree.score(X_train, y_train)
        test_acc = tree.score(X_test, y_test)
        results[name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'tree': tree
        }
    
    # Demonstrate the greedy nature with information gain calculation
    def calculate_info_gain_for_split(X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        def entropy(labels):
            if len(labels) == 0:
                return 0
            _, counts = np.unique(labels, return_counts=True)
            probs = counts / len(labels)
            return -np.sum(probs * np.log2(probs + 1e-8))
        
        parent_entropy = entropy(y)
        n = len(y)
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        left_entropy = entropy(y[left_mask])
        right_entropy = entropy(y[right_mask])
        
        weighted_entropy = (np.sum(left_mask)/n) * left_entropy + (np.sum(right_mask)/n) * right_entropy
        return parent_entropy - weighted_entropy
    
    # Test different first splits
    thresholds = np.linspace(-0.8, 0.8, 20)
    feature_0_gains = []
    feature_1_gains = []
    
    for thresh in thresholds:
        gain_0 = calculate_info_gain_for_split(X_train, y_train, 0, thresh)
        gain_1 = calculate_info_gain_for_split(X_train, y_train, 1, thresh)
        feature_0_gains.append(gain_0)
        feature_1_gains.append(gain_1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Data distribution (XOR pattern)
    scatter = axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.7)
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    axes[0, 0].set_title('XOR-like Data Pattern')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Plot 2: Information gain for different first splits
    axes[0, 1].plot(thresholds, feature_0_gains, 'b-o', label='Feature 1 splits')
    axes[0, 1].plot(thresholds, feature_1_gains, 'r-s', label='Feature 2 splits')
    axes[0, 1].set_xlabel('Split Threshold')
    axes[0, 1].set_ylabel('Information Gain')
    axes[0, 1].set_title('Information Gain for First Split Options')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Tree performance comparison
    tree_names = list(results.keys())
    train_accs = [results[name]['train_acc'] for name in tree_names]
    test_accs = [results[name]['test_acc'] for name in tree_names]
    
    x = np.arange(len(tree_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, train_accs, width, label='Training Accuracy', alpha=0.7)
    axes[1, 0].bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.7)
    axes[1, 0].set_xlabel('Tree Configuration')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Performance vs Tree Depth')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(tree_names, rotation=45)
    axes[1, 0].legend()
    
    # Plot 4: Decision boundary for best tree
    best_tree = results['Greedy (depth 3)']['tree']
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
    y_min, y_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = best_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[1, 1].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = axes[1, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')
    axes[1, 1].set_title('Decision Boundary (Greedy Tree, Depth 3)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement10_global_optimality.png'), dpi=300, bbox_inches='tight')
    
    print(f"Performance Results:")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Training accuracy: {result['train_acc']:.4f}")
        print(f"  Test accuracy: {result['test_acc']:.4f}")
    
    print(f"\nInformation Gain Analysis:")
    print(f"Best information gain for Feature 1: {max(feature_0_gains):.4f}")
    print(f"Best information gain for Feature 2: {max(feature_1_gains):.4f}")
    print(f"Best threshold for Feature 1: {thresholds[np.argmax(feature_0_gains)]:.3f}")
    print(f"Best threshold for Feature 2: {thresholds[np.argmax(feature_1_gains)]:.3f}")
    
    # Theoretical analysis
    print(f"\nGreedy vs Global Optimality:")
    print("- Decision trees use greedy algorithms that make locally optimal choices")
    print("- At each split, they choose the feature and threshold with highest information gain")
    print("- This does not guarantee finding the globally optimal tree structure")
    print("- The XOR problem demonstrates this: no single split significantly improves purity")
    print("- But combining multiple splits can solve the problem effectively")
    print("- The globally optimal solution might require looking ahead multiple splits")
    
    result = {
        'statement': "Decision tree algorithms guarantee finding the globally optimal tree structure",
        'is_true': False,
        'explanation': "Decision tree algorithms use greedy approaches that make locally optimal decisions at each split. They do not guarantee finding the globally optimal tree structure. The algorithms choose the best split at each node based on immediate criteria (like information gain) without considering future splits, which may prevent finding the optimal overall solution.",
        'results': results,
        'max_info_gain_feature1': max(feature_0_gains),
        'max_info_gain_feature2': max(feature_1_gains)
    }
    
    return result

def run_all_statements_part2():
    """Run statements 6-10 analysis"""
    results = []
    
    print("Starting analysis of Decision Tree statements 6-10...")
    
    results.append(statement6_multiway_vs_binary_splits())
    results.append(statement7_gain_ratio_vs_information_gain())
    results.append(statement8_regression_tree_mse())
    results.append(statement9_feature_bagging_correlation())
    results.append(statement10_global_optimal_tree())
    
    print("\n" + "="*80)
    print("SUMMARY OF STATEMENTS 6-10")
    print("="*80)
    
    for i, result in enumerate(results, 6):
        print(f"\nStatement {i}: {result['statement']}")
        print(f"Answer: {'TRUE' if result['is_true'] else 'FALSE'}")
        print(f"Explanation: {result['explanation']}")
    
    return results

if __name__ == "__main__":
    results = run_all_statements_part2()
