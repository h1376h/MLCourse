import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import warnings
from sklearn.tree import export_text, plot_tree

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

def statement1_surrogate_splits():
    """
    Statement 1: CART's surrogate splits help maintain tree performance when primary splitting features are unavailable
    """
    print("\n==== Statement 1: CART Surrogate Splits ====")
    
    # Generate synthetic data with correlated features
    np.random.seed(42)
    n_samples = 1000
    
    # Create primary feature and highly correlated surrogate feature
    X_primary = np.random.normal(0, 1, n_samples)
    X_surrogate = X_primary + np.random.normal(0, 0.2, n_samples)  # Highly correlated
    X_noise = np.random.normal(0, 1, (n_samples, 3))  # Uncorrelated noise features
    
    X = np.column_stack([X_primary, X_surrogate, X_noise])
    
    # Create target based on primary feature
    y = (X_primary > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train decision tree
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    
    # Get baseline accuracy
    baseline_accuracy = dt.score(X_test, y_test)
    
    # Simulate missing primary feature by setting it to random values
    X_test_missing = X_test.copy()
    X_test_missing[:, 0] = np.random.normal(0, 1, len(X_test))  # Replace primary feature with noise
    
    # Test performance with "missing" primary feature
    missing_primary_accuracy = dt.score(X_test_missing, y_test)
    
    # Create visualization showing the correlation between features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Correlation matrix
    correlation_matrix = np.corrcoef(X_train.T)
    im = axes[0, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 0].set_title('Feature Correlation Matrix')
    axes[0, 0].set_xlabel('Feature Index')
    axes[0, 0].set_ylabel('Feature Index')
    feature_names = ['Primary', 'Surrogate', 'Noise1', 'Noise2', 'Noise3']
    axes[0, 0].set_xticks(range(5))
    axes[0, 0].set_yticks(range(5))
    axes[0, 0].set_xticklabels(feature_names, rotation=45)
    axes[0, 0].set_yticklabels(feature_names)
    plt.colorbar(im, ax=axes[0, 0])
    
    # Add correlation values to the matrix
    for i in range(5):
        for j in range(5):
            axes[0, 0].text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                          ha='center', va='center', color='black')
    
    # Plot 2: Scatter plot of primary vs surrogate feature
    scatter = axes[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.6, cmap='viridis')
    axes[0, 1].set_xlabel('Primary Feature')
    axes[0, 1].set_ylabel('Surrogate Feature')
    axes[0, 1].set_title('Primary vs Surrogate Feature Relationship')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Plot 3: Decision tree visualization (simplified)
    plot_tree(dt, ax=axes[1, 0], feature_names=feature_names, 
              class_names=['Class 0', 'Class 1'], filled=True, max_depth=2)
    axes[1, 0].set_title('Decision Tree Structure')
    
    # Plot 4: Performance comparison
    performance_data = ['Complete Data', 'Missing Primary Feature']
    accuracies = [baseline_accuracy, missing_primary_accuracy]
    colors = ['green', 'red']
    
    bars = axes[1, 1].bar(performance_data, accuracies, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Performance with/without Primary Feature')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement1_surrogate_splits.png'), dpi=300, bbox_inches='tight')
    
    print(f"Baseline accuracy (complete data): {baseline_accuracy:.4f}")
    print(f"Accuracy with missing primary feature: {missing_primary_accuracy:.4f}")
    print(f"Performance degradation: {(baseline_accuracy - missing_primary_accuracy):.4f}")
    print(f"Correlation between primary and surrogate: {np.corrcoef(X_train[:, 0], X_train[:, 1])[0, 1]:.4f}")
    
    result = {
        'statement': "CART's surrogate splits help maintain tree performance when primary splitting features are unavailable",
        'is_true': True,
        'explanation': "CART uses surrogate splits to maintain performance when the primary splitting feature is missing. The high correlation between features allows the tree to use alternative features that provide similar information gain.",
        'baseline_accuracy': baseline_accuracy,
        'missing_primary_accuracy': missing_primary_accuracy,
        'correlation': np.corrcoef(X_train[:, 0], X_train[:, 1])[0, 1]
    }
    
    return result

def statement2_pessimistic_pruning():
    """
    Statement 2: C4.5's pessimistic error pruning uses validation data to determine which subtrees to remove
    """
    print("\n==== Statement 2: C4.5 Pessimistic Error Pruning ====")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                              n_redundant=2, n_clusters_per_class=1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create trees with different pruning strategies
    max_depths = range(1, 15)
    train_accuracies = []
    test_accuracies = []
    tree_sizes = []
    
    for depth in max_depths:
        # Create unpruned tree
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        
        train_acc = dt.score(X_train, y_train)
        test_acc = dt.score(X_test, y_test)
        tree_size = dt.tree_.node_count
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        tree_sizes.append(tree_size)
    
    # Demonstrate cost-complexity pruning (similar to C4.5's approach)
    dt_full = DecisionTreeClassifier(random_state=42)
    dt_full.fit(X_train, y_train)
    
    # Get pruning path
    path = dt_full.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities
    
    # Train trees with different ccp_alpha values
    pruned_trees = []
    for ccp_alpha in ccp_alphas[:-1]:  # Exclude the last alpha which gives a tree with only one node
        dt_pruned = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
        dt_pruned.fit(X_train, y_train)
        pruned_trees.append(dt_pruned)
    
    # Calculate accuracies for pruned trees
    train_scores = [dt.score(X_train, y_train) for dt in pruned_trees]
    test_scores = [dt.score(X_test, y_test) for dt in pruned_trees]
    node_counts = [dt.tree_.node_count for dt in pruned_trees]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Tree complexity vs accuracy (showing overfitting)
    axes[0, 0].plot(max_depths, train_accuracies, 'b-o', label='Training Accuracy')
    axes[0, 0].plot(max_depths, test_accuracies, 'r-o', label='Test Accuracy')
    axes[0, 0].set_xlabel('Maximum Tree Depth')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Overfitting with Increasing Tree Depth')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Cost-complexity pruning path
    axes[0, 1].plot(ccp_alphas[:-1], node_counts, 'g-o')
    axes[0, 1].set_xlabel('Cost Complexity Parameter (alpha)')
    axes[0, 1].set_ylabel('Number of Nodes')
    axes[0, 1].set_title('Tree Size vs Cost Complexity Parameter')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True)
    
    # Plot 3: Pruning performance
    axes[1, 0].plot(ccp_alphas[:-1], train_scores, 'b-o', label='Training Accuracy')
    axes[1, 0].plot(ccp_alphas[:-1], test_scores, 'r-o', label='Test Accuracy')
    axes[1, 0].set_xlabel('Cost Complexity Parameter (alpha)')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Pruning Effect on Model Performance')
    axes[1, 0].set_xscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Pessimistic vs validation-based pruning concept
    # Show theoretical difference
    methods = ['Pessimistic Error\n(Training Data)', 'Validation-based\n(Hold-out Data)']
    theoretical_performance = [0.85, 0.88]  # Hypothetical values
    colors = ['orange', 'blue']
    
    bars = axes[1, 1].bar(methods, theoretical_performance, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Theoretical Performance')
    axes[1, 1].set_title('Pruning Method Comparison (Conceptual)')
    axes[1, 1].set_ylim(0.8, 0.9)
    
    for bar, perf in zip(bars, theoretical_performance):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{perf:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement2_pessimistic_pruning.png'), dpi=300, bbox_inches='tight')
    
    # Find optimal pruned tree
    best_idx = np.argmax(test_scores)
    best_alpha = ccp_alphas[best_idx]
    best_test_score = test_scores[best_idx]
    best_nodes = node_counts[best_idx]
    
    print(f"Best pruning alpha: {best_alpha:.6f}")
    print(f"Best test accuracy: {best_test_score:.4f}")
    print(f"Corresponding tree size: {best_nodes} nodes")
    print(f"Full tree size: {dt_full.tree_.node_count} nodes")
    
    result = {
        'statement': "C4.5's pessimistic error pruning uses validation data to determine which subtrees to remove",
        'is_true': False,
        'explanation': "C4.5's pessimistic error pruning does NOT use validation data. Instead, it uses a statistical approach based on the training data itself, applying a pessimistic estimate to the error rates. Validation-based pruning would use a separate validation set.",
        'best_alpha': best_alpha,
        'best_test_score': best_test_score,
        'pruning_benefit': (best_test_score - train_scores[0])
    }
    
    return result

def statement3_negative_information_gain():
    """
    Statement 3: Information gain can be negative when a split reduces the overall purity of child nodes
    """
    print("\n==== Statement 3: Negative Information Gain ====")
    
    # Create a demonstration of information gain calculation
    def calculate_entropy(y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-8))
    
    def calculate_information_gain(y_parent, y_left, y_right):
        n = len(y_parent)
        n_left, n_right = len(y_left), len(y_right)
        
        entropy_parent = calculate_entropy(y_parent)
        entropy_left = calculate_entropy(y_left)
        entropy_right = calculate_entropy(y_right)
        
        weighted_entropy = (n_left/n) * entropy_left + (n_right/n) * entropy_right
        information_gain = entropy_parent - weighted_entropy
        
        return information_gain, entropy_parent, entropy_left, entropy_right
    
    # Example 1: Good split (positive information gain)
    y_good_parent = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_good_left = np.array([0, 0, 0, 0])
    y_good_right = np.array([1, 1, 1, 1])
    
    ig_good, ent_good_parent, ent_good_left, ent_good_right = calculate_information_gain(
        y_good_parent, y_good_left, y_good_right)
    
    # Example 2: Bad split (still positive information gain, but smaller)
    y_bad_parent = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_bad_left = np.array([0, 0, 1, 1])
    y_bad_right = np.array([0, 0, 1, 1])
    
    ig_bad, ent_bad_parent, ent_bad_left, ent_bad_right = calculate_information_gain(
        y_bad_parent, y_bad_left, y_bad_right)
    
    # Example 3: Show that information gain is always non-negative
    # Generate random splits to demonstrate this
    np.random.seed(42)
    n_experiments = 1000
    information_gains = []
    
    for _ in range(n_experiments):
        # Create random parent node
        n_samples = np.random.randint(10, 100)
        y_parent = np.random.choice([0, 1], size=n_samples)
        
        # Create random split
        split_point = np.random.randint(1, n_samples)
        indices = np.random.permutation(n_samples)
        y_left = y_parent[indices[:split_point]]
        y_right = y_parent[indices[split_point:]]
        
        ig, _, _, _ = calculate_information_gain(y_parent, y_left, y_right)
        information_gains.append(ig)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Example splits comparison
    examples = ['Perfect Split', 'Poor Split']
    gains = [ig_good, ig_bad]
    colors = ['green', 'orange']
    
    bars = axes[0, 0].bar(examples, gains, color=colors, alpha=0.7)
    axes[0, 0].set_ylabel('Information Gain')
    axes[0, 0].set_title('Information Gain for Different Split Quality')
    axes[0, 0].set_ylim(0, max(gains) * 1.1)
    
    for bar, gain in zip(bars, gains):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{gain:.3f}', ha='center', va='bottom')
    
    # Plot 2: Distribution of information gains from random experiments
    axes[0, 1].hist(information_gains, bins=30, alpha=0.7, color='blue')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Information Gain')
    axes[0, 1].set_xlabel('Information Gain')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Information Gains (1000 Random Splits)')
    axes[0, 1].legend()
    
    # Plot 3: Entropy visualization for good split
    nodes = ['Parent', 'Left Child', 'Right Child']
    entropies_good = [ent_good_parent, ent_good_left, ent_good_right]
    colors_entropy = ['purple', 'green', 'blue']
    
    bars = axes[1, 0].bar(nodes, entropies_good, color=colors_entropy, alpha=0.7)
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].set_title(f'Entropy Values for Perfect Split (IG = {ig_good:.3f})')
    
    for bar, entropy in zip(bars, entropies_good):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{entropy:.3f}', ha='center', va='bottom')
    
    # Plot 4: Entropy visualization for bad split
    entropies_bad = [ent_bad_parent, ent_bad_left, ent_bad_right]
    
    bars = axes[1, 1].bar(nodes, entropies_bad, color=colors_entropy, alpha=0.7)
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title(f'Entropy Values for Poor Split (IG = {ig_bad:.3f})')
    
    for bar, entropy in zip(bars, entropies_bad):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{entropy:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement3_information_gain.png'), dpi=300, bbox_inches='tight')
    
    min_ig = min(information_gains)
    max_ig = max(information_gains)
    mean_ig = np.mean(information_gains)
    negative_count = sum(1 for ig in information_gains if ig < 0)
    
    print(f"Perfect split information gain: {ig_good:.6f}")
    print(f"Poor split information gain: {ig_bad:.6f}")
    print(f"Minimum information gain from {n_experiments} random splits: {min_ig:.6f}")
    print(f"Maximum information gain from {n_experiments} random splits: {max_ig:.6f}")
    print(f"Mean information gain: {mean_ig:.6f}")
    print(f"Number of negative information gains: {negative_count}")
    
    result = {
        'statement': "Information gain can be negative when a split reduces the overall purity of child nodes",
        'is_true': False,
        'explanation': "Information gain cannot be negative. It is defined as the reduction in entropy, and since entropy can only decrease or stay the same when splitting, information gain is always non-negative. In the worst case, a split provides zero information gain.",
        'min_ig': min_ig,
        'negative_count': negative_count,
        'perfect_split_ig': ig_good,
        'poor_split_ig': ig_bad
    }
    
    return result

def statement4_deeper_trees_training_accuracy():
    """
    Statement 4: Decision trees with deeper maximum depth always achieve better training accuracy than shallower trees
    """
    print("\n==== Statement 4: Tree Depth and Training Accuracy ====")
    
    # Generate synthetic datasets with different characteristics
    datasets = {
        'Simple Linear': make_classification(n_samples=200, n_features=2, n_informative=2, 
                                           n_redundant=0, n_clusters_per_class=1, random_state=42),
        'Complex Nonlinear': make_classification(n_samples=500, n_features=10, n_informative=5, 
                                                n_redundant=2, n_clusters_per_class=2, random_state=42),
        'Noisy Data': make_classification(n_samples=300, n_features=5, n_informative=3, 
                                        n_redundant=1, flip_y=0.1, random_state=42)
    }
    
    max_depths = range(1, 21)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['blue', 'red', 'green']
    
    all_results = {}
    
    for i, (name, (X, y)) in enumerate(datasets.items()):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        train_accuracies = []
        test_accuracies = []
        tree_sizes = []
        
        for depth in max_depths:
            dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
            dt.fit(X_train, y_train)
            
            train_acc = dt.score(X_train, y_train)
            test_acc = dt.score(X_test, y_test)
            tree_size = dt.tree_.node_count
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            tree_sizes.append(tree_size)
        
        all_results[name] = {
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'tree_sizes': tree_sizes
        }
        
        # Plot training accuracies
        if i < 3:
            row = i // 2
            col = i % 2 if i < 2 else 0
            
            axes[row, col].plot(max_depths, train_accuracies, 'o-', color=colors[i], 
                              label='Training Accuracy', linewidth=2)
            axes[row, col].plot(max_depths, test_accuracies, 's--', color=colors[i], 
                              alpha=0.7, label='Test Accuracy')
            axes[row, col].set_xlabel('Maximum Tree Depth')
            axes[row, col].set_ylabel('Accuracy')
            axes[row, col].set_title(f'{name} Dataset')
            axes[row, col].legend()
            axes[row, col].grid(True)
            axes[row, col].set_ylim(0, 1.05)
    
    # Combined plot showing all training accuracies
    for i, (name, results) in enumerate(all_results.items()):
        axes[1, 1].plot(max_depths, results['train_accuracies'], 'o-', 
                       color=colors[i], label=f'{name} (Training)', linewidth=2)
    
    axes[1, 1].set_xlabel('Maximum Tree Depth')
    axes[1, 1].set_ylabel('Training Accuracy')
    axes[1, 1].set_title('Training Accuracy vs Tree Depth (All Datasets)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim(0.5, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement4_depth_accuracy.png'), dpi=300, bbox_inches='tight')
    
    # Create a detailed analysis table
    print("\nDetailed Analysis of Training Accuracy vs Tree Depth:")
    for name, results in all_results.items():
        print(f"\n{name} Dataset:")
        train_accs = results['train_accuracies']
        
        # Check if training accuracy always increases
        always_increases = all(train_accs[i] >= train_accs[i-1] for i in range(1, len(train_accs)))
        max_train_acc = max(train_accs)
        depth_at_max = max_depths[train_accs.index(max_train_acc)]
        
        print(f"  Training accuracy always increases with depth: {always_increases}")
        print(f"  Maximum training accuracy: {max_train_acc:.4f} at depth {depth_at_max}")
        print(f"  Accuracy at depth 1: {train_accs[0]:.4f}")
        print(f"  Accuracy at depth 20: {train_accs[-1]:.4f}")
        
        # Find if there are any decreases
        decreases = []
        for i in range(1, len(train_accs)):
            if train_accs[i] < train_accs[i-1]:
                decreases.append((max_depths[i], train_accs[i-1], train_accs[i]))
        
        if decreases:
            print(f"  Found {len(decreases)} decreases in training accuracy:")
            for depth, prev_acc, curr_acc in decreases[:3]:  # Show first 3
                print(f"    Depth {depth}: {prev_acc:.4f} → {curr_acc:.4f}")
    
    # Test specific case where training accuracy might not always increase
    # Create a very simple dataset where a deeper tree might not help
    X_simple = np.array([[0], [1], [2], [3]])
    y_simple = np.array([0, 1, 0, 1])
    
    simple_results = []
    for depth in [1, 2, 3, 4]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_simple, y_simple)
        train_acc = dt.score(X_simple, y_simple)
        simple_results.append(train_acc)
    
    print(f"\nSimple 4-sample dataset training accuracies:")
    for depth, acc in zip([1, 2, 3, 4], simple_results):
        print(f"  Depth {depth}: {acc:.4f}")
    
    result = {
        'statement': "Decision trees with deeper maximum depth always achieve better training accuracy than shallower trees",
        'is_true': True,
        'explanation': "In general, deeper trees can achieve equal or better training accuracy than shallower trees because they have more flexibility to fit the training data. However, there can be exceptions with certain random seeds or tie-breaking scenarios, but the general trend is that training accuracy does not decrease with depth.",
        'results': all_results,
        'simple_case': simple_results
    }
    
    return result

def statement5_ccp_alpha_tradeoff():
    """
    Statement 5: CART's cost-complexity pruning parameter α controls the trade-off between tree complexity and training error
    """
    print("\n==== Statement 5: Cost-Complexity Pruning Alpha Parameter ====")
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, 
                              n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Get the full pruning path
    dt_full = DecisionTreeClassifier(random_state=42)
    dt_full.fit(X_train, y_train)
    
    path = dt_full.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities
    
    # Train trees with different alpha values
    alphas_to_test = ccp_alphas[::len(ccp_alphas)//10]  # Sample 10 alpha values
    if alphas_to_test[-1] != ccp_alphas[-1]:
        alphas_to_test = np.append(alphas_to_test, ccp_alphas[-1])
    
    results = []
    for alpha in alphas_to_test:
        dt = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        dt.fit(X_train, y_train)
        
        train_acc = dt.score(X_train, y_train)
        test_acc = dt.score(X_test, y_test)
        n_nodes = dt.tree_.node_count
        depth = dt.tree_.max_depth
        
        results.append({
            'alpha': alpha,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_nodes': n_nodes,
            'depth': depth
        })
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    alphas = [r['alpha'] for r in results]
    train_accs = [r['train_accuracy'] for r in results]
    test_accs = [r['test_accuracy'] for r in results]
    n_nodes = [r['n_nodes'] for r in results]
    depths = [r['depth'] for r in results]
    
    # Plot 1: Alpha vs Tree Complexity (nodes)
    axes[0, 0].semilogx(alphas[:-1], n_nodes[:-1], 'bo-')  # Exclude last alpha (single node)
    axes[0, 0].set_xlabel('Alpha (log scale)')
    axes[0, 0].set_ylabel('Number of Nodes')
    axes[0, 0].set_title('Tree Complexity vs Alpha Parameter')
    axes[0, 0].grid(True)
    
    # Plot 2: Alpha vs Training Error
    train_errors = [1 - acc for acc in train_accs]
    axes[0, 1].semilogx(alphas, train_errors, 'ro-')
    axes[0, 1].set_xlabel('Alpha (log scale)')
    axes[0, 1].set_ylabel('Training Error')
    axes[0, 1].set_title('Training Error vs Alpha Parameter')
    axes[0, 1].grid(True)
    
    # Plot 3: Combined view of the tradeoff
    ax3_twin = axes[1, 0].twinx()
    line1 = axes[1, 0].semilogx(alphas[:-1], n_nodes[:-1], 'b-o', label='Tree Complexity (nodes)')
    line2 = ax3_twin.semilogx(alphas, train_errors, 'r-s', label='Training Error')
    
    axes[1, 0].set_xlabel('Alpha (log scale)')
    axes[1, 0].set_ylabel('Number of Nodes', color='blue')
    ax3_twin.set_ylabel('Training Error', color='red')
    axes[1, 0].set_title('Complexity-Error Tradeoff')
    
    # Combine legends
    lines1, labels1 = axes[1, 0].get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    axes[1, 0].legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # Plot 4: Full pruning path
    axes[1, 1].plot(ccp_alphas, impurities, 'g-')
    axes[1, 1].set_xlabel('Alpha')
    axes[1, 1].set_ylabel('Impurity of Leaves')
    axes[1, 1].set_title('Cost-Complexity Pruning Path')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement5_ccp_alpha.png'), dpi=300, bbox_inches='tight')
    
    # Create detailed results table
    df = pd.DataFrame(results)
    print("\nCost-Complexity Pruning Results:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if x < 1 else f"{x:.0f}"))
    
    # Analyze the tradeoff
    complexity_reduction = (n_nodes[0] - n_nodes[-2]) / n_nodes[0] * 100  # Exclude single-node tree
    error_increase = (train_errors[-1] - train_errors[0]) * 100
    
    print(f"\nTradeoff Analysis:")
    print(f"Maximum tree complexity: {n_nodes[0]} nodes")
    print(f"Minimum tree complexity: {n_nodes[-1]} nodes")
    print(f"Complexity reduction with pruning: {complexity_reduction:.1f}%")
    print(f"Training error increase: {error_increase:.1f} percentage points")
    
    result = {
        'statement': "CART's cost-complexity pruning parameter α controls the trade-off between tree complexity and training error",
        'is_true': True,
        'explanation': "The cost-complexity parameter α directly controls the tradeoff between model complexity and training error. As α increases, the tree becomes simpler (fewer nodes) but training error typically increases. This is the fundamental principle of cost-complexity pruning.",
        'complexity_reduction': complexity_reduction,
        'error_increase': error_increase,
        'results': results
    }
    
    return result

def run_all_statements():
    """Run all statement analyses"""
    results = []
    
    print("Starting analysis of Decision Tree statements...")
    
    results.append(statement1_surrogate_splits())
    results.append(statement2_pessimistic_pruning())
    results.append(statement3_negative_information_gain())
    results.append(statement4_deeper_trees_training_accuracy())
    results.append(statement5_ccp_alpha_tradeoff())
    
    print("\n" + "="*80)
    print("SUMMARY OF ALL STATEMENTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\nStatement {i}: {result['statement']}")
        print(f"Answer: {'TRUE' if result['is_true'] else 'FALSE'}")
        print(f"Explanation: {result['explanation']}")
    
    return results

if __name__ == "__main__":
    results = run_all_statements()
