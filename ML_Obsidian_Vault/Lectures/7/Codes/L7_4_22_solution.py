import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import time

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_22")
os.makedirs(save_dir, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def generate_computer_vision_dataset():
    """Generate a synthetic computer vision dataset with HOG, SIFT, and color features."""
    print_step_header(1, "Generating Computer Vision Dataset")
    
    np.random.seed(42)
    
    # Dataset parameters
    n_images = 10000
    n_hog_features = 3780  # Typical HOG descriptor size
    n_sift_features = 128  # SIFT descriptor size
    n_color_features = 64  # Color histogram features
    
    total_features = n_hog_features + n_sift_features + n_color_features
    
    print(f"Dataset Parameters:")
    print(f"- Total images: {n_images}")
    print(f"- HOG features: {n_hog_features}")
    print(f"- SIFT features: {n_sift_features}")
    print(f"- Color features: {n_color_features}")
    print(f"- Total features: {total_features}")
    
    # Generate synthetic features
    # HOG features (typically sparse and normalized)
    hog_features = np.random.exponential(0.1, (n_images, n_hog_features))
    hog_features = hog_features / np.linalg.norm(hog_features, axis=1, keepdims=True)
    
    # SIFT features (typically dense and normalized)
    sift_features = np.random.normal(0, 1, (n_images, n_sift_features))
    sift_features = np.abs(sift_features)  # SIFT features are non-negative
    sift_features = sift_features / np.linalg.norm(sift_features, axis=1, keepdims=True)
    
    # Color histogram features (normalized histograms)
    color_features = np.random.gamma(2, 1, (n_images, n_color_features))
    color_features = color_features / np.sum(color_features, axis=1, keepdims=True)
    
    # Combine all features
    X = np.hstack([hog_features, sift_features, color_features])
    
    # Generate labels based on feature patterns
    # Create realistic patterns for object detection
    hog_weights = np.random.normal(0, 0.5, n_hog_features)
    sift_weights = np.random.normal(0, 0.8, n_sift_features)
    color_weights = np.random.normal(0, 0.3, n_color_features)
    
    all_weights = np.hstack([hog_weights, sift_weights, color_weights])
    
    # Calculate scores and convert to binary labels
    scores = np.dot(X, all_weights) + np.random.normal(0, 0.1, n_images)
    y = (scores > np.median(scores)).astype(int)
    
    print(f"\nDataset Statistics:")
    print(f"- Positive samples (object present): {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    print(f"- Negative samples (no object): {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
    print(f"- Feature matrix shape: {X.shape}")
    print(f"- Feature matrix memory: {X.nbytes / 1024**2:.1f} MB")
    
    # Create feature type mapping
    feature_types = (['HOG'] * n_hog_features + 
                    ['SIFT'] * n_sift_features + 
                    ['Color'] * n_color_features)
    
    return X, y, feature_types, all_weights

def analyze_feature_characteristics(X, feature_types):
    """Analyze characteristics of different feature types."""
    print_step_header(2, "Analyzing Feature Characteristics")
    
    # Split features by type
    hog_mask = np.array(feature_types) == 'HOG'
    sift_mask = np.array(feature_types) == 'SIFT'
    color_mask = np.array(feature_types) == 'Color'
    
    hog_features = X[:, hog_mask]
    sift_features = X[:, sift_mask]
    color_features = X[:, color_mask]
    
    # Calculate statistics for each feature type
    feature_stats = {
        'HOG': {
            'mean': np.mean(hog_features),
            'std': np.std(hog_features),
            'sparsity': np.mean(hog_features == 0),
            'max_value': np.max(hog_features),
            'dimensionality': hog_features.shape[1]
        },
        'SIFT': {
            'mean': np.mean(sift_features),
            'std': np.std(sift_features),
            'sparsity': np.mean(sift_features == 0),
            'max_value': np.max(sift_features),
            'dimensionality': sift_features.shape[1]
        },
        'Color': {
            'mean': np.mean(color_features),
            'std': np.std(color_features),
            'sparsity': np.mean(color_features == 0),
            'max_value': np.max(color_features),
            'dimensionality': color_features.shape[1]
        }
    }
    
    print("Feature Type Analysis:")
    print("-" * 30)
    for feature_type, stats in feature_stats.items():
        print(f"\n{feature_type} Features:")
        print(f"  Dimensionality: {stats['dimensionality']}")
        print(f"  Mean value: {stats['mean']:.4f}")
        print(f"  Standard deviation: {stats['std']:.4f}")
        print(f"  Sparsity: {stats['sparsity']:.1%}")
        print(f"  Max value: {stats['max_value']:.4f}")
    
    # Visualize feature characteristics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Feature value distributions
    feature_data = [hog_features.flatten(), sift_features.flatten(), color_features.flatten()]
    feature_names = ['HOG', 'SIFT', 'Color']
    colors = ['blue', 'green', 'red']
    
    for i, (data, name, color) in enumerate(zip(feature_data, feature_names, colors)):
        # Sample data for visualization (too many points otherwise)
        sample_data = np.random.choice(data, size=min(10000, len(data)), replace=False)
        axes[0, i].hist(sample_data, bins=50, alpha=0.7, color=color, density=True)
        axes[0, i].set_xlabel('Feature Value')
        axes[0, i].set_ylabel('Density')
        axes[0, i].set_title(f'{name} Feature Distribution')
        axes[0, i].grid(True, alpha=0.3)
    
    # Feature statistics comparison
    stats_names = ['Mean', 'Std', 'Sparsity']
    stats_data = [
        [feature_stats[ft]['mean'] for ft in feature_names],
        [feature_stats[ft]['std'] for ft in feature_names],
        [feature_stats[ft]['sparsity'] for ft in feature_names]
    ]
    
    x = np.arange(len(feature_names))
    width = 0.25
    
    for i, (stat_name, stat_values) in enumerate(zip(stats_names, stats_data)):
        axes[1, i].bar(x, stat_values, color=colors, alpha=0.7)
        axes[1, i].set_xlabel('Feature Type')
        axes[1, i].set_ylabel(stat_name)
        axes[1, i].set_title(f'{stat_name} Comparison')
        axes[1, i].set_xticks(x)
        axes[1, i].set_xticklabels(feature_names)
        axes[1, i].grid(True, alpha=0.3)
        
        # Add value labels
        for j, v in enumerate(stat_values):
            axes[1, i].text(j, v + max(stat_values) * 0.01, f'{v:.3f}', 
                           ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_characteristics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return feature_stats

def design_weak_learners_for_images():
    """Design and compare weak learners suitable for image features."""
    print_step_header(3, "Designing Weak Learners for Image Features")
    
    weak_learners = {
        'Decision Stump': {
            'classifier': DecisionTreeClassifier(max_depth=1, random_state=42),
            'description': 'Single threshold on one feature',
            'pros': ['Very fast', 'Interpretable', 'Handles high dimensions'],
            'cons': ['Limited expressiveness', 'May need many iterations'],
            'training_time': 0.001,  # seconds per learner
            'memory_usage': 0.1,     # MB per learner
            'interpretability': 10
        },
        'Shallow Tree': {
            'classifier': DecisionTreeClassifier(max_depth=3, random_state=42),
            'description': 'Small decision tree with multiple splits',
            'pros': ['More expressive', 'Still fast', 'Good for feature interactions'],
            'cons': ['Less interpretable', 'May overfit'],
            'training_time': 0.01,
            'memory_usage': 0.5,
            'interpretability': 7
        },
        'Random Projection Tree': {
            'classifier': DecisionTreeClassifier(max_depth=2, random_state=42),
            'description': 'Tree on random linear combinations',
            'pros': ['Handles high dimensions', 'Diverse weak learners', 'Fast'],
            'cons': ['Less interpretable', 'Random performance'],
            'training_time': 0.005,
            'memory_usage': 0.3,
            'interpretability': 4
        },
        'Feature Subset Tree': {
            'classifier': DecisionTreeClassifier(max_depth=2, max_features='sqrt', random_state=42),
            'description': 'Tree trained on random feature subset',
            'pros': ['Reduces overfitting', 'Fast', 'Handles irrelevant features'],
            'cons': ['May miss important features', 'Random performance'],
            'training_time': 0.008,
            'memory_usage': 0.4,
            'interpretability': 6
        }
    }
    
    print("Weak Learner Analysis for Computer Vision:")
    print("-" * 50)
    
    for name, info in weak_learners.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Pros: {', '.join(info['pros'])}")
        print(f"  Cons: {', '.join(info['cons'])}")
        print(f"  Training time: {info['training_time']*1000:.1f} ms")
        print(f"  Memory usage: {info['memory_usage']:.1f} MB")
        print(f"  Interpretability: {info['interpretability']}/10")
    
    # Visualize weak learner comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    learner_names = list(weak_learners.keys())
    training_times = [info['training_time'] for info in weak_learners.values()]
    memory_usage = [info['memory_usage'] for info in weak_learners.values()]
    interpretability = [info['interpretability'] for info in weak_learners.values()]
    
    # Training time comparison
    bars1 = axes[0, 0].bar(range(len(learner_names)), training_times, 
                          color='skyblue', alpha=0.7)
    axes[0, 0].set_xlabel('Weak Learner Type')
    axes[0, 0].set_ylabel('Training Time (seconds)')
    axes[0, 0].set_title('Training Time Comparison')
    axes[0, 0].set_xticks(range(len(learner_names)))
    axes[0, 0].set_xticklabels([name.replace(' ', '\n') for name in learner_names])
    axes[0, 0].grid(True, alpha=0.3)
    
    for bar, time_val in zip(bars1, training_times):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(training_times)*0.01,
                       f'{time_val*1000:.1f}ms', ha='center', va='bottom')
    
    # Memory usage comparison
    bars2 = axes[0, 1].bar(range(len(learner_names)), memory_usage, 
                          color='lightcoral', alpha=0.7)
    axes[0, 1].set_xlabel('Weak Learner Type')
    axes[0, 1].set_ylabel('Memory Usage (MB)')
    axes[0, 1].set_title('Memory Usage Comparison')
    axes[0, 1].set_xticks(range(len(learner_names)))
    axes[0, 1].set_xticklabels([name.replace(' ', '\n') for name in learner_names])
    axes[0, 1].grid(True, alpha=0.3)
    
    for bar, mem_val in zip(bars2, memory_usage):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(memory_usage)*0.01,
                       f'{mem_val:.1f}MB', ha='center', va='bottom')
    
    # Interpretability comparison
    bars3 = axes[1, 0].bar(range(len(learner_names)), interpretability, 
                          color='lightgreen', alpha=0.7)
    axes[1, 0].set_xlabel('Weak Learner Type')
    axes[1, 0].set_ylabel('Interpretability Score (1-10)')
    axes[1, 0].set_title('Interpretability Comparison')
    axes[1, 0].set_xticks(range(len(learner_names)))
    axes[1, 0].set_xticklabels([name.replace(' ', '\n') for name in learner_names])
    axes[1, 0].grid(True, alpha=0.3)
    
    for bar, interp_val in zip(bars3, interpretability):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                       f'{interp_val}', ha='center', va='bottom')
    
    # Overall score (weighted combination)
    weights = {'time': 0.3, 'memory': 0.2, 'interpretability': 0.5}
    
    # Normalize scores (lower is better for time and memory, higher for interpretability)
    norm_time = 1 - np.array(training_times) / max(training_times)
    norm_memory = 1 - np.array(memory_usage) / max(memory_usage)
    norm_interp = np.array(interpretability) / 10
    
    overall_scores = (weights['time'] * norm_time + 
                     weights['memory'] * norm_memory + 
                     weights['interpretability'] * norm_interp)
    
    bars4 = axes[1, 1].bar(range(len(learner_names)), overall_scores, 
                          color='gold', alpha=0.7)
    axes[1, 1].set_xlabel('Weak Learner Type')
    axes[1, 1].set_ylabel('Overall Score (0-1)')
    axes[1, 1].set_title('Overall Performance Score')
    axes[1, 1].set_xticks(range(len(learner_names)))
    axes[1, 1].set_xticklabels([name.replace(' ', '\n') for name in learner_names])
    axes[1, 1].grid(True, alpha=0.3)
    
    for bar, score in zip(bars4, overall_scores):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weak_learner_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Recommendation
    best_learner_idx = np.argmax(overall_scores)
    best_learner = learner_names[best_learner_idx]
    
    print(f"\nRecommendation for Computer Vision:")
    print(f"Best overall weak learner: {best_learner}")
    print(f"Overall score: {overall_scores[best_learner_idx]:.2f}")
    
    return weak_learners, best_learner

def handle_high_dimensional_features(X, y, feature_types):
    """Analyze strategies for handling high-dimensional feature space."""
    print_step_header(4, "Handling High-Dimensional Feature Space")

    print(f"Original feature space: {X.shape[1]} dimensions")
    print("Strategies for dimensionality reduction:")
    print("-" * 40)

    # Strategy 1: PCA
    print("1. Principal Component Analysis (PCA)")
    pca_components = [100, 500, 1000, 2000]
    pca_results = {}

    for n_comp in pca_components:
        if n_comp < X.shape[1]:
            pca = PCA(n_components=n_comp, random_state=42)
            X_pca = pca.fit_transform(X)
            explained_var = np.sum(pca.explained_variance_ratio_)
            pca_results[n_comp] = {
                'explained_variance': explained_var,
                'shape': X_pca.shape,
                'compression_ratio': X.shape[1] / n_comp
            }
            print(f"  {n_comp} components: {explained_var:.1%} variance explained")

    # Strategy 2: Feature Selection based on variance
    print("\n2. Variance-based Feature Selection")
    feature_variances = np.var(X, axis=0)
    variance_thresholds = [0.001, 0.01, 0.1]
    variance_results = {}

    for threshold in variance_thresholds:
        selected_features = feature_variances > threshold
        n_selected = np.sum(selected_features)
        variance_results[threshold] = {
            'n_features': n_selected,
            'reduction_ratio': X.shape[1] / n_selected if n_selected > 0 else np.inf
        }
        print(f"  Threshold {threshold}: {n_selected} features selected")

    # Strategy 3: Random Projection
    print("\n3. Random Projection")
    from sklearn.random_projection import GaussianRandomProjection

    projection_dims = [100, 500, 1000]
    projection_results = {}

    for n_dim in projection_dims:
        if n_dim < X.shape[1]:
            rp = GaussianRandomProjection(n_components=n_dim, random_state=42)
            X_rp = rp.fit_transform(X)
            projection_results[n_dim] = {
                'shape': X_rp.shape,
                'compression_ratio': X.shape[1] / n_dim
            }
            print(f"  {n_dim} dimensions: {X.shape[1] / n_dim:.1f}x compression")

    # Visualize dimensionality reduction strategies
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # PCA explained variance
    components = list(pca_results.keys())
    explained_vars = [pca_results[c]['explained_variance'] for c in components]

    axes[0, 0].plot(components, explained_vars, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of PCA Components')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('PCA Explained Variance')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
    axes[0, 0].legend()

    # Feature variance distribution
    axes[0, 1].hist(feature_variances, bins=50, alpha=0.7, color='green', density=True)
    axes[0, 1].set_xlabel('Feature Variance')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution of Feature Variances')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # Add threshold lines
    for threshold in variance_thresholds:
        axes[0, 1].axvline(x=threshold, color='red', linestyle='--', alpha=0.7)

    # Compression ratios comparison
    strategies = ['PCA\n(500)', 'Variance\n(0.01)', 'Random Proj\n(500)']
    compression_ratios = [
        pca_results[500]['compression_ratio'] if 500 in pca_results else 1,
        variance_results[0.01]['reduction_ratio'],
        projection_results[500]['compression_ratio'] if 500 in projection_results else 1
    ]

    bars = axes[1, 0].bar(strategies, compression_ratios,
                         color=['blue', 'green', 'orange'], alpha=0.7)
    axes[1, 0].set_ylabel('Compression Ratio')
    axes[1, 0].set_title('Dimensionality Reduction Comparison')
    axes[1, 0].grid(True, alpha=0.3)

    for bar, ratio in zip(bars, compression_ratios):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                       f'{ratio:.1f}x', ha='center', va='bottom')

    # Training time estimation
    original_time = X.shape[1] * 0.001  # Simulated time per feature
    reduced_times = [
        500 * 0.001 if 500 in pca_results else original_time,
        variance_results[0.01]['n_features'] * 0.001,
        500 * 0.001 if 500 in projection_results else original_time
    ]

    bars2 = axes[1, 1].bar(strategies, reduced_times,
                          color=['blue', 'green', 'orange'], alpha=0.7)
    axes[1, 1].set_ylabel('Estimated Training Time (s)')
    axes[1, 1].set_title('Training Time with Reduced Dimensions')
    axes[1, 1].grid(True, alpha=0.3)

    for bar, time_val in zip(bars2, reduced_times):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(reduced_times)*0.01,
                       f'{time_val:.2f}s', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dimensionality_reduction.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return pca_results, variance_results, projection_results

def preprocessing_recommendations():
    """Provide preprocessing recommendations for computer vision features."""
    print_step_header(5, "Preprocessing Recommendations")

    preprocessing_steps = {
        'Normalization': {
            'description': 'Scale features to unit norm or [0,1] range',
            'importance': 9,
            'computational_cost': 2,
            'benefits': ['Prevents feature dominance', 'Improves convergence', 'Standard practice'],
            'implementation': 'StandardScaler or MinMaxScaler'
        },
        'Feature Selection': {
            'description': 'Remove low-variance or irrelevant features',
            'importance': 8,
            'computational_cost': 3,
            'benefits': ['Reduces overfitting', 'Faster training', 'Better generalization'],
            'implementation': 'Variance threshold or univariate selection'
        },
        'Dimensionality Reduction': {
            'description': 'Project to lower-dimensional space',
            'importance': 7,
            'computational_cost': 6,
            'benefits': ['Handles curse of dimensionality', 'Noise reduction', 'Faster computation'],
            'implementation': 'PCA, Random Projection, or Truncated SVD'
        },
        'Outlier Removal': {
            'description': 'Remove or clip extreme feature values',
            'importance': 6,
            'computational_cost': 4,
            'benefits': ['Improves robustness', 'Reduces noise impact', 'Better weak learners'],
            'implementation': 'IQR method or robust scaling'
        },
        'Feature Engineering': {
            'description': 'Create new features from existing ones',
            'importance': 8,
            'computational_cost': 7,
            'benefits': ['Better representation', 'Domain knowledge', 'Improved performance'],
            'implementation': 'Polynomial features or domain-specific transforms'
        }
    }

    print("Preprocessing Steps for Computer Vision:")
    print("-" * 45)

    for step, info in preprocessing_steps.items():
        print(f"\n{step}:")
        print(f"  Description: {info['description']}")
        print(f"  Importance: {info['importance']}/10")
        print(f"  Computational Cost: {info['computational_cost']}/10")
        print(f"  Benefits: {', '.join(info['benefits'])}")
        print(f"  Implementation: {info['implementation']}")

    # Visualize preprocessing recommendations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    step_names = list(preprocessing_steps.keys())
    importance_scores = [info['importance'] for info in preprocessing_steps.values()]
    cost_scores = [info['computational_cost'] for info in preprocessing_steps.values()]

    # Importance vs Cost scatter plot
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, imp, cost) in enumerate(zip(step_names, importance_scores, cost_scores)):
        axes[0, 0].scatter(cost, imp, s=200, c=colors[i], alpha=0.7, label=name)

    axes[0, 0].set_xlabel('Computational Cost (1-10)')
    axes[0, 0].set_ylabel('Importance (1-10)')
    axes[0, 0].set_title('Preprocessing Steps: Importance vs Cost')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Importance ranking
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_names = [step_names[i] for i in sorted_indices]
    sorted_importance = [importance_scores[i] for i in sorted_indices]

    bars1 = axes[0, 1].barh(range(len(sorted_names)), sorted_importance,
                           color=colors, alpha=0.7)
    axes[0, 1].set_yticks(range(len(sorted_names)))
    axes[0, 1].set_yticklabels(sorted_names)
    axes[0, 1].set_xlabel('Importance Score')
    axes[0, 1].set_title('Preprocessing Steps by Importance')
    axes[0, 1].grid(True, alpha=0.3)

    # Cost analysis
    bars2 = axes[1, 0].bar(range(len(step_names)), cost_scores,
                          color=colors, alpha=0.7)
    axes[1, 0].set_xticks(range(len(step_names)))
    axes[1, 0].set_xticklabels([name.replace(' ', '\n') for name in step_names], rotation=45)
    axes[1, 0].set_ylabel('Computational Cost')
    axes[1, 0].set_title('Computational Cost by Preprocessing Step')
    axes[1, 0].grid(True, alpha=0.3)

    # Efficiency score (importance / cost)
    efficiency_scores = [imp / cost for imp, cost in zip(importance_scores, cost_scores)]

    bars3 = axes[1, 1].bar(range(len(step_names)), efficiency_scores,
                          color=colors, alpha=0.7)
    axes[1, 1].set_xticks(range(len(step_names)))
    axes[1, 1].set_xticklabels([name.replace(' ', '\n') for name in step_names], rotation=45)
    axes[1, 1].set_ylabel('Efficiency (Importance/Cost)')
    axes[1, 1].set_title('Preprocessing Efficiency')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'preprocessing_recommendations.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Recommended preprocessing pipeline
    print(f"\nRecommended Preprocessing Pipeline:")
    print("-" * 35)

    # Sort by efficiency
    efficiency_indices = np.argsort(efficiency_scores)[::-1]

    print("Priority order (by efficiency):")
    for i, idx in enumerate(efficiency_indices):
        step = step_names[idx]
        efficiency = efficiency_scores[idx]
        print(f"{i+1}. {step} (efficiency: {efficiency:.2f})")

    return preprocessing_steps, efficiency_scores

def real_time_performance_analysis():
    """Analyze real-time performance requirements."""
    print_step_header(6, "Real-Time Performance Analysis")

    # Performance requirements
    target_fps = 100  # images per second
    max_latency = 1000 / target_fps  # milliseconds per image

    print(f"Real-Time Requirements:")
    print(f"- Target throughput: {target_fps} images/second")
    print(f"- Maximum latency: {max_latency:.1f} ms per image")
    print()

    # Analyze different ensemble sizes
    ensemble_sizes = [10, 25, 50, 100, 200, 500]

    # Simulated performance metrics
    weak_learner_times = {
        'Decision Stump': 0.1,      # ms per weak learner
        'Shallow Tree': 0.5,
        'Random Projection': 0.3,
        'Feature Subset': 0.4
    }

    performance_results = {}

    for learner_type, base_time in weak_learner_times.items():
        performance_results[learner_type] = {}

        for size in ensemble_sizes:
            total_time = size * base_time
            achievable_fps = 1000 / total_time if total_time > 0 else np.inf
            meets_requirement = achievable_fps >= target_fps

            performance_results[learner_type][size] = {
                'total_time': total_time,
                'fps': achievable_fps,
                'meets_requirement': meets_requirement
            }

    print("Performance Analysis by Weak Learner Type:")
    print("-" * 45)

    for learner_type, results in performance_results.items():
        print(f"\n{learner_type}:")
        print(f"  Base time per weak learner: {weak_learner_times[learner_type]:.1f} ms")

        # Find maximum ensemble size that meets requirements
        max_size = 0
        for size in ensemble_sizes:
            if results[size]['meets_requirement']:
                max_size = size

        print(f"  Maximum ensemble size for real-time: {max_size}")

        if max_size > 0:
            best_result = results[max_size]
            print(f"  Achievable FPS with max size: {best_result['fps']:.1f}")
            print(f"  Total latency: {best_result['total_time']:.1f} ms")

    # Visualize performance analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Latency vs ensemble size
    colors = ['blue', 'green', 'red', 'orange']
    for i, (learner_type, color) in enumerate(zip(weak_learner_times.keys(), colors)):
        latencies = [performance_results[learner_type][size]['total_time']
                    for size in ensemble_sizes]
        axes[0, 0].plot(ensemble_sizes, latencies, color=color, linewidth=2,
                       marker='o', label=learner_type)

    axes[0, 0].axhline(y=max_latency, color='red', linestyle='--', alpha=0.7,
                      label=f'Max latency ({max_latency:.1f} ms)')
    axes[0, 0].set_xlabel('Ensemble Size')
    axes[0, 0].set_ylabel('Total Latency (ms)')
    axes[0, 0].set_title('Latency vs Ensemble Size')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')

    # FPS vs ensemble size
    for i, (learner_type, color) in enumerate(zip(weak_learner_times.keys(), colors)):
        fps_values = [performance_results[learner_type][size]['fps']
                     for size in ensemble_sizes]
        # Cap FPS at 1000 for visualization
        fps_capped = [min(fps, 1000) for fps in fps_values]
        axes[0, 1].plot(ensemble_sizes, fps_capped, color=color, linewidth=2,
                       marker='o', label=learner_type)

    axes[0, 1].axhline(y=target_fps, color='red', linestyle='--', alpha=0.7,
                      label=f'Target FPS ({target_fps})')
    axes[0, 1].set_xlabel('Ensemble Size')
    axes[0, 1].set_ylabel('Achievable FPS')
    axes[0, 1].set_title('FPS vs Ensemble Size')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')

    # Maximum ensemble sizes
    max_sizes = []
    learner_types = list(weak_learner_times.keys())

    for learner_type in learner_types:
        max_size = 0
        for size in ensemble_sizes:
            if performance_results[learner_type][size]['meets_requirement']:
                max_size = size
        max_sizes.append(max_size)

    bars = axes[1, 0].bar(range(len(learner_types)), max_sizes,
                         color=colors, alpha=0.7)
    axes[1, 0].set_xticks(range(len(learner_types)))
    axes[1, 0].set_xticklabels([name.replace(' ', '\n') for name in learner_types])
    axes[1, 0].set_ylabel('Maximum Ensemble Size')
    axes[1, 0].set_title('Max Ensemble Size for Real-Time Performance')
    axes[1, 0].grid(True, alpha=0.3)

    for bar, size in zip(bars, max_sizes):
        if size > 0:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                           f'{size}', ha='center', va='bottom')

    # Performance efficiency (max_size / base_time)
    efficiency = [max_size / weak_learner_times[learner_type]
                 for learner_type, max_size in zip(learner_types, max_sizes)]

    bars2 = axes[1, 1].bar(range(len(learner_types)), efficiency,
                          color=colors, alpha=0.7)
    axes[1, 1].set_xticks(range(len(learner_types)))
    axes[1, 1].set_xticklabels([name.replace(' ', '\n') for name in learner_types])
    axes[1, 1].set_ylabel('Performance Efficiency')
    axes[1, 1].set_title('Performance Efficiency (Max Size / Base Time)')
    axes[1, 1].grid(True, alpha=0.3)

    for bar, eff in zip(bars2, efficiency):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(efficiency)*0.01,
                       f'{eff:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'real_time_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return performance_results, max_sizes

def main():
    """Main function to run the complete computer vision analysis."""
    print("Question 22: AdaBoost Computer Vision Task")
    print("=" * 60)

    # Generate dataset
    X, y, feature_types, true_weights = generate_computer_vision_dataset()

    # Analyze feature characteristics
    feature_stats = analyze_feature_characteristics(X, feature_types)

    # Design weak learners
    weak_learners, best_learner = design_weak_learners_for_images()

    # Handle high-dimensional features
    pca_results, variance_results, projection_results = handle_high_dimensional_features(X, y, feature_types)

    # Preprocessing recommendations
    preprocessing_steps, efficiency_scores = preprocessing_recommendations()

    # Real-time performance analysis
    performance_results, max_ensemble_sizes = real_time_performance_analysis()

    # Summary
    print_step_header(7, "Summary and Recommendations")

    print("Key Findings:")
    print("-" * 20)
    print(f"1. Best weak learner for computer vision: {best_learner}")
    print(f"2. Original feature space: {X.shape[1]} dimensions")
    print(f"3. PCA with 500 components explains {pca_results[500]['explained_variance']:.1%} variance" if 500 in pca_results else "")
    print(f"4. Most efficient preprocessing: {list(preprocessing_steps.keys())[np.argmax(efficiency_scores)]}")
    print(f"5. Maximum ensemble size for real-time: {max(max_ensemble_sizes)} (with decision stumps)")

    print(f"\nAll visualizations saved to: {save_dir}")

if __name__ == "__main__":
    main()
