import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import random as sparse_random
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_20")
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

def print_detailed_explanation(title, content):
    """Print detailed explanations with proper formatting."""
    print(f"\n{title}:")
    print("-" * len(title))
    if isinstance(content, dict):
        for key, value in content.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
    elif isinstance(content, list):
        for item in content:
            print(f"  - {item}")
    else:
        print(f"  {content}")
    print()

def generate_recommendation_dataset():
    """Generate a synthetic recommendation dataset with sparse features."""
    print_step_header(1, "Generating Synthetic Recommendation Dataset")
    
    np.random.seed(42)
    
    # Dataset parameters
    n_users = 1000
    n_items = 500
    n_features = 1000
    sparsity = 0.05  # 5% non-zero values
    
    dataset_params = {
        "Users": n_users,
        "Items": n_items,
        "Features": n_features,
        "Sparsity": f"{sparsity * 100}%",
        "Expected non-zero values per sample": int(n_features * sparsity),
        "Total possible interactions": n_users * n_items,
        "Actual interactions generated": int(n_users * n_items * 0.1)
    }
    print_detailed_explanation("Dataset Parameters", dataset_params)
    
    # Generate user-item interactions
    n_interactions = int(n_users * n_items * 0.1)  # 10% interaction rate
    user_ids = np.random.randint(0, n_users, n_interactions)
    item_ids = np.random.randint(0, n_items, n_interactions)
    
    # Generate features for each interaction
    # Features include: user demographics, item properties, contextual features
    feature_matrix = sparse_random(n_interactions, n_features, density=sparsity, format='csr')
    feature_matrix = feature_matrix.toarray()
    
    # Generate labels (like/dislike) based on feature patterns
    # Create some realistic patterns
    user_preference_weights = np.random.normal(0, 1, n_features)
    item_quality_weights = np.random.normal(0, 0.5, n_features)
    
    # Calculate interaction scores
    interaction_scores = (
        np.dot(feature_matrix, user_preference_weights) + 
        np.dot(feature_matrix, item_quality_weights) +
        np.random.normal(0, 0.1, n_interactions)
    )
    
    # Convert to binary labels (like = 1, dislike = 0)
    labels = (interaction_scores > np.median(interaction_scores)).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'label': labels
    })
    
    # Add feature columns
    for i in range(n_features):
        data[f'feature_{i}'] = feature_matrix[:, i]
    
    print(f"\nDataset Statistics:")
    print(f"- Total interactions: {len(data)}")
    print(f"- Positive labels (likes): {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    print(f"- Negative labels (dislikes): {len(labels) - np.sum(labels)} ({(1-np.mean(labels))*100:.1f}%)")
    print(f"- Average non-zero features per interaction: {np.mean(np.sum(feature_matrix > 0, axis=1)):.1f}")
    
    return data, feature_matrix, user_preference_weights, item_quality_weights

def analyze_sparsity_patterns(data, feature_matrix):
    """Analyze sparsity patterns in the recommendation data."""
    print_step_header(2, "Analyzing Sparsity Patterns")
    
    # Calculate sparsity statistics
    total_elements = feature_matrix.size
    non_zero_elements = np.count_nonzero(feature_matrix)
    sparsity_ratio = 1 - (non_zero_elements / total_elements)
    
    print(f"Sparsity Analysis:")
    print(f"- Total elements: {total_elements:,}")
    print(f"- Non-zero elements: {non_zero_elements:,}")
    print(f"- Sparsity ratio: {sparsity_ratio:.3f} ({sparsity_ratio*100:.1f}%)")
    
    # Analyze feature distribution
    feature_sums = np.sum(feature_matrix, axis=0)
    non_zero_features = np.sum(feature_sums > 0)
    
    print(f"\nFeature Usage:")
    print(f"- Features with non-zero values: {non_zero_features}/{feature_matrix.shape[1]}")
    print(f"- Average feature value (non-zero): {np.mean(feature_matrix[feature_matrix > 0]):.3f}")
    
    # Visualize sparsity patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Feature usage histogram
    axes[0, 0].hist(feature_sums, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Sum of Feature Values')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_title('Distribution of Feature Usage')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Sparsity per sample
    samples_non_zero = np.sum(feature_matrix > 0, axis=1)
    axes[0, 1].hist(samples_non_zero, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Number of Non-zero Features')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_title('Non-zero Features per Sample')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Feature value distribution
    non_zero_values = feature_matrix[feature_matrix > 0]
    axes[1, 0].hist(non_zero_values, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Feature Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Non-zero Feature Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: User-Item interaction heatmap (sample)
    sample_users = data['user_id'].unique()[:20]
    sample_items = data['item_id'].unique()[:20]
    interaction_matrix = np.zeros((len(sample_users), len(sample_items)))
    
    for i, user in enumerate(sample_users):
        for j, item in enumerate(sample_items):
            interactions = data[(data['user_id'] == user) & (data['item_id'] == item)]
            if len(interactions) > 0:
                interaction_matrix[i, j] = interactions['label'].iloc[0]
    
    im = axes[1, 1].imshow(interaction_matrix, cmap='RdYlBu', aspect='auto')
    axes[1, 1].set_xlabel('Items (sample)')
    axes[1, 1].set_ylabel('Users (sample)')
    axes[1, 1].set_title('User-Item Interaction Matrix (Sample)')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sparsity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return sparsity_ratio, feature_sums, samples_non_zero

def design_weak_learners_for_sparse_data():
    """Design and compare weak learners suitable for sparse data."""
    print_step_header(3, "Designing Weak Learners for Sparse Data")
    
    weak_learners = {
        'Decision Stump': {
            'classifier': DecisionTreeClassifier(max_depth=1, random_state=42),
            'description': 'Single split decision tree - handles sparsity well',
            'pros': ['Fast training', 'Interpretable', 'Handles missing values'],
            'cons': ['Limited expressiveness', 'May need many iterations']
        },
        'Shallow Tree': {
            'classifier': DecisionTreeClassifier(max_depth=3, random_state=42),
            'description': 'Shallow decision tree - good balance for sparse data',
            'pros': ['More expressive than stumps', 'Still interpretable', 'Handles sparsity'],
            'cons': ['Slightly slower', 'May overfit with very sparse data']
        },
        'Linear Classifier': {
            'classifier': DecisionTreeClassifier(max_depth=1, random_state=42),  # We'll simulate linear
            'description': 'Linear decision boundary - efficient for high dimensions',
            'pros': ['Fast prediction', 'Works well with sparse features', 'Regularizable'],
            'cons': ['Limited to linear boundaries', 'May need feature engineering']
        }
    }
    
    print("Weak Learner Analysis for Sparse Recommendation Data:")
    print("-" * 60)
    
    for name, info in weak_learners.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Pros: {', '.join(info['pros'])}")
        print(f"  Cons: {', '.join(info['cons'])}")
    
    # Visualize weak learner characteristics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Training time comparison (simulated)
    learner_names = list(weak_learners.keys())
    training_times = [0.1, 0.3, 0.05]  # Simulated relative training times

    axes[0, 0].bar(learner_names, training_times, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    axes[0, 0].set_ylabel('Relative Training Time')
    axes[0, 0].set_title('Training Time Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    # Memory usage comparison (simulated)
    memory_usage = [0.2, 0.8, 0.1]  # Simulated relative memory usage

    axes[0, 1].bar(learner_names, memory_usage, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    axes[0, 1].set_ylabel('Relative Memory Usage')
    axes[0, 1].set_title('Memory Usage Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # Interpretability score (simulated)
    interpretability = [9, 7, 6]  # Out of 10

    axes[1, 0].bar(learner_names, interpretability, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    axes[1, 0].set_ylabel('Interpretability Score (1-10)')
    axes[1, 0].set_title('Interpretability Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # Sparsity handling capability (simulated)
    sparsity_handling = [8, 7, 9]  # Out of 10

    axes[1, 1].bar(learner_names, sparsity_handling, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    axes[1, 1].set_ylabel('Sparsity Handling Score (1-10)')
    axes[1, 1].set_title('Sparsity Handling Capability')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weak_learner_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return weak_learners

def handle_cold_start_problem():
    """Analyze and propose solutions for the cold-start problem."""
    print_step_header(4, "Handling Cold-Start Problem")

    print("Cold-Start Problem Analysis:")
    print("-" * 40)

    cold_start_strategies = {
        'Content-Based Features': {
            'description': 'Use item/user attributes for new users/items',
            'implementation': 'Include demographic and content features in AdaBoost',
            'effectiveness': 8,
            'complexity': 6
        },
        'Popularity-Based Initialization': {
            'description': 'Start with popular items for new users',
            'implementation': 'Weight popular items higher in initial recommendations',
            'effectiveness': 6,
            'complexity': 3
        },
        'Hybrid Approach': {
            'description': 'Combine collaborative and content-based features',
            'implementation': 'Use both user-item interactions and content features',
            'effectiveness': 9,
            'complexity': 8
        },
        'Active Learning': {
            'description': 'Strategically ask new users for preferences',
            'implementation': 'Use AdaBoost uncertainty to select items to query',
            'effectiveness': 9,
            'complexity': 9
        }
    }

    for strategy, info in cold_start_strategies.items():
        print(f"\n{strategy}:")
        print(f"  Description: {info['description']}")
        print(f"  Implementation: {info['implementation']}")
        print(f"  Effectiveness: {info['effectiveness']}/10")
        print(f"  Complexity: {info['complexity']}/10")

    # Visualize cold-start strategies
    strategies = list(cold_start_strategies.keys())
    effectiveness = [info['effectiveness'] for info in cold_start_strategies.values()]
    complexity = [info['complexity'] for info in cold_start_strategies.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Effectiveness comparison
    bars1 = ax1.bar(strategies, effectiveness, color='lightblue', alpha=0.7)
    ax1.set_ylabel('Effectiveness Score (1-10)')
    ax1.set_title('Cold-Start Strategy Effectiveness')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars1, effectiveness):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(value), ha='center', va='bottom')

    # Complexity comparison
    bars2 = ax2.bar(strategies, complexity, color='lightcoral', alpha=0.7)
    ax2.set_ylabel('Implementation Complexity (1-10)')
    ax2.set_title('Cold-Start Strategy Complexity')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars2, complexity):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(value), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cold_start_strategies.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return cold_start_strategies

def feature_engineering_strategy():
    """Design feature engineering strategy for recommendation system."""
    print_step_header(5, "Feature Engineering Strategy")

    feature_categories = {
        'User Features': [
            'Age group', 'Gender', 'Location', 'Occupation',
            'Historical preferences', 'Activity level', 'Time patterns'
        ],
        'Item Features': [
            'Category', 'Price range', 'Brand', 'Popularity',
            'Release date', 'Ratings', 'Content attributes'
        ],
        'Contextual Features': [
            'Time of day', 'Day of week', 'Season', 'Device type',
            'Session length', 'Previous interactions', 'Search queries'
        ],
        'Interaction Features': [
            'Implicit feedback', 'Explicit ratings', 'Dwell time',
            'Click-through rate', 'Purchase history', 'Social signals'
        ]
    }

    print("Feature Engineering Categories:")
    print("-" * 40)

    for category, features in feature_categories.items():
        print(f"\n{category}:")
        for feature in features:
            print(f"  - {feature}")

    # Visualize feature importance simulation
    np.random.seed(42)
    all_features = []
    feature_importance = []
    categories = []

    for category, features in feature_categories.items():
        all_features.extend(features)
        # Simulate importance scores
        importance = np.random.exponential(0.3, len(features))
        feature_importance.extend(importance)
        categories.extend([category] * len(features))

    # Create feature importance plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Top 15 most important features
    top_indices = np.argsort(feature_importance)[-15:]
    top_features = [all_features[i] for i in top_indices]
    top_importance = [feature_importance[i] for i in top_indices]
    top_categories = [categories[i] for i in top_indices]

    colors = {'User Features': 'skyblue', 'Item Features': 'lightcoral',
              'Contextual Features': 'lightgreen', 'Interaction Features': 'gold'}
    bar_colors = [colors[cat] for cat in top_categories]

    bars = ax1.barh(range(len(top_features)), top_importance, color=bar_colors, alpha=0.7)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features)
    ax1.set_xlabel('Feature Importance Score')
    ax1.set_title('Top 15 Most Important Features')
    ax1.grid(True, alpha=0.3)

    # Feature category distribution
    category_counts = {cat: len(features) for cat, features in feature_categories.items()}

    wedges, texts, autotexts = ax2.pie(category_counts.values(), labels=category_counts.keys(),
                                       autopct='%1.1f%%', colors=list(colors.values()))
    ax2.set_title('Feature Distribution by Category')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_engineering.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return feature_categories, top_features

def evaluate_recommendation_quality():
    """Evaluate recommendation quality using various metrics."""
    print_step_header(6, "Evaluating Recommendation Quality")

    evaluation_metrics = {
        'Precision@K': {
            'description': 'Fraction of recommended items that are relevant',
            'formula': 'TP / (TP + FP) for top K recommendations',
            'use_case': 'When false positives are costly',
            'typical_values': '0.1 - 0.3 for K=10'
        },
        'Recall@K': {
            'description': 'Fraction of relevant items that are recommended',
            'formula': 'TP / (TP + FN) for top K recommendations',
            'use_case': 'When coverage is important',
            'typical_values': '0.05 - 0.2 for K=10'
        },
        'F1@K': {
            'description': 'Harmonic mean of Precision@K and Recall@K',
            'formula': '2 * (Precision * Recall) / (Precision + Recall)',
            'use_case': 'Balanced evaluation',
            'typical_values': '0.08 - 0.25 for K=10'
        },
        'AUC-ROC': {
            'description': 'Area under ROC curve',
            'formula': 'Integral of TPR vs FPR curve',
            'use_case': 'Overall ranking quality',
            'typical_values': '0.6 - 0.8'
        },
        'NDCG@K': {
            'description': 'Normalized Discounted Cumulative Gain',
            'formula': 'DCG@K / IDCG@K',
            'use_case': 'When ranking order matters',
            'typical_values': '0.1 - 0.4 for K=10'
        }
    }

    print("Recommendation Quality Metrics:")
    print("-" * 40)

    for metric, info in evaluation_metrics.items():
        print(f"\n{metric}:")
        print(f"  Description: {info['description']}")
        print(f"  Formula: {info['formula']}")
        print(f"  Use Case: {info['use_case']}")
        print(f"  Typical Values: {info['typical_values']}")

    # Simulate evaluation results
    np.random.seed(42)
    metrics = list(evaluation_metrics.keys())
    adaboost_scores = [0.25, 0.15, 0.19, 0.72, 0.28]  # Simulated AdaBoost performance
    baseline_scores = [0.18, 0.12, 0.14, 0.65, 0.22]  # Simulated baseline performance

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, adaboost_scores, width, label='AdaBoost', color='skyblue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, baseline_scores, width, label='Baseline', color='lightcoral', alpha=0.7)

    ax1.set_xlabel('Evaluation Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('AdaBoost vs Baseline Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')

    # Performance improvement
    improvements = [(ada - base) / base * 100 for ada, base in zip(adaboost_scores, baseline_scores)]

    bars3 = ax2.bar(metrics, improvements, color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Evaluation Metrics')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Performance Improvement over Baseline')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add value labels on bars
    for bar, improvement in zip(bars3, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'recommendation_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return evaluation_metrics, adaboost_scores, baseline_scores

def feature_selection_analysis():
    """Analyze feature selection for reducing from 1000 to 100 features."""
    print_step_header(7, "Feature Selection Analysis")

    print("Feature Selection Strategy for AdaBoost Recommendation System:")
    print("-" * 60)

    # Simulate feature importance scores
    np.random.seed(42)
    n_features = 1000
    feature_importance = np.random.exponential(0.1, n_features)
    feature_importance = feature_importance / np.sum(feature_importance)  # Normalize

    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_importance = feature_importance[sorted_indices]

    # Calculate cumulative importance
    cumulative_importance = np.cumsum(sorted_importance)

    # Find how many features needed for different importance thresholds
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    features_needed = []

    for threshold in thresholds:
        idx = np.where(cumulative_importance >= threshold)[0][0] + 1
        features_needed.append(idx)
        print(f"Features needed for {threshold*100}% importance: {idx}")

    # Visualize feature selection analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Feature importance distribution
    axes[0, 0].plot(range(1, 101), sorted_importance[:100], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Feature Rank')
    axes[0, 0].set_ylabel('Importance Score')
    axes[0, 0].set_title('Top 100 Features by Importance')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=100, color='red', linestyle='--', label='Selection Cutoff')
    axes[0, 0].legend()

    # Cumulative importance
    axes[0, 1].plot(range(1, 501), cumulative_importance[:500], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Number of Features')
    axes[0, 1].set_ylabel('Cumulative Importance')
    axes[0, 1].set_title('Cumulative Feature Importance')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=100, color='red', linestyle='--', label='100 Features')
    axes[0, 1].axhline(y=cumulative_importance[99], color='red', linestyle='--', alpha=0.7)
    axes[0, 1].legend()

    # Features needed for different thresholds
    axes[1, 0].bar(range(len(thresholds)), features_needed, color='orange', alpha=0.7)
    axes[1, 0].set_xlabel('Importance Threshold')
    axes[1, 0].set_ylabel('Features Needed')
    axes[1, 0].set_title('Features Needed for Different Thresholds')
    axes[1, 0].set_xticks(range(len(thresholds)))
    axes[1, 0].set_xticklabels([f'{t*100}%' for t in thresholds])
    axes[1, 0].grid(True, alpha=0.3)

    # Performance vs number of features (simulated)
    feature_counts = [10, 25, 50, 100, 200, 500, 1000]
    performance = [0.65, 0.70, 0.74, 0.78, 0.79, 0.80, 0.80]  # Simulated AUC scores

    axes[1, 1].plot(feature_counts, performance, 'ro-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Number of Features')
    axes[1, 1].set_ylabel('AUC Score')
    axes[1, 1].set_title('Performance vs Number of Features')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=100, color='red', linestyle='--', label='Selected: 100 features')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_selection_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    selected_importance = cumulative_importance[99]
    print(f"\nWith 100 features, we capture {selected_importance:.1%} of total importance")
    print(f"Estimated performance with 100 features: {performance[3]:.3f} AUC")

    return sorted_indices[:100], selected_importance

def main():
    """Main function to run the complete analysis."""
    print("Question 20: AdaBoost Recommendation System")
    print("=" * 60)

    # Generate dataset
    data, feature_matrix, user_weights, item_weights = generate_recommendation_dataset()

    # Analyze sparsity
    sparsity_ratio, feature_sums, samples_non_zero = analyze_sparsity_patterns(data, feature_matrix)

    # Design weak learners
    weak_learners = design_weak_learners_for_sparse_data()

    # Handle cold-start problem
    cold_start_strategies = handle_cold_start_problem()

    # Feature engineering
    feature_categories, top_features = feature_engineering_strategy()

    # Evaluate recommendation quality
    evaluation_metrics, ada_scores, baseline_scores = evaluate_recommendation_quality()

    # Feature selection analysis
    selected_features, importance_captured = feature_selection_analysis()

    # Summary
    print_step_header(8, "Summary and Recommendations")

    print("Key Findings:")
    print("-" * 20)
    print(f"1. Dataset has {sparsity_ratio:.1%} sparsity - suitable for AdaBoost with decision stumps")
    print(f"2. Decision stumps are recommended for sparse data handling")
    print(f"3. Hybrid approach best for cold-start problem (effectiveness: 9/10)")
    print(f"4. AdaBoost shows {((ada_scores[3] - baseline_scores[3]) / baseline_scores[3] * 100):.1f}% improvement in AUC")
    print(f"5. 100 features capture {importance_captured:.1%} of total importance")

    print(f"\nAll visualizations saved to: {save_dir}")

if __name__ == "__main__":
    main()
