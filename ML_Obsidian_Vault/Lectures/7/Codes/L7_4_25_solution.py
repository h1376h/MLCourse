import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification, load_breast_cancer, load_wine, load_iris
from scipy import stats
import seaborn as sns
from itertools import combinations

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_25")
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

def design_comprehensive_datasets():
    """Design datasets with different characteristics for comprehensive evaluation."""
    print_step_header(1, "Designing Comprehensive Evaluation Datasets")
    
    datasets = {}
    
    # Dataset 1: Balanced binary classification
    print("Creating Dataset 1: Balanced Binary Classification")
    X1, y1 = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                n_redundant=5, n_clusters_per_class=1, 
                                class_sep=1.0, random_state=42)
    datasets['balanced_binary'] = {
        'X': X1, 'y': y1, 
        'description': 'Balanced binary classification with moderate difficulty',
        'n_samples': len(X1), 'n_features': X1.shape[1], 'n_classes': len(np.unique(y1)),
        'class_balance': np.bincount(y1) / len(y1)
    }
    
    # Dataset 2: Imbalanced binary classification
    print("Creating Dataset 2: Imbalanced Binary Classification")
    X2, y2 = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                n_redundant=5, weights=[0.9, 0.1], 
                                class_sep=0.8, random_state=42)
    datasets['imbalanced_binary'] = {
        'X': X2, 'y': y2,
        'description': 'Imbalanced binary classification (90%-10% split)',
        'n_samples': len(X2), 'n_features': X2.shape[1], 'n_classes': len(np.unique(y2)),
        'class_balance': np.bincount(y2) / len(y2)
    }
    
    # Dataset 3: Multi-class classification
    print("Creating Dataset 3: Multi-class Classification")
    X3, y3 = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                n_redundant=5, n_classes=5, n_clusters_per_class=1,
                                class_sep=0.8, random_state=42)
    datasets['multiclass'] = {
        'X': X3, 'y': y3,
        'description': 'Multi-class classification with 5 classes',
        'n_samples': len(X3), 'n_features': X3.shape[1], 'n_classes': len(np.unique(y3)),
        'class_balance': np.bincount(y3) / len(y3)
    }
    
    # Dataset 4: High-dimensional, low sample
    print("Creating Dataset 4: High-dimensional, Low Sample")
    X4, y4 = make_classification(n_samples=200, n_features=100, n_informative=50,
                                n_redundant=20, n_clusters_per_class=1,
                                class_sep=1.2, random_state=42)
    datasets['high_dim_low_sample'] = {
        'X': X4, 'y': y4,
        'description': 'High-dimensional (100 features), low sample (200) dataset',
        'n_samples': len(X4), 'n_features': X4.shape[1], 'n_classes': len(np.unique(y4)),
        'class_balance': np.bincount(y4) / len(y4)
    }
    
    # Dataset 5: Noisy dataset
    print("Creating Dataset 5: Noisy Dataset")
    X5, y5 = make_classification(n_samples=1000, n_features=20, n_informative=10,
                                n_redundant=5, n_clusters_per_class=1,
                                class_sep=0.5, flip_y=0.1, random_state=42)
    datasets['noisy'] = {
        'X': X5, 'y': y5,
        'description': 'Noisy dataset with 10% label noise and low separability',
        'n_samples': len(X5), 'n_features': X5.shape[1], 'n_classes': len(np.unique(y5)),
        'class_balance': np.bincount(y5) / len(y5)
    }
    
    # Real-world datasets
    print("Loading Real-world Datasets")
    
    # Breast cancer dataset
    bc_data = load_breast_cancer()
    datasets['breast_cancer'] = {
        'X': bc_data.data, 'y': bc_data.target,
        'description': 'Breast cancer diagnosis (real-world medical data)',
        'n_samples': len(bc_data.data), 'n_features': bc_data.data.shape[1], 
        'n_classes': len(np.unique(bc_data.target)),
        'class_balance': np.bincount(bc_data.target) / len(bc_data.target)
    }
    
    # Wine dataset
    wine_data = load_wine()
    datasets['wine'] = {
        'X': wine_data.data, 'y': wine_data.target,
        'description': 'Wine classification (real-world chemical analysis)',
        'n_samples': len(wine_data.data), 'n_features': wine_data.data.shape[1],
        'n_classes': len(np.unique(wine_data.target)),
        'class_balance': np.bincount(wine_data.target) / len(wine_data.target)
    }
    
    print(f"\nDataset Summary:")
    print("-" * 50)
    for name, info in datasets.items():
        print(f"{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Samples: {info['n_samples']}, Features: {info['n_features']}, Classes: {info['n_classes']}")
        print(f"  Class balance: {info['class_balance']}")
        print()
    
    return datasets

def define_evaluation_metrics():
    """Define comprehensive evaluation metrics for different problem types."""
    print_step_header(2, "Defining Evaluation Metrics")
    
    metrics = {
        'binary_classification': {
            'accuracy': {
                'function': accuracy_score,
                'description': 'Overall correctness',
                'range': '[0, 1]',
                'higher_better': True
            },
            'precision': {
                'function': lambda y_true, y_pred: precision_score(y_true, y_pred, average='binary'),
                'description': 'True positives / (True positives + False positives)',
                'range': '[0, 1]',
                'higher_better': True
            },
            'recall': {
                'function': lambda y_true, y_pred: recall_score(y_true, y_pred, average='binary'),
                'description': 'True positives / (True positives + False negatives)',
                'range': '[0, 1]',
                'higher_better': True
            },
            'f1_score': {
                'function': lambda y_true, y_pred: f1_score(y_true, y_pred, average='binary'),
                'description': 'Harmonic mean of precision and recall',
                'range': '[0, 1]',
                'higher_better': True
            }
        },
        'multiclass_classification': {
            'accuracy': {
                'function': accuracy_score,
                'description': 'Overall correctness',
                'range': '[0, 1]',
                'higher_better': True
            },
            'precision_macro': {
                'function': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
                'description': 'Average precision across all classes',
                'range': '[0, 1]',
                'higher_better': True
            },
            'recall_macro': {
                'function': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
                'description': 'Average recall across all classes',
                'range': '[0, 1]',
                'higher_better': True
            },
            'f1_macro': {
                'function': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
                'description': 'Average F1 score across all classes',
                'range': '[0, 1]',
                'higher_better': True
            }
        },
        'imbalanced_classification': {
            'accuracy': {
                'function': accuracy_score,
                'description': 'Overall correctness (may be misleading)',
                'range': '[0, 1]',
                'higher_better': True
            },
            'precision_weighted': {
                'function': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
                'description': 'Weighted average precision',
                'range': '[0, 1]',
                'higher_better': True
            },
            'recall_weighted': {
                'function': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
                'description': 'Weighted average recall',
                'range': '[0, 1]',
                'higher_better': True
            },
            'f1_weighted': {
                'function': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
                'description': 'Weighted average F1 score',
                'range': '[0, 1]',
                'higher_better': True
            }
        }
    }
    
    print("Evaluation Metrics by Problem Type:")
    print("-" * 40)
    
    for problem_type, metric_dict in metrics.items():
        print(f"\n{problem_type.replace('_', ' ').title()}:")
        for metric_name, metric_info in metric_dict.items():
            print(f"  {metric_name}: {metric_info['description']}")
            print(f"    Range: {metric_info['range']}, Higher better: {metric_info['higher_better']}")
    
    return metrics

def design_weak_learner_configurations():
    """Design different weak learner configurations for testing."""
    print_step_header(3, "Designing Weak Learner Configurations")
    
    weak_learner_configs = {
        'decision_stump': {
            'classifier': DecisionTreeClassifier(max_depth=1, random_state=42),
            'description': 'Single-level decision tree (stump)',
            'complexity': 'Very Low',
            'interpretability': 'Very High'
        },
        'shallow_tree_depth2': {
            'classifier': DecisionTreeClassifier(max_depth=2, random_state=42),
            'description': 'Shallow decision tree (depth 2)',
            'complexity': 'Low',
            'interpretability': 'High'
        },
        'shallow_tree_depth3': {
            'classifier': DecisionTreeClassifier(max_depth=3, random_state=42),
            'description': 'Shallow decision tree (depth 3)',
            'complexity': 'Medium',
            'interpretability': 'Medium'
        },
        'limited_features': {
            'classifier': DecisionTreeClassifier(max_depth=2, max_features='sqrt', random_state=42),
            'description': 'Shallow tree with feature subsampling',
            'complexity': 'Medium',
            'interpretability': 'Medium'
        },
        'min_samples_split': {
            'classifier': DecisionTreeClassifier(max_depth=2, min_samples_split=20, random_state=42),
            'description': 'Shallow tree with minimum samples constraint',
            'complexity': 'Low',
            'interpretability': 'High'
        }
    }
    
    print("Weak Learner Configurations:")
    print("-" * 35)
    
    for config_name, config_info in weak_learner_configs.items():
        print(f"\n{config_name}:")
        print(f"  Description: {config_info['description']}")
        print(f"  Complexity: {config_info['complexity']}")
        print(f"  Interpretability: {config_info['interpretability']}")
    
    return weak_learner_configs

def design_ensemble_size_experiments():
    """Design experiments with different ensemble sizes."""
    print_step_header(4, "Designing Ensemble Size Experiments")
    
    ensemble_sizes = [10, 25, 50, 100, 200, 500]
    
    print("Ensemble Size Experiment Design:")
    print("-" * 35)
    print(f"Testing ensemble sizes: {ensemble_sizes}")
    print("Objectives:")
    print("- Find optimal ensemble size for different datasets")
    print("- Analyze overfitting vs underfitting trade-offs")
    print("- Measure computational cost vs performance")
    print("- Identify diminishing returns point")
    
    return ensemble_sizes

def run_comprehensive_evaluation(datasets, metrics, weak_learner_configs, ensemble_sizes):
    """Run comprehensive evaluation across all configurations."""
    print_step_header(5, "Running Comprehensive Evaluation")
    
    results = {}
    
    # Select a subset for demonstration (to avoid excessive computation)
    selected_datasets = ['balanced_binary', 'imbalanced_binary', 'multiclass']
    selected_configs = ['decision_stump', 'shallow_tree_depth2']
    selected_sizes = [25, 50, 100]
    
    print(f"Running evaluation on:")
    print(f"- Datasets: {selected_datasets}")
    print(f"- Weak learners: {selected_configs}")
    print(f"- Ensemble sizes: {selected_sizes}")
    print()
    
    for dataset_name in selected_datasets:
        print(f"Evaluating dataset: {dataset_name}")
        dataset = datasets[dataset_name]
        X, y = dataset['X'], dataset['y']
        
        # Determine problem type for metrics
        if dataset['n_classes'] == 2:
            if min(dataset['class_balance']) < 0.3:
                problem_type = 'imbalanced_classification'
            else:
                problem_type = 'binary_classification'
        else:
            problem_type = 'multiclass_classification'
        
        dataset_metrics = metrics[problem_type]
        
        for config_name in selected_configs:
            config = weak_learner_configs[config_name]
            
            for ensemble_size in selected_sizes:
                print(f"  Config: {config_name}, Size: {ensemble_size}")
                
                # Create AdaBoost classifier
                ada_clf = AdaBoostClassifier(
                    estimator=config['classifier'],
                    n_estimators=ensemble_size,
                    random_state=42
                )
                
                # Perform cross-validation
                cv_scores = {}
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                for metric_name, metric_info in dataset_metrics.items():
                    scores = cross_val_score(ada_clf, X, y, cv=skf, 
                                           scoring='accuracy' if metric_name == 'accuracy' else None)
                    
                    # For non-accuracy metrics, we need to compute manually
                    if metric_name != 'accuracy':
                        manual_scores = []
                        for train_idx, test_idx in skf.split(X, y):
                            X_train, X_test = X[train_idx], X[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]
                            
                            ada_clf.fit(X_train, y_train)
                            y_pred = ada_clf.predict(X_test)
                            score = metric_info['function'](y_test, y_pred)
                            manual_scores.append(score)
                        scores = np.array(manual_scores)
                    
                    cv_scores[metric_name] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'scores': scores
                    }
                
                # Store results
                key = (dataset_name, config_name, ensemble_size)
                results[key] = {
                    'cv_scores': cv_scores,
                    'problem_type': problem_type
                }
    
    print(f"\nEvaluation completed! Collected {len(results)} result sets.")
    return results

def statistical_significance_testing(results):
    """Perform statistical significance testing on evaluation results."""
    print_step_header(6, "Statistical Significance Testing")

    print("Statistical Tests for Performance Differences:")
    print("-" * 45)

    # Extract results for statistical testing
    dataset_names = list(set([key[0] for key in results.keys()]))
    config_names = list(set([key[1] for key in results.keys()]))
    ensemble_sizes = list(set([key[2] for key in results.keys()]))

    statistical_results = {}

    for dataset_name in dataset_names:
        print(f"\nDataset: {dataset_name}")
        dataset_results = {key: value for key, value in results.items() if key[0] == dataset_name}

        # Compare different configurations at the same ensemble size
        for ensemble_size in ensemble_sizes:
            size_results = {key: value for key, value in dataset_results.items() if key[2] == ensemble_size}

            if len(size_results) >= 2:
                print(f"  Ensemble size {ensemble_size}:")

                # Get accuracy scores for comparison
                config_scores = {}
                for key, result in size_results.items():
                    config_name = key[1]
                    accuracy_scores = result['cv_scores']['accuracy']['scores']
                    config_scores[config_name] = accuracy_scores

                # Perform pairwise t-tests
                config_pairs = list(combinations(config_scores.keys(), 2))

                for config1, config2 in config_pairs:
                    scores1 = config_scores[config1]
                    scores2 = config_scores[config2]

                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(scores1, scores2)

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std

                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

                    print(f"    {config1} vs {config2}:")
                    print(f"      Mean diff: {np.mean(scores1) - np.mean(scores2):.4f}")
                    print(f"      t-stat: {t_stat:.3f}, p-value: {p_value:.4f} {significance}")
                    print(f"      Effect size (Cohen's d): {cohens_d:.3f}")

                    # Store results
                    test_key = (dataset_name, ensemble_size, config1, config2)
                    statistical_results[test_key] = {
                        't_stat': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'mean_diff': np.mean(scores1) - np.mean(scores2),
                        'significance': significance
                    }

    return statistical_results

def visualize_evaluation_results(results, statistical_results):
    """Visualize comprehensive evaluation results."""
    print_step_header(7, "Visualizing Evaluation Results")

    # Prepare data for visualization
    dataset_names = list(set([key[0] for key in results.keys()]))
    config_names = list(set([key[1] for key in results.keys()]))
    ensemble_sizes = list(set([key[2] for key in results.keys()]))

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Performance by dataset and configuration
    performance_data = []
    for key, result in results.items():
        dataset_name, config_name, ensemble_size = key
        accuracy_mean = result['cv_scores']['accuracy']['mean']
        accuracy_std = result['cv_scores']['accuracy']['std']

        performance_data.append({
            'dataset': dataset_name,
            'config': config_name,
            'ensemble_size': ensemble_size,
            'accuracy_mean': accuracy_mean,
            'accuracy_std': accuracy_std
        })

    perf_df = pd.DataFrame(performance_data)

    # Heatmap of mean accuracy
    pivot_data = perf_df.pivot_table(values='accuracy_mean',
                                    index=['dataset', 'config'],
                                    columns='ensemble_size')

    im1 = axes[0, 0].imshow(pivot_data.values, cmap='viridis', aspect='auto')
    axes[0, 0].set_xticks(range(len(pivot_data.columns)))
    axes[0, 0].set_xticklabels(pivot_data.columns)
    axes[0, 0].set_yticks(range(len(pivot_data.index)))
    axes[0, 0].set_yticklabels([f"{idx[0]}\n{idx[1]}" for idx in pivot_data.index])
    axes[0, 0].set_xlabel('Ensemble Size')
    axes[0, 0].set_title('Mean Accuracy Heatmap')
    plt.colorbar(im1, ax=axes[0, 0])

    # Plot 2: Performance vs ensemble size
    colors = ['blue', 'red', 'green']
    for i, dataset in enumerate(dataset_names):
        dataset_data = perf_df[perf_df['dataset'] == dataset]

        for j, config in enumerate(config_names):
            config_data = dataset_data[dataset_data['config'] == config]
            if not config_data.empty:
                linestyle = '-' if j == 0 else '--'
                axes[0, 1].plot(config_data['ensemble_size'], config_data['accuracy_mean'],
                               color=colors[i], linestyle=linestyle, marker='o',
                               label=f'{dataset}_{config}')

    axes[0, 1].set_xlabel('Ensemble Size')
    axes[0, 1].set_ylabel('Mean Accuracy')
    axes[0, 1].set_title('Performance vs Ensemble Size')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Statistical significance summary
    sig_counts = {'***': 0, '**': 0, '*': 0, 'ns': 0}
    for result in statistical_results.values():
        sig_counts[result['significance']] += 1

    wedges, texts, autotexts = axes[0, 2].pie(sig_counts.values(),
                                             labels=['p<0.001', 'p<0.01', 'p<0.05', 'not sig.'],
                                             autopct='%1.1f%%',
                                             colors=['darkred', 'red', 'orange', 'lightgray'])
    axes[0, 2].set_title('Statistical Significance Distribution')

    # Plot 4: Effect sizes
    effect_sizes = [result['cohens_d'] for result in statistical_results.values()]
    axes[1, 0].hist(effect_sizes, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(x=0.2, color='green', linestyle='--', label='Small effect')
    axes[1, 0].axvline(x=0.5, color='orange', linestyle='--', label='Medium effect')
    axes[1, 0].axvline(x=0.8, color='red', linestyle='--', label='Large effect')
    axes[1, 0].set_xlabel("Cohen's d")
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Effect Sizes')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Performance variance analysis
    variance_data = []
    for key, result in results.items():
        dataset_name, config_name, ensemble_size = key
        accuracy_std = result['cv_scores']['accuracy']['std']
        variance_data.append({
            'dataset': dataset_name,
            'config': config_name,
            'ensemble_size': ensemble_size,
            'std': accuracy_std
        })

    var_df = pd.DataFrame(variance_data)

    # Box plot of standard deviations by configuration
    config_stds = [var_df[var_df['config'] == config]['std'].values for config in config_names]
    bp = axes[1, 1].boxplot(config_stds, labels=config_names, patch_artist=True)

    colors_box = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)

    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_title('Performance Variance by Configuration')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Best configuration summary
    best_configs = {}
    for dataset in dataset_names:
        dataset_results = {key: value for key, value in results.items() if key[0] == dataset}
        best_key = max(dataset_results.keys(),
                      key=lambda k: dataset_results[k]['cv_scores']['accuracy']['mean'])
        best_configs[dataset] = best_key[1]  # config name

    config_wins = {config: 0 for config in config_names}
    for config in best_configs.values():
        config_wins[config] += 1

    bars = axes[1, 2].bar(config_wins.keys(), config_wins.values(),
                         color=['lightblue', 'lightcoral'], alpha=0.7)
    axes[1, 2].set_ylabel('Number of Datasets Won')
    axes[1, 2].set_title('Best Configuration by Dataset')
    axes[1, 2].grid(True, alpha=0.3)

    for bar, count in zip(bars, config_wins.values()):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                       f'{count}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return perf_df, best_configs

def computational_constraints_analysis():
    """Analyze evaluation under computational constraints."""
    print_step_header(8, "Computational Constraints Analysis")

    # Simulate 24-hour computational budget
    total_budget_hours = 24
    total_budget_seconds = total_budget_hours * 3600

    print(f"24-Hour Computational Budget Analysis:")
    print(f"Total budget: {total_budget_hours} hours ({total_budget_seconds:,} seconds)")
    print()

    # Estimate computational costs
    computational_costs = {
        'dataset_loading': 60,  # seconds per dataset
        'preprocessing': 120,   # seconds per dataset
        'model_training': {
            'decision_stump': {25: 30, 50: 60, 100: 120},      # seconds per config
            'shallow_tree_depth2': {25: 45, 50: 90, 100: 180}
        },
        'cross_validation_multiplier': 5,  # 5-fold CV
        'statistical_testing': 300,        # seconds total
        'visualization': 600               # seconds total
    }

    # Calculate total time for different experiment configurations
    experiment_configs = [
        {
            'name': 'Minimal Experiment',
            'datasets': 3,
            'weak_learners': 1,
            'ensemble_sizes': 2,
            'cv_folds': 3
        },
        {
            'name': 'Standard Experiment',
            'datasets': 5,
            'weak_learners': 2,
            'ensemble_sizes': 3,
            'cv_folds': 5
        },
        {
            'name': 'Comprehensive Experiment',
            'datasets': 7,
            'weak_learners': 5,
            'ensemble_sizes': 6,
            'cv_folds': 10
        }
    ]

    print("Experiment Configuration Analysis:")
    print("-" * 40)

    feasible_configs = []

    for config in experiment_configs:
        # Calculate total time
        total_time = 0

        # Dataset costs
        total_time += config['datasets'] * (computational_costs['dataset_loading'] +
                                          computational_costs['preprocessing'])

        # Training costs
        training_combinations = (config['datasets'] * config['weak_learners'] *
                               config['ensemble_sizes'])

        # Estimate average training time (using decision_stump with 50 estimators as baseline)
        avg_training_time = computational_costs['model_training']['decision_stump'][50]
        total_training_time = (training_combinations * avg_training_time *
                             config['cv_folds'])
        total_time += total_training_time

        # Additional costs
        total_time += computational_costs['statistical_testing']
        total_time += computational_costs['visualization']

        # Convert to hours
        total_hours = total_time / 3600
        feasible = total_time <= total_budget_seconds

        print(f"\n{config['name']}:")
        print(f"  Datasets: {config['datasets']}")
        print(f"  Weak learners: {config['weak_learners']}")
        print(f"  Ensemble sizes: {config['ensemble_sizes']}")
        print(f"  CV folds: {config['cv_folds']}")
        print(f"  Total combinations: {training_combinations}")
        print(f"  Estimated time: {total_hours:.1f} hours")
        print(f"  Feasible in 24h: {'Yes' if feasible else 'No'}")

        if feasible:
            feasible_configs.append(config)

    # Recommend optimal configuration
    if feasible_configs:
        # Choose the most comprehensive feasible configuration
        recommended_config = max(feasible_configs,
                               key=lambda x: x['datasets'] * x['weak_learners'] * x['ensemble_sizes'])

        print(f"\nRecommended Configuration: {recommended_config['name']}")
        print("This provides the most comprehensive evaluation within the time budget.")
    else:
        print(f"\nWarning: No configurations are feasible within 24 hours!")
        print("Consider reducing the scope or using parallel processing.")

    return feasible_configs, recommended_config if feasible_configs else None

def main():
    """Main function to run the comprehensive evaluation framework."""
    print("Question 25: AdaBoost Evaluation Framework")
    print("=" * 60)

    # Design datasets
    datasets = design_comprehensive_datasets()

    # Define metrics
    metrics = define_evaluation_metrics()

    # Design weak learner configurations
    weak_learner_configs = design_weak_learner_configurations()

    # Design ensemble size experiments
    ensemble_sizes = design_ensemble_size_experiments()

    # Run comprehensive evaluation
    results = run_comprehensive_evaluation(datasets, metrics, weak_learner_configs, ensemble_sizes)

    # Statistical significance testing
    statistical_results = statistical_significance_testing(results)

    # Visualize results
    perf_df, best_configs = visualize_evaluation_results(results, statistical_results)

    # Computational constraints analysis
    feasible_configs, recommended_config = computational_constraints_analysis()

    # Summary
    print_step_header(9, "Evaluation Framework Summary")

    print("Key Findings:")
    print("-" * 20)
    print(f"1. Evaluated {len(results)} configuration combinations")
    print(f"2. Performed {len(statistical_results)} statistical tests")
    print(f"3. Best configurations by dataset:")
    for dataset, config in best_configs.items():
        print(f"   - {dataset}: {config}")

    if recommended_config:
        print(f"4. Recommended 24h experiment: {recommended_config['name']}")

    print(f"\nAll visualizations saved to: {save_dir}")

if __name__ == "__main__":
    main()
