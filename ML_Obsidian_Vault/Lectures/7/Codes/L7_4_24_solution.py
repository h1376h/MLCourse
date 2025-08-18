import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.datasets import make_classification
import time
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_24")
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

def analyze_boosting_algorithms():
    """Analyze key differences between boosting algorithms."""
    print_step_header(1, "Analyzing Boosting Algorithm Differences")
    
    algorithms = {
        'AdaBoost': {
            'full_name': 'Adaptive Boosting',
            'loss_function': 'Exponential Loss',
            'weight_update': 'Sample reweighting',
            'weak_learner': 'Decision stumps (typically)',
            'learning_rate': 'Adaptive (based on error)',
            'regularization': 'Limited (early stopping)',
            'interpretability': 9,
            'speed': 8,
            'accuracy': 7,
            'robustness': 6,
            'year_introduced': 1995
        },
        'Gradient Boosting': {
            'full_name': 'Gradient Boosting Machine',
            'loss_function': 'Various (MSE, log-loss, etc.)',
            'weight_update': 'Gradient-based residual fitting',
            'weak_learner': 'Shallow trees (typically)',
            'learning_rate': 'Fixed hyperparameter',
            'regularization': 'Learning rate, tree depth',
            'interpretability': 7,
            'speed': 6,
            'accuracy': 8,
            'robustness': 7,
            'year_introduced': 1999
        },
        'XGBoost': {
            'full_name': 'Extreme Gradient Boosting',
            'loss_function': 'Various + regularization terms',
            'weight_update': 'Second-order gradient optimization',
            'weak_learner': 'Optimized trees',
            'learning_rate': 'Fixed + adaptive options',
            'regularization': 'L1/L2 + tree complexity',
            'interpretability': 5,
            'speed': 9,
            'accuracy': 9,
            'robustness': 8,
            'year_introduced': 2014
        },
        'LightGBM': {
            'full_name': 'Light Gradient Boosting Machine',
            'loss_function': 'Various + regularization',
            'weight_update': 'Gradient-based with optimizations',
            'weak_learner': 'Leaf-wise trees',
            'learning_rate': 'Fixed + adaptive options',
            'regularization': 'Multiple regularization techniques',
            'interpretability': 5,
            'speed': 10,
            'accuracy': 9,
            'robustness': 8,
            'year_introduced': 2017
        }
    }
    
    print("Boosting Algorithm Comparison:")
    print("-" * 40)
    
    for algo, info in algorithms.items():
        print(f"\n{algo} ({info['full_name']}):")
        print(f"  Loss Function: {info['loss_function']}")
        print(f"  Weight Update: {info['weight_update']}")
        print(f"  Weak Learner: {info['weak_learner']}")
        print(f"  Learning Rate: {info['learning_rate']}")
        print(f"  Regularization: {info['regularization']}")
        print(f"  Year Introduced: {info['year_introduced']}")
        print(f"  Scores - Interpretability: {info['interpretability']}/10, "
              f"Speed: {info['speed']}/10, "
              f"Accuracy: {info['accuracy']}/10, "
              f"Robustness: {info['robustness']}/10")
    
    # Visualize algorithm comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    algo_names = list(algorithms.keys())
    interpretability = [info['interpretability'] for info in algorithms.values()]
    speed = [info['speed'] for info in algorithms.values()]
    accuracy = [info['accuracy'] for info in algorithms.values()]
    robustness = [info['robustness'] for info in algorithms.values()]
    
    # Radar chart data
    metrics = ['Interpretability', 'Speed', 'Accuracy', 'Robustness']
    
    # Plot 1: Performance comparison
    x = np.arange(len(algo_names))
    width = 0.2
    
    axes[0, 0].bar(x - 1.5*width, interpretability, width, label='Interpretability', alpha=0.7)
    axes[0, 0].bar(x - 0.5*width, speed, width, label='Speed', alpha=0.7)
    axes[0, 0].bar(x + 0.5*width, accuracy, width, label='Accuracy', alpha=0.7)
    axes[0, 0].bar(x + 1.5*width, robustness, width, label='Robustness', alpha=0.7)
    
    axes[0, 0].set_xlabel('Algorithm')
    axes[0, 0].set_ylabel('Score (1-10)')
    axes[0, 0].set_title('Algorithm Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(algo_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Evolution timeline
    years = [info['year_introduced'] for info in algorithms.values()]
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, (algo, year, color) in enumerate(zip(algo_names, years, colors)):
        axes[0, 1].scatter(year, i, s=200, c=color, alpha=0.7, label=algo)
        axes[0, 1].text(year + 0.5, i, algo, fontsize=10, va='center')
    
    axes[0, 1].set_xlabel('Year Introduced')
    axes[0, 1].set_ylabel('Algorithm')
    axes[0, 1].set_title('Boosting Algorithm Evolution Timeline')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yticks(range(len(algo_names)))
    axes[0, 1].set_yticklabels(algo_names)
    
    # Plot 3: Speed vs Accuracy trade-off
    for i, (algo, color) in enumerate(zip(algo_names, colors)):
        axes[1, 0].scatter(speed[i], accuracy[i], s=200, c=color, alpha=0.7, label=algo)
        axes[1, 0].text(speed[i] + 0.1, accuracy[i], algo, fontsize=9, va='center')
    
    axes[1, 0].set_xlabel('Speed Score')
    axes[1, 0].set_ylabel('Accuracy Score')
    axes[1, 0].set_title('Speed vs Accuracy Trade-off')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Overall score (weighted average)
    weights = {'interpretability': 0.2, 'speed': 0.3, 'accuracy': 0.3, 'robustness': 0.2}
    overall_scores = [
        weights['interpretability'] * interp + 
        weights['speed'] * spd + 
        weights['accuracy'] * acc + 
        weights['robustness'] * rob
        for interp, spd, acc, rob in zip(interpretability, speed, accuracy, robustness)
    ]
    
    bars = axes[1, 1].bar(algo_names, overall_scores, color=colors, alpha=0.7)
    axes[1, 1].set_xlabel('Algorithm')
    axes[1, 1].set_ylabel('Overall Score')
    axes[1, 1].set_title('Overall Performance Score (Weighted)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, overall_scores):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                       f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return algorithms, overall_scores

def when_to_choose_adaboost():
    """Analyze when to choose AdaBoost over other boosting methods."""
    print_step_header(2, "When to Choose AdaBoost Over Other Methods")
    
    scenarios = {
        'Small Dataset (< 1000 samples)': {
            'adaboost_score': 9,
            'gradient_boosting_score': 7,
            'xgboost_score': 6,
            'lightgbm_score': 5,
            'reasoning': 'AdaBoost less prone to overfitting on small datasets'
        },
        'High Interpretability Required': {
            'adaboost_score': 9,
            'gradient_boosting_score': 7,
            'xgboost_score': 4,
            'lightgbm_score': 4,
            'reasoning': 'AdaBoost with decision stumps is highly interpretable'
        },
        'Limited Computational Resources': {
            'adaboost_score': 8,
            'gradient_boosting_score': 6,
            'xgboost_score': 7,
            'lightgbm_score': 9,
            'reasoning': 'AdaBoost is simple and fast to train'
        },
        'Binary Classification': {
            'adaboost_score': 9,
            'gradient_boosting_score': 8,
            'xgboost_score': 8,
            'lightgbm_score': 8,
            'reasoning': 'AdaBoost was originally designed for binary classification'
        },
        'Noisy Data': {
            'adaboost_score': 4,
            'gradient_boosting_score': 6,
            'xgboost_score': 7,
            'lightgbm_score': 7,
            'reasoning': 'AdaBoost sensitive to noise and outliers'
        },
        'Large Dataset (> 100k samples)': {
            'adaboost_score': 5,
            'gradient_boosting_score': 6,
            'xgboost_score': 9,
            'lightgbm_score': 10,
            'reasoning': 'Modern boosting methods scale better'
        },
        'Feature Selection Important': {
            'adaboost_score': 8,
            'gradient_boosting_score': 7,
            'xgboost_score': 9,
            'lightgbm_score': 9,
            'reasoning': 'AdaBoost naturally performs feature selection'
        },
        'Real-time Prediction': {
            'adaboost_score': 8,
            'gradient_boosting_score': 6,
            'xgboost_score': 7,
            'lightgbm_score': 9,
            'reasoning': 'Simple ensemble structure enables fast prediction'
        }
    }
    
    print("Scenario-based Algorithm Selection:")
    print("-" * 40)
    
    for scenario, scores in scenarios.items():
        print(f"\n{scenario}:")
        print(f"  AdaBoost: {scores['adaboost_score']}/10")
        print(f"  Gradient Boosting: {scores['gradient_boosting_score']}/10")
        print(f"  XGBoost: {scores['xgboost_score']}/10")
        print(f"  LightGBM: {scores['lightgbm_score']}/10")
        print(f"  Reasoning: {scores['reasoning']}")
        
        # Determine best algorithm for this scenario
        algo_scores = {
            'AdaBoost': scores['adaboost_score'],
            'Gradient Boosting': scores['gradient_boosting_score'],
            'XGBoost': scores['xgboost_score'],
            'LightGBM': scores['lightgbm_score']
        }
        best_algo = max(algo_scores.keys(), key=lambda x: algo_scores[x])
        print(f"  Recommended: {best_algo}")
    
    # Visualize scenario analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    scenario_names = list(scenarios.keys())
    adaboost_scores = [s['adaboost_score'] for s in scenarios.values()]
    gb_scores = [s['gradient_boosting_score'] for s in scenarios.values()]
    xgb_scores = [s['xgboost_score'] for s in scenarios.values()]
    lgb_scores = [s['lightgbm_score'] for s in scenarios.values()]
    
    # Heatmap of all scores
    score_matrix = np.array([adaboost_scores, gb_scores, xgb_scores, lgb_scores])
    
    im = axes[0, 0].imshow(score_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)
    axes[0, 0].set_xticks(range(len(scenario_names)))
    axes[0, 0].set_xticklabels([name.replace(' ', '\n') for name in scenario_names], rotation=45)
    axes[0, 0].set_yticks(range(4))
    axes[0, 0].set_yticklabels(['AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM'])
    axes[0, 0].set_title('Algorithm Suitability by Scenario')
    
    # Add text annotations
    for i in range(4):
        for j in range(len(scenario_names)):
            text = axes[0, 0].text(j, i, f'{score_matrix[i, j]}',
                                  ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=axes[0, 0])
    
    # AdaBoost advantage scenarios
    adaboost_advantages = []
    for scenario, scores in scenarios.items():
        if scores['adaboost_score'] >= max(scores['gradient_boosting_score'], 
                                          scores['xgboost_score'], 
                                          scores['lightgbm_score']):
            adaboost_advantages.append(scenario)
    
    advantage_scores = [scenarios[scenario]['adaboost_score'] for scenario in adaboost_advantages]
    
    if adaboost_advantages:
        bars = axes[0, 1].bar(range(len(adaboost_advantages)), advantage_scores, 
                             color='blue', alpha=0.7)
        axes[0, 1].set_xticks(range(len(adaboost_advantages)))
        axes[0, 1].set_xticklabels([name.replace(' ', '\n') for name in adaboost_advantages])
        axes[0, 1].set_ylabel('AdaBoost Score')
        axes[0, 1].set_title('Scenarios Where AdaBoost Excels')
        axes[0, 1].grid(True, alpha=0.3)
        
        for bar, score in zip(bars, advantage_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                           f'{score}', ha='center', va='bottom')
    
    # Algorithm win count
    win_counts = {'AdaBoost': 0, 'Gradient Boosting': 0, 'XGBoost': 0, 'LightGBM': 0}
    
    for scenario, scores in scenarios.items():
        algo_scores = {
            'AdaBoost': scores['adaboost_score'],
            'Gradient Boosting': scores['gradient_boosting_score'],
            'XGBoost': scores['xgboost_score'],
            'LightGBM': scores['lightgbm_score']
        }
        winner = max(algo_scores.keys(), key=lambda x: algo_scores[x])
        win_counts[winner] += 1
    
    wedges, texts, autotexts = axes[1, 0].pie(win_counts.values(), 
                                             labels=win_counts.keys(),
                                             autopct='%1.1f%%',
                                             colors=['blue', 'green', 'red', 'orange'])
    axes[1, 0].set_title('Algorithm Wins by Scenario')
    
    # Average scores across all scenarios
    avg_scores = {
        'AdaBoost': np.mean(adaboost_scores),
        'Gradient Boosting': np.mean(gb_scores),
        'XGBoost': np.mean(xgb_scores),
        'LightGBM': np.mean(lgb_scores)
    }
    
    bars2 = axes[1, 1].bar(avg_scores.keys(), avg_scores.values(), 
                          color=['blue', 'green', 'red', 'orange'], alpha=0.7)
    axes[1, 1].set_ylabel('Average Score')
    axes[1, 1].set_title('Average Performance Across All Scenarios')
    axes[1, 1].grid(True, alpha=0.3)
    
    for bar, score in zip(bars2, avg_scores.values()):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                       f'{score:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scenario_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return scenarios, adaboost_advantages, win_counts

def computational_trade_offs():
    """Analyze computational trade-offs between boosting methods."""
    print_step_header(3, "Computational Trade-offs Analysis")
    
    # Simulated computational metrics (relative to AdaBoost = 1.0)
    computational_metrics = {
        'AdaBoost': {
            'training_time': 1.0,
            'prediction_time': 1.0,
            'memory_usage': 1.0,
            'hyperparameter_tuning': 1.0,
            'scalability': 1.0
        },
        'Gradient Boosting': {
            'training_time': 2.5,
            'prediction_time': 1.2,
            'memory_usage': 1.5,
            'hyperparameter_tuning': 3.0,
            'scalability': 2.0
        },
        'XGBoost': {
            'training_time': 1.8,
            'prediction_time': 1.1,
            'memory_usage': 1.3,
            'hyperparameter_tuning': 4.0,
            'scalability': 4.0
        },
        'LightGBM': {
            'training_time': 1.2,
            'prediction_time': 0.9,
            'memory_usage': 1.1,
            'hyperparameter_tuning': 3.5,
            'scalability': 5.0
        }
    }
    
    print("Computational Trade-offs (relative to AdaBoost):")
    print("-" * 50)
    
    for algo, metrics in computational_metrics.items():
        print(f"\n{algo}:")
        print(f"  Training Time: {metrics['training_time']:.1f}x")
        print(f"  Prediction Time: {metrics['prediction_time']:.1f}x")
        print(f"  Memory Usage: {metrics['memory_usage']:.1f}x")
        print(f"  Hyperparameter Tuning Complexity: {metrics['hyperparameter_tuning']:.1f}x")
        print(f"  Scalability: {metrics['scalability']:.1f}x")
    
    return computational_metrics

def visualize_computational_tradeoffs(computational_metrics):
    """Visualize computational trade-offs between algorithms."""
    print_step_header(4, "Visualizing Computational Trade-offs")

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    algo_names = list(computational_metrics.keys())
    colors = ['blue', 'green', 'red', 'orange']

    # Extract metrics
    training_times = [metrics['training_time'] for metrics in computational_metrics.values()]
    prediction_times = [metrics['prediction_time'] for metrics in computational_metrics.values()]
    memory_usage = [metrics['memory_usage'] for metrics in computational_metrics.values()]
    tuning_complexity = [metrics['hyperparameter_tuning'] for metrics in computational_metrics.values()]
    scalability = [metrics['scalability'] for metrics in computational_metrics.values()]

    # Plot 1: Training time comparison
    bars1 = axes[0, 0].bar(algo_names, training_times, color=colors, alpha=0.7)
    axes[0, 0].set_ylabel('Relative Training Time')
    axes[0, 0].set_title('Training Time Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='AdaBoost baseline')

    for bar, time_val in zip(bars1, training_times):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                       f'{time_val:.1f}x', ha='center', va='bottom')

    # Plot 2: Prediction time comparison
    bars2 = axes[0, 1].bar(algo_names, prediction_times, color=colors, alpha=0.7)
    axes[0, 1].set_ylabel('Relative Prediction Time')
    axes[0, 1].set_title('Prediction Time Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

    for bar, time_val in zip(bars2, prediction_times):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                       f'{time_val:.1f}x', ha='center', va='bottom')

    # Plot 3: Memory usage comparison
    bars3 = axes[0, 2].bar(algo_names, memory_usage, color=colors, alpha=0.7)
    axes[0, 2].set_ylabel('Relative Memory Usage')
    axes[0, 2].set_title('Memory Usage Comparison')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

    for bar, mem_val in zip(bars3, memory_usage):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                       f'{mem_val:.1f}x', ha='center', va='bottom')

    # Plot 4: Hyperparameter tuning complexity
    bars4 = axes[1, 0].bar(algo_names, tuning_complexity, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('Relative Tuning Complexity')
    axes[1, 0].set_title('Hyperparameter Tuning Complexity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

    for bar, tune_val in zip(bars4, tuning_complexity):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                       f'{tune_val:.1f}x', ha='center', va='bottom')

    # Plot 5: Scalability comparison
    bars5 = axes[1, 1].bar(algo_names, scalability, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Relative Scalability')
    axes[1, 1].set_title('Scalability Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

    for bar, scale_val in zip(bars5, scalability):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                       f'{scale_val:.1f}x', ha='center', va='bottom')

    # Plot 6: Overall computational efficiency
    # Lower is better for most metrics except scalability
    efficiency_scores = []
    for i, algo in enumerate(algo_names):
        # Inverse for time and memory (lower is better), direct for scalability
        score = (1/training_times[i] + 1/prediction_times[i] + 1/memory_usage[i] +
                1/tuning_complexity[i] + scalability[i]) / 5
        efficiency_scores.append(score)

    bars6 = axes[1, 2].bar(algo_names, efficiency_scores, color=colors, alpha=0.7)
    axes[1, 2].set_ylabel('Computational Efficiency Score')
    axes[1, 2].set_title('Overall Computational Efficiency')
    axes[1, 2].grid(True, alpha=0.3)

    for bar, eff_val in zip(bars6, efficiency_scores):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{eff_val:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'computational_tradeoffs.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return efficiency_scores

def small_dataset_analysis():
    """Analyze performance on small datasets (< 1000 samples)."""
    print_step_header(5, "Small Dataset Analysis")

    print("Why AdaBoost excels on small datasets:")
    print("-" * 40)

    small_dataset_factors = {
        'Overfitting Resistance': {
            'adaboost': 'High - simple weak learners reduce overfitting',
            'gradient_boosting': 'Medium - requires careful regularization',
            'xgboost': 'Medium - many hyperparameters to tune',
            'lightgbm': 'Low - can easily overfit small datasets'
        },
        'Hyperparameter Sensitivity': {
            'adaboost': 'Low - few hyperparameters, robust defaults',
            'gradient_boosting': 'Medium - learning rate and depth important',
            'xgboost': 'High - many hyperparameters to optimize',
            'lightgbm': 'High - requires careful tuning'
        },
        'Training Stability': {
            'adaboost': 'High - consistent performance across runs',
            'gradient_boosting': 'Medium - some variance in results',
            'xgboost': 'Medium - depends on hyperparameter settings',
            'lightgbm': 'Low - can be unstable on small data'
        },
        'Interpretability': {
            'adaboost': 'High - simple decision stumps are interpretable',
            'gradient_boosting': 'Medium - tree structure somewhat interpretable',
            'xgboost': 'Low - complex ensemble difficult to interpret',
            'lightgbm': 'Low - complex ensemble difficult to interpret'
        }
    }

    for factor, comparisons in small_dataset_factors.items():
        print(f"\n{factor}:")
        for algo, description in comparisons.items():
            print(f"  {algo.replace('_', ' ').title()}: {description}")

    # Simulate small dataset performance
    np.random.seed(42)
    dataset_sizes = [100, 250, 500, 750, 1000]

    # Simulated performance metrics (accuracy)
    performance_data = {
        'AdaBoost': [0.82, 0.85, 0.87, 0.88, 0.89],
        'Gradient Boosting': [0.78, 0.83, 0.86, 0.88, 0.90],
        'XGBoost': [0.75, 0.81, 0.85, 0.88, 0.91],
        'LightGBM': [0.72, 0.79, 0.84, 0.87, 0.90]
    }

    # Simulated variance (higher for complex methods on small data)
    variance_data = {
        'AdaBoost': [0.03, 0.025, 0.02, 0.018, 0.015],
        'Gradient Boosting': [0.05, 0.04, 0.03, 0.025, 0.02],
        'XGBoost': [0.08, 0.06, 0.04, 0.03, 0.025],
        'LightGBM': [0.10, 0.08, 0.05, 0.04, 0.03]
    }

    # Visualize small dataset analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    colors = ['blue', 'green', 'red', 'orange']

    # Plot 1: Performance vs dataset size
    for i, (algo, performance) in enumerate(performance_data.items()):
        axes[0, 0].plot(dataset_sizes, performance, color=colors[i],
                       linewidth=2, marker='o', label=algo)

    axes[0, 0].set_xlabel('Dataset Size')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Performance vs Dataset Size')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Variance vs dataset size
    for i, (algo, variance) in enumerate(variance_data.items()):
        axes[0, 1].plot(dataset_sizes, variance, color=colors[i],
                       linewidth=2, marker='s', label=algo)

    axes[0, 1].set_xlabel('Dataset Size')
    axes[0, 1].set_ylabel('Performance Variance')
    axes[0, 1].set_title('Performance Stability vs Dataset Size')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Small dataset advantage (performance at 250 samples)
    small_performance = [perf[1] for perf in performance_data.values()]  # 250 samples
    small_variance = [var[1] for var in variance_data.values()]  # 250 samples

    algo_names = list(performance_data.keys())

    bars = axes[1, 0].bar(algo_names, small_performance, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('Accuracy (250 samples)')
    axes[1, 0].set_title('Performance on Small Dataset (250 samples)')
    axes[1, 0].grid(True, alpha=0.3)

    for bar, perf in zip(bars, small_performance):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                       f'{perf:.3f}', ha='center', va='bottom')

    # Plot 4: Stability comparison (inverse of variance)
    stability_scores = [1/var for var in small_variance]

    bars2 = axes[1, 1].bar(algo_names, stability_scores, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Stability Score (1/variance)')
    axes[1, 1].set_title('Training Stability on Small Dataset')
    axes[1, 1].grid(True, alpha=0.3)

    for bar, stab in zip(bars2, stability_scores):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                       f'{stab:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'small_dataset_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSmall Dataset Performance Summary:")
    print(f"Best performer on 250 samples: {algo_names[np.argmax(small_performance)]}")
    print(f"Most stable on 250 samples: {algo_names[np.argmax(stability_scores)]}")

    return performance_data, variance_data

def business_stakeholder_explanation():
    """Provide explanations suitable for business stakeholders."""
    print_step_header(6, "Explaining to Business Stakeholders")

    business_explanations = {
        'AdaBoost': {
            'simple_description': 'Learns from mistakes by focusing on difficult cases',
            'business_analogy': 'Like a teacher who spends extra time with struggling students',
            'key_benefits': ['Easy to understand', 'Works well with small data', 'Fast results'],
            'when_to_use': 'Small datasets, need interpretability, limited resources',
            'business_value': 'Quick insights with minimal complexity'
        },
        'Gradient Boosting': {
            'simple_description': 'Builds models step-by-step, each fixing previous errors',
            'business_analogy': 'Like iterative product development - each version improves on the last',
            'key_benefits': ['High accuracy', 'Handles complex patterns', 'Proven track record'],
            'when_to_use': 'Medium datasets, accuracy is priority, have ML expertise',
            'business_value': 'Reliable high-performance predictions'
        },
        'XGBoost': {
            'simple_description': 'Advanced machine learning with many optimization tricks',
            'business_analogy': 'Like a high-performance sports car - powerful but needs expert driver',
            'key_benefits': ['State-of-the-art accuracy', 'Handles large data', 'Competition winner'],
            'when_to_use': 'Large datasets, maximum accuracy needed, have ML team',
            'business_value': 'Best-in-class predictions for critical decisions'
        },
        'LightGBM': {
            'simple_description': 'Fastest advanced machine learning for large datasets',
            'business_analogy': 'Like a Formula 1 car - extremely fast and efficient',
            'key_benefits': ['Very fast training', 'Handles huge datasets', 'Low memory usage'],
            'when_to_use': 'Very large datasets, speed is critical, real-time applications',
            'business_value': 'Rapid insights from big data'
        }
    }

    print("Business-Friendly Algorithm Explanations:")
    print("-" * 45)

    for algo, explanation in business_explanations.items():
        print(f"\n{algo}:")
        print(f"  What it does: {explanation['simple_description']}")
        print(f"  Think of it like: {explanation['business_analogy']}")
        print(f"  Key benefits: {', '.join(explanation['key_benefits'])}")
        print(f"  When to use: {explanation['when_to_use']}")
        print(f"  Business value: {explanation['business_value']}")

    # Create decision matrix for business users
    decision_factors = {
        'Data Size': {
            'Small (< 1K)': 'AdaBoost',
            'Medium (1K-100K)': 'Gradient Boosting',
            'Large (> 100K)': 'XGBoost/LightGBM'
        },
        'Accuracy Priority': {
            'Good enough': 'AdaBoost',
            'High accuracy': 'Gradient Boosting',
            'Maximum accuracy': 'XGBoost'
        },
        'Speed Priority': {
            'Not important': 'Any',
            'Important': 'AdaBoost',
            'Critical': 'LightGBM'
        },
        'Interpretability': {
            'Must explain': 'AdaBoost',
            'Some explanation': 'Gradient Boosting',
            'Black box OK': 'XGBoost/LightGBM'
        },
        'Team Expertise': {
            'Basic': 'AdaBoost',
            'Intermediate': 'Gradient Boosting',
            'Advanced': 'XGBoost/LightGBM'
        }
    }

    print(f"\nDecision Matrix for Business Users:")
    print("-" * 35)

    for factor, options in decision_factors.items():
        print(f"\n{factor}:")
        for situation, recommendation in options.items():
            print(f"  {situation}: {recommendation}")

    return business_explanations, decision_factors

def main():
    """Main function to run the complete boosting algorithm comparison."""
    print("Question 24: Boosting Algorithm Comparison")
    print("=" * 60)

    # Analyze algorithm differences
    algorithms, overall_scores = analyze_boosting_algorithms()

    # When to choose AdaBoost
    scenarios, ada_advantages, win_counts = when_to_choose_adaboost()

    # Computational trade-offs
    comp_metrics = computational_trade_offs()

    # Visualize computational trade-offs
    efficiency_scores = visualize_computational_tradeoffs(comp_metrics)

    # Small dataset analysis
    performance_data, variance_data = small_dataset_analysis()

    # Business explanations
    business_explanations, decision_factors = business_stakeholder_explanation()

    # Summary
    print_step_header(7, "Summary and Recommendations")

    print("Key Findings:")
    print("-" * 20)
    best_overall = list(algorithms.keys())[np.argmax(overall_scores)]
    most_efficient = list(comp_metrics.keys())[np.argmax(efficiency_scores)]
    adaboost_wins = len(ada_advantages)

    print(f"1. Best overall algorithm: {best_overall}")
    print(f"2. Most computationally efficient: {most_efficient}")
    print(f"3. AdaBoost wins in {adaboost_wins}/{len(scenarios)} scenarios")
    print(f"4. AdaBoost best for: {', '.join(ada_advantages[:3])}")
    print(f"5. Choose AdaBoost when: interpretability and small data are priorities")

    print(f"\nAll visualizations saved to: {save_dir}")

if __name__ == "__main__":
    main()
