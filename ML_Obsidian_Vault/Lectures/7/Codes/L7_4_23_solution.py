import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import deque
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_23")
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

class OnlineAdaBoost:
    """Online AdaBoost implementation for streaming data."""
    
    def __init__(self, max_learners=50, learning_rate=1.0, memory_limit=1000):
        self.max_learners = max_learners
        self.learning_rate = learning_rate
        self.memory_limit = memory_limit
        
        self.weak_learners = []
        self.alphas = []
        self.sample_weights = None
        self.memory_buffer = deque(maxlen=memory_limit)
        
        # Performance tracking
        self.performance_history = []
        self.drift_detections = []
        
    def partial_fit(self, X, y):
        """Incrementally fit the model with new data."""
        n_samples = len(X)
        
        # Initialize sample weights if first batch
        if self.sample_weights is None:
            self.sample_weights = np.ones(n_samples) / n_samples
        else:
            # Extend weights for new samples
            new_weights = np.ones(n_samples) / n_samples
            self.sample_weights = np.concatenate([self.sample_weights, new_weights])
        
        # Add to memory buffer
        for i in range(n_samples):
            self.memory_buffer.append((X[i], y[i]))
        
        # Train new weak learner if we have enough data
        if len(self.memory_buffer) >= 100:  # Minimum batch size
            self._train_weak_learner()
        
        # Limit ensemble size
        if len(self.weak_learners) > self.max_learners:
            self._remove_oldest_learner()
    
    def _train_weak_learner(self):
        """Train a new weak learner on current memory buffer."""
        # Extract data from memory buffer
        buffer_data = list(self.memory_buffer)
        X_buffer = np.array([item[0] for item in buffer_data])
        y_buffer = np.array([item[1] for item in buffer_data])
        
        # Use recent sample weights (last len(buffer_data) weights)
        recent_weights = self.sample_weights[-len(buffer_data):]
        
        # Train weak learner
        weak_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
        weak_learner.fit(X_buffer, y_buffer, sample_weight=recent_weights)
        
        # Calculate weighted error
        predictions = weak_learner.predict(X_buffer)
        errors = (predictions != y_buffer).astype(float)
        weighted_error = np.average(errors, weights=recent_weights)
        
        # Only add if better than random
        if weighted_error < 0.5:
            # Calculate alpha
            alpha = self.learning_rate * 0.5 * np.log((1 - weighted_error) / max(weighted_error, 1e-10))
            
            # Add to ensemble
            self.weak_learners.append(weak_learner)
            self.alphas.append(alpha)
            
            # Update sample weights
            self._update_sample_weights(X_buffer, y_buffer, predictions, alpha)
    
    def _update_sample_weights(self, X, y, predictions, alpha):
        """Update sample weights based on predictions."""
        # Calculate weight updates for recent samples
        weight_updates = np.exp(-alpha * y * predictions)
        
        # Update the recent weights
        start_idx = len(self.sample_weights) - len(X)
        self.sample_weights[start_idx:] *= weight_updates
        
        # Normalize weights
        self.sample_weights /= np.sum(self.sample_weights)
    
    def _remove_oldest_learner(self):
        """Remove the oldest weak learner to maintain memory constraints."""
        if self.weak_learners:
            self.weak_learners.pop(0)
            self.alphas.pop(0)
    
    def predict(self, X):
        """Make predictions using the current ensemble."""
        if not self.weak_learners:
            return np.zeros(len(X))
        
        # Combine predictions from all weak learners
        ensemble_predictions = np.zeros(len(X))
        
        for learner, alpha in zip(self.weak_learners, self.alphas):
            predictions = learner.predict(X)
            ensemble_predictions += alpha * predictions
        
        return np.sign(ensemble_predictions)
    
    def get_ensemble_size(self):
        """Get current ensemble size."""
        return len(self.weak_learners)

def generate_streaming_data_with_drift():
    """Generate streaming data with concept drift."""
    print_step_header(1, "Generating Streaming Data with Concept Drift")
    
    np.random.seed(42)
    
    # Parameters
    n_total_samples = 5000
    n_features = 10
    drift_points = [1500, 3000, 4000]  # Points where concept drift occurs
    
    print(f"Streaming Data Parameters:")
    print(f"- Total samples: {n_total_samples}")
    print(f"- Features: {n_features}")
    print(f"- Concept drift points: {drift_points}")
    
    # Generate data with different concepts
    X_all = []
    y_all = []
    concept_labels = []
    
    current_sample = 0
    concept_id = 0
    
    while current_sample < n_total_samples:
        # Determine next drift point
        next_drift = drift_points[concept_id] if concept_id < len(drift_points) else n_total_samples
        batch_size = min(next_drift - current_sample, n_total_samples - current_sample)
        
        # Generate data for current concept
        X_batch, y_batch = generate_concept_data(batch_size, n_features, concept_id)
        
        X_all.append(X_batch)
        y_all.append(y_batch)
        concept_labels.extend([concept_id] * batch_size)
        
        current_sample += batch_size
        concept_id += 1
    
    X = np.vstack(X_all)
    y = np.hstack(y_all)
    concept_labels = np.array(concept_labels)
    
    print(f"\nGenerated Data Statistics:")
    for i in range(concept_id):
        concept_mask = concept_labels == i
        concept_size = np.sum(concept_mask)
        concept_balance = np.mean(y[concept_mask])
        print(f"- Concept {i}: {concept_size} samples, {concept_balance:.1%} positive")
    
    return X, y, concept_labels, drift_points

def generate_concept_data(n_samples, n_features, concept_id):
    """Generate data for a specific concept."""
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Different concepts have different decision boundaries
    if concept_id == 0:
        # Linear boundary: x1 + x2 > 0
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    elif concept_id == 1:
        # Rotated boundary: x1 - x2 > 0
        y = (X[:, 0] - X[:, 1] > 0).astype(int)
    elif concept_id == 2:
        # Quadratic boundary: x1^2 + x2^2 > 1
        y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int)
    else:
        # Complex boundary: sin(x1) + cos(x2) > 0
        y = (np.sin(X[:, 0]) + np.cos(X[:, 1]) > 0).astype(int)
    
    # Convert to {-1, +1} labels
    y = 2 * y - 1
    
    return X, y

def analyze_streaming_challenges():
    """Analyze challenges posed by streaming data for AdaBoost."""
    print_step_header(2, "Analyzing Streaming Data Challenges")
    
    challenges = {
        'Memory Constraints': {
            'description': 'Limited memory to store all historical data',
            'impact': 'High',
            'solutions': ['Sliding window', 'Sample selection', 'Forgetting mechanisms'],
            'severity': 9
        },
        'Concept Drift': {
            'description': 'Data distribution changes over time',
            'impact': 'High',
            'solutions': ['Drift detection', 'Model adaptation', 'Ensemble pruning'],
            'severity': 9
        },
        'Real-time Processing': {
            'description': 'Need to process data as it arrives',
            'impact': 'Medium',
            'solutions': ['Incremental learning', 'Batch processing', 'Parallel processing'],
            'severity': 7
        },
        'Limited Training Data': {
            'description': 'Cannot wait for large batches to train',
            'impact': 'Medium',
            'solutions': ['Online learning', 'Transfer learning', 'Active learning'],
            'severity': 6
        },
        'Model Staleness': {
            'description': 'Model becomes outdated as data evolves',
            'impact': 'High',
            'solutions': ['Continuous updating', 'Model versioning', 'Adaptive learning rates'],
            'severity': 8
        }
    }
    
    print("Streaming Data Challenges for AdaBoost:")
    print("-" * 45)
    
    for challenge, info in challenges.items():
        print(f"\n{challenge}:")
        print(f"  Description: {info['description']}")
        print(f"  Impact: {info['impact']}")
        print(f"  Severity: {info['severity']}/10")
        print(f"  Solutions: {', '.join(info['solutions'])}")
    
    # Visualize challenges
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    challenge_names = list(challenges.keys())
    severities = [info['severity'] for info in challenges.values()]
    
    # Severity comparison
    colors = ['red', 'orange', 'blue', 'green', 'purple']
    bars = axes[0, 0].bar(range(len(challenge_names)), severities, 
                         color=colors, alpha=0.7)
    axes[0, 0].set_xticks(range(len(challenge_names)))
    axes[0, 0].set_xticklabels([name.replace(' ', '\n') for name in challenge_names])
    axes[0, 0].set_ylabel('Severity (1-10)')
    axes[0, 0].set_title('Challenge Severity Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    
    for bar, severity in zip(bars, severities):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                       f'{severity}', ha='center', va='bottom')
    
    # Impact distribution
    impact_counts = {'High': 0, 'Medium': 0, 'Low': 0}
    for info in challenges.values():
        impact_counts[info['impact']] += 1
    
    wedges, texts, autotexts = axes[0, 1].pie(impact_counts.values(), 
                                             labels=impact_counts.keys(),
                                             autopct='%1.1f%%',
                                             colors=['red', 'orange', 'green'])
    axes[0, 1].set_title('Challenge Impact Distribution')
    
    # Solutions count
    all_solutions = []
    for info in challenges.values():
        all_solutions.extend(info['solutions'])
    
    unique_solutions = list(set(all_solutions))
    solution_counts = [all_solutions.count(sol) for sol in unique_solutions]
    
    axes[1, 0].barh(range(len(unique_solutions)), solution_counts, 
                   color='lightblue', alpha=0.7)
    axes[1, 0].set_yticks(range(len(unique_solutions)))
    axes[1, 0].set_yticklabels(unique_solutions)
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_title('Most Common Solutions')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Challenge timeline (simulated)
    time_points = np.arange(0, 100, 10)
    challenge_evolution = {
        'Memory Constraints': 5 + 3 * np.sin(time_points * 0.1),
        'Concept Drift': 3 + 4 * np.sin(time_points * 0.05 + 1),
        'Real-time Processing': 6 + 2 * np.cos(time_points * 0.08),
        'Model Staleness': 4 + 3 * np.sin(time_points * 0.12 + 2)
    }
    
    for challenge, evolution in challenge_evolution.items():
        axes[1, 1].plot(time_points, evolution, linewidth=2, label=challenge)
    
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Challenge Intensity')
    axes[1, 1].set_title('Challenge Evolution Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'streaming_challenges.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return challenges

def implement_concept_drift_detection():
    """Implement concept drift detection strategies."""
    print_step_header(3, "Concept Drift Detection Strategies")
    
    drift_detection_methods = {
        'ADWIN': {
            'description': 'Adaptive Windowing - detects changes in data distribution',
            'complexity': 'Medium',
            'sensitivity': 8,
            'false_positive_rate': 'Low',
            'implementation': 'Maintains adaptive window size based on change detection'
        },
        'DDM': {
            'description': 'Drift Detection Method - monitors error rate changes',
            'complexity': 'Low',
            'sensitivity': 6,
            'false_positive_rate': 'Medium',
            'implementation': 'Tracks error rate and standard deviation'
        },
        'EDDM': {
            'description': 'Early Drift Detection Method - improved version of DDM',
            'complexity': 'Medium',
            'sensitivity': 7,
            'false_positive_rate': 'Low',
            'implementation': 'Monitors distance between classification errors'
        },
        'Page-Hinkley': {
            'description': 'Sequential change detection test',
            'complexity': 'Low',
            'sensitivity': 9,
            'false_positive_rate': 'High',
            'implementation': 'Cumulative sum of deviations from mean'
        },
        'Statistical Test': {
            'description': 'Kolmogorov-Smirnov or other statistical tests',
            'complexity': 'High',
            'sensitivity': 8,
            'false_positive_rate': 'Low',
            'implementation': 'Compare distributions using statistical tests'
        }
    }
    
    print("Concept Drift Detection Methods:")
    print("-" * 35)
    
    for method, info in drift_detection_methods.items():
        print(f"\n{method}:")
        print(f"  Description: {info['description']}")
        print(f"  Complexity: {info['complexity']}")
        print(f"  Sensitivity: {info['sensitivity']}/10")
        print(f"  False Positive Rate: {info['false_positive_rate']}")
        print(f"  Implementation: {info['implementation']}")
    
    return drift_detection_methods

def simulate_online_adaboost_performance():
    """Simulate Online AdaBoost performance on streaming data."""
    print_step_header(4, "Simulating Online AdaBoost Performance")
    
    # Generate streaming data
    X, y, concept_labels, drift_points = generate_streaming_data_with_drift()
    
    # Initialize online AdaBoost
    online_ada = OnlineAdaBoost(max_learners=50, memory_limit=500)
    
    # Simulate streaming process
    batch_size = 100
    n_batches = len(X) // batch_size
    
    performance_metrics = {
        'accuracy': [],
        'ensemble_size': [],
        'concept': [],
        'batch_number': []
    }
    
    print(f"Simulating streaming process:")
    print(f"- Total samples: {len(X)}")
    print(f"- Batch size: {batch_size}")
    print(f"- Number of batches: {n_batches}")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(X))
        
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        
        # Train on current batch
        online_ada.partial_fit(X_batch, y_batch)
        
        # Evaluate on current batch
        if online_ada.get_ensemble_size() > 0:
            predictions = online_ada.predict(X_batch)
            accuracy = accuracy_score(y_batch, predictions)
        else:
            accuracy = 0.5  # Random performance
        
        # Record metrics
        current_concept = concept_labels[start_idx]
        performance_metrics['accuracy'].append(accuracy)
        performance_metrics['ensemble_size'].append(online_ada.get_ensemble_size())
        performance_metrics['concept'].append(current_concept)
        performance_metrics['batch_number'].append(batch_idx)
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}: Accuracy = {accuracy:.3f}, "
                  f"Ensemble size = {online_ada.get_ensemble_size()}, "
                  f"Concept = {current_concept}")
    
    return performance_metrics, drift_points

def visualize_streaming_performance(performance_metrics, drift_points):
    """Visualize the performance of online AdaBoost on streaming data."""
    print_step_header(5, "Visualizing Streaming Performance")

    # Convert to arrays for easier plotting
    accuracies = np.array(performance_metrics['accuracy'])
    ensemble_sizes = np.array(performance_metrics['ensemble_size'])
    concepts = np.array(performance_metrics['concept'])
    batch_numbers = np.array(performance_metrics['batch_number'])

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Accuracy over time with concept drift points
    axes[0, 0].plot(batch_numbers, accuracies, 'b-', linewidth=2, alpha=0.7)

    # Add concept drift indicators
    for drift_point in drift_points:
        drift_batch = drift_point // 100  # Convert to batch number
        if drift_batch < len(batch_numbers):
            axes[0, 0].axvline(x=drift_batch, color='red', linestyle='--',
                              alpha=0.7, linewidth=2)

    # Color background by concept
    unique_concepts = np.unique(concepts)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']

    for i, concept in enumerate(unique_concepts):
        concept_mask = concepts == concept
        if np.any(concept_mask):
            concept_batches = batch_numbers[concept_mask]
            if len(concept_batches) > 0:
                start_batch = concept_batches[0]
                end_batch = concept_batches[-1]
                axes[0, 0].axvspan(start_batch, end_batch, alpha=0.2,
                                  color=colors[i % len(colors)],
                                  label=f'Concept {concept}')

    axes[0, 0].set_xlabel('Batch Number')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Online AdaBoost Accuracy Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)

    # Plot 2: Ensemble size evolution
    axes[0, 1].plot(batch_numbers, ensemble_sizes, 'g-', linewidth=2, marker='o', markersize=3)

    # Add drift points
    for drift_point in drift_points:
        drift_batch = drift_point // 100
        if drift_batch < len(batch_numbers):
            axes[0, 1].axvline(x=drift_batch, color='red', linestyle='--',
                              alpha=0.7, linewidth=2)

    axes[0, 1].set_xlabel('Batch Number')
    axes[0, 1].set_ylabel('Ensemble Size')
    axes[0, 1].set_title('Ensemble Size Evolution')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Accuracy distribution by concept
    concept_accuracies = {}
    for concept in unique_concepts:
        concept_mask = concepts == concept
        concept_accuracies[concept] = accuracies[concept_mask]

    box_data = [concept_accuracies[concept] for concept in unique_concepts]
    box_labels = [f'Concept {concept}' for concept in unique_concepts]

    bp = axes[1, 0].boxplot(box_data, labels=box_labels, patch_artist=True)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors[:len(unique_concepts)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy Distribution by Concept')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Performance degradation around drift points
    drift_analysis = []
    window_size = 5  # Batches before and after drift

    for drift_point in drift_points:
        drift_batch = drift_point // 100

        # Get accuracy before and after drift
        before_start = max(0, drift_batch - window_size)
        before_end = drift_batch
        after_start = drift_batch
        after_end = min(len(accuracies), drift_batch + window_size)

        if before_end > before_start and after_end > after_start:
            acc_before = np.mean(accuracies[before_start:before_end])
            acc_after = np.mean(accuracies[after_start:after_end])
            degradation = acc_before - acc_after
            drift_analysis.append({
                'drift_point': drift_point,
                'acc_before': acc_before,
                'acc_after': acc_after,
                'degradation': degradation
            })

    if drift_analysis:
        drift_points_plot = [d['drift_point'] for d in drift_analysis]
        degradations = [d['degradation'] for d in drift_analysis]

        bars = axes[1, 1].bar(range(len(drift_points_plot)), degradations,
                             color='red', alpha=0.7)
        axes[1, 1].set_xticks(range(len(drift_points_plot)))
        axes[1, 1].set_xticklabels([f'Drift {i+1}' for i in range(len(drift_points_plot))])
        axes[1, 1].set_ylabel('Accuracy Degradation')
        axes[1, 1].set_title('Performance Impact of Concept Drift')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Add value labels
        for bar, deg in zip(bars, degradations):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2.,
                           height + (0.01 if height >= 0 else -0.02),
                           f'{deg:.3f}', ha='center',
                           va='bottom' if height >= 0 else 'top')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'streaming_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print performance summary
    print("Performance Summary:")
    print("-" * 25)
    print(f"Overall accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"Final ensemble size: {ensemble_sizes[-1]}")

    for concept in unique_concepts:
        concept_mask = concepts == concept
        concept_acc = np.mean(accuracies[concept_mask])
        print(f"Concept {concept} accuracy: {concept_acc:.3f}")

    if drift_analysis:
        avg_degradation = np.mean([d['degradation'] for d in drift_analysis])
        print(f"Average drift degradation: {avg_degradation:.3f}")

    return drift_analysis

def memory_management_strategies():
    """Analyze memory management strategies for streaming AdaBoost."""
    print_step_header(6, "Memory Management Strategies")

    strategies = {
        'Sliding Window': {
            'description': 'Keep only the most recent N samples',
            'memory_usage': 'Fixed',
            'adaptation_speed': 'Fast',
            'information_loss': 'High',
            'implementation_complexity': 'Low',
            'score': 7
        },
        'Reservoir Sampling': {
            'description': 'Maintain representative sample of all data',
            'memory_usage': 'Fixed',
            'adaptation_speed': 'Medium',
            'information_loss': 'Medium',
            'implementation_complexity': 'Medium',
            'score': 8
        },
        'Forgetting Factor': {
            'description': 'Exponentially decay importance of old samples',
            'memory_usage': 'Growing',
            'adaptation_speed': 'Medium',
            'information_loss': 'Low',
            'implementation_complexity': 'Medium',
            'score': 7
        },
        'Ensemble Pruning': {
            'description': 'Remove weak learners based on performance',
            'memory_usage': 'Variable',
            'adaptation_speed': 'Slow',
            'information_loss': 'Medium',
            'implementation_complexity': 'High',
            'score': 6
        },
        'Hierarchical Sampling': {
            'description': 'Multi-level sampling with different time scales',
            'memory_usage': 'Fixed',
            'adaptation_speed': 'Fast',
            'information_loss': 'Low',
            'implementation_complexity': 'High',
            'score': 9
        }
    }

    print("Memory Management Strategies:")
    print("-" * 35)

    for strategy, info in strategies.items():
        print(f"\n{strategy}:")
        print(f"  Description: {info['description']}")
        print(f"  Memory usage: {info['memory_usage']}")
        print(f"  Adaptation speed: {info['adaptation_speed']}")
        print(f"  Information loss: {info['information_loss']}")
        print(f"  Implementation complexity: {info['implementation_complexity']}")
        print(f"  Overall score: {info['score']}/10")

    # Visualize strategies
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    strategy_names = list(strategies.keys())
    scores = [info['score'] for info in strategies.values()]

    # Overall scores
    bars = axes[0, 0].bar(range(len(strategy_names)), scores,
                         color='skyblue', alpha=0.7)
    axes[0, 0].set_xticks(range(len(strategy_names)))
    axes[0, 0].set_xticklabels([name.replace(' ', '\n') for name in strategy_names])
    axes[0, 0].set_ylabel('Overall Score (1-10)')
    axes[0, 0].set_title('Memory Management Strategy Scores')
    axes[0, 0].grid(True, alpha=0.3)

    for bar, score in zip(bars, scores):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                       f'{score}', ha='center', va='bottom')

    # Memory usage distribution
    memory_types = ['Fixed', 'Growing', 'Variable']
    memory_counts = {}
    for mem_type in memory_types:
        memory_counts[mem_type] = sum(1 for info in strategies.values()
                                     if info['memory_usage'] == mem_type)

    wedges, texts, autotexts = axes[0, 1].pie(memory_counts.values(),
                                             labels=memory_counts.keys(),
                                             autopct='%1.1f%%',
                                             colors=['lightblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].set_title('Memory Usage Patterns')

    # Adaptation speed vs Implementation complexity
    adaptation_mapping = {'Fast': 3, 'Medium': 2, 'Slow': 1}
    complexity_mapping = {'Low': 1, 'Medium': 2, 'High': 3}

    adaptation_scores = [adaptation_mapping[info['adaptation_speed']]
                        for info in strategies.values()]
    complexity_scores = [complexity_mapping[info['implementation_complexity']]
                        for info in strategies.values()]

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, adapt, complex_score) in enumerate(zip(strategy_names, adaptation_scores, complexity_scores)):
        axes[1, 0].scatter(complex_score, adapt, s=200, c=colors[i], alpha=0.7, label=name)

    axes[1, 0].set_xlabel('Implementation Complexity')
    axes[1, 0].set_ylabel('Adaptation Speed')
    axes[1, 0].set_title('Adaptation Speed vs Implementation Complexity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].set_xticks([1, 2, 3])
    axes[1, 0].set_xticklabels(['Low', 'Medium', 'High'])
    axes[1, 0].set_yticks([1, 2, 3])
    axes[1, 0].set_yticklabels(['Slow', 'Medium', 'Fast'])

    # Information loss analysis
    loss_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    loss_scores = [loss_mapping[info['information_loss']] for info in strategies.values()]

    bars2 = axes[1, 1].bar(range(len(strategy_names)), loss_scores,
                          color='lightcoral', alpha=0.7)
    axes[1, 1].set_xticks(range(len(strategy_names)))
    axes[1, 1].set_xticklabels([name.replace(' ', '\n') for name in strategy_names])
    axes[1, 1].set_ylabel('Information Loss')
    axes[1, 1].set_title('Information Loss by Strategy')
    axes[1, 1].set_yticks([1, 2, 3])
    axes[1, 1].set_yticklabels(['Low', 'Medium', 'High'])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'memory_management.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Recommendation
    best_strategy_idx = np.argmax(scores)
    best_strategy = strategy_names[best_strategy_idx]

    print(f"\nRecommended Strategy: {best_strategy}")
    print(f"Score: {scores[best_strategy_idx]}/10")
    print(f"Rationale: {strategies[best_strategy]['description']}")

    return strategies, best_strategy

def visualize_concept_drift_analysis():
    """Generate visualizations for concept drift analysis."""
    print_step_header(8, "Generating Concept Drift Visualizations")
    
    # Create concept drift analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Concept drift timeline
    time_points = np.arange(0, 5000, 100)
    drift_indicators = np.zeros_like(time_points)
    drift_points = [1500, 3000, 4000]
    
    for drift_point in drift_points:
        drift_idx = drift_point // 100
        if drift_idx < len(drift_indicators):
            drift_indicators[drift_idx] = 1
    
    axes[0, 0].plot(time_points, drift_indicators, 'r-', linewidth=3, label='Concept Drift')
    axes[0, 0].set_xlabel('Sample Number')
    axes[0, 0].set_ylabel('Drift Indicator')
    axes[0, 0].set_title('Concept Drift Timeline')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Data distribution changes
    concept_ranges = [(0, 1500), (1500, 3000), (3000, 4000), (4000, 5000)]
    concept_means = []
    concept_stds = []
    
    for start, end in concept_ranges:
        if start < len(time_points):
            concept_means.append(np.mean(drift_indicators[start//100:end//100]))
            concept_stds.append(np.std(drift_indicators[start//100:end//100]))
    
    concept_labels = ['Concept 0', 'Concept 1', 'Concept 2', 'Concept 3']
    bars = axes[0, 1].bar(range(len(concept_labels)), concept_means, 
                         yerr=concept_stds, capsize=5, alpha=0.7, color='skyblue')
    axes[0, 1].set_xticks(range(len(concept_labels)))
    axes[0, 1].set_xticklabels(concept_labels)
    axes[0, 1].set_ylabel('Drift Intensity')
    axes[0, 1].set_title('Concept Drift Intensity by Period')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Drift detection sensitivity
    detection_methods = ['ADWIN', 'DDM', 'EDDM', 'Page-Hinkley', 'Statistical']
    sensitivity_scores = [8, 6, 7, 9, 8]
    false_positive_rates = [0.1, 0.3, 0.2, 0.4, 0.15]
    
    scatter = axes[1, 0].scatter(sensitivity_scores, false_positive_rates, 
                               s=100, c=range(len(detection_methods)), 
                               cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('Sensitivity (1-10)')
    axes[1, 0].set_ylabel('False Positive Rate')
    axes[1, 0].set_title('Drift Detection Method Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add method labels
    for i, method in enumerate(detection_methods):
        axes[1, 0].annotate(method, (sensitivity_scores[i], false_positive_rates[i]),
                           xytext=(5, 5), textcoords='offset points')
    
    # Plot 4: Drift recovery analysis
    recovery_times = [5, 8, 12, 6, 10]  # Batches to recover
    drift_types = ['Gradual', 'Abrupt', 'Mixed', 'Incremental', 'Cyclic']
    
    bars2 = axes[1, 1].bar(range(len(drift_types)), recovery_times, 
                          color='lightcoral', alpha=0.7)
    axes[1, 1].set_xticks(range(len(drift_types)))
    axes[1, 1].set_xticklabels(drift_types)
    axes[1, 1].set_ylabel('Recovery Time (batches)')
    axes[1, 1].set_title('Drift Recovery Time by Type')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'concept_drift_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Concept drift analysis visualizations saved.")

def visualize_ensemble_evolution():
    """Generate visualizations for ensemble evolution analysis."""
    print_step_header(9, "Generating Ensemble Evolution Visualizations")
    
    # Simulate ensemble evolution data
    n_batches = 50
    batch_numbers = np.arange(n_batches)
    
    # Ensemble size evolution
    ensemble_sizes = np.minimum(50, np.cumsum(np.random.exponential(0.3, n_batches)))
    
    # Alpha values evolution (simulated)
    alpha_values = np.random.gamma(2, 0.5, n_batches)
    
    # Performance evolution
    base_performance = 0.7
    performance = base_performance + 0.1 * np.sin(batch_numbers * 0.2) + 0.05 * np.random.randn(n_batches)
    performance = np.clip(performance, 0.5, 0.9)
    
    # Create ensemble evolution plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Ensemble size growth
    axes[0, 0].plot(batch_numbers, ensemble_sizes, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Memory Limit')
    axes[0, 0].set_xlabel('Batch Number')
    axes[0, 0].set_ylabel('Ensemble Size')
    axes[0, 0].set_title('Ensemble Size Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Alpha values distribution
    axes[0, 1].hist(alpha_values, bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Alpha Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Weak Learner Weights (α)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Performance over time
    axes[1, 0].plot(batch_numbers, performance, 'g-', linewidth=2, marker='s', markersize=4)
    axes[1, 0].set_xlabel('Batch Number')
    axes[1, 0].set_ylabel('Performance (Accuracy)')
    axes[1, 0].set_title('Performance Evolution Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0.4, 1.0)
    
    # Plot 4: Ensemble diversity analysis
    diversity_metrics = []
    for i in range(0, n_batches, 5):
        if i + 5 <= n_batches:
            # Simulate diversity metric (correlation between learners)
            diversity = 1 - np.mean(np.random.uniform(0.3, 0.7, 5))
            diversity_metrics.append(diversity)
    
    batch_indices = np.arange(0, n_batches, 5)[:len(diversity_metrics)]
    axes[1, 1].plot(batch_indices, diversity_metrics, 'purple', linewidth=2, marker='^', markersize=6)
    axes[1, 1].set_xlabel('Batch Number')
    axes[1, 1].set_ylabel('Ensemble Diversity')
    axes[1, 1].set_title('Ensemble Diversity Evolution')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ensemble_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Ensemble evolution visualizations saved.")

def visualize_memory_usage_patterns():
    """Generate visualizations for memory usage patterns."""
    print_step_header(10, "Generating Memory Usage Pattern Visualizations")
    
    # Simulate memory usage data for different strategies
    time_points = np.arange(0, 100, 1)
    
    # Different memory management strategies
    sliding_window = np.full_like(time_points, 1000)  # Fixed memory
    reservoir_sampling = np.full_like(time_points, 1000)  # Fixed memory
    forgetting_factor = 800 + 200 * np.exp(-0.05 * time_points)  # Growing but controlled
    ensemble_pruning = 600 + 400 * np.sin(0.1 * time_points)  # Variable
    hierarchical = np.full_like(time_points, 1000)  # Fixed memory
    
    # Information retention rates
    sliding_retention = 0.3 + 0.1 * np.sin(0.1 * time_points)
    reservoir_retention = 0.6 + 0.1 * np.cos(0.08 * time_points)
    forgetting_retention = 0.8 * np.exp(-0.02 * time_points)
    pruning_retention = 0.5 + 0.2 * np.sin(0.15 * time_points)
    hierarchical_retention = 0.75 + 0.1 * np.cos(0.05 * time_points)
    
    # Create memory usage pattern plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Memory usage over time
    axes[0, 0].plot(time_points, sliding_window, 'b-', linewidth=2, label='Sliding Window')
    axes[0, 0].plot(time_points, reservoir_sampling, 'g-', linewidth=2, label='Reservoir Sampling')
    axes[0, 0].plot(time_points, forgetting_factor, 'r-', linewidth=2, label='Forgetting Factor')
    axes[0, 0].plot(time_points, ensemble_pruning, 'orange', linewidth=2, label='Ensemble Pruning')
    axes[0, 0].plot(time_points, hierarchical, 'purple', linewidth=2, label='Hierarchical Sampling')
    
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Memory Usage (MB)')
    axes[0, 0].set_title('Memory Usage Patterns Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Information retention comparison
    axes[0, 1].plot(time_points, sliding_retention, 'b-', linewidth=2, label='Sliding Window')
    axes[0, 1].plot(time_points, reservoir_retention, 'g-', linewidth=2, label='Reservoir Sampling')
    axes[0, 1].plot(time_points, forgetting_retention, 'r-', linewidth=2, label='Forgetting Factor')
    axes[0, 1].plot(time_points, pruning_retention, 'orange', linewidth=2, label='Ensemble Pruning')
    axes[0, 1].plot(time_points, hierarchical_retention, 'purple', linewidth=2, label='Hierarchical Sampling')
    
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Information Retention Rate')
    axes[0, 1].set_title('Information Retention Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Memory efficiency vs adaptation speed
    strategies = ['Sliding\nWindow', 'Reservoir\nSampling', 'Forgetting\nFactor', 'Ensemble\nPruning', 'Hierarchical\nSampling']
    efficiency_scores = [0.7, 0.8, 0.6, 0.5, 0.9]
    adaptation_scores = [0.9, 0.6, 0.6, 0.3, 0.8]
    
    scatter = axes[1, 0].scatter(efficiency_scores, adaptation_scores, 
                               s=200, c=range(len(strategies)), 
                               cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('Memory Efficiency')
    axes[1, 0].set_ylabel('Adaptation Speed')
    axes[1, 0].set_title('Memory Efficiency vs Adaptation Speed')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add strategy labels
    for i, strategy in enumerate(strategies):
        axes[1, 0].annotate(strategy, (efficiency_scores[i], adaptation_scores[i]),
                           xytext=(5, 5), textcoords='offset points', ha='center')
    
    # Plot 4: Memory allocation breakdown for hierarchical sampling
    memory_breakdown = [400, 350, 250]  # Short-term, Medium-term, Long-term
    memory_labels = ['Short-term\n(λ₁=0.1)', 'Medium-term\n(λ₂=0.05)', 'Long-term\n(λ₃=0.01)']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    wedges, texts, autotexts = axes[1, 1].pie(memory_breakdown, labels=memory_labels, 
                                             autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 1].set_title('Hierarchical Sampling Memory Allocation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'memory_usage_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Memory usage pattern visualizations saved.")

def visualize_online_learning_characteristics():
    """Generate visualizations for online learning characteristics."""
    print_step_header(11, "Generating Online Learning Characteristic Visualizations")
    
    # Simulate online learning data
    n_samples = 1000
    sample_numbers = np.arange(n_samples)
    
    # Learning curves for different scenarios
    # Scenario 1: Stable environment
    stable_performance = 0.6 + 0.3 * (1 - np.exp(-sample_numbers / 200))
    
    # Scenario 2: Concept drift
    drift_performance = 0.6 + 0.3 * (1 - np.exp(-sample_numbers / 200))
    drift_performance[400:600] += 0.1 * np.sin((sample_numbers[400:600] - 400) * 0.1)
    drift_performance[600:] = 0.6 + 0.2 * (1 - np.exp(-(sample_numbers[600:] - 600) / 150))
    
    # Scenario 3: Noisy environment
    noisy_performance = 0.6 + 0.3 * (1 - np.exp(-sample_numbers / 200)) + 0.05 * np.random.randn(n_samples)
    noisy_performance = np.clip(noisy_performance, 0.5, 0.9)
    
    # Create online learning characteristic plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Learning curves comparison
    axes[0, 0].plot(sample_numbers, stable_performance, 'b-', linewidth=2, label='Stable Environment')
    axes[0, 0].plot(sample_numbers, drift_performance, 'r-', linewidth=2, label='Concept Drift')
    axes[0, 0].plot(sample_numbers, noisy_performance, 'g-', linewidth=2, label='Noisy Environment')
    axes[0, 0].set_xlabel('Number of Samples')
    axes[0, 0].set_ylabel('Performance (Accuracy)')
    axes[0, 0].set_title('Online Learning Curves')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0.5, 0.9)
    
    # Plot 2: Adaptation speed analysis
    adaptation_windows = [50, 100, 200, 500, 1000]
    adaptation_scores = [0.9, 0.8, 0.6, 0.4, 0.2]
    stability_scores = [0.2, 0.4, 0.6, 0.8, 0.9]
    
    x_pos = np.arange(len(adaptation_windows))
    width = 0.35
    
    bars1 = axes[0, 1].bar(x_pos - width/2, adaptation_scores, width, label='Adaptation Speed', alpha=0.7)
    bars2 = axes[0, 1].bar(x_pos + width/2, stability_scores, width, label='Stability', alpha=0.7)
    
    axes[0, 1].set_xlabel('Memory Window Size')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Adaptation Speed vs Stability Trade-off')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(adaptation_windows)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Batch size impact
    batch_sizes = [10, 25, 50, 100, 200]
    training_times = [0.1, 0.25, 0.5, 1.0, 2.0]
    convergence_rates = [0.8, 0.7, 0.6, 0.5, 0.4]
    
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(batch_sizes, training_times, 'b-', linewidth=2, marker='o', label='Training Time')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Training Time (seconds)', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    
    line2 = ax3_twin.plot(batch_sizes, convergence_rates, 'r-', linewidth=2, marker='s', label='Convergence Rate')
    ax3_twin.set_ylabel('Convergence Rate', color='r')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    
    ax3.set_title('Batch Size Impact on Training')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # Plot 4: Online vs batch learning comparison
    sample_sizes = [100, 200, 500, 1000, 2000]
    online_accuracy = [0.65, 0.68, 0.71, 0.73, 0.74]
    batch_accuracy = [0.70, 0.72, 0.75, 0.76, 0.77]
    online_time = [0.1, 0.2, 0.5, 1.0, 2.0]
    batch_time = [0.5, 1.0, 2.5, 5.0, 10.0]
    
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line3 = ax4.plot(sample_sizes, online_accuracy, 'g-', linewidth=2, marker='o', label='Online Accuracy')
    line4 = ax4.plot(sample_sizes, batch_accuracy, 'b-', linewidth=2, marker='s', label='Batch Accuracy')
    ax4.set_xlabel('Sample Size')
    ax4.set_ylabel('Accuracy', color='black')
    ax4.tick_params(axis='y', labelcolor='black')
    
    line5 = ax4_twin.plot(sample_sizes, online_time, 'g--', linewidth=2, marker='^', label='Online Time')
    line6 = ax4_twin.plot(sample_sizes, batch_time, 'b--', linewidth=2, marker='v', label='Batch Time')
    ax4_twin.set_ylabel('Training Time (seconds)', color='gray')
    ax4_twin.tick_params(axis='y', labelcolor='gray')
    
    ax4.set_title('Online vs Batch Learning Comparison')
    ax4.grid(True, alpha=0.3)
    
    # Combine legends
    lines_all = line3 + line4 + line5 + line6
    labels_all = [l.get_label() for l in lines_all]
    ax4.legend(lines_all, labels_all, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'online_learning_characteristics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Online learning characteristic visualizations saved.")

def main():
    """Main function to run the complete streaming data analysis."""
    print("Question 23: AdaBoost Streaming Data")
    print("=" * 60)

    # Analyze streaming challenges
    challenges = analyze_streaming_challenges()

    # Concept drift detection
    drift_methods = implement_concept_drift_detection()

    # Simulate online performance
    performance_metrics, drift_points = simulate_online_adaboost_performance()

    # Visualize performance
    drift_analysis = visualize_streaming_performance(performance_metrics, drift_points)

    # Memory management strategies
    memory_strategies, best_strategy = memory_management_strategies()

    # Visualize concept drift analysis
    visualize_concept_drift_analysis()

    # Visualize ensemble evolution
    visualize_ensemble_evolution()

    # Visualize memory usage patterns
    visualize_memory_usage_patterns()

    # Visualize online learning characteristics
    visualize_online_learning_characteristics()

    # Summary
    print_step_header(7, "Summary and Recommendations")

    print("Key Findings:")
    print("-" * 20)
    print(f"1. Most severe challenge: {max(challenges.keys(), key=lambda x: challenges[x]['severity'])}")
    print(f"2. Best memory strategy: {best_strategy}")
    print(f"3. Average accuracy: {np.mean(performance_metrics['accuracy']):.3f}")
    print(f"4. Final ensemble size: {performance_metrics['ensemble_size'][-1]}")
    if drift_analysis:
        avg_degradation = np.mean([d['degradation'] for d in drift_analysis])
        print(f"5. Average drift impact: {avg_degradation:.3f} accuracy loss")

    print(f"\nAll visualizations saved to: {save_dir}")

if __name__ == "__main__":
    main()
