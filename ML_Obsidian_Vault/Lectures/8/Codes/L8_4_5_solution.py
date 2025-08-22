import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from itertools import combinations
import os
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_4_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

def create_sample_dataset():
    """Create a sample dataset for consistency analysis"""
    np.random.seed(42)

    # Create dataset with some inconsistencies
    X = np.array([
        [1, 2], [1, 2], [2, 3], [2, 3], [3, 4], [3, 4],  # Consistent pairs
        [4, 5], [4, 6], [5, 7], [5, 8], [6, 9], [6, 9],  # Some inconsistencies
        [7, 10], [7, 10], [8, 11], [8, 11], [9, 12], [9, 12]
    ])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Inconsistent labels for identical features

    return X, y

def calculate_consistency_score(X, y, feature_subset):
    """Calculate consistency score for a given feature subset"""
    if not feature_subset:
        return 0.0

    # Extract selected features
    X_subset = X[:, feature_subset] if len(feature_subset) > 1 else X[:, feature_subset].reshape(-1, 1)

    # Find samples with identical feature values
    unique_samples = {}
    for i, sample in enumerate(X_subset):
        sample_tuple = tuple(sample)
        if sample_tuple not in unique_samples:
            unique_samples[sample_tuple] = []
        unique_samples[sample_tuple].append(y[i])

    # Calculate consistency
    total_consistent = 0
    total_samples = len(X)

    for sample_features, labels in unique_samples.items():
        if len(labels) > 1:
            # Check if all labels are the same
            unique_labels = set(labels)
            if len(unique_labels) == 1:
                total_consistent += len(labels)
            else:
                # Partial consistency: samples with same features get majority label
                most_common = max(set(labels), key=labels.count)
                total_consistent += labels.count(most_common)
        else:
            total_consistent += 1

    return total_consistent / total_samples

def find_minimal_consistent_subset(X, y, all_features):
    """Find the minimal consistent feature subset"""
    n_features = len(all_features)
    subsets_scores = {}

    # Test all possible subsets
    for k in range(1, n_features + 1):
        for subset in combinations(range(n_features), k):
            score = calculate_consistency_score(X, y, list(subset))
            subsets_scores[subset] = score

    # Find minimal subset with perfect consistency
    min_size = float('inf')
    best_subset = None

    for subset, score in subsets_scores.items():
        if score == 1.0 and len(subset) < min_size:
            min_size = len(subset)
            best_subset = subset

    return best_subset, subsets_scores

def compare_evaluation_criteria(X, y):
    """Compare consistency with other evaluation criteria"""
    n_features = X.shape[1]
    results = {}

    # Test all possible subsets
    for k in range(1, n_features + 1):
        for subset in combinations(range(n_features), k):
            subset_list = list(subset)

            # Consistency score
            consistency = calculate_consistency_score(X, y, subset_list)

            # Information gain approximation (using decision tree)
            if len(subset_list) > 0:
                X_subset = X[:, subset_list] if len(subset_list) > 1 else X[:, subset_list].reshape(-1, 1)
                clf = DecisionTreeClassifier(random_state=42, max_depth=3)
                clf.fit(X_subset, y)
                info_gain = accuracy_score(y, clf.predict(X_subset))
            else:
                info_gain = 0

            # Distance measure (simplified)
            if len(subset_list) == 1:
                feature_values = X[:, subset_list[0]]
                distance_score = np.var(feature_values)  # Higher variance = better separation
            else:
                distance_score = np.linalg.det(np.cov(X[:, subset_list].T)) if len(subset_list) > 1 else 0

            results[subset] = {
                'consistency': consistency,
                'information_gain': info_gain,
                'distance_measure': distance_score,
                'size': len(subset)
            }

    return results

def visualize_consistency_analysis():
    """Create comprehensive visualizations for consistency analysis"""

    # Create sample dataset
    X, y = create_sample_dataset()

    print("=" * 60)
    print("CONSISTENCY MEASURES ANALYSIS - QUESTION 5")
    print("=" * 60)

    # Task 1: Min-features bias demonstration
    print("\n1. MIN-FEATURES BIAS IN CONSISTENCY MEASURES")
    print("-" * 50)

    all_features = list(range(X.shape[1]))
    best_subset, all_scores = find_minimal_consistent_subset(X, y, all_features)

    print(f"Original dataset shape: {X.shape}")
    print("Feature values and labels:")
    for i in range(len(X)):
        print(f"Sample {i+1}: Features {X[i]}, Label: {y[i]}")

    if best_subset is not None:
        print(f"\nMinimal consistent subset: Features {list(best_subset)}")
        print(f"Size of minimal subset: {len(best_subset)}")
    else:
        print("\nNo perfectly consistent subset found (dataset contains inconsistencies)")
        print("Finding the most consistent subset instead...")

        # Find the most consistent subset
        best_score = 0
        best_subset = None
        for subset, score in all_scores.items():
            if score > best_score:
                best_score = score
                best_subset = subset

        if best_subset is not None:
            print(f"Most consistent subset: Features {list(best_subset)}")
            print(f"Consistency score: {best_score:.3f}")

    # Show all subset scores
    print("\nAll subset consistency scores:")
    for subset, score in all_scores.items():
        print(f"Features {list(subset)}: Consistency = {score:.3f}")

    # Task 2: Classification consistency measurement
    print("\n\n2. CLASSIFICATION CONSISTENCY MEASUREMENT")
    print("-" * 50)

    print("Method: For each unique feature combination, check if all samples")
    print("with identical features have the same class label.")

    # Analyze each possible subset
    for subset in [(0,), (1,), (0, 1)]:
        score = calculate_consistency_score(X, y, list(subset))
        print(f"\nSubset {list(subset)} consistency: {score:.3f}")

        # Show detailed analysis
        X_subset = X[:, subset] if len(subset) > 1 else X[:, subset].reshape(-1, 1)
        unique_combinations = {}
        for i, sample in enumerate(X_subset):
            sample_tuple = tuple(sample)
            if sample_tuple not in unique_combinations:
                unique_combinations[sample_tuple] = []
            unique_combinations[sample_tuple].append((i+1, y[i]))

        print("  Unique combinations and their labels:")
        for combo, samples in unique_combinations.items():
            if len(samples) > 1:
                labels = [s[1] for s in samples]
                consistent = len(set(labels)) == 1
                print(f"    {combo} -> Labels {labels}, Consistent: {consistent}")
            else:
                print(f"    {combo} -> Label {samples[0][1]} (single sample)")

    # Task 3: What does it mean for a feature subset to be consistent?
    print("\n\n3. WHAT DOES IT MEAN FOR A FEATURE SUBSET TO BE CONSISTENT?")
    print("-" * 60)

    print("A feature subset is consistent if:")
    print("1. No two samples with identical feature values have different labels")
    print("2. The subset maintains the original classification relationships")
    print("3. Classification accuracy is preserved with fewer features")

    # Show inconsistent examples
    print("\nInconsistent examples in our dataset:")
    X_subset = X[:, [0]]  # Feature 1 only
    unique_combinations = {}
    for i, sample in enumerate(X_subset):
        sample_tuple = tuple(sample)
        if sample_tuple not in unique_combinations:
            unique_combinations[sample_tuple] = []
        unique_combinations[sample_tuple].append((i+1, y[i]))

    for combo, samples in unique_combinations.items():
        if len(samples) > 1:
            labels = [s[1] for s in samples]
            if len(set(labels)) > 1:
                print(f"Feature value {combo[0]} has inconsistent labels: {labels}")

    # Task 4: What happens when samples have identical features but different labels?
    print("\n\n4. IDENTICAL FEATURES WITH DIFFERENT LABELS")
    print("-" * 50)

    print("This situation indicates:")
    print("1. DATA INCONSISTENCY: The dataset contains conflicting examples")
    print("2. NOISE: Possible measurement errors or genuine ambiguity")
    print("3. INSUFFICIENT FEATURES: Current features don't capture all relevant information")
    print("4. CLASS OVERLAP: Natural overlap between classes in feature space")

    print("\nExamples of inconsistent samples:")
    inconsistent_pairs = []
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if np.array_equal(X[i], X[j]) and y[i] != y[j]:
                inconsistent_pairs.append((i, j))

    if inconsistent_pairs:
        for pair in inconsistent_pairs:
            i, j = pair
            print(f"Samples {i+1} and {j+1}: Both have features {X[i]}, but labels {y[i]} vs {y[j]}")
    else:
        print("No identical feature pairs with different labels found.")

    # Task 5: Compare consistency vs other evaluation criteria
    print("\n\n5. COMPARISON: CONSISTENCY VS OTHER EVALUATION CRITERIA")
    print("-" * 60)

    comparison_results = compare_evaluation_criteria(X, y)

    # Create comparison table
    print("\nComparison of evaluation criteria:")
    print("Subset\t\tConsistency\tInfo Gain\tDistance\tSize")
    print("-" * 65)

    for subset, scores in comparison_results.items():
        subset_str = str(list(subset)).ljust(15)
        print(f"{subset_str}\t{scores['consistency']:.3f}\t\t{scores['information_gain']:.3f}\t\t{scores['distance_measure']:.3f}\t\t{scores['size']}")

    # Analyze trade-offs
    print("\nKey insights from comparison:")
    print("1. CONSISTENCY focuses on data integrity and minimal features")
    print("2. INFORMATION GAIN maximizes predictive power")
    print("3. DISTANCE MEASURE maximizes class separability")
    print("4. Different criteria may select different optimal subsets")

    # Find best subset for each criterion
    best_consistency = max(comparison_results.items(), key=lambda x: x[1]['consistency'])
    best_info_gain = max(comparison_results.items(), key=lambda x: x[1]['information_gain'])
    best_distance = max(comparison_results.items(), key=lambda x: x[1]['distance_measure'])

    print("\nBest subsets by criterion:")
    print(f"  Consistency: {list(best_consistency[0])} (score: {best_consistency[1]['consistency']:.3f})")
    print(f"  Information Gain: {list(best_info_gain[0])} (score: {best_info_gain[1]['information_gain']:.3f})")
    print(f"  Distance: {list(best_distance[0])} (score: {best_distance[1]['distance_measure']:.3f})")

    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    # Visualization 1: Consistency vs Feature Subset Size
    plt.figure(figsize=(12, 8))

    sizes = []
    consistencies = []
    info_gains = []
    distances = []

    for subset, scores in comparison_results.items():
        sizes.append(scores['size'])
        consistencies.append(scores['consistency'])
        info_gains.append(scores['information_gain'])
        distances.append(scores['distance_measure'])

    plt.subplot(2, 2, 1)
    plt.scatter(sizes, consistencies, alpha=0.7, s=100, c='blue', edgecolors='black')
    plt.xlabel('Feature Subset Size')
    plt.ylabel('Consistency Score')
    plt.title('Consistency vs Feature Subset Size')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)

    plt.subplot(2, 2, 2)
    plt.scatter(sizes, info_gains, alpha=0.7, s=100, c='green', edgecolors='black')
    plt.xlabel('Feature Subset Size')
    plt.ylabel('Information Gain')
    plt.title('Information Gain vs Feature Subset Size')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.scatter(sizes, distances, alpha=0.7, s=100, c='red', edgecolors='black')
    plt.xlabel('Feature Subset Size')
    plt.ylabel('Distance Measure')
    plt.title('Distance Measure vs Feature Subset Size')
    plt.grid(True, alpha=0.3)

    # Radar chart comparing criteria
    plt.subplot(2, 2, 4)
    criteria = ['Consistency', 'Information\nGain', 'Distance\nMeasure']
    angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False)

    # Normalize scores for radar chart
    max_scores = [1.0, 1.0, max(distances)]
    normalized_scores = [
        [best_consistency[1]['consistency'], best_info_gain[1]['information_gain'], best_distance[1]['distance_measure']/max_scores[2]]
    ]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    ax.plot(angles, normalized_scores[0], 'o-', linewidth=2, label='Best Subsets')
    ax.fill(angles, normalized_scores[0], alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(criteria)
    ax.set_title('Comparison of Best Subsets by Criteria', size=12, pad=20)
    ax.grid(True)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'consistency_criteria_comparison.png'), dpi=300, bbox_inches='tight')

    # Visualization 2: Dataset with inconsistencies highlighted
    plt.figure(figsize=(10, 8))

    # Plot all points
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.7, edgecolors='black')

    # Highlight inconsistent points
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if np.array_equal(X[i], X[j]) and y[i] != y[j]:
                # Draw line between inconsistent points
                plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k--', linewidth=2, alpha=0.7)
                # Mark points with special markers
                plt.scatter(X[i, 0], X[i, 1], marker='x', s=200, c='yellow', linewidth=3)
                plt.scatter(X[j, 0], X[j, 1], marker='x', s=200, c='yellow', linewidth=3)

    # Add labels
    for i, (x, label) in enumerate(zip(X, y)):
        plt.annotate(f'({x[0]}, {x[1]})\nClass {label}', (x[0], x[1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Dataset with Inconsistent Samples Highlighted')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'inconsistent_samples_visualization.png'), dpi=300, bbox_inches='tight')

    # Visualization 3: Min-features bias demonstration
    plt.figure(figsize=(12, 6))

    subset_sizes = []
    consistency_scores = []

    for subset, score in all_scores.items():
        subset_sizes.append(len(subset))
        consistency_scores.append(score)

    plt.subplot(1, 2, 1)
    plt.bar(range(len(all_scores)), consistency_scores, color='skyblue', edgecolor='black')
    plt.xlabel('Feature Subset Index')
    plt.ylabel('Consistency Score')
    plt.title('All Subset Consistency Scores')
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(1, 2, 2)
    unique_sizes = sorted(set(subset_sizes))
    avg_consistency_by_size = []

    for size in unique_sizes:
        size_consistencies = [score for s, score in zip(subset_sizes, consistency_scores) if s == size]
        avg_consistency_by_size.append(np.mean(size_consistencies))

    plt.plot(unique_sizes, avg_consistency_by_size, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Feature Subset Size')
    plt.ylabel('Average Consistency Score')
    plt.title('Min-Features Bias: Smaller Subsets Tend to Be More Consistent')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'min_features_bias.png'), dpi=300, bbox_inches='tight')

    print("\nVisualizations saved:")
    print(f"1. consistency_criteria_comparison.png - Comparison of evaluation criteria")
    print(f"2. inconsistent_samples_visualization.png - Dataset with inconsistencies highlighted")
    print(f"3. min_features_bias.png - Demonstration of min-features bias")

    print(f"\nAll results saved to: {save_dir}")

if __name__ == "__main__":
    visualize_consistency_analysis()
