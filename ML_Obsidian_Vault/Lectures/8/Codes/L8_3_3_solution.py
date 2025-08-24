import numpy as np
import matplotlib.pyplot as plt
import math
import os
from itertools import combinations
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_3_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

def format_number(num):
    """Format large numbers for better readability"""
    if num >= 1e12:
        return ".1e"
    elif num >= 1e9:
        return ".1e"
    elif num >= 1e6:
        return ".1e"
    elif num >= 1e3:
        return ","
    else:
        return "d"

def calculate_search_space(n_features, k_min=5, k_max=10):
    """Calculate search space statistics"""
    print("="*80)
    print("FEATURE SELECTION SEARCH SPACE ANALYSIS")
    print("="*80)
    print(f"Total number of features: {n_features}")
    print(f"Target subset size range: {k_min} - {k_max} features")
    print()

    # 1. Total possible subsets (excluding empty set)
    total_subsets = 2**n_features - 1
    print("1. TOTAL NUMBER OF POSSIBLE FEATURE SUBSETS")
    print(f"   Formula: 2^{n_features} - 1 = 2^{n_features} - 1")
    print(",")
    print(",")
    print()

    # 2. Subsets with exactly 7 features
    subsets_7 = math.comb(n_features, 7)
    print("2. NUMBER OF SUBSETS WITH EXACTLY 7 FEATURES")
    print(f"   Formula: C({n_features}, 7) = {n_features}! / (7! × ({n_features}-7)!)")
    print(",")
    print(",")
    print()

    # 3. Time for exhaustive search
    eval_time = 0.1  # seconds per evaluation
    total_time_seconds = total_subsets * eval_time
    total_time_hours = total_time_seconds / 3600
    total_time_days = total_time_hours / 24
    total_time_years = total_time_days / 365

    print("3. EXHAUSTIVE SEARCH TIME ESTIMATION")
    print(".1f")
    print(",")
    print(".1f")
    print(".1f")
    print(".1f")
    print(".1f")
    print()

    # 4. Forward selection evaluations
    print("4. FORWARD SELECTION EVALUATION COUNT")
    print("   Starting with 1 feature, adding one at a time...")

    forward_evals = 0
    remaining_features = n_features

    for k in range(1, n_features + 1):
        if k == 1:
            evals_at_step = n_features  # Choose first feature
        else:
            evals_at_step = remaining_features  # Choose next feature from remaining
        forward_evals += evals_at_step
        remaining_features -= 1
        print(f"   Step {k}: {evals_at_step} evaluations, cumulative: {forward_evals}")

    print(",")
    print()

    # 5. Heuristic search strategy
    max_evals = 1000
    print("5. HEURISTIC SEARCH STRATEGY (≤1000 evaluations)")
    print(f"   Maximum evaluations allowed: {max_evals}")

    # Strategy: Sample from different subset sizes proportionally
    target_sizes = list(range(k_min, k_max + 1))
    size_range = k_max - k_min + 1

    # Distribute evaluations across target sizes
    evals_per_size = max_evals // size_range
    remainder = max_evals % size_range

    heuristic_strategy = {}
    total_heuristic_evals = 0

    for i, k in enumerate(target_sizes):
        allocated = evals_per_size + (1 if i < remainder else 0)
        total_possible = math.comb(n_features, k)
        coverage = min(allocated, total_possible)
        heuristic_strategy[k] = {
            'allocated': allocated,
            'possible': total_possible,
            'coverage': coverage,
            'percentage': (coverage / total_possible * 100) if total_possible > 0 else 0
        }
        total_heuristic_evals += coverage
        print(",.1f")

    print(",")
    print()

    return {
        'total_subsets': total_subsets,
        'subsets_7': subsets_7,
        'exhaustive_time_years': total_time_years,
        'forward_evals': forward_evals,
        'heuristic_strategy': heuristic_strategy
    }

def smart_search_analysis(n_features=20, n_clusters=4, cluster_size=5):
    """Analyze smart search that skips correlated feature combinations"""
    print("6. SMART SEARCH ANALYSIS")
    print("="*50)
    print(f"Total features: {n_features}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Features per cluster: {cluster_size}")
    print()

    # Verify clustering setup
    assert n_clusters * cluster_size == n_features, f"Clustering setup invalid: {n_clusters} × {cluster_size} ≠ {n_features}"

    # Create feature-to-cluster mapping
    feature_clusters = {}
    cluster_features = {}

    for cluster_id in range(n_clusters):
        start_feature = cluster_id * cluster_size
        end_feature = start_feature + cluster_size
        cluster_features[cluster_id] = list(range(start_feature, end_feature))

        for feature in range(start_feature, end_feature):
            feature_clusters[feature] = cluster_id

    print("Feature-to-cluster mapping:")
    for cluster_id in range(n_clusters):
        features = cluster_features[cluster_id]
        print(f"   Cluster {cluster_id}: Features {features}")
    print()

    # We want subsets of size 5-10, but analyze for 7 features (as in question)
    target_k = 7

    print(f"Analyzing subsets of size {target_k}:")
    print(",")

    # Generate all possible 7-feature subsets
    all_subsets = list(combinations(range(n_features), target_k))

    # Count subsets where first 3 features are from same cluster
    skipped_subsets = 0
    valid_subsets = 0

    print("Analyzing each subset...")
    for i, subset in enumerate(all_subsets):
        subset_list = list(subset)
        first_three = subset_list[:3]
        first_three_clusters = [feature_clusters[f] for f in first_three]

        # Check if all first 3 features are from same cluster
        if len(set(first_three_clusters)) == 1:
            skipped_subsets += 1
            skip_reason = f"First 3 features {first_three} all in cluster {first_three_clusters[0]}"
        else:
            valid_subsets += 1
            skip_reason = "Valid subset"

        if i < 10:  # Show first 10 examples
            print(f"   Subset {i+1}: {subset_list} -> {skip_reason}")

    print("   ... (showing first 10 subsets)")
    print()

    print("SUMMARY:")
    print(",")
    print(",")
    print(",")

    percentage_skipped = (skipped_subsets / len(all_subsets)) * 100
    print(".2f")

    # Calculate how many subsets we can skip in the target range
    target_range_subsets = 0
    target_range_skipped = 0

    for k in range(5, 11):  # 5-10 features
        k_subsets = list(combinations(range(n_features), k))
        target_range_subsets += len(k_subsets)

        k_skipped = 0
        for subset in k_subsets:
            subset_list = list(subset)
            if len(subset_list) >= 3:  # Only check if subset has at least 3 features
                first_three = subset_list[:3]
                first_three_clusters = [feature_clusters[f] for f in first_three]
                if len(set(first_three_clusters)) == 1:
                    k_skipped += 1

        target_range_skipped += k_skipped
        print(",")

    print()
    print("TARGET RANGE (5-10 features) SUMMARY:")
    print(",")
    print(",")
    print(",.2f")

    return {
        'total_7_feature_subsets': len(all_subsets),
        'skipped_7_feature_subsets': skipped_subsets,
        'percentage_skipped': percentage_skipped,
        'target_range_subsets': target_range_subsets,
        'target_range_skipped': target_range_skipped
    }

def genetic_algorithm_analysis():
    """Analyze genetic algorithm parameters"""
    print("7. GENETIC ALGORITHM ANALYSIS")
    print("="*50)

    population_size = 50
    mutation_rate = 0.1
    crossover_rate = 0.8
    generations = 10
    offspring_per_generation = 50
    duplicate_percentage = 0.10

    print(f"Population size: {population_size}")
    print(f"Mutation rate: {mutation_rate}")
    print(f"Crossover rate: {crossover_rate}")
    print(f"Number of generations: {generations}")
    print(f"Offspring per generation: {offspring_per_generation}")
    print(f"Duplicate percentage: {duplicate_percentage}")
    print()

    # Calculate unique offspring per generation
    unique_offspring_per_gen = int(offspring_per_generation * (1 - duplicate_percentage))
    print(f"Expected unique offspring per generation: {unique_offspring_per_gen}")

    # Track unique subsets across generations
    total_unique_subsets = population_size  # Initial population
    cumulative_unique = population_size

    print("Generation-by-generation analysis:")
    print("Gen | New Offspring | Duplicates | Unique New | Cumulative Unique")
    print("----|---------------|------------|------------|------------------")

    for gen in range(1, generations + 1):
        new_offspring = offspring_per_generation
        duplicates = int(new_offspring * duplicate_percentage)
        unique_new = new_offspring - duplicates

        # Assume some unique new subsets (conservative estimate)
        # In practice, this would depend on selection pressure and diversity
        actual_unique_new = min(unique_new, unique_new // 2)  # Conservative estimate

        cumulative_unique += actual_unique_new

        print("3d")

    print()
    print(",")
    print("Note: This is a conservative estimate. In practice, the actual number")
    print("      may be lower due to selection pressure and convergence.")

    return {
        'generations': generations,
        'population_size': population_size,
        'unique_offspring_per_gen': unique_offspring_per_gen,
        'cumulative_unique': cumulative_unique
    }

def create_visualizations(n_features=20):
    """Create visualizations for the search space problem"""

    # Figure 1: Search space growth
    plt.figure(figsize=(12, 8))

    feature_counts = list(range(1, n_features + 1))
    subset_counts = [2**n - 1 for n in feature_counts]
    subset_7_counts = [math.comb(n, 7) if n >= 7 else 0 for n in feature_counts]

    plt.subplot(2, 2, 1)
    plt.semilogy(feature_counts, subset_counts, 'b-', linewidth=2, label='All subsets')
    plt.semilogy(feature_counts, subset_7_counts, 'r--', linewidth=2, label='7-feature subsets')
    plt.xlabel('Number of Features')
    plt.ylabel('Number of Subsets')
    plt.title('Search Space Growth')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axvline(x=20, color='green', linestyle=':', alpha=0.7, label='Our problem (20 features)')
    plt.legend()

    # Figure 2: Time complexity
    plt.subplot(2, 2, 2)
    eval_time = 0.1  # seconds
    time_seconds = [2**n * eval_time for n in feature_counts]
    time_hours = [t / 3600 for t in time_seconds]
    time_years = [t / (3600 * 24 * 365) for t in time_seconds]

    plt.loglog(feature_counts, time_years, 'r-', linewidth=2)
    plt.xlabel('Number of Features')
    plt.ylabel('Time (years)')
    plt.title('Exhaustive Search Time')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=20, color='green', linestyle=':', alpha=0.7)

    # Figure 3: Forward selection efficiency
    plt.subplot(2, 2, 3)
    forward_evals = []
    cumulative = 0
    for n in feature_counts:
        remaining = n
        for k in range(1, n + 1):
            if k == 1:
                cumulative += n
            else:
                remaining -= 1
                cumulative += remaining
        forward_evals.append(cumulative)
        cumulative = 0

    plt.semilogy(feature_counts, forward_evals, 'g-', linewidth=2)
    plt.xlabel('Number of Features')
    plt.ylabel('Forward Selection Evaluations')
    plt.title('Forward Selection Complexity')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=20, color='green', linestyle=':', alpha=0.7)

    # Figure 4: Cluster-based skipping
    plt.subplot(2, 2, 4)
    n_clusters = 4
    cluster_size = 5
    features = list(range(n_features))

    # Create cluster visualization
    cluster_colors = ['red', 'blue', 'green', 'orange']
    cluster_labels = []

    for i, feature in enumerate(features):
        cluster_id = i // cluster_size
        cluster_labels.append(cluster_id)

    plt.scatter(range(n_features), [1] * n_features, c=[cluster_colors[i] for i in cluster_labels], s=100)
    plt.xlabel('Feature Index')
    plt.ylabel('')
    plt.title('Feature Clustering (4 clusters of 5)')
    plt.yticks([])
    plt.grid(True, alpha=0.3)

    # Add cluster boundaries
    for i in range(1, n_clusters):
        plt.axvline(x=i*cluster_size - 0.5, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'search_space_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 5: Genetic algorithm diversity
    plt.figure(figsize=(10, 6))

    generations = list(range(11))
    pop_size = 50
    cumulative_unique = [pop_size]

    for gen in range(1, 11):
        offspring = 50
        duplicates = int(offspring * 0.1)
        unique_new = offspring - duplicates
        actual_unique_new = min(unique_new, unique_new // 2)  # Conservative
        cumulative_unique.append(cumulative_unique[-1] + actual_unique_new)

    plt.plot(generations, cumulative_unique, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Generation')
    plt.ylabel('Cumulative Unique Subsets')
    plt.title('Genetic Algorithm Diversity Over Time')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=math.comb(20, 7), color='red', linestyle='--', alpha=0.7,
                label='Total 7-feature subsets')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'genetic_algorithm_diversity.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function"""
    n_features = 20
    k_min, k_max = 5, 10

    print("FEATURE SELECTION SEARCH SPACE ANALYSIS")
    print("Dataset: 20 features, target subset size: 5-10 features")
    print("="*80)

    # Run all analyses
    search_results = calculate_search_space(n_features, k_min, k_max)
    print("\n" + "="*80 + "\n")

    smart_results = smart_search_analysis(n_features, 4, 5)
    print("\n" + "="*80 + "\n")

    ga_results = genetic_algorithm_analysis()
    print("\n" + "="*80 + "\n")

    # Create visualizations
    print("CREATING VISUALIZATIONS...")
    create_visualizations(n_features)
    print(f"Plots saved to: {save_dir}")

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(",")
    print(",")
    print(",")
    print(",")
    print(",")
    print(".2f")
    print(",")

if __name__ == "__main__":
    main()
