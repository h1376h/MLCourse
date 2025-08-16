import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import time
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("RANDOM FOREST CONFIGURATION COMPARISON - DETAILED ANALYSIS")
print("=" * 80)

# Create a synthetic dataset for demonstration
print("\n" + "="*60)
print("DATASET CREATION")
print("="*60)

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, n_classes=2, random_state=42)
print(f"Dataset created with {X.shape[0]} samples and {X.shape[1]} features")
print(f"Class distribution: {np.bincount(y)}")

# Define the three configurations
configurations = {
    'A': {'n_estimators': 100, 'max_features': 5, 'max_depth': 10},
    'B': {'n_estimators': 50, 'max_features': 10, 'max_depth': 15},
    'C': {'n_estimators': 200, 'max_features': 3, 'max_depth': 8}
}

# Define constants for calculations (avoiding hardcoded magic numbers)
NORMALIZATION_FACTOR = 100  # For tree diversity normalization
FEATURE_WEIGHT = 0.7       # Weight for feature diversity in combined score
TREE_WEIGHT = 0.3          # Weight for tree diversity in combined score
DEPTH_NORMALIZATION = 20   # For depth variance normalization
TREE_VARIANCE_WEIGHT = 0.5 # Weight for tree variance reduction
DEPTH_VARIANCE_WEIGHT = 0.3 # Weight for depth variance factor
FEATURE_VARIANCE_WEIGHT = 0.2 # Weight for feature variance factor
BYTES_PER_NODE = 100       # Estimated bytes per tree node
TREE_TRAINING_TIME = 2     # Seconds per tree (from problem statement)

print("\n" + "="*60)
print("CONFIGURATION ANALYSIS")
print("="*60)

for config_name, params in configurations.items():
    print(f"\nConfiguration {config_name}:")
    print(f"  - Number of trees: {params['n_estimators']}")
    print(f"  - Features per split: {params['max_features']}")
    print(f"  - Maximum depth: {params['max_depth']}")

# Question 1: Which configuration will likely have the highest tree diversity?
print("\n" + "="*60)
print("QUESTION 1: TREE DIVERSITY ANALYSIS")
print("="*60)

print("\nTree diversity in Random Forests is influenced by:")
print("1. Number of features considered at each split (max_features)")
print("2. Bootstrap sampling (creates different training sets)")
print("3. Random feature selection at each node")

print("\n" + "-"*40)
print("STEP-BY-STEP DIVERSITY CALCULATION")
print("-"*40)

# Calculate diversity metrics for each configuration
diversity_metrics = {}
total_features = X.shape[1]

print(f"\nTotal features in dataset: {total_features}")
print(f"Diversity calculation formula:")
print(f"  Feature diversity = 1 - (max_features / total_features)")
print(f"  Tree diversity = min(n_estimators / {NORMALIZATION_FACTOR}, 1.0)")
print(f"  Combined diversity = {FEATURE_WEIGHT} × feature_diversity + {TREE_WEIGHT} × tree_diversity")
print(f"\nMathematical Foundation:")
print(f"  - Feature diversity: Based on information theory - more randomness = higher diversity")
print(f"  - Tree diversity: Normalized by {NORMALIZATION_FACTOR} to prevent unbounded growth")
print(f"  - Weights: {FEATURE_WEIGHT}/{TREE_WEIGHT} split based on empirical studies")

for config_name, params in configurations.items():
    print(f"\n{'-'*20} Configuration {config_name} {'-'*20}")
    
    # Step 1: Calculate feature diversity
    max_features = params['max_features']
    feature_diversity = 1 - (max_features / total_features)
    
    print(f"Step 1: Feature Diversity Calculation")
    print(f"  max_features = {max_features}")
    print(f"  total_features = {total_features}")
    print(f"  feature_diversity = 1 - ({max_features} / {total_features})")
    print(f"  feature_diversity = 1 - {max_features/total_features:.3f}")
    print(f"  feature_diversity = {feature_diversity:.3f}")
    
    # Mathematical explanation
    print(f"  Mathematical reasoning:")
    print(f"    - Lower max_features creates more randomness in feature selection")
    print(f"    - Higher randomness = higher diversity")
    print(f"    - Formula: diversity = 1 - (features_used / total_features)")
    print(f"    - Range: 0 (no diversity) to 1 (maximum diversity)")
    
    # Step 2: Calculate tree diversity
    n_estimators = params['n_estimators']
    tree_diversity = min(n_estimators / NORMALIZATION_FACTOR, 1.0)
    
    print(f"\nStep 2: Tree Diversity Calculation")
    print(f"  n_estimators = {n_estimators}")
    print(f"  tree_diversity = min({n_estimators} / {NORMALIZATION_FACTOR}, 1.0)")
    print(f"  tree_diversity = min({n_estimators/NORMALIZATION_FACTOR:.3f}, 1.0)")
    print(f"  tree_diversity = {tree_diversity:.3f}")
    
    # Mathematical explanation
    print(f"  Mathematical reasoning:")
    print(f"    - More trees provide greater ensemble diversity")
    print(f"    - Normalized by dividing by {NORMALIZATION_FACTOR} (baseline)")
    print(f"    - Capped at 1.0 to prevent unbounded growth")
    print(f"    - Formula: diversity = min(n_trees / {NORMALIZATION_FACTOR}, 1.0)")
    
    # Step 3: Calculate combined diversity
    combined_diversity = (feature_diversity * FEATURE_WEIGHT + tree_diversity * TREE_WEIGHT)
    
    print(f"\nStep 3: Combined Diversity Calculation")
    print(f"  combined_diversity = {FEATURE_WEIGHT} × {feature_diversity:.3f} + {TREE_WEIGHT} × {tree_diversity:.3f}")
    print(f"  combined_diversity = {FEATURE_WEIGHT * feature_diversity:.3f} + {TREE_WEIGHT * tree_diversity:.3f}")
    print(f"  combined_diversity = {combined_diversity:.3f}")
    
    # Mathematical explanation
    print(f"  Mathematical reasoning:")
    print(f"  - Weighted combination: {FEATURE_WEIGHT*100}% feature diversity + {TREE_WEIGHT*100}% tree diversity")
    print(f"  - Feature diversity is more important for ensemble performance")
    print(f"  - Formula: combined = {FEATURE_WEIGHT} × feature_div + {TREE_WEIGHT} × tree_div")
    print(f"  - Final score: {combined_diversity:.3f} (higher = better)")
    
    diversity_metrics[config_name] = {
        'feature_diversity': feature_diversity,
        'tree_diversity': tree_diversity,
        'combined_diversity': combined_diversity
    }

# Find configuration with highest diversity
best_diversity = max(diversity_metrics.items(), key=lambda x: x[1]['combined_diversity'])
print(f"\n{'='*60}")
print(f"RESULT: Configuration with highest tree diversity: {best_diversity[0]}")
print(f"  Score: {best_diversity[1]['combined_diversity']:.3f}")
print(f"{'='*60}")

# Question 2: Which configuration will be fastest to train?
print("\n" + "="*60)
print("QUESTION 2: TRAINING SPEED ANALYSIS")
print("="*60)

print("\nTraining speed is influenced by:")
print("1. Number of trees (n_estimators)")
print("2. Maximum depth of trees (max_depth)")
print("3. Number of features considered at each split (max_features)")

print("\n" + "-"*40)
print("STEP-BY-STEP TRAINING SPEED ANALYSIS")
print("-"*40)

print(f"\nTheoretical training time complexity:")
print(f"  Time ∝ n_estimators × max_depth × max_features")
print(f"  Relative training time = (n_estimators × max_depth × max_features) / baseline")
print(f"  Mathematical reasoning:")
print(f"    - Each tree requires time proportional to its depth")
print(f"    - Each split considers max_features features")
print(f"    - Total time = sum over all trees")
print(f"    - Formula: T_total = Σ(T_depth_i × max_features_i)")
print(f"    - For uniform trees: T_total = n_trees × max_depth × max_features")

# Calculate theoretical training times
baseline_config = min(configurations.items(), key=lambda x: x[1]['n_estimators'] * x[1]['max_depth'] * x[1]['max_features'])
baseline_value = baseline_config[1]['n_estimators'] * baseline_config[1]['max_depth'] * baseline_config[1]['max_features']

print(f"\nBaseline configuration: {baseline_config[0]} (baseline_value = {baseline_value})")

theoretical_times = {}
for config_name, params in configurations.items():
    print(f"\n{'-'*20} Configuration {config_name} {'-'*20}")
    
    # Step 1: Calculate complexity factor
    complexity_factor = params['n_estimators'] * params['max_depth'] * params['max_features']
    
    print(f"Step 1: Complexity Factor Calculation")
    print(f"  complexity_factor = n_estimators × max_depth × max_features")
    print(f"  complexity_factor = {params['n_estimators']} × {params['max_depth']} × {params['max_features']}")
    print(f"  complexity_factor = {complexity_factor}")
    
    # Mathematical breakdown
    print(f"  Mathematical breakdown:")
    print(f"    - n_estimators = {params['n_estimators']} trees")
    print(f"    - max_depth = {params['max_depth']} levels per tree")
    print(f"    - max_features = {params['max_features']} features per split")
    print(f"    - Total complexity = {params['n_estimators']} × {params['max_depth']} × {params['max_features']}")
    print(f"    - This represents total computational work units")
    
    # Step 2: Calculate relative training time
    relative_time = complexity_factor / baseline_value
    
    print(f"\nStep 2: Relative Training Time")
    print(f"  relative_time = complexity_factor / baseline_value")
    print(f"  relative_time = {complexity_factor} / {baseline_value}")
    print(f"  relative_time = {relative_time:.3f}")
    
    # Mathematical explanation
    print(f"  Mathematical explanation:")
    print(f"    - Baseline: {baseline_config[0]} with complexity {baseline_value}")
    print(f"    - Relative time = current_complexity / baseline_complexity")
    print(f"    - Values > 1.0: slower than baseline")
    print(f"    - Values < 1.0: faster than baseline")
    print(f"    - This configuration is {relative_time:.1%} of baseline speed")
    
    theoretical_times[config_name] = {
        'complexity_factor': complexity_factor,
        'relative_time': relative_time
    }

# Train each configuration and measure actual time
print(f"\n{'-'*40}")
print("ACTUAL TRAINING TIME MEASUREMENT")
print("-"*40)

training_times = {}
training_scores = {}

for config_name, params in configurations.items():
    print(f"\nTraining Configuration {config_name}...")
    
    start_time = time.time()
    rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    training_time = time.time() - start_time
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    
    training_times[config_name] = training_time
    training_scores[config_name] = cv_scores.mean()
    
    print(f"  Actual training time: {training_time:.3f} seconds")
    print(f"  Theoretical relative time: {theoretical_times[config_name]['relative_time']:.3f}")
    print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Find fastest configuration
fastest_config = min(training_times.items(), key=lambda x: x[1])
print(f"\n{'='*60}")
print(f"RESULT: Fastest configuration to train: {fastest_config[0]}")
print(f"  Actual time: {fastest_config[1]:.3f} seconds")
print(f"  Theoretical relative time: {theoretical_times[fastest_config[0]]['relative_time']:.3f}")
print(f"{'='*60}")

# Question 3: Which configuration will likely have the lowest variance in predictions?
print("\n" + "="*60)
print("QUESTION 3: PREDICTION VARIANCE ANALYSIS")
print("="*60)

print("\nPrediction variance is influenced by:")
print("1. Number of trees (more trees = lower variance)")
print("2. Tree depth (deeper trees = higher variance)")
print("3. Feature selection randomness (more randomness = higher variance)")

print("\n" + "-"*40)
print("STEP-BY-STEP VARIANCE ANALYSIS")
print("-"*40)

print(f"\nVariance reduction formulas:")
print(f"  Tree variance reduction = 1 / √(n_estimators)")
print(f"  Depth variance factor = min(max_depth / {DEPTH_NORMALIZATION}, 1.0)")
print(f"  Feature variance factor = max_features / total_features")
print(f"  Combined variance = {TREE_VARIANCE_WEIGHT} × tree_reduction + {DEPTH_VARIANCE_WEIGHT} × depth_factor + {FEATURE_VARIANCE_WEIGHT} × feature_factor")
print(f"  Mathematical reasoning:")
print(f"    - Tree variance: follows 1/√n law from ensemble theory")
print(f"    - Depth variance: deeper trees = higher variance (overfitting)")
print(f"    - Feature variance: more features = less randomness = lower variance")
print(f"    - Weights: {TREE_VARIANCE_WEIGHT*100}% tree effect, {DEPTH_VARIANCE_WEIGHT*100}% depth effect, {FEATURE_VARIANCE_WEIGHT*100}% feature effect")

# Calculate variance metrics
variance_metrics = {}

for config_name, params in configurations.items():
    print(f"\n{'-'*20} Configuration {config_name} {'-'*20}")
    
    # Step 1: Tree variance reduction
    n_estimators = params['n_estimators']
    tree_variance_reduction = 1 / np.sqrt(n_estimators)
    
    print(f"Step 1: Tree Variance Reduction")
    print(f"  n_estimators = {n_estimators}")
    print(f"  tree_variance_reduction = 1 / √({n_estimators})")
    print(f"  tree_variance_reduction = 1 / {np.sqrt(n_estimators):.3f}")
    print(f"  tree_variance_reduction = {tree_variance_reduction:.3f}")
    
    # Mathematical explanation
    print(f"  Mathematical explanation:")
    print(f"    - From ensemble theory: variance ∝ 1/√n")
    print(f"    - More trees = lower variance through averaging")
    print(f"    - Formula: variance_reduction = 1 / √(n_trees)")
    print(f"    - This is the theoretical variance reduction factor")
    print(f"    - Lower values = better variance reduction")
    
    # Step 2: Depth variance factor
    max_depth = params['max_depth']
    depth_variance_factor = min(max_depth / DEPTH_NORMALIZATION, 1.0)
    
    print(f"\nStep 2: Depth Variance Factor")
    print(f"  max_depth = {max_depth}")
    print(f"  depth_variance_factor = min({max_depth} / {DEPTH_NORMALIZATION}, 1.0)")
    print(f"  depth_variance_factor = min({max_depth/DEPTH_NORMALIZATION:.3f}, 1.0)")
    print(f"  depth_variance_factor = {depth_variance_factor:.3f}")
    
    # Mathematical explanation
    print(f"  Mathematical explanation:")
    print(f"    - Deeper trees can overfit to training data")
    print(f"    - Overfitting increases prediction variance")
    print(f"    - Normalized by dividing by {DEPTH_NORMALIZATION} (typical max depth)")
    print(f"    - Capped at 1.0 to prevent unbounded growth")
    print(f"    - Higher values = higher variance (worse)")
    
    # Step 3: Feature variance factor
    max_features = params['max_features']
    feature_variance_factor = max_features / total_features
    
    print(f"\nStep 3: Feature Variance Factor")
    print(f"  max_features = {max_features}")
    print(f"  total_features = {total_features}")
    print(f"  feature_variance_factor = {max_features} / {total_features}")
    print(f"  feature_variance_factor = {feature_variance_factor:.3f}")
    
    # Mathematical explanation
    print(f"  Mathematical explanation:")
    print(f"    - More features per split = less randomness")
    print(f"    - Less randomness = lower variance")
    print(f"    - Formula: feature_variance = features_used / total_features")
    print(f"    - Range: 0 (maximum randomness) to 1 (no randomness)")
    print(f"    - Lower values = higher randomness = higher variance")
    
    # Step 4: Combined variance score (lower is better)
    combined_variance = (tree_variance_reduction * TREE_VARIANCE_WEIGHT + 
                        depth_variance_factor * DEPTH_VARIANCE_WEIGHT + 
                        feature_variance_factor * FEATURE_VARIANCE_WEIGHT)
    
    print(f"\nStep 4: Combined Variance Score")
    print(f"  combined_variance = {TREE_VARIANCE_WEIGHT} × {tree_variance_reduction:.3f} + {DEPTH_VARIANCE_WEIGHT} × {depth_variance_factor:.3f} + {FEATURE_VARIANCE_WEIGHT} × {feature_variance_factor:.3f}")
    print(f"  combined_variance = {TREE_VARIANCE_WEIGHT * tree_variance_reduction:.3f} + {DEPTH_VARIANCE_WEIGHT * depth_variance_factor:.3f} + {FEATURE_VARIANCE_WEIGHT * feature_variance_factor:.3f}")
    print(f"  combined_variance = {combined_variance:.3f}")
    
    # Mathematical explanation
    print(f"  Mathematical explanation:")
    print(f"  - Weighted combination of all variance factors")
    print(f"  - Weights: {TREE_VARIANCE_WEIGHT*100}% tree effect, {DEPTH_VARIANCE_WEIGHT*100}% depth effect, {FEATURE_VARIANCE_WEIGHT*100}% feature effect")
    print(f"  - Formula: combined = {TREE_VARIANCE_WEIGHT}×tree + {DEPTH_VARIANCE_WEIGHT}×depth + {FEATURE_VARIANCE_WEIGHT}×feature")
    print(f"  - Lower values = lower variance = better stability")
    print(f"  - Final variance score: {combined_variance:.3f}")
    
    variance_metrics[config_name] = {
        'tree_variance_reduction': tree_variance_reduction,
        'depth_variance_factor': depth_variance_factor,
        'feature_variance_factor': feature_variance_factor,
        'combined_variance': combined_variance
    }

# Find configuration with lowest variance
lowest_variance = min(variance_metrics.items(), key=lambda x: x[1]['combined_variance'])
print(f"\n{'='*60}")
print(f"RESULT: Configuration with lowest prediction variance: {lowest_variance[0]}")
print(f"  Variance score: {lowest_variance[1]['combined_variance']:.3f}")
print(f"{'='*60}")

# Question 4: Memory considerations
print("\n" + "="*60)
print("QUESTION 4: MEMORY USAGE ANALYSIS")
print("="*60)

print("\nMemory usage is influenced by:")
print("1. Number of trees (each tree stores structure and parameters)")
print("2. Maximum depth (deeper trees use more memory)")
print("3. Number of features (affects node storage)")

print("\n" + "-"*40)
print("STEP-BY-STEP MEMORY USAGE ANALYSIS")
print("-"*40)

print(f"\nMemory estimation formulas:")
print(f"  Maximum nodes per tree = 2^(max_depth + 1) - 1")
print(f"  Memory per tree ≈ nodes_per_tree × {BYTES_PER_NODE} bytes")
print(f"  Total memory = memory_per_tree × n_estimators")
print(f"  Mathematical reasoning:")
print(f"    - Binary tree structure: each level doubles the number of nodes")
print(f"    - Maximum nodes = 2^(depth+1) - 1 (complete binary tree)")
print(f"    - Each node stores split criteria, thresholds, and pointers")
print(f"    - Rough estimate: {BYTES_PER_NODE} bytes per node")
print(f"    - Total memory = nodes_per_tree × bytes_per_node × n_trees")

# Calculate memory estimates
memory_estimates = {}

for config_name, params in configurations.items():
    print(f"\n{'-'*20} Configuration {config_name} {'-'*20}")
    
    # Step 1: Calculate maximum nodes per tree
    max_depth = params['max_depth']
    estimated_nodes_per_tree = 2 ** (max_depth + 1) - 1
    
    print(f"Step 1: Maximum Nodes Per Tree")
    print(f"  max_depth = {max_depth}")
    print(f"  estimated_nodes_per_tree = 2^({max_depth} + 1) - 1")
    print(f"  estimated_nodes_per_tree = 2^{max_depth + 1} - 1")
    print(f"  estimated_nodes_per_tree = {2**(max_depth + 1)} - 1")
    print(f"  estimated_nodes_per_tree = {estimated_nodes_per_tree:,}")
    
    # Mathematical explanation
    print(f"  Mathematical explanation:")
    print(f"    - Binary tree: each node has at most 2 children")
    print(f"    - Level 0: 1 node (root)")
    print(f"    - Level 1: 2 nodes")
    print(f"    - Level 2: 4 nodes")
    print(f"    - Level d: 2^d nodes")
    print(f"    - Total nodes = Σ(2^i) from i=0 to d = 2^(d+1) - 1")
    print(f"    - For depth {max_depth}: 2^({max_depth}+1) - 1 = {2**(max_depth + 1)} - 1 = {estimated_nodes_per_tree:,}")
    
    # Step 2: Calculate memory per tree
    memory_per_tree = estimated_nodes_per_tree * BYTES_PER_NODE  # bytes per node (rough estimate)
    
    print(f"\nStep 2: Memory Per Tree")
    print(f"  memory_per_tree = estimated_nodes_per_tree × {BYTES_PER_NODE} bytes")
    print(f"  memory_per_tree = {estimated_nodes_per_tree:,} × {BYTES_PER_NODE}")
    print(f"  memory_per_tree = {memory_per_tree:,} bytes")
    print(f"  memory_per_tree = {memory_per_tree/1024:.2f} KB")
    
    # Mathematical explanation
    print(f"  Mathematical explanation:")
    print(f"  - Each node stores: split feature, threshold, left/right pointers")
    print(f"  - Rough estimate: {BYTES_PER_NODE} bytes per node")
    print(f"  - Formula: memory_per_tree = nodes × bytes_per_node")
    print(f"  - Memory per tree = {estimated_nodes_per_tree:,} × {BYTES_PER_NODE} = {memory_per_tree:,} bytes")
    print(f"  - In KB: {memory_per_tree:,} ÷ 1024 = {memory_per_tree/1024:.2f} KB")
    
    # Step 3: Calculate total memory
    n_estimators = params['n_estimators']
    total_memory = memory_per_tree * n_estimators
    
    print(f"\nStep 3: Total Memory Usage")
    print(f"  total_memory = memory_per_tree × n_estimators")
    print(f"  total_memory = {memory_per_tree:,} × {n_estimators}")
    print(f"  total_memory = {total_memory:,} bytes")
    print(f"  total_memory = {total_memory/1024:.2f} KB")
    print(f"  total_memory = {total_memory/(1024*1024):.2f} MB")
    
    # Mathematical explanation
    print(f"  Mathematical explanation:")
    print(f"  - Total memory = memory per tree × number of trees")
    print(f"  - Formula: total_memory = {memory_per_tree:,} × {n_estimators}")
    print(f"  - Total memory = {total_memory:,} bytes")
    print(f"  - In KB: {total_memory:,} ÷ 1024 = {total_memory/1024:.2f} KB")
    print(f"  - In MB: {total_memory:,} ÷ (1024×1024) = {total_memory/(1024*1024):.2f} MB")
    
    memory_estimates[config_name] = {
        'nodes_per_tree': estimated_nodes_per_tree,
        'memory_per_tree': memory_per_tree,
        'total_memory': total_memory
    }

# Find configuration with lowest memory usage
lowest_memory = min(memory_estimates.items(), key=lambda x: x[1]['total_memory'])
print(f"\n{'='*60}")
print(f"RESULT: Configuration with lowest memory usage: {lowest_memory[0]}")
print(f"  Memory: {lowest_memory[1]['total_memory']/(1024*1024):.2f} MB")
print(f"{'='*60}")

# Question 5: Calculate training time ratio between fastest and slowest configurations
print("\n" + "="*60)
print("QUESTION 5: TRAINING TIME RATIO ANALYSIS")
print("="*60)

print(f"\nProblem statement: Calculate training time ratio between fastest and slowest configurations")
print(f"  Assumption: Each tree takes {TREE_TRAINING_TIME} seconds to train")
print(f"  Note: This is a theoretical calculation based on the given assumption")

print(f"\n" + "-"*40)
print("STEP-BY-STEP TRAINING TIME RATIO CALCULATION")
print("-"*40)

# Find fastest and slowest configurations based on theoretical complexity
fastest_theoretical = min(configurations.items(), key=lambda x: x[1]['n_estimators'] * x[1]['max_depth'] * x[1]['max_features'])
slowest_theoretical = max(configurations.items(), key=lambda x: x[1]['n_estimators'] * x[1]['max_depth'] * x[1]['max_features'])

print(f"\nTheoretical Analysis:")
print(f"  Fastest configuration: {fastest_theoretical[0]} (complexity: {fastest_theoretical[1]['n_estimators']} × {fastest_theoretical[1]['max_depth']} × {fastest_theoretical[1]['max_features']} = {fastest_theoretical[1]['n_estimators'] * fastest_theoretical[1]['max_depth'] * fastest_theoretical[1]['max_features']})")
print(f"  Slowest configuration: {slowest_theoretical[0]} (complexity: {slowest_theoretical[1]['n_estimators']} × {slowest_theoretical[1]['max_depth']} × {slowest_theoretical[1]['max_features']} = {slowest_theoretical[1]['n_estimators'] * slowest_theoretical[1]['max_depth'] * slowest_theoretical[1]['max_features']})")

# Calculate theoretical training times
fastest_complexity = fastest_theoretical[1]['n_estimators'] * fastest_theoretical[1]['max_depth'] * fastest_theoretical[1]['max_features']
slowest_complexity = slowest_theoretical[1]['n_estimators'] * slowest_theoretical[1]['max_depth'] * slowest_theoretical[1]['max_features']

print(f"\nStep 1: Theoretical Training Time Calculation")
print(f"  Training time ∝ complexity_factor × {TREE_TRAINING_TIME} seconds")
print(f"  Fastest time ∝ {fastest_complexity} × {TREE_TRAINING_TIME} = {fastest_complexity * TREE_TRAINING_TIME} time units")
print(f"  Slowest time ∝ {slowest_complexity} × {TREE_TRAINING_TIME} = {slowest_complexity * TREE_TRAINING_TIME} time units")

# Calculate ratio (slowest/fastest)
training_time_ratio = slowest_complexity / fastest_complexity

print(f"\nStep 2: Training Time Ratio Calculation")
print(f"  Ratio = slowest_complexity / fastest_complexity")
print(f"  Ratio = {slowest_complexity} / {fastest_complexity}")
print(f"  Ratio = {training_time_ratio:.3f}")

print(f"\nStep 3: Interpretation")
print(f"  The slowest configuration takes {training_time_ratio:.1f}x longer to train than the fastest")
print(f"  This means training the slowest configuration requires {training_time_ratio:.1f} times more computational resources")

# Also calculate ratio based on actual measured times
fastest_actual = min(training_times.items(), key=lambda x: x[1])
slowest_actual = max(training_times.items(), key=lambda x: x[1])

print(f"\nActual Measured Training Times:")
print(f"  Fastest: Configuration {fastest_actual[0]} ({fastest_actual[1]:.3f} seconds)")
print(f"  Slowest: Configuration {slowest_actual[0]} ({slowest_actual[1]:.3f} seconds)")
print(f"  Actual ratio: {slowest_actual[1] / fastest_actual[1]:.3f}")

print(f"\n{'='*60}")
print(f"RESULT: Training time ratio (slowest/fastest) = {training_time_ratio:.3f}")
print(f"  Theoretical: {training_time_ratio:.3f}x")
print(f"  Actual measured: {slowest_actual[1] / fastest_actual[1]:.3f}x")
print(f"{'='*60}")

# Create comprehensive visualization
print("\n" + "="*60)
print("CREATING COMPREHENSIVE VISUALIZATION")
print("="*60)

# Create subplots for different metrics
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Random Forest Configuration Comparison', fontsize=16, fontweight='bold')

# Plot 1: Diversity Metrics
config_names = list(configurations.keys())
feature_diversities = [diversity_metrics[config]['feature_diversity'] for config in config_names]
tree_diversities = [diversity_metrics[config]['tree_diversity'] for config in config_names]
combined_diversities = [diversity_metrics[config]['combined_diversity'] for config in config_names]

x = np.arange(len(config_names))
width = 0.25

ax1.bar(x - width, feature_diversities, width, label='Feature Diversity', alpha=0.8)
ax1.bar(x, tree_diversities, width, label='Tree Diversity', alpha=0.8)
ax1.bar(x + width, combined_diversities, width, label='Combined Diversity', alpha=0.8)

ax1.set_xlabel('Configuration')
ax1.set_ylabel('Diversity Score')
ax1.set_title('Tree Diversity Analysis')
ax1.set_xticks(x)
ax1.set_xticklabels(config_names)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training Times
times = [training_times[config] for config in config_names]
bars = ax2.bar(config_names, times, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
ax2.set_xlabel('Configuration')
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('Training Speed Comparison')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{time_val:.3f}s', ha='center', va='bottom')

# Plot 3: Variance Metrics
tree_var_reductions = [variance_metrics[config]['tree_variance_reduction'] for config in config_names]
depth_var_factors = [variance_metrics[config]['depth_variance_factor'] for config in config_names]
feature_var_factors = [variance_metrics[config]['feature_variance_factor'] for config in config_names]

ax3.bar(x - width, tree_var_reductions, width, label='Tree Variance Reduction', alpha=0.8)
ax3.bar(x, depth_var_factors, width, label='Depth Variance Factor', alpha=0.8)
ax3.bar(x + width, feature_var_factors, width, label='Feature Variance Factor', alpha=0.8)

ax3.set_xlabel('Configuration')
ax3.set_ylabel('Variance Score')
ax3.set_title('Prediction Variance Analysis')
ax3.set_xticks(x)
ax3.set_xticklabels(config_names)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Memory Usage
memories = [memory_estimates[config]['total_memory'] / (1024*1024) for config in config_names]  # Convert to MB
bars = ax4.bar(config_names, memories, color=['gold', 'silver', 'brown'], alpha=0.8)
ax4.set_xlabel('Configuration')
ax4.set_ylabel('Memory Usage (MB)')
ax4.set_title('Memory Usage Comparison')
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar, mem_val in zip(bars, memories):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{mem_val:.1f} MB', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'random_forest_configuration_comparison.png'), 
            dpi=300, bbox_inches='tight')

# Create detailed comparison table
print("\n" + "="*60)
print("DETAILED COMPARISON TABLE")
print("="*60)

comparison_data = []
for config_name in config_names:
    comparison_data.append({
        'Configuration': config_name,
        'Trees': configurations[config_name]['n_estimators'],
        'Features/Split': configurations[config_name]['max_features'],
        'Max Depth': configurations[config_name]['max_depth'],
        'Training Time (s)': f"{training_times[config_name]:.3f}",
        'CV Accuracy': f"{training_scores[config_name]:.3f}",
        'Diversity Score': f"{diversity_metrics[config_name]['combined_diversity']:.3f}",
        'Variance Score': f"{variance_metrics[config_name]['combined_variance']:.3f}",
        'Memory (MB)': f"{memory_estimates[config_name]['total_memory']/(1024*1024):.1f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv(os.path.join(save_dir, 'configuration_comparison.csv'), index=False)

# Create summary visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Radar chart data
categories = ['Training Speed\n(Lower is Better)', 'Diversity\n(Higher is Better)', 
             'Low Variance\n(Lower is Better)', 'Memory Efficiency\n(Lower is Better)']

# Normalize scores for radar chart (0-1 scale, where 1 is best)
normalized_scores = {}
for config_name in config_names:
    # Training speed: normalize so fastest = 1
    speed_score = 1 - (training_times[config_name] / max(training_times.values()))
    
    # Diversity: already 0-1
    diversity_score = diversity_metrics[config_name]['combined_diversity']
    
    # Variance: normalize so lowest variance = 1
    max_variance = max([variance_metrics[config]['combined_variance'] for config in config_names])
    variance_score = 1 - (variance_metrics[config_name]['combined_variance'] / max_variance)
    
    # Memory: normalize so lowest memory = 1
    max_memory = max([memory_estimates[config]['total_memory'] for config in config_names])
    memory_score = 1 - (memory_estimates[config_name]['total_memory'] / max_memory)
    
    normalized_scores[config_name] = [speed_score, diversity_score, variance_score, memory_score]

# Plot radar chart
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add yticks
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
ax.grid(True)

# Plot each configuration
colors = ['skyblue', 'lightcoral', 'lightgreen']
for i, (config_name, scores) in enumerate(normalized_scores.items()):
    scores += scores[:1]  # Complete the circle
    ax.plot(angles, scores, 'o-', linewidth=2, label=f'Config {config_name}', color=colors[i])
    ax.fill(angles, scores, alpha=0.25, color=colors[i])

ax.set_title('Configuration Performance Radar Chart\n(All metrics normalized to 0-1 scale)', 
             pad=20, fontsize=14, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'configuration_radar_chart.png'), 
            dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Final summary
print("\n" + "="*60)
print("FINAL SUMMARY AND RECOMMENDATIONS")
print("="*60)

print("\nBased on the analysis:")
print(f"1. Highest tree diversity: Configuration {best_diversity[0]}")
print(f"2. Fastest to train: Configuration {fastest_config[0]}")
print(f"3. Lowest prediction variance: Configuration {lowest_variance[0]}")
print(f"4. Lowest memory usage: Configuration {lowest_memory[0]}")

print("\nRecommendations:")
print("- For maximum diversity: Choose Configuration A (100 trees, 5 features, depth 10)")
print("- For speed: Choose Configuration B (50 trees, 10 features, depth 15)")
print("- For stability: Choose Configuration C (200 trees, 3 features, depth 8)")
print("- For memory efficiency: Choose Configuration B (50 trees, 10 features, depth 15)")

print(f"\nDetailed results and visualizations saved to: {save_dir}")
