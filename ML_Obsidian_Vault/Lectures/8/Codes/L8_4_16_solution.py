import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean, cosine
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.special import gamma
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_4_Quiz_16")
os.makedirs(save_dir, exist_ok=True)

# Set non-interactive backend to prevent plots from opening
import matplotlib
matplotlib.use('Agg')

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

print("Question 16: Curse of Dimensionality and Evaluation Criteria")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

def generate_high_dimensional_data(n_samples, n_features, noise_level=0.1):
    """Generate synthetic data with controlled relationships between features and target."""
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create target variable with some features being informative
    n_informative = max(1, n_features // 4)  # 25% of features are informative
    
    # First n_informative features have strong relationship with target
    informative_features = X[:, :n_informative]
    target = np.sum(informative_features, axis=1) + noise_level * np.random.randn(n_samples)
    
    return X, target

def analyze_distance_measures(X, dimensions):
    """Analyze how distance measures change with dimensionality."""
    print("\n1. Distance Measures Analysis:")
    print("-" * 40)
    
    # Sample two points and compute distances
    point1 = X[0, :]
    point2 = X[1, :]
    
    euclidean_distances = []
    cosine_distances = []
    manhattan_distances = []
    
    for d in dimensions:
        # Take first d dimensions
        p1_d = point1[:d]
        p2_d = point2[:d]
        
        # Compute distances
        euclidean_distances.append(euclidean(p1_d, p2_d))
        cosine_distances.append(cosine(p1_d, p2_d))
        manhattan_distances.append(np.sum(np.abs(p1_d - p2_d)))  # Manhattan distance
    
    # Plot distance measures vs dimensionality
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(dimensions, euclidean_distances, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Dimensionality')
    plt.ylabel('Euclidean Distance')
    plt.title('Euclidean Distance vs Dimensionality')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(dimensions, cosine_distances, 'r-s', linewidth=2, markersize=6)
    plt.xlabel('Dimensionality')
    plt.ylabel('Cosine Distance')
    plt.title('Cosine Distance vs Dimensionality')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(dimensions, manhattan_distances, 'g-^', linewidth=2, markersize=6)
    plt.xlabel('Dimensionality')
    plt.ylabel('Manhattan Distance')
    plt.title('Manhattan Distance vs Dimensionality')
    plt.grid(True, alpha=0.3)
    
    # Distance ratio analysis
    plt.subplot(2, 2, 4)
    euclidean_ratios = [euclidean_distances[i]/euclidean_distances[0] for i in range(len(dimensions))]
    plt.plot(dimensions, euclidean_ratios, 'purple', linewidth=2, marker='o', markersize=6)
    plt.xlabel('Dimensionality')
    plt.ylabel('Distance Ratio (Normalized)')
    plt.title('Distance Growth Rate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distance_measures_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"Euclidean distance growth: {euclidean_distances[-1]/euclidean_distances[0]:.2f}x")
    print(f"Cosine distance behavior: {cosine_distances[-1]:.4f} (stable)")
    print(f"Manhattan distance growth: {manhattan_distances[-1]/manhattan_distances[0]:.2f}x")
    
    return euclidean_distances, cosine_distances, manhattan_distances

def analyze_information_measures(X, y, dimensions):
    """Analyze how information measures change with dimensionality."""
    print("\n2. Information Measures Analysis:")
    print("-" * 40)
    
    # Mutual information scores
    mi_scores = []
    f_scores = []
    
    for d in dimensions:
        # Take first d dimensions
        X_d = X[:, :d]
        
        # Compute mutual information
        mi_score = np.mean(mutual_info_regression(X_d, y))
        mi_scores.append(mi_score)
        
        # Compute F-statistic
        f_score = np.mean(f_regression(X_d, y)[0])
        f_scores.append(f_score)
    
    # Plot information measures vs dimensionality
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(dimensions, mi_scores, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Dimensionality')
    plt.ylabel('Mutual Information Score')
    plt.title('Mutual Information vs Dimensionality')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(dimensions, f_scores, 'r-s', linewidth=2, markersize=6)
    plt.xlabel('Dimensionality')
    plt.ylabel('F-Statistic Score')
    plt.title('F-Statistic vs Dimensionality')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'information_measures_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"Mutual Information change: {mi_scores[-1]/mi_scores[0]:.2f}x")
    print(f"F-statistic change: {f_scores[-1]/f_scores[0]:.2f}x")
    
    return mi_scores, f_scores

def analyze_dependency_measures(X, y, dimensions):
    """Analyze how dependency measures change with dimensionality."""
    print("\n3. Dependency Measures Analysis:")
    print("-" * 40)
    
    # Correlation and dependency measures
    pearson_corrs = []
    spearman_corrs = []
    r2_scores = []
    
    for d in dimensions:
        # Take first d dimensions
        X_d = X[:, :d]
        
        # Compute correlations with target
        pearson_corr = np.mean([abs(pearsonr(X_d[:, i], y)[0]) for i in range(X_d.shape[1])])
        spearman_corr = np.mean([abs(spearmanr(X_d[:, i], y)[0]) for i in range(X_d.shape[1])])
        
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        
        # Compute R² score using linear regression
        if d > 1:
            model = LinearRegression()
            model.fit(X_d, y)
            y_pred = model.predict(X_d)
            r2 = r2_score(y, y_pred)
            r2_scores.append(r2)
        else:
            r2_scores.append(0)
    
    # Plot dependency measures vs dimensionality
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(dimensions, pearson_corrs, 'b-o', linewidth=2, markersize=6, label='Pearson')
    plt.plot(dimensions, spearman_corrs, 'r-s', linewidth=2, markersize=6, label='Spearman')
    plt.xlabel('Dimensionality')
    plt.ylabel('Correlation Coefficient')
    plt.title('Correlation Measures vs Dimensionality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(dimensions, r2_scores, 'g-^', linewidth=2, markersize=6)
    plt.xlabel('Dimensionality')
    plt.ylabel('R² Score')
    plt.title('R² Score vs Dimensionality')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dependency_measures_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"Pearson correlation change: {pearson_corrs[-1]/pearson_corrs[0]:.2f}x")
    print(f"Spearman correlation change: {spearman_corrs[-1]/spearman_corrs[0]:.2f}x")
    print(f"R² score change: {r2_scores[-1]/r2_scores[0] if r2_scores[0] > 0 else 'N/A'}")
    
    return pearson_corrs, spearman_corrs, r2_scores

def compare_robustness(euclidean_distances, cosine_distances, mi_scores, f_scores, 
                       pearson_corrs, spearman_corrs, r2_scores, dimensions):
    """Compare the robustness of different evaluation criteria."""
    print("\n4. Robustness Comparison:")
    print("-" * 40)
    
    # Normalize all measures to [0,1] scale for comparison
    def normalize_measure(measure):
        measure_array = np.array(measure)
        if np.max(measure_array) == np.min(measure_array):
            return np.ones_like(measure_array)
        return (measure_array - np.min(measure_array)) / (np.max(measure_array) - np.min(measure_array))
    
    # Normalize measures (higher values = more robust)
    euclidean_norm = 1 - normalize_measure(euclidean_distances)  # Invert since we want stability
    cosine_norm = normalize_measure(cosine_distances)
    mi_norm = normalize_measure(mi_scores)
    f_norm = normalize_measure(f_scores)
    pearson_norm = normalize_measure(pearson_corrs)
    spearman_norm = normalize_measure(spearman_corrs)
    r2_norm = normalize_measure(r2_scores)
    
    # Plot robustness comparison
    plt.figure(figsize=(14, 8))
    
    # Main robustness plot
    plt.subplot(2, 2, 1)
    plt.plot(dimensions, euclidean_norm, 'b-o', linewidth=2, markersize=6, label='Euclidean Distance')
    plt.plot(dimensions, cosine_norm, 'r-s', linewidth=2, markersize=6, label='Cosine Distance')
    plt.plot(dimensions, mi_norm, 'g-^', linewidth=2, markersize=6, label='Mutual Information')
    plt.plot(dimensions, f_norm, 'purple', linewidth=2, marker='d', markersize=6, label='F-Statistic')
    plt.xlabel('Dimensionality')
    plt.ylabel('Robustness Score (Normalized)')
    plt.title('Robustness Comparison of Evaluation Criteria')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distance measures comparison
    plt.subplot(2, 2, 2)
    plt.plot(dimensions, euclidean_norm, 'b-o', linewidth=2, markersize=6, label='Euclidean')
    plt.plot(dimensions, cosine_norm, 'r-s', linewidth=2, markersize=6, label='Cosine')
    plt.xlabel('Dimensionality')
    plt.ylabel('Robustness Score')
    plt.title('Distance Measures Robustness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Information measures comparison
    plt.subplot(2, 2, 3)
    plt.plot(dimensions, mi_norm, 'g-^', linewidth=2, markersize=6, label='Mutual Information')
    plt.plot(dimensions, f_norm, 'purple', linewidth=2, marker='d', markersize=6, label='F-Statistic')
    plt.xlabel('Dimensionality')
    plt.ylabel('Robustness Score')
    plt.title('Information Measures Robustness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dependency measures comparison
    plt.subplot(2, 2, 4)
    plt.plot(dimensions, pearson_norm, 'orange', linewidth=2, marker='o', markersize=6, label='Pearson')
    plt.plot(dimensions, spearman_norm, 'brown', linewidth=2, marker='s', markersize=6, label='Spearman')
    plt.plot(dimensions, r2_norm, 'pink', linewidth=2, marker='^', markersize=6, label='R² Score')
    plt.xlabel('Dimensionality')
    plt.ylabel('Robustness Score')
    plt.title('Dependency Measures Robustness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'robustness_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate overall robustness scores
    robustness_scores = {
        'Euclidean Distance': np.mean(euclidean_norm),
        'Cosine Distance': np.mean(cosine_norm),
        'Mutual Information': np.mean(mi_norm),
        'F-Statistic': np.mean(f_norm),
        'Pearson Correlation': np.mean(pearson_norm),
        'Spearman Correlation': np.mean(spearman_norm),
        'R² Score': np.mean(r2_norm)
    }
    
    # Sort by robustness
    sorted_robustness = sorted(robustness_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n5. Overall Robustness Ranking:")
    print("-" * 40)
    for i, (measure, score) in enumerate(sorted_robustness, 1):
        print(f"{i}. {measure}: {score:.3f}")
    
    # Create final summary plot
    plt.figure(figsize=(10, 6))
    measures = [item[0] for item in sorted_robustness]
    scores = [item[1] for item in sorted_robustness]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(measures)))
    bars = plt.bar(measures, scores, color=colors, alpha=0.8)
    
    plt.xlabel('Evaluation Criteria')
    plt.ylabel('Robustness Score')
    plt.title('Overall Robustness to High Dimensionality')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_robustness_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return robustness_scores

def demonstrate_sparsity_effect():
    """Demonstrate the sparsity effect in high dimensions."""
    print("\n6. Sparsity Effect Demonstration:")
    print("-" * 40)
    
    # Generate data with increasing sparsity
    n_samples = 1000
    dimensions = [2, 5, 10, 20, 50, 100]
    
    sparsity_ratios = []
    volume_ratios = []
    
    for d in dimensions:
        # Generate random points in d-dimensional unit cube
        points = np.random.random((n_samples, d))
        
        # Calculate sparsity (fraction of points near the surface)
        # Points are "near surface" if they're within 0.1 of any boundary
        near_surface = np.sum(np.any((points < 0.1) | (points > 0.9), axis=1))
        sparsity_ratio = near_surface / n_samples
        sparsity_ratios.append(sparsity_ratio)
        
        # Calculate volume ratio (volume of inscribed sphere to cube)
        # In d dimensions, inscribed sphere has radius 0.5
        sphere_volume = (np.pi**(d/2) / gamma(d/2 + 1)) * (0.5**d)
        cube_volume = 1.0
        volume_ratio = sphere_volume / cube_volume
        volume_ratios.append(volume_ratio)
    
    # Plot sparsity effects
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(dimensions, sparsity_ratios, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Dimensionality')
    plt.ylabel('Sparsity Ratio')
    plt.title('Data Sparsity vs Dimensionality')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(dimensions, volume_ratios, 'r-s', linewidth=2, markersize=6)
    plt.xlabel('Dimensionality')
    plt.ylabel('Volume Ratio')
    plt.title('Sphere-to-Cube Volume Ratio')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sparsity_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sparsity ratio at d=100: {sparsity_ratios[-1]:.3f}")
    print(f"Volume ratio at d=100: {volume_ratios[-1]:.2e}")

# Main execution
if __name__ == "__main__":
    # Parameters
    n_samples = 1000
    max_dimensions = 100
    dimensions = np.arange(2, max_dimensions + 1, 2)
    
    print(f"Generating data with {n_samples} samples and up to {max_dimensions} dimensions...")
    
    # Generate high-dimensional data
    X, y = generate_high_dimensional_data(n_samples, max_dimensions)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # 1. Analyze distance measures
    euclidean_distances, cosine_distances, manhattan_distances = analyze_distance_measures(X, dimensions)
    
    # 2. Analyze information measures
    mi_scores, f_scores = analyze_information_measures(X, y, dimensions)
    
    # 3. Analyze dependency measures
    pearson_corrs, spearman_corrs, r2_scores = analyze_dependency_measures(X, y, dimensions)
    
    # 4. Compare robustness
    robustness_scores = compare_robustness(
        euclidean_distances, cosine_distances, mi_scores, f_scores,
        pearson_corrs, spearman_corrs, r2_scores, dimensions
    )
    
    # 5. Demonstrate sparsity effects
    demonstrate_sparsity_effect()
    
    print(f"\nAll plots saved to: {save_dir}")
    print("\nAnalysis complete!")
