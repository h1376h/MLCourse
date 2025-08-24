import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.special import gamma
from scipy.stats import uniform
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 2: CURSE OF DIMENSIONALITY - DETAILED SOLUTION")
print("=" * 80)

# =============================================================================
# TASK 1: What is the curse of dimensionality in one sentence?
# =============================================================================
print("\n" + "="*60)
print("TASK 1: Definition of Curse of Dimensionality")
print("="*60)

curse_definition = """The curse of dimensionality refers to the phenomenon where the performance of 
machine learning algorithms deteriorates as the number of features (dimensions) increases, 
due to the exponential growth in data sparsity and the increasing difficulty of finding 
meaningful patterns in high-dimensional spaces."""

print(f"Answer: {curse_definition}")

# =============================================================================
# TASK 2: How does the curse affect nearest neighbor algorithms?
# =============================================================================
print("\n" + "="*60)
print("TASK 2: Impact on Nearest Neighbor Algorithms")
print("="*60)

nn_impact = """1. **Distance Concentration**: As dimensions increase, all pairwise distances between 
   points become more similar, making it harder to distinguish between 'near' and 'far' points.
2. **Data Sparsity**: Points become increasingly isolated in high-dimensional space, 
   reducing the effectiveness of local neighborhood information.
3. **Irrelevant Features**: More dimensions often mean more noise, diluting the signal 
   that distance-based algorithms rely on.
4. **Computational Cost**: Distance calculations become more expensive with more dimensions."""

print("Answer:")
print(nn_impact)

# =============================================================================
# TASK 3: What happens to the volume of a hypercube as dimensions increase?
# =============================================================================
print("\n" + "="*60)
print("TASK 3: Hypercube Volume Analysis")
print("="*60)

# Calculate volume and surface area for different dimensions
dimensions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
unit_hypercube_volume = []
unit_hypercube_surface_area = []

for d in dimensions:
    # Volume of unit hypercube: V = 1^d = 1 for all d
    volume = 1.0
    
    # Surface area of unit hypercube: S = 2d (2^(d-1)) = 2d
    # Each face has area 1, and there are 2d faces in d dimensions
    surface_area = 2 * d
    
    unit_hypercube_volume.append(volume)
    unit_hypercube_surface_area.append(surface_area)

print("Volume and Surface Area of Unit Hypercube:")
print(f"{'Dimensions':<12} {'Volume':<10} {'Surface Area':<15} {'V/S Ratio':<15}")
print("-" * 60)
for i, d in enumerate(dimensions):
    v_s_ratio = unit_hypercube_volume[i] / unit_hypercube_surface_area[i]
    print(f"{d:<12} {unit_hypercube_volume[i]:<10.3f} {unit_hypercube_surface_area[i]:<15.3f} {v_s_ratio:<15.6f}")

print(f"\nKey Insight: Volume remains constant (1) while surface area grows linearly with dimensions.")
print(f"This means the ratio of volume to surface area decreases as 1/(2d), approaching 0 as d → ∞.")

# =============================================================================
# TASK 4: Sample density calculation
# =============================================================================
print("\n" + "="*60)
print("TASK 4: Sample Density Calculation")
print("="*60)

# Given: 1000 samples in 2D
samples_2d = 1000
dim_2d = 2
dim_10d = 10

# For similar density, we need samples proportional to volume
# If we want to maintain similar spacing between points
# The number of samples needed grows exponentially with dimensions

# Method 1: Volume-based scaling
# Assuming we want to maintain similar point density
volume_2d = 1.0  # Unit square
volume_10d = 1.0  # Unit hypercube

# For similar density: samples_10d / volume_10d ≈ samples_2d / volume_2d
# Since volumes are both 1, this gives us the same number of samples
# But this doesn't account for the curse of dimensionality

# Method 2: Distance-based scaling (more realistic)
# If we want to maintain similar average distance between points
# The number of samples needed grows as O(d^d) for uniform distribution

# For a more practical estimate, let's use the fact that in high dimensions,
# most of the volume is concentrated near the surface
estimated_samples_10d = samples_2d * (dim_10d / dim_2d)**2  # Conservative estimate

print(f"Given: {samples_2d} samples in {dim_2d}D")
print(f"Question: How many samples needed in {dim_10d}D for similar density?")

print(f"\nMethod 1 (Volume-based): {samples_2d} samples")
print(f"   - Both 2D and 10D unit hypercubes have volume = 1")
print(f"   - But this doesn't account for the curse of dimensionality")

print(f"\nMethod 2 (Distance-based estimate): {estimated_samples_10d:,.0f} samples")
print(f"   - This is a conservative estimate based on maintaining similar point spacing")
print(f"   - In reality, you might need exponentially more samples")

print(f"\nPractical Answer: You would need significantly more than {samples_2d} samples")
print(f"in {dim_10d}D to achieve similar effective density due to the curse of dimensionality.")

# =============================================================================
# TASK 5: Volume to Surface Area Ratio
# =============================================================================
print("\n" + "="*60)
print("TASK 5: Volume to Surface Area Ratio")
print("="*60)

# Calculate the ratio for 2D vs 10D
d_2d = 2
d_10d = 10

# Volume = 1 for both (unit hypercube)
volume_2d = 1.0
volume_10d = 1.0

# Surface area = 2d
surface_2d = 2 * d_2d
surface_10d = 2 * d_10d

# Ratio = Volume / Surface Area
ratio_2d = volume_2d / surface_2d
ratio_10d = volume_10d / surface_10d

print(f"2D Unit Square:")
print(f"  Volume = {volume_2d}")
print(f"  Surface Area = {surface_2d} (perimeter)")
print(f"  V/S Ratio = {volume_2d} / {surface_2d} = {ratio_2d:.6f}")

print(f"\n10D Unit Hypercube:")
print(f"  Volume = {volume_10d}")
print(f"  Surface Area = {surface_10d}")
print(f"  V/S Ratio = {volume_10d} / {surface_10d} = {ratio_10d:.6f}")

print(f"\nRatio Comparison:")
print(f"  2D Ratio / 10D Ratio = {ratio_2d:.6f} / {ratio_10d:.6f} = {ratio_2d/ratio_10d:.2f}")

print(f"\nKey Insight: The V/S ratio decreases by a factor of {ratio_2d/ratio_10d:.1f}")
print(f"This means that in higher dimensions, the volume becomes increasingly concentrated")
print(f"near the surface, making the 'interior' of the hypercube less significant.")

# =============================================================================
# TASK 6: Expected Distance Calculation
# =============================================================================
print("\n" + "="*60)
print("TASK 6: Expected Distance Analysis")
print("="*60)

# Given formula: E[d] = sqrt(d/6) where d is dimensions
def expected_distance(dim):
    return np.sqrt(dim / 6)

# Calculate for different dimensions
dimensions_to_test = [2, 5, 10]
expected_distances = []
max_distances = []
ratios = []

print("Expected Distance Analysis:")
print(f"{'Dimensions':<12} {'E[d]':<10} {'Max Distance':<15} {'Max/E[d] Ratio':<15}")
print("-" * 60)

for d in dimensions_to_test:
    e_d = expected_distance(d)
    # Maximum distance in unit hypercube = sqrt(d) (diagonal)
    max_d = np.sqrt(d)
    ratio = max_d / e_d
    
    expected_distances.append(e_d)
    max_distances.append(max_d)
    ratios.append(ratio)
    
    print(f"{d:<12} {e_d:<10.4f} {max_d:<15.4f} {ratio:<15.4f}")

print(f"\nAnalysis:")
print(f"1. Expected distance increases as √(d/6)")
print(f"2. Maximum distance increases as √d")
print(f"3. The ratio of maximum to expected distance increases as dimensions increase")

print(f"\nMathematical Explanation:")
print(f"  E[d] = √(d/6) ≈ √(d) × 0.408")
print(f"  Max distance = √d")
print(f"  Ratio = √d / (√d × 0.408) = 1/0.408 ≈ 2.45")

print(f"\nThis means that as dimensions increase:")
print(f"- The expected distance between random points increases")
print(f"- The maximum possible distance increases faster")
print(f"- The ratio approaches a constant value of approximately 2.45")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Visualization 1: Volume vs Surface Area
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(dimensions, unit_hypercube_volume, 'b-', linewidth=2, marker='o', label='Volume')
plt.plot(dimensions, unit_hypercube_surface_area, 'r-', linewidth=2, marker='s', label='Surface Area')
plt.xlabel('Dimensions')
plt.ylabel('Value')
plt.title('Unit Hypercube: Volume vs Surface Area')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
v_s_ratios = [v/s for v, s in zip(unit_hypercube_volume, unit_hypercube_surface_area)]
plt.plot(dimensions, v_s_ratios, 'g-', linewidth=2, marker='^')
plt.xlabel('Dimensions')
plt.ylabel('Volume/Surface Area Ratio')
plt.title('Volume to Surface Area Ratio')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(dimensions_to_test, expected_distances, 'purple', linewidth=2, marker='o', label='Expected Distance')
plt.plot(dimensions_to_test, max_distances, 'orange', linewidth=2, marker='s', label='Max Distance')
plt.xlabel('Dimensions')
plt.ylabel('Distance')
plt.title('Distance Analysis in Unit Hypercube')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(dimensions_to_test, ratios, 'brown', linewidth=2, marker='^')
plt.xlabel('Dimensions')
plt.ylabel('Max Distance / Expected Distance')
plt.title('Ratio of Maximum to Expected Distance')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'curse_of_dimensionality_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 2: 2D vs 3D Hypercube Comparison
fig = plt.figure(figsize=(15, 6))

# 2D Square
ax1 = fig.add_subplot(1, 3, 1)
square = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.7)
ax1.add_patch(square)
ax1.set_xlim(-0.1, 1.1)
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('2D Unit Square\nVolume = 1, Perimeter = 4')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# 3D Cube
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
x = [0, 1, 1, 0, 0, 1, 1, 0]
y = [0, 0, 1, 1, 0, 0, 1, 1]
z = [0, 0, 0, 0, 1, 1, 1, 1]
ax2.scatter(x, y, z, c='red', s=100)
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_zlabel('$x_3$')
ax2.set_title('3D Unit Cube\nVolume = 1, Surface Area = 6')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_zlim(0, 1)

# 4D+ Conceptual Representation
ax3 = fig.add_subplot(1, 3, 3)
# Create a conceptual representation of higher dimensions
dimensions_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
volume_plot = [1] * len(dimensions_plot)
surface_plot = [2 * d for d in dimensions_plot]

ax3.plot(dimensions_plot, volume_plot, 'b-', linewidth=3, label='Volume (constant)')
ax3.plot(dimensions_plot, surface_plot, 'r-', linewidth=3, label='Surface Area (linear)')
ax3.fill_between(dimensions_plot, volume_plot, surface_plot, alpha=0.3, color='gray')
ax3.set_xlabel('Dimensions')
ax3.set_ylabel('Value')
ax3.set_title('Higher Dimensions\nVolume vs Surface Area')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'hypercube_comparison.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Distance Distribution Simulation
plt.figure(figsize=(15, 10))

# Simulate random points in different dimensions and calculate distances
np.random.seed(42)
n_points = 1000
n_pairs = 500

dimensions_sim = [2, 5, 10]
distance_distributions = []

for d in dimensions_sim:
    # Generate random points in unit hypercube
    points = np.random.uniform(0, 1, (n_points, d))
    
    # Calculate pairwise distances
    distances = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n_points, 2, replace=False)
        dist = np.linalg.norm(points[i] - points[j])
        distances.append(dist)
    
    distance_distributions.append(distances)

# Plot distance distributions
for i, (d, distances) in enumerate(zip(dimensions_sim, distance_distributions)):
    plt.subplot(2, 3, i+1)
    plt.hist(distances, bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
    plt.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.3f}')
    plt.axvline(np.sqrt(d), color='green', linestyle='--', linewidth=2, label=f'Max: {np.sqrt(d):.3f}')
    plt.axvline(expected_distance(d), color='orange', linestyle='--', linewidth=2, label=f'Theoretical E[d]: {expected_distance(d):.3f}')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title(f'{d}D: Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot theoretical vs empirical expected distances
plt.subplot(2, 3, 4)
theoretical_distances = [expected_distance(d) for d in dimensions_sim]
empirical_distances = [np.mean(dist) for dist in distance_distributions]

plt.plot(dimensions_sim, theoretical_distances, 'b-o', linewidth=2, label='Theoretical E[d]')
plt.plot(dimensions_sim, empirical_distances, 'r-s', linewidth=2, label='Empirical Mean')
plt.xlabel('Dimensions')
plt.ylabel('Expected Distance')
plt.title('Theoretical vs Empirical Expected Distances')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot ratio analysis
plt.subplot(2, 3, 5)
max_distances_sim = [np.sqrt(d) for d in dimensions_sim]
ratios_sim = [max_d / exp_d for max_d, exp_d in zip(max_distances_sim, theoretical_distances)]

plt.plot(dimensions_sim, ratios_sim, 'g-^', linewidth=2, marker='^')
plt.xlabel('Dimensions')
plt.ylabel('Max Distance / Expected Distance')
plt.title('Ratio Analysis')
plt.grid(True, alpha=0.3)

# Plot volume concentration
plt.subplot(2, 3, 6)
# Show how volume concentrates near surface in higher dimensions
dimensions_vol = np.linspace(1, 20, 100)
volume_ratios = [1/(2*d) for d in dimensions_vol]

plt.plot(dimensions_vol, volume_ratios, 'purple', linewidth=2)
plt.xlabel('Dimensions')
plt.ylabel('Volume/Surface Area Ratio')
plt.title('Volume Concentration Near Surface')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'distance_analysis_simulation.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")
print("\n" + "="*80)
print("SOLUTION COMPLETE!")
print("="*80)
