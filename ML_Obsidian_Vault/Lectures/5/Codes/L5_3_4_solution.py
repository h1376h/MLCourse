import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Disable LaTeX to avoid compilation issues
plt.rcParams['font.size'] = 12

print("=" * 60)
print("QUESTION 4: RBF KERNEL PROPERTIES")
print("=" * 60)

# Given vectors
x = np.array([1, 0])
z = np.array([0, 1])

print(f"Given vectors:")
print(f"x = {x}")
print(f"z = {z}")

# Task 1: Calculate K(x,z) for different gamma values
print("\n" + "="*50)
print("TASK 1: RBF KERNEL CALCULATIONS")
print("="*50)

def rbf_kernel(x, z, gamma):
    """
    Calculate RBF kernel K(x,z) = exp(-gamma * ||x - z||^2)
    """
    diff = x - z
    squared_distance = np.dot(diff, diff)
    kernel_value = np.exp(-gamma * squared_distance)
    return squared_distance, kernel_value

# Calculate for different gamma values
gamma_values = [0.5, 1, 2]

print(f"\nCalculating K(x,z) = exp(-γ||x - z||²) for x = {x}, z = {z}")

# First calculate the squared distance
diff = x - z
squared_dist = np.dot(diff, diff)
print(f"\nStep 1: Calculate ||x - z||²")
print(f"x - z = {x} - {z} = {diff}")
print(f"||x - z||² = {diff[0]}² + {diff[1]}² = {diff[0]**2} + {diff[1]**2} = {squared_dist}")

print(f"\nStep 2: Calculate K(x,z) for different γ values:")
for gamma in gamma_values:
    _, kernel_val = rbf_kernel(x, z, gamma)
    print(f"γ = {gamma}: K(x,z) = exp(-{gamma} × {squared_dist}) = exp({-gamma * squared_dist}) = {kernel_val:.6f}")

# Task 2: Show that K(x,x) = 1 for any x
print("\n" + "="*50)
print("TASK 2: PROOF THAT K(x,x) = 1")
print("="*50)

def prove_self_kernel():
    """
    Prove that K(x,x) = 1 for any x
    """
    print("\nProof that K(x,x) = 1 for any x:")
    print("K(x,x) = exp(-γ||x - x||²)")
    print("       = exp(-γ||0||²)")
    print("       = exp(-γ × 0)")
    print("       = exp(0)")
    print("       = 1")
    
    print("\nVerification with examples:")
    test_vectors = [
        np.array([1, 0]),
        np.array([0, 1]), 
        np.array([2, -3]),
        np.array([-1, 5, 2])
    ]
    
    for gamma in [0.1, 1, 10]:
        print(f"\nFor γ = {gamma}:")
        for vec in test_vectors:
            _, kernel_val = rbf_kernel(vec, vec, gamma)
            print(f"  K({vec}, {vec}) = {kernel_val:.10f}")

prove_self_kernel()

# Task 3: Prove that 0 ≤ K(x,z) ≤ 1
print("\n" + "="*50)
print("TASK 3: PROOF THAT 0 ≤ K(x,z) ≤ 1")
print("="*50)

def prove_bounds():
    """
    Prove that 0 ≤ K(x,z) ≤ 1 for any x, z
    """
    print("\nProof that 0 ≤ K(x,z) ≤ 1 for any x, z:")
    print("\nK(x,z) = exp(-γ||x - z||²)")
    
    print("\nUpper bound:")
    print("Since γ > 0 and ||x - z||² ≥ 0, we have:")
    print("-γ||x - z||² ≤ 0")
    print("Therefore: exp(-γ||x - z||²) ≤ exp(0) = 1")
    print("So K(x,z) ≤ 1")
    
    print("\nLower bound:")
    print("Since the exponential function is always positive:")
    print("exp(-γ||x - z||²) > 0 for any finite ||x - z||²")
    print("So K(x,z) > 0")
    
    print("\nAs ||x - z||² → ∞:")
    print("-γ||x - z||² → -∞")
    print("exp(-γ||x - z||²) → 0")
    print("So K(x,z) → 0 (but never reaches 0 for finite distances)")
    
    print("\nTherefore: 0 ≤ K(x,z) ≤ 1")

prove_bounds()

# Task 4: Behavior as ||x - z|| → ∞
print("\n" + "="*50)
print("TASK 4: BEHAVIOR AS DISTANCE → ∞")
print("="*50)

def analyze_distance_behavior():
    """
    Analyze the behavior of K(x,z) as ||x - z|| → ∞
    """
    print("\nBehavior as ||x - z|| → ∞:")
    print("K(x,z) = exp(-γ||x - z||²)")
    print("\nAs ||x - z|| → ∞:")
    print("||x - z||² → ∞")
    print("-γ||x - z||² → -∞")
    print("exp(-γ||x - z||²) → 0")
    
    print("\nRate of decay depends on γ:")
    print("- Larger γ: faster decay to 0")
    print("- Smaller γ: slower decay to 0")
    
    # Visualization
    distances = np.linspace(0, 5, 100)
    
    plt.figure(figsize=(12, 8))
    
    for gamma in [0.1, 0.5, 1, 2, 5]:
        kernel_values = np.exp(-gamma * distances**2)
        plt.plot(distances, kernel_values, linewidth=2, label=f'γ = {gamma}')
    
    plt.xlabel('Distance ||x - z||')
    plt.ylabel('Kernel Value K(x,z)')
    plt.title('RBF Kernel Decay with Distance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 5)
    plt.ylim(0, 1.1)
    
    # Add annotations
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Maximum value')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Asymptotic limit')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rbf_distance_decay.png'), dpi=300, bbox_inches='tight')
    plt.show()

analyze_distance_behavior()

# Task 5: Infinite-dimensional feature space
print("\n" + "="*50)
print("TASK 5: INFINITE-DIMENSIONAL FEATURE SPACE")
print("="*50)

def explain_infinite_dimension():
    """
    Explain why RBF kernel corresponds to infinite-dimensional feature space
    """
    print("\nWhy RBF kernel corresponds to infinite-dimensional feature space:")
    print("\nThe RBF kernel can be expanded using Taylor series:")
    print("K(x,z) = exp(-γ||x - z||²)")
    print("        = exp(-γ||x||²) × exp(-γ||z||²) × exp(2γx^T z)")
    
    print("\nThe term exp(2γx^T z) can be expanded as:")
    print("exp(2γx^T z) = Σ(k=0 to ∞) [(2γx^T z)^k / k!]")
    print("             = Σ(k=0 to ∞) [(2γ)^k / k!] × (x^T z)^k")
    
    print("\nEach term (x^T z)^k corresponds to polynomial features of degree k")
    print("Since k goes to infinity, we have infinitely many features!")
    
    print("\nThis means the RBF kernel implicitly maps to a feature space with:")
    print("- All polynomial features of all degrees")
    print("- Infinite dimensionality")
    print("- Perfect separation capability (universal approximator)")
    
    # Demonstrate with finite approximation
    print("\nFinite approximation example:")
    x_val = 0.5
    z_val = 0.3
    gamma = 1
    
    true_kernel = np.exp(-gamma * (x_val - z_val)**2)
    
    print(f"True RBF kernel for x={x_val}, z={z_val}, γ={gamma}: {true_kernel:.6f}")
    
    # Approximate using Taylor series
    approximations = []
    for max_degree in [1, 2, 5, 10, 20]:
        approx = 0
        for k in range(max_degree + 1):
            term = ((2*gamma*x_val*z_val)**k) / math.factorial(k)
            approx += term
        
        # Include the exponential prefactors
        approx *= np.exp(-gamma*x_val**2) * np.exp(-gamma*z_val**2)
        approximations.append(approx)
        
        print(f"Approximation with degree ≤ {max_degree}: {approx:.6f}")
    
    print(f"Error decreases as we include more polynomial terms!")

explain_infinite_dimension()

# Task 6: Recommendation system design
print("\n" + "="*50)
print("TASK 6: RECOMMENDATION SYSTEM DESIGN")
print("="*50)

def design_recommendation_system():
    """
    Design a recommendation system using RBF kernel for user similarity
    """
    print("\n🎬 MOVIE RECOMMENDATION SYSTEM 🎬")
    print("="*50)
    
    # User preferences (Action, Romance ratings out of 10)
    users = {
        'User A': np.array([8, 2]),  # Loves action, dislikes romance
        'User B': np.array([2, 8]),  # Loves romance, dislikes action  
        'User C': np.array([5, 5])   # Moderate preferences
    }
    
    print("\nUser Genre Preferences (Action, Romance):")
    for name, prefs in users.items():
        print(f"{name}: Action={prefs[0]}, Romance={prefs[1]}")
    
    # Calculate similarities for different gamma values
    gamma_values = [0.01, 0.05, 0.1]
    
    print(f"\nSimilarity Analysis using K(x,z) = exp(-γ||x - z||²):")
    
    user_names = list(users.keys())
    user_vectors = list(users.values())
    
    for gamma in gamma_values:
        print(f"\nγ = {gamma}:")
        print("Similarity Matrix:")
        print("        ", end="")
        for name in user_names:
            print(f"{name:>8}", end="")
        print()
        
        similarities = np.zeros((3, 3))
        
        for i, name_i in enumerate(user_names):
            print(f"{name_i:>8}", end="")
            for j, name_j in enumerate(user_names):
                _, sim = rbf_kernel(user_vectors[i], user_vectors[j], gamma)
                similarities[i, j] = sim
                print(f"{sim:>8.3f}", end="")
            print()
    
    # Recommendation confidence system
    print(f"\n🎯 RECOMMENDATION CONFIDENCE SYSTEM:")
    print("="*40)
    
    confidence_threshold = 0.7
    print(f"Similarity threshold for recommendations: {confidence_threshold}")
    
    for gamma in gamma_values:
        print(f"\nFor γ = {gamma}:")
        
        for i, name_i in enumerate(user_names):
            print(f"\n{name_i} recommendations:")
            
            for j, name_j in enumerate(user_names):
                if i != j:
                    _, sim = rbf_kernel(user_vectors[i], user_vectors[j], gamma)
                    
                    if sim >= confidence_threshold:
                        confidence = "HIGH"
                        action = "RECOMMEND"
                    elif sim >= 0.5:
                        confidence = "MEDIUM" 
                        action = "CONSIDER"
                    else:
                        confidence = "LOW"
                        action = "AVOID"
                    
                    print(f"  {name_j}: similarity={sim:.3f}, confidence={confidence}, action={action}")
    
    # Find optimal gamma for 70% threshold
    print(f"\n🔍 OPTIMAL γ FOR 70% SIMILARITY THRESHOLD:")
    print("="*45)
    
    target_similarity = 0.7
    
    # Test different gamma values
    gamma_range = np.linspace(0.1, 3, 100)
    
    # Focus on User A and User C similarity (moderate case)
    user_a = users['User A']
    user_c = users['User C']
    
    similarities_ac = []
    for g in gamma_range:
        _, sim = rbf_kernel(user_a, user_c, g)
        similarities_ac.append(sim)
    
    # Find gamma that gives closest to 70% similarity
    similarities_ac = np.array(similarities_ac)
    closest_idx = np.argmin(np.abs(similarities_ac - target_similarity))
    optimal_gamma = gamma_range[closest_idx]
    achieved_similarity = similarities_ac[closest_idx]
    
    print(f"For User A and User C to have {target_similarity:.0%} similarity:")
    print(f"Optimal γ ≈ {optimal_gamma:.3f}")
    print(f"Achieved similarity: {achieved_similarity:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: User preferences in 2D space
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue', 'green']
    for i, (name, prefs) in enumerate(users.items()):
        plt.scatter(prefs[0], prefs[1], c=colors[i], s=200, label=name)
        plt.annotate(name, (prefs[0], prefs[1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    
    plt.xlabel('Action Rating')
    plt.ylabel('Romance Rating')
    plt.title('User Preferences')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    # Plot 2: Similarity vs gamma
    plt.subplot(1, 3, 2)
    plt.plot(gamma_range, similarities_ac, 'b-', linewidth=2, label='User A vs User C')
    plt.axhline(y=target_similarity, color='red', linestyle='--', 
                label=f'{target_similarity:.0%} threshold')
    plt.axvline(x=optimal_gamma, color='green', linestyle='--', 
                label=f'Optimal γ={optimal_gamma:.3f}')
    plt.xlabel('γ parameter')
    plt.ylabel('Similarity')
    plt.title('Similarity vs γ Parameter')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Similarity heatmap for optimal gamma
    plt.subplot(1, 3, 3)
    sim_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            _, sim = rbf_kernel(user_vectors[i], user_vectors[j], optimal_gamma)
            sim_matrix[i, j] = sim
    
    im = plt.imshow(sim_matrix, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(3), user_names)
    plt.yticks(range(3), user_names)
    plt.title(f'Similarity Matrix (γ={optimal_gamma:.3f})')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{sim_matrix[i,j]:.3f}', 
                    ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'recommendation_system.png'), dpi=300, bbox_inches='tight')
    plt.show()

design_recommendation_system()

print(f"\nPlots saved to: {save_dir}")
print("\n" + "="*60)
print("SOLUTION COMPLETE!")
print("="*60)
