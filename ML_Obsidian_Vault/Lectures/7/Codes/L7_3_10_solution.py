import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Random Forest Decision Boundary Analysis - Question 10")
print("=" * 60)

# Define the Random Forest with 3 trees
print("\nRandom Forest Configuration:")
print("Tree 1: Splits on X at x = 5, then Y at y = 3")
print("Tree 2: Splits on Y at y = 4, then X at x = 6")
print("Tree 3: Splits on X at x = 4, then Y at y = 2")

# Define the trees and their decision rules
trees = [
    {
        'name': 'Tree 1',
        'splits': [
            {'feature': 'X', 'value': 5, 'direction': 'left'},
            {'feature': 'Y', 'value': 3, 'direction': 'left'}
        ]
    },
    {
        'name': 'Tree 2',
        'splits': [
            {'feature': 'Y', 'value': 4, 'direction': 'left'},
            {'feature': 'X', 'value': 6, 'direction': 'left'}
        ]
    },
    {
        'name': 'Tree 3',
        'splits': [
            {'feature': 'X', 'value': 4, 'direction': 'left'},
            {'feature': 'Y', 'value': 2, 'direction': 'left'}
        ]
    }
]

def predict_tree(point, tree):
    """
    Predict class for a point using a single tree
    Returns: 1 for positive class, -1 for negative class
    """
    x, y = point
    
    for split in tree['splits']:
        if split['feature'] == 'X':
            if split['direction'] == 'left':
                if x <= split['value']:
                    continue
                else:
                    return -1
            else:
                if x > split['value']:
                    continue
                else:
                    return -1
        elif split['feature'] == 'Y':
            if split['direction'] == 'left':
                if y <= split['value']:
                    continue
                else:
                    return -1
            else:
                if y > split['value']:
                    continue
                else:
                    return -1
    
    return 1

def predict_ensemble(point, trees):
    """
    Predict class for a point using ensemble voting
    Returns: 1 for positive class, -1 for negative class
    """
    predictions = [predict_tree(point, tree) for tree in trees]
    return 1 if sum(predictions) > 0 else -1

def get_tree_decision_regions(tree):
    """
    Get the decision regions for a single tree
    Returns: list of regions with their class predictions
    """
    regions = []
    
    if tree['name'] == 'Tree 1':
        # Tree 1: X ≤ 5, then Y ≤ 3
        regions = [
            {'coords': [(0, 0), (5, 0), (5, 3), (0, 3)], 'class': 1, 'name': r'$X \leq 5, Y \leq 3$'},
            {'coords': [(0, 3), (5, 3), (5, 10), (0, 10)], 'class': -1, 'name': r'$X \leq 5, Y > 3$'},
            {'coords': [(5, 0), (10, 0), (10, 10), (5, 10)], 'class': -1, 'name': r'$X > 5$'}
        ]
    elif tree['name'] == 'Tree 2':
        # Tree 2: Y ≤ 4, then X ≤ 6
        regions = [
            {'coords': [(0, 0), (6, 0), (6, 4), (0, 4)], 'class': 1, 'name': r'$Y \leq 4, X \leq 6$'},
            {'coords': [(0, 4), (6, 4), (6, 10), (0, 10)], 'class': -1, 'name': r'$Y > 4$'},
            {'coords': [(6, 0), (10, 0), (10, 4), (6, 4)], 'class': -1, 'name': r'$Y \leq 4, X > 6$'}
        ]
    elif tree['name'] == 'Tree 3':
        # Tree 3: X ≤ 4, then Y ≤ 2
        regions = [
            {'coords': [(0, 0), (4, 0), (4, 2), (0, 2)], 'class': 1, 'name': r'$X \leq 4, Y \leq 2$'},
            {'coords': [(0, 2), (4, 2), (4, 10), (0, 10)], 'class': -1, 'name': r'$X \leq 4, Y > 2$'},
            {'coords': [(4, 0), (10, 0), (10, 10), (4, 10)], 'class': -1, 'name': r'$X > 4$'}
        ]
    
    return regions

def plot_individual_trees():
    """Plot decision boundaries for each individual tree"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Individual Tree Decision Boundaries', fontsize=16)
    
    colors = {1: 'lightblue', -1: 'lightpink'}
    
    for i, tree in enumerate(trees):
        ax = axes[i]
        regions = get_tree_decision_regions(tree)
        
        # Plot decision boundaries
        for region in regions:
            coords = region['coords']
            class_pred = region['class']
            name = region['name']
            
            # Create polygon for region
            polygon = Polygon(coords, facecolor=colors[class_pred], 
                           edgecolor='black', linewidth=1, alpha=0.7)
            ax.add_patch(polygon)
            
            # Add text label
            center_x = sum([p[0] for p in coords]) / len(coords)
            center_y = sum([p[1] for p in coords]) / len(coords)
            ax.text(center_x, center_y, f'{name}\nClass {class_pred}', 
                   ha='center', va='center', fontsize=10, weight='bold', usetex=True)
        
        # Add decision boundary lines
        if tree['name'] == 'Tree 1':
            ax.axvline(x=5, color='red', linestyle='--', linewidth=2, label='X = 5')
            ax.axhline(y=3, color='red', linestyle='--', linewidth=2, label='Y = 3')
        elif tree['name'] == 'Tree 2':
            ax.axhline(y=4, color='red', linestyle='--', linewidth=2, label='Y = 4')
            ax.axvline(x=6, color='red', linestyle='--', linewidth=2, label='X = 6')
        elif tree['name'] == 'Tree 3':
            ax.axvline(x=4, color='red', linestyle='--', linewidth=2, label='X = 4')
            ax.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Y = 2')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_title(f'{tree["name"]} Decision Boundaries')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'individual_tree_boundaries.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to prevent it from opening

def plot_ensemble_decision_boundary():
    """Plot the ensemble decision boundary"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create a fine grid
    x = np.linspace(0, 10, 200)
    y = np.linspace(0, 10, 200)
    X, Y = np.meshgrid(x, y)
    
    # Get predictions for each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = predict_ensemble([X[i, j], Y[i, j]], trees)
    
    # Plot the decision regions
    colors = {1: 'lightblue', -1: 'lightpink'}
    ax.contourf(X, Y, Z, levels=[-1.5, -0.5, 0.5, 1.5], 
                colors=[colors[-1], colors[-1], colors[1], colors[1]], alpha=0.7)
    
    # Add decision boundary lines from individual trees
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Tree 1: X = 5')
    ax.axhline(y=3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Tree 1: Y = 3')
    ax.axhline(y=4, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Tree 2: Y = 4')
    ax.axvline(x=6, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Tree 2: X = 6')
    ax.axvline(x=4, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Tree 3: X = 4')
    ax.axhline(y=2, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Tree 3: Y = 2')
    
    # Add legend for regions
    pos_patch = mpatches.Patch(color=colors[1], label='Class 1 (Positive)')
    neg_patch = mpatches.Patch(color=colors[-1], label='Class -1 (Negative)')
    ax.legend(handles=[pos_patch, neg_patch], loc='upper right')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_title('Ensemble Decision Boundary (Majority Voting)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ensemble_decision_boundary.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to prevent it from opening

def analyze_test_points():
    """Analyze the test points step by step"""
    test_points = [(3, 1), (7, 5)]
    
    print("\n" + "="*60)
    print("STEP-BY-STEP ANALYSIS OF TEST POINTS")
    print("="*60)
    
    for i, point in enumerate(test_points):
        x, y = point
        print(f"\nPoint {i+1}: ({x}, {y})")
        print("-" * 40)
        
        # Get predictions from each tree
        tree_predictions = []
        print("Step 1: Evaluate each tree individually")
        print("=" * 40)
        
        for j, tree in enumerate(trees):
            print(f"\n  {tree['name']} Analysis:")
            print(f"  ┌─ Starting at root node")
            
            pred = predict_tree(point, tree)
            tree_predictions.append(pred)
            
            # Detailed decision path analysis
            print(f"  ├─ Decision path:")
            for k, split in enumerate(tree['splits']):
                if split['feature'] == 'X':
                    if split['direction'] == 'left':
                        condition = f"X ≤ {split['value']}"
                        result = x <= split['value']
                        comparison = f"{x} ≤ {split['value']}"
                    else:
                        condition = f"X > {split['value']}"
                        result = x > split['value']
                        comparison = f"{x} > {split['value']}"
                else:  # Y
                    if split['direction'] == 'left':
                        condition = f"Y ≤ {split['value']}"
                        result = y <= split['value']
                        comparison = f"{y} ≤ {split['value']}"
                    else:
                        condition = f"Y > {split['value']}"
                        result = y > split['value']
                        comparison = f"{y} > {split['value']}"
                
                print(f"  │  ├─ Split {k+1}: {condition}")
                print(f"  │  │  ├─ Test: {comparison}")
                print(f"  │  │  ├─ Result: {result}")
                print(f"  │  │  └─ Action: {'Go LEFT' if result else 'Go RIGHT'}")
                
                if not result and k == 0:  # First split failed
                    print(f"  │  └─ → Leaf node reached: Class -1")
                    break
                elif k == len(tree['splits']) - 1:  # Last split
                    print(f"  │  └─ → Leaf node reached: Class 1")
                else:
                    print(f"  │  ├─ Continue to next split...")
            
            print(f"  └─ Final prediction: Class {pred}")
        
        print(f"\nStep 2: Ensemble Decision (Majority Voting)")
        print("=" * 40)
        
        # Get ensemble prediction
        ensemble_pred = predict_ensemble(point, trees)
        
        # Show voting breakdown
        positive_votes = sum(1 for p in tree_predictions if p == 1)
        negative_votes = sum(1 for p in tree_predictions if p == -1)
        
        print(f"  Tree predictions: {tree_predictions}")
        print(f"  Positive votes (Class 1): {positive_votes}")
        print(f"  Negative votes (Class -1): {negative_votes}")
        print(f"  Majority rule: Class 1 if positive votes > negative votes")
        print(f"  Calculation: {positive_votes} > {negative_votes} → {positive_votes > negative_votes}")
        print(f"  Final ensemble prediction: Class {ensemble_pred}")
        
        print(f"\nStep 3: Verification")
        print("=" * 40)
        print(f"  Point ({x}, {y}) is classified as Class {ensemble_pred}")
        print(f"  This means the majority of trees ({max(positive_votes, negative_votes)} out of 3) predict Class {ensemble_pred}")
        
        if positive_votes == negative_votes:
            print(f"  Note: In case of a tie, we default to Class -1")
        
        print(f"\n" + "─" * 60)

def calculate_agreement_regions():
    """Calculate regions where all trees agree"""
    print("\n" + "="*60)
    print("CALCULATING AGREEMENT REGIONS")
    print("="*60)
    
    print("\nStep 1: Grid Setup")
    print("-" * 40)
    grid_size = 100
    x_range = 10
    y_range = 10
    
    print(f"  Total area: {x_range} × {y_range} = {x_range * y_range} square units")
    print(f"  Grid resolution: {grid_size} × {grid_size} = {grid_size * grid_size} cells")
    print(f"  Grid cell area: ({x_range}/{grid_size}) × ({y_range}/{grid_size}) = {x_range/grid_size:.4f} × {y_range/grid_size:.4f} = {(x_range/grid_size) * (y_range/grid_size):.4f} square units")
    
    # Create a fine grid to find agreement regions
    x = np.linspace(0, x_range, grid_size)
    y = np.linspace(0, y_range, grid_size)
    X, Y = np.meshgrid(x, y)
    
    print(f"\nStep 2: Tree Prediction Analysis")
    print("-" * 40)
    print(f"  For each grid point, we evaluate all three trees")
    print(f"  We count how many points have unanimous agreement")
    
    # Get predictions from each tree
    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)
    Z3 = np.zeros_like(X)
    
    print(f"  Computing predictions for {grid_size * grid_size} points...")
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z1[i, j] = predict_tree([X[i, j], Y[i, j]], trees[0])
            Z2[i, j] = predict_tree([X[i, j], Y[i, j]], trees[1])
            Z3[i, j] = predict_tree([X[i, j], Y[i, j]], trees[2])
    
    print(f"  All tree predictions computed successfully")
    
    print(f"\nStep 3: Finding Agreement Regions")
    print("-" * 40)
    
    # Find regions where all trees agree
    agreement_positive = (Z1 == 1) & (Z2 == 1) & (Z3 == 1)
    agreement_negative = (Z1 == -1) & (Z2 == -1) & (Z3 == -1)
    
    positive_count = np.sum(agreement_positive)
    negative_count = np.sum(agreement_negative)
    
    print(f"  Points where all trees predict Class 1: {positive_count}")
    print(f"  Points where all trees predict Class -1: {negative_count}")
    print(f"  Total points with unanimous agreement: {positive_count + negative_count}")
    print(f"  Points with disagreement: {grid_size * grid_size - (positive_count + negative_count)}")
    
    print(f"\nStep 4: Area Calculation")
    print("-" * 40)
    
    # Calculate areas
    grid_area = (x_range/grid_size) * (y_range/grid_size)  # Area of each grid cell
    
    positive_area = positive_count * grid_area
    negative_area = negative_count * grid_area
    total_agreement_area = positive_area + negative_area
    
    print(f"  Grid cell area: {grid_area:.4f} square units")
    print(f"  Area where all trees predict Class 1: {positive_count} × {grid_area:.4f} = {positive_area:.2f} square units")
    print(f"  Area where all trees predict Class -1: {negative_count} × {grid_area:.4f} = {negative_area:.2f} square units")
    print(f"  Total agreement area: {positive_area:.2f} + {negative_area:.2f} = {total_agreement_area:.2f} square units")
    
    print(f"\nStep 5: Percentage Analysis")
    print("-" * 40)
    total_area = x_range * y_range
    agreement_percentage = (total_agreement_area / total_area) * 100
    disagreement_percentage = 100 - agreement_percentage
    
    print(f"  Total area: {total_area} square units")
    print(f"  Agreement percentage: ({total_agreement_area:.2f} / {total_area}) × 100 = {agreement_percentage:.1f}%")
    print(f"  Disagreement percentage: 100% - {agreement_percentage:.1f}% = {disagreement_percentage:.1f}%")
    
    print(f"\nStep 6: Interpretation")
    print("-" * 40)
    print(f"  High agreement ({agreement_percentage:.1f}%) indicates:")
    print(f"    • Trees make similar predictions for most of the feature space")
    print(f"    • Ensemble is confident in its predictions for these regions")
    print(f"    • Random Forest is stable and robust")
    print(f"  Disagreement regions ({disagreement_percentage:.1f}%) indicate:")
    print(f"    • Trees have different opinions about these areas")
    print(f"    • Ensemble predictions may be less certain")
    print(f"    • These are often near decision boundaries")
    
    # Plot agreement regions
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot agreement regions
    ax.contourf(X, Y, agreement_positive, levels=[0.5, 1.5], colors=['lightgreen'], alpha=0.7, label='All trees predict Class 1')
    ax.contourf(X, Y, agreement_negative, levels=[-1.5, -0.5], colors=['lightcoral'], alpha=0.7, label='All trees predict Class -1')
    
    # Add decision boundary lines
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Tree 1: X = 5')
    ax.axhline(y=3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Tree 1: Y = 3')
    ax.axhline(y=4, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Tree 2: Y = 4')
    ax.axvline(x=6, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Tree 2: X = 6')
    ax.axvline(x=4, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Tree 3: X = 4')
    ax.axhline(y=2, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Tree 3: Y = 2')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_title('Regions Where All Trees Agree')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'agreement_regions.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to prevent it from opening
    
    return positive_area, negative_area, total_agreement_area

def derive_decision_boundaries():
    """Derive decision boundaries mathematically"""
    print("\n" + "="*60)
    print("MATHEMATICAL DERIVATION OF DECISION BOUNDARIES")
    print("="*60)
    
    print("\nStep 1: Understanding Tree Structure")
    print("-" * 40)
    print("Each tree follows a hierarchical decision structure:")
    print("Root → Split 1 → Split 2 → Leaf Node")
    print("Each split creates a binary decision: go LEFT or RIGHT")
    
    print("\nStep 2: Mathematical Formulation")
    print("-" * 40)
    
    for i, tree in enumerate(trees):
        print(f"\n{tree['name']} Decision Function:")
        print(f"  Let f_{i+1}(x, y) be the decision function for {tree['name']}")
        
        if tree['name'] == 'Tree 1':
            print(f"  f_1(x, y) = 1 if (x ≤ 5) AND (y ≤ 3)")
            print(f"  f_1(x, y) = -1 otherwise")
            print(f"  Decision boundary: x = 5 OR y = 3")
            print(f"  Positive region: {{(x, y) | x ≤ 5 AND y ≤ 3}}")
            
        elif tree['name'] == 'Tree 2':
            print(f"  f_2(x, y) = 1 if (y ≤ 4) AND (x ≤ 6)")
            print(f"  f_2(x, y) = -1 otherwise")
            print(f"  Decision boundary: y = 4 OR x = 6")
            print(f"  Positive region: {{(x, y) | y ≤ 4 AND x ≤ 6}}")
            
        elif tree['name'] == 'Tree 3':
            print(f"  f_3(x, y) = 1 if (x ≤ 4) AND (y ≤ 2)")
            print(f"  f_3(x, y) = -1 otherwise")
            print(f"  Decision boundary: x = 4 OR y = 2")
            print(f"  Positive region: {{(x, y) | x ≤ 4 AND y ≤ 2}}")
    
    print(f"\nStep 3: Ensemble Decision Function")
    print("-" * 40)
    print(f"  Ensemble function: F(x, y) = sign(Σᵢ fᵢ(x, y))")
    print(f"  Where sign(z) = 1 if z > 0, -1 if z ≤ 0")
    print(f"  For 3 trees: F(x, y) = sign(f₁(x, y) + f₂(x, y) + f₃(x, y))")
    print(f"  Decision rule: Class 1 if majority of trees predict 1")
    print(f"  Decision rule: Class -1 if majority of trees predict -1")
    
    print(f"\nStep 4: Boundary Analysis")
    print("-" * 40)
    print(f"  Individual boundaries are simple: x = c or y = c")
    print(f"  Ensemble boundary is complex due to majority voting")
    print(f"  Points near individual boundaries may have uncertain predictions")
    print(f"  Points far from all boundaries have confident predictions")

def main():
    """Main function to run the complete analysis"""
    print("Starting Random Forest Decision Boundary Analysis...")
    
    # Step 1: Mathematical derivation
    print("\nStep 1: Mathematical derivation of decision boundaries...")
    derive_decision_boundaries()
    
    # Step 2: Plot individual tree boundaries
    print("\nStep 2: Plotting individual tree decision boundaries...")
    plot_individual_trees()
    
    # Step 3: Plot ensemble decision boundary
    print("\nStep 3: Plotting ensemble decision boundary...")
    plot_ensemble_decision_boundary()
    
    # Step 4: Analyze test points
    print("\nStep 4: Analyzing test points...")
    analyze_test_points()
    
    # Step 5: Calculate agreement regions
    print("\nStep 5: Calculating agreement regions...")
    positive_area, negative_area, total_agreement = calculate_agreement_regions()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print("1. Mathematical derivation of decision boundaries completed")
    print("2. Individual tree decision boundaries have been plotted")
    print("3. Ensemble decision boundary shows the combined effect of majority voting")
    print("4. Test points have been analyzed step-by-step")
    print("5. Agreement regions have been calculated and visualized")
    print(f"6. Total area where all trees agree: {total_agreement:.2f} square units")
    
    print(f"\nAll plots have been saved to: {save_dir}")

if __name__ == "__main__":
    main()
