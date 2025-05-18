import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from collections import Counter
import pandas as pd
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_6_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots and enable LaTeX support
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'

# Step 1: Calculate the number of binary classifiers for n classes in OVO
def calculate_ovo_classifiers(n_classes):
    """Calculate the number of binary classifiers needed for OVO"""
    return int(n_classes * (n_classes - 1) / 2)

# Step 2: Generate all pairs of classes for OVO
def generate_class_pairs(classes):
    """Generate all pairs of classes for OVO"""
    return list(itertools.combinations(classes, 2))

# Step 3: Implement voting scheme for OVO
def ovo_voting(pairwise_results, classes):
    """
    Implement voting scheme for OVO
    
    Args:
        pairwise_results: Dictionary with class pairs as keys and winning class as values
        classes: List of all classes
        
    Returns:
        Dictionary with vote counts for each class
    """
    votes = {cls: 0 for cls in classes}
    
    for pair, winner in pairwise_results.items():
        votes[winner] += 1
    
    return votes

# Step 4: Visualize voting results
def visualize_voting(votes, save_path):
    """Visualize voting results as a bar chart"""
    plt.figure(figsize=(10, 6))
    classes = list(votes.keys())
    vote_counts = list(votes.values())
    
    plt.bar(classes, vote_counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Votes')
    plt.title('OVO Voting Results')
    plt.xticks(range(len(classes)), classes)
    
    # Add value labels on top of each bar
    for i, v in enumerate(vote_counts):
        plt.text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Step 5: Demonstrate scaling issue with many classes
def demonstrate_scaling(max_classes=20, save_path=None):
    """Demonstrate how OVO and OVA scale with number of classes"""
    n_classes_range = range(2, max_classes + 1)
    ovo_classifiers = [calculate_ovo_classifiers(n) for n in n_classes_range]
    ova_classifiers = list(n_classes_range)  # OVA requires n classifiers
    
    plt.figure(figsize=(12, 7))
    plt.plot(n_classes_range, ovo_classifiers, 'o-', label='OVO', linewidth=2, markersize=8)
    plt.plot(n_classes_range, ova_classifiers, 's-', label='OVA', linewidth=2, markersize=8)
    plt.xlabel('Number of Classes')
    plt.ylabel('Number of Binary Classifiers')
    plt.title('Scaling of OVO vs OVA with Number of Classes')
    plt.grid(True)
    plt.legend()
    
    # Add annotations for specific points
    for n in [4, 10, 20]:
        if n <= max_classes:
            ovo = calculate_ovo_classifiers(n)
            ova = n
            plt.annotate(f'({n}, {ovo})', 
                        xy=(n, ovo), 
                        xytext=(n+0.5, ovo+5), 
                        arrowprops=dict(arrowstyle='->'))
            plt.annotate(f'({n}, {ova})', 
                        xy=(n, ova), 
                        xytext=(n+0.5, ova+2), 
                        arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Step 6: Class imbalance comparison between OVO and OVA
def class_imbalance_simulation(n_samples_per_class, save_path=None):
    """
    Simulate class imbalance effects on OVO vs OVA
    
    Args:
        n_samples_per_class: Dictionary with classes as keys and sample counts as values
        save_path: Path to save the visualization
    """
    classes = list(n_samples_per_class.keys())
    n_classes = len(classes)
    
    # Calculate effective training samples for OVA
    ova_training_samples = {}
    for cls in classes:
        # For OVA, a classifier for class C uses all samples
        # Class C samples are positive, all others are negative
        ova_training_samples[cls] = sum(n_samples_per_class.values())
    
    # Calculate effective training samples for OVO
    ovo_training_samples = {}
    for cls in classes:
        # For OVO, we only use samples from the two classes being compared
        # Each class C is used in (n_classes - 1) binary classifiers
        samples_used = 0
        for other_cls in classes:
            if other_cls != cls:
                samples_used += n_samples_per_class[cls] + n_samples_per_class[other_cls]
        ovo_training_samples[cls] = samples_used
    
    # Calculate class imbalance ratio for OVA
    ova_imbalance = {}
    for cls in classes:
        # Ratio of negative to positive examples for each binary classifier
        positive_samples = n_samples_per_class[cls]
        negative_samples = sum(n_samples_per_class.values()) - positive_samples
        ova_imbalance[cls] = negative_samples / positive_samples if positive_samples > 0 else float('inf')
    
    # Calculate average class imbalance ratio for OVO
    ovo_imbalance = {}
    for cls in classes:
        imbalance_ratios = []
        for other_cls in classes:
            if other_cls != cls:
                ratio = n_samples_per_class[other_cls] / n_samples_per_class[cls] if n_samples_per_class[cls] > 0 else float('inf')
                imbalance_ratios.append(ratio)
        ovo_imbalance[cls] = np.mean(imbalance_ratios) if imbalance_ratios else 0
    
    # Create a dataframe for easier visualization
    data = {
        'Class': classes * 2,
        'Approach': ['OVA'] * n_classes + ['OVO'] * n_classes,
        'Imbalance Ratio': list(ova_imbalance.values()) + list(ovo_imbalance.values())
    }
    df = pd.DataFrame(data)
    
    # Visualization
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Class', y='Imbalance Ratio', hue='Approach', data=df, palette='muted')
    plt.title('Class Imbalance Comparison: OVA vs OVO')
    plt.xlabel('Class')
    plt.ylabel('Imbalance Ratio (higher is more imbalanced)')
    plt.grid(True, axis='y')
    plt.legend(title='')
    
    # Add text to explain what the ratios mean
    plt.figtext(0.5, 0.01, 
                'Lower imbalance ratio is better.\nOVA imbalance ratio = negative/positive samples\n'
                'OVO imbalance ratio = average ratio of competing class samples to this class samples',
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Now use the functions to answer the specific questions

# Q1: How many binary classifiers for OVO with 4 classes?
classes = ['A', 'B', 'C', 'D']
n_classes = len(classes)
n_classifiers = calculate_ovo_classifiers(n_classes)
class_pairs = generate_class_pairs(classes)

# Print out Q1 solution
print(f"Q1: Number of binary classifiers needed for OVO with {n_classes} classes: {n_classifiers}")
print("All class pairs:", class_pairs)

# Q2: Voting scheme with given predictions
# Given prediction results
pairwise_results = {
    ('A', 'B'): 'A',
    ('A', 'C'): 'C',
    ('A', 'D'): 'A',
    ('B', 'C'): 'C',
    ('B', 'D'): 'B',
    ('C', 'D'): 'C'
}

# Count votes from pairwise predictions
votes = ovo_voting(pairwise_results, classes)
print("\nQ2: Voting results:")
for cls, vote_count in votes.items():
    print(f"Class {cls}: {vote_count} votes")

winner = max(votes, key=votes.get)
print(f"The predicted class is: {winner}")

# Visualize the voting results
visualize_voting(votes, os.path.join(save_dir, 'ovo_voting_results.png'))

# Q3: Potential issues with OVO voting scheme for large number of classes
# Demonstrate scaling of OVO vs OVA
demonstrate_scaling(max_classes=20, save_path=os.path.join(save_dir, 'ovo_ova_scaling.png'))

# Q4: Class imbalance comparison
# Simulate a dataset with class imbalance
imbalanced_dataset = {
    'A': 1000,  # Majority class
    'B': 500,
    'C': 200,
    'D': 50     # Minority class
}

# Visualize how OVO and OVA handle class imbalance
class_imbalance_simulation(imbalanced_dataset, save_path=os.path.join(save_dir, 'ovo_ova_imbalance.png'))

print("\nQ3: Potential issue with OVO voting:")
print("OVO requires quadratically more classifiers as the number of classes increases,")
print(f"which can be computationally expensive for many classes (e.g., {calculate_ovo_classifiers(100)} classifiers for 100 classes).")

print("\nQ4: OVO vs OVA for class imbalance:")
print("OVO typically handles class imbalance better than OVA because each classifier")
print("only uses data from two classes at a time, leading to more balanced training sets.")
print("With OVA, a classifier for a minority class uses all samples from other classes as negatives,")
print("which results in a highly imbalanced training set.")

print("\nAll visualizations saved to:", save_dir) 