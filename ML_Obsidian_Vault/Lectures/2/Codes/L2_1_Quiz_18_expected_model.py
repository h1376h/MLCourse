import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
import matplotlib as mpl

# Set global matplotlib style for prettier plots
plt.style.use('seaborn-v0_8-pastel')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['figure.figsize'] = (10, 6)

def calculate_model_performance(p_class_a=0.6, acc_a=0.85, acc_b=0.75, var_a=0.02, var_b=0.02):
    """
    Calculate various metrics for model performance using Laws of Total Expectation and Variance.
    
    Args:
        p_class_a: Proportion of samples from class A
        acc_a: Accuracy on class A
        acc_b: Accuracy on class B
        var_a: Variance of accuracy within class A
        var_b: Variance of accuracy within class B
    
    Returns:
        Dictionary with calculated metrics
    """
    # Derived values
    p_class_b = 1 - p_class_a
    
    # Task 1: Overall expected accuracy using Law of Total Expectation
    # E[Accuracy] = P(Class A) * E[Accuracy|Class A] + P(Class B) * E[Accuracy|Class B]
    overall_accuracy = p_class_a * acc_a + p_class_b * acc_b
    
    # Task 2: Probability that a sample was from class A given prediction is correct
    # Using Bayes' theorem: P(A|correct) = P(correct|A) * P(A) / P(correct)
    # P(correct) = overall_accuracy
    p_class_a_given_correct = (acc_a * p_class_a) / overall_accuracy
    
    # Task 3: Variance of model's accuracy using Law of Total Variance
    # Var(Accuracy) = E[Var(Accuracy|Class)] + Var(E[Accuracy|Class])
    # First term: P(A) * Var(Accuracy|A) + P(B) * Var(Accuracy|B)
    # Second term: P(A) * (E[Accuracy|A] - E[Accuracy])^2 + P(B) * (E[Accuracy|B] - E[Accuracy])^2
    expected_conditional_variance = p_class_a * var_a + p_class_b * var_b
    variance_of_conditional_expectation = (
        p_class_a * (acc_a - overall_accuracy)**2 + 
        p_class_b * (acc_b - overall_accuracy)**2
    )
    total_variance = expected_conditional_variance + variance_of_conditional_expectation
    
    # Task 4: New expected accuracy with balanced classes (p_class_a = 0.5)
    balanced_accuracy = 0.5 * acc_a + 0.5 * acc_b
    
    # Return all calculated metrics
    return {
        'p_class_a': p_class_a,
        'p_class_b': p_class_b,
        'accuracy_a': acc_a,
        'accuracy_b': acc_b,
        'variance_a': var_a,
        'variance_b': var_b,
        'overall_accuracy': overall_accuracy,
        'p_class_a_given_correct': p_class_a_given_correct,
        'expected_conditional_variance': expected_conditional_variance,
        'variance_of_conditional_expectation': variance_of_conditional_expectation,
        'total_variance': total_variance,
        'balanced_accuracy': balanced_accuracy
    }

def create_accuracy_by_class_visualization(metrics, save_path=None):
    """
    Create a visualization of accuracy by class with class distribution.
    """
    plt.figure(figsize=(9, 5))
    
    # Create a stacked bar for class distribution
    ax1 = plt.subplot(1, 2, 1)
    class_sizes = [metrics['p_class_a'], metrics['p_class_b']]
    labels = ['Class A', 'Class B']
    ax1.bar(0, class_sizes, color=['#3498DB', '#E74C3C'], width=0.5)
    
    # Add percentage labels
    for i, (size, label) in enumerate(zip(class_sizes, labels)):
        bottom = sum(class_sizes[:i])
        ax1.text(0, bottom + size/2, f"{label}\n{size:.0%}", 
                ha='center', va='center', color='white', fontweight='bold')
    
    ax1.set_title('Class Distribution')
    ax1.set_ylabel('Proportion')
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks([])
    
    # Create bar chart for accuracy by class
    ax2 = plt.subplot(1, 2, 2)
    accuracies = [metrics['accuracy_a'], metrics['accuracy_b'], metrics['overall_accuracy']]
    x_positions = [0, 1, 2]
    labels = ['Class A', 'Class B', 'Overall']
    colors = ['#3498DB', '#E74C3C', '#2ECC71']
    
    bars = ax2.bar(x_positions, accuracies, color=colors, width=0.6)
    
    # Add accuracy values on bars
    for bar, accuracy in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{accuracy:.2f}", ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('Accuracy by Class')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.0)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy by class visualization saved to {save_path}")
        print(f"Class A: {metrics['p_class_a']:.1%}, accuracy: {metrics['accuracy_a']:.2f}")
        print(f"Class B: {metrics['p_class_b']:.1%}, accuracy: {metrics['accuracy_b']:.2f}")
        print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    
    plt.close()

def create_bayes_theorem_visualization(metrics, save_path=None):
    """
    Create a visualization demonstrating Bayes' theorem for class given correctness.
    """
    # Calculate joint probabilities
    p_a_correct = metrics['p_class_a'] * metrics['accuracy_a']
    p_a_incorrect = metrics['p_class_a'] * (1 - metrics['accuracy_a'])
    p_b_correct = metrics['p_class_b'] * metrics['accuracy_b']
    p_b_incorrect = metrics['p_class_b'] * (1 - metrics['accuracy_b'])
    
    plt.figure(figsize=(8, 6))
    
    # Create pie chart with better wedge styling
    sizes = [p_a_correct, p_a_incorrect, p_b_correct, p_b_incorrect]
    labels = ['A, Correct', 'A, Incorrect', 'B, Correct', 'B, Incorrect']
    colors = ['#3498DB', '#93c5fd', '#EF4444', '#fca5a5']
    
    wedges, texts, autotexts = plt.pie(
        sizes, 
        labels=None,
        colors=colors, 
        autopct='%1.1f%%',
        pctdistance=0.7,
        startangle=90, 
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    # Add a circle to create a donut chart
    centre_circle = plt.Circle((0,0), 0.5, fc='white', ec='lightgray')
    plt.gca().add_artist(centre_circle)
    
    # Add custom labels with better positioning
    plt.legend(
        wedges, 
        labels,
        title="Joint Probabilities",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    # Add the conditional probability in the center with improved visibility
    plt.text(0, 0, f"{metrics['p_class_a_given_correct']:.4f}", ha='center', va='center', 
            fontsize=22, fontweight='bold', color='#1d4ed8',
            bbox=dict(boxstyle="circle", fc="white", ec="lightgray", alpha=0.9, pad=0.5))
    
    plt.title('Joint Probabilities')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bayes' theorem visualization saved to {save_path}")
        print(f"P(Class A | Correct) = P(Correct|A) * P(A) / P(Correct)")
        print(f"P(Class A | Correct) = {metrics['accuracy_a']} * {metrics['p_class_a']} / {metrics['overall_accuracy']}")
        print(f"P(Class A | Correct) = {p_a_correct:.4f} / {p_a_correct + p_b_correct:.4f} = {metrics['p_class_a_given_correct']:.6f}")
    
    plt.close()

def create_confusion_matrix_visualization(metrics, save_path=None):
    """
    Create a visualization of the probabilistic confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    
    # Calculate joint probabilities
    p_a_correct = metrics['p_class_a'] * metrics['accuracy_a']
    p_a_incorrect = metrics['p_class_a'] * (1 - metrics['accuracy_a'])
    p_b_correct = metrics['p_class_b'] * metrics['accuracy_b']
    p_b_incorrect = metrics['p_class_b'] * (1 - metrics['accuracy_b'])
    
    # Calculate precision and recall
    precision_a = p_a_correct / (p_a_correct + p_b_incorrect)
    recall_a = p_a_correct / (p_a_correct + p_a_incorrect)
    f1_a = 2 * precision_a * recall_a / (precision_a + recall_a)
    
    precision_b = p_b_correct / (p_b_correct + p_a_incorrect)
    recall_b = p_b_correct / (p_b_correct + p_b_incorrect)
    f1_b = 2 * precision_b * recall_b / (precision_b + recall_b)
    
    # Create confusion matrix
    cm = np.array([
        [p_a_correct, p_a_incorrect],
        [p_b_incorrect, p_b_correct]
    ])
    
    # Plot confusion matrix
    plt.subplot(1, 1, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Probabilistic Confusion Matrix')
    plt.colorbar()
    
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Predict A', 'Predict B'])
    plt.yticks(tick_marks, ['True A', 'True B'])
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix visualization saved to {save_path}")
        print(f"Joint probabilities:")
        print(f"  P(A, Correct) = {p_a_correct:.4f}")
        print(f"  P(A, Incorrect) = {p_a_incorrect:.4f}")
        print(f"  P(B, Correct) = {p_b_correct:.4f}")
        print(f"  P(B, Incorrect) = {p_b_incorrect:.4f}")
        print(f"Performance metrics:")
        print(f"  Precision A: {precision_a:.4f}")
        print(f"  Recall A: {recall_a:.4f}")
        print(f"  F1 Score A: {f1_a:.4f}")
        print(f"  Precision B: {precision_b:.4f}")
        print(f"  Recall B: {recall_b:.4f}")
        print(f"  F1 Score B: {f1_b:.4f}")
    
    plt.close()

def create_total_variance_visualization(metrics, save_path=None):
    """
    Create a visualization demonstrating the Law of Total Variance.
    """
    plt.figure(figsize=(9, 5))
    
    # Extract values
    ecv = metrics['expected_conditional_variance']
    voce = metrics['variance_of_conditional_expectation']
    total = metrics['total_variance']
    
    # Create stacked bar chart
    components = ['E[Var(Acc|Class)]', 'Var(E[Acc|Class])', 'Total Variance']
    values = [ecv, voce, total]
    colors = ['#3498DB', '#E74C3C', '#2ECC71']
    
    # Create the bar plot
    bars = plt.bar([0, 1, 2], values, color=colors, width=0.6)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.0005,
               f"{value:.4f}", ha='center', va='bottom', fontweight='bold')
    
    # Add styling
    plt.title('Components of Variance')
    plt.ylabel('Variance')
    plt.xticks([0, 1, 2], components)
    plt.ylim(0, max(values) * 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Total variance visualization saved to {save_path}")
        print(f"Variance components:")
        print(f"  Expected conditional variance: {ecv:.6f}")
        print(f"  Variance of conditional expectation: {voce:.6f}")
        print(f"  Total variance: {total:.6f}")
        print(f"  Formula: Total Variance = E[Var(Acc|Class)] + Var(E[Acc|Class]) = {ecv:.4f} + {voce:.4f} = {total:.4f}")
    
    plt.close()

def create_class_balance_effect_visualization(metrics, save_path=None):
    """
    Create a visualization showing how class balance affects expected accuracy.
    """
    plt.figure(figsize=(9, 5))
    
    # Extract values
    acc_a = metrics['accuracy_a']
    acc_b = metrics['accuracy_b']
    orig_p_a = metrics['p_class_a']
    
    # Generate a range of class A proportions
    p_a_values = np.linspace(0, 1, 100)
    expected_accuracies = [p_a * acc_a + (1 - p_a) * acc_b for p_a in p_a_values]
    
    # Create the plot
    plt.plot(p_a_values, expected_accuracies, color='#3498DB', linewidth=2.5)
    
    # Mark key points
    # Original class distribution
    orig_acc = metrics['overall_accuracy']
    plt.scatter([orig_p_a], [orig_acc], color='#E74C3C', s=100, zorder=3)
    
    # Balanced classes
    balanced_acc = metrics['balanced_accuracy']
    plt.scatter([0.5], [balanced_acc], color='#2ECC71', s=100, zorder=3)
    
    # Add styling
    plt.title('Effect of Class Balance on Accuracy')
    plt.xlabel('Proportion of Class A')
    plt.ylabel('Expected Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='#3498DB', label='Accuracy'),
        Patch(facecolor='#E74C3C', label='Original'),
        Patch(facecolor='#2ECC71', label='Balanced')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class balance effect visualization saved to {save_path}")
        print(f"Original class distribution: Class A: {orig_p_a:.2f}, Accuracy: {orig_acc:.4f}")
        print(f"Balanced class distribution: Class A: 0.50, Accuracy: {balanced_acc:.4f}")
        print(f"Accuracy difference: {orig_acc - balanced_acc:.4f}")
        print(f"Class A accuracy: {acc_a:.2f}, Class B accuracy: {acc_b:.2f}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 18 of the L2.1 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_18")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 18 of the L2.1 Probability quiz: Expected Model Performance...")
    
    # Calculate metrics
    metrics = calculate_model_performance(p_class_a=0.6, acc_a=0.85, acc_b=0.75, var_a=0.02, var_b=0.02)
    
    # Print results
    print("\nModel Performance Metrics:")
    print(f"Class distribution: Class A: {metrics['p_class_a']:.2f}, Class B: {metrics['p_class_b']:.2f}")
    print(f"Accuracy by class: Class A: {metrics['accuracy_a']:.4f}, Class B: {metrics['accuracy_b']:.4f}")
    print(f"Task 1: Overall expected accuracy: {metrics['overall_accuracy']:.6f}")
    print(f"Task 2: P(Class A | Correct prediction): {metrics['p_class_a_given_correct']:.6f}")
    print(f"Task 3: Variance components:")
    print(f"  - Expected conditional variance: {metrics['expected_conditional_variance']:.6f}")
    print(f"  - Variance of conditional expectation: {metrics['variance_of_conditional_expectation']:.6f}")
    print(f"  - Total variance: {metrics['total_variance']:.6f}")
    print(f"Task 4: Expected accuracy with balanced classes: {metrics['balanced_accuracy']:.6f}")
    
    # Generate visualizations
    create_accuracy_by_class_visualization(metrics, save_path=os.path.join(save_dir, "accuracy_by_class.png"))
    print("1. Accuracy by class visualization created")
    
    create_bayes_theorem_visualization(metrics, save_path=os.path.join(save_dir, "bayes_theorem.png"))
    print("2. Bayes' theorem visualization created")
    
    create_confusion_matrix_visualization(metrics, save_path=os.path.join(save_dir, "confusion_matrix.png"))
    print("3. Confusion matrix visualization created")
    
    create_total_variance_visualization(metrics, save_path=os.path.join(save_dir, "total_variance.png"))
    print("4. Total variance visualization created")
    
    create_class_balance_effect_visualization(metrics, save_path=os.path.join(save_dir, "class_balance_effect.png"))
    print("5. Class balance effect visualization created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 