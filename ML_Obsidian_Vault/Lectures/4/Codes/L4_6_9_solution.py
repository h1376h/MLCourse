import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_6_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Set up the plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    # Create table comparing multi-class classification strategies
    table_data = {
        'Strategy': [
            'One-vs-All (OVA)', 
            'One-vs-One (OVO)', 
            'Error-Correcting Output Codes (ECOC)', 
            'Direct Multi-class (Softmax)'
        ],
        'Handles Large # of Classes': ['No', 'No', 'Yes', 'Yes'],
        'Provides Probability Estimates': ['Yes', 'Yes', 'No', 'Yes'],
        'Training Efficiency': ['Medium', 'Low', 'Low', 'High'],
        'Prediction Efficiency': ['Medium', 'Low', 'Medium', 'High'],
        'Robust to Class Imbalance': ['Medium', 'High', 'High', 'Low']
    }
    
    summary_df = pd.DataFrame(table_data)
    print("\nSummary of Multi-class Classification Strategies:")
    print(summary_df.to_markdown(index=False))
    
    # Generate plots illustrating key concepts
    
    # 1. Number of binary classifiers needed for each strategy
    plt.figure(figsize=(10, 6))
    
    n_classes = np.arange(2, 101)
    ova_classifiers = n_classes
    ovo_classifiers = n_classes * (n_classes - 1) / 2
    ecoc_classifiers = np.ceil(np.log2(n_classes)) * 10  # Approximation of ECOC classifier count
    direct_classifiers = np.ones_like(n_classes)
    
    plt.plot(n_classes, ova_classifiers, label='One-vs-All (OVA)', linewidth=2, color='#1f77b4')
    plt.plot(n_classes, ovo_classifiers, label='One-vs-One (OVO)', linewidth=2, color='#ff7f0e')
    plt.plot(n_classes, ecoc_classifiers, label='Error-Correcting Output Codes (ECOC)', linewidth=2, color='#2ca02c')
    plt.plot(n_classes, direct_classifiers, label='Direct Multi-class (Softmax)', linewidth=2, color='#d62728')
    
    # Highlight specific class counts
    for n in [3, 10, 50, 100]:
        plt.axvline(x=n, color='gray', linestyle='--', alpha=0.3)
    
    # Annotation for key points
    plt.annotate(f'OVO (n=100): {int(100*99/2)} classifiers', xy=(100, 100*99/2), 
                xytext=(70, 4500), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.annotate(f'OVA (n=100): 100 classifiers', xy=(100, 100), 
                xytext=(70, 1500), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.xlabel('Number of Classes (n)')
    plt.ylabel('Number of Binary Classifiers')
    plt.title('Number of Binary Classifiers Required vs Number of Classes')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.ylim(0, 5000)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'classifiers_vs_classes.png'), dpi=300)
    
    # 2. Training time complexity visualization
    plt.figure(figsize=(10, 6))
    
    # Approximate training time complexity (arbitrary units)
    # Assume N samples with n classes, with each binary classifier taking O(N) time
    samples = 1000
    ova_time = ova_classifiers * samples  # n * N
    ovo_time = ovo_classifiers * (samples / n_classes * 2)  # n(n-1)/2 * 2N/n = N(n-1)
    ecoc_time = ecoc_classifiers * samples  # log(n) * 10 * N
    direct_time = samples * np.power(n_classes, 0.5) * 5  # N * sqrt(n) * constant
    
    plt.plot(n_classes, ova_time / 1000, label='One-vs-All (OVA)', linewidth=2, color='#1f77b4')
    plt.plot(n_classes, ovo_time / 1000, label='One-vs-One (OVO)', linewidth=2, color='#ff7f0e')
    plt.plot(n_classes, ecoc_time / 1000, label='Error-Correcting Output Codes (ECOC)', linewidth=2, color='#2ca02c')
    plt.plot(n_classes, direct_time / 1000, label='Direct Multi-class (Softmax)', linewidth=2, color='#d62728')
    
    # Add shaded regions to indicate efficiency zones
    plt.fill_between(n_classes, 0, 10, alpha=0.2, color='green', label='High Efficiency')
    plt.fill_between(n_classes, 10, 30, alpha=0.2, color='yellow', label='Medium Efficiency')
    plt.fill_between(n_classes, 30, 120, alpha=0.2, color='orange', label='Low Efficiency')
    plt.fill_between(n_classes, 120, 5000, alpha=0.2, color='red')
    
    plt.xlabel('Number of Classes (n)')
    plt.ylabel('Relative Training Time (Arbitrary Units)')
    plt.title('Approximate Training Time Complexity vs Number of Classes')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.ylim(0, 120)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_time_complexity.png'), dpi=300)
    
    # 3. Class imbalance visualization
    plt.figure(figsize=(10, 6))
    
    # Create a sample class distribution with imbalance
    class_sizes = np.array([100, 30, 20, 10, 5])
    class_labels = [f'Class {i+1}' for i in range(len(class_sizes))]
    
    # Calculate accuracy impact (simulated values based on strategy robustness)
    # Higher values mean more robust to imbalance (less accuracy drop)
    imbalance_impact = {
        'Balanced': [0.95, 0.95, 0.95, 0.95, 0.95],
        'OVA': [0.92, 0.85, 0.80, 0.75, 0.70],
        'OVO': [0.94, 0.90, 0.88, 0.85, 0.83],
        'ECOC': [0.93, 0.90, 0.87, 0.84, 0.82],
        'Direct': [0.90, 0.78, 0.70, 0.60, 0.50]
    }
    
    x = np.arange(len(class_labels))
    width = 0.15
    offsets = [-2, -1, 0, 1, 2]
    
    colors = ['#333333', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (strategy, accuracies) in enumerate(imbalance_impact.items()):
        plt.bar(x + offsets[i]*width, accuracies, width, label=strategy, color=colors[i], alpha=0.8)
    
    plt.xlabel('Class')
    plt.ylabel('Classification Accuracy')
    plt.title('Strategy Performance on Imbalanced Classes')
    plt.xticks(x, class_labels)
    plt.ylim(0.4, 1.0)
    
    # Add text annotations showing class sizes
    for i, size in enumerate(class_sizes):
        plt.text(i, 0.45, f'n={size}', ha='center', fontweight='bold')
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_imbalance_performance.png'), dpi=300)
    
    # 4. Computational complexity visualization
    plt.figure(figsize=(10, 8))
    
    strategies = ['OVA', 'OVO', 'ECOC', 'Direct\nMulti-class']
    n_values = [3, 10, 100]
    
    # Training complexity - rows are strategies, columns are n values
    # Higher value means more complex/slower
    train_complexity = np.array([
        [3, 10, 100],       # OVA - O(n)
        [3, 45, 4950],      # OVO - O(n²)
        [6, 10, 70],        # ECOC - O(log(n))
        [2, 3, 10]          # Direct - O(1) but with class factor
    ])
    
    # Normalize for visualization
    train_complexity = train_complexity / np.max(train_complexity) * 100
    
    # Prediction complexity (similar pattern but can differ)
    predict_complexity = np.array([
        [3, 10, 100],      # OVA - O(n) 
        [3, 45, 4950],     # OVO - O(n²)
        [6, 10, 70],       # ECOC - O(log(n))
        [1, 1, 1]          # Direct - O(1)
    ])
    predict_complexity = predict_complexity / np.max(predict_complexity) * 100
    
    # Plot as a heatmap with both train and predict complexity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training complexity heatmap
    sns.heatmap(train_complexity, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax1,
                xticklabels=[f'n={n}' for n in n_values],
                yticklabels=strategies)
    ax1.set_title('Training Complexity (Relative)')
    
    # Prediction complexity heatmap
    sns.heatmap(predict_complexity, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax2,
                xticklabels=[f'n={n}' for n in n_values],
                yticklabels=strategies)
    ax2.set_title('Prediction Complexity (Relative)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'computational_complexity.png'), dpi=300)
    
    # 5. Probability estimation capability visualization
    plt.figure(figsize=(10, 6))
    
    # Simple visual showing which methods naturally provide probabilities
    strategies = ['OVA', 'OVO', 'ECOC', 'Direct Multi-class']
    prob_capable = [1, 0.7, 0, 1]  # 1 = Yes, 0 = No, 0.7 = Partial/indirect
    
    plt.bar(strategies, prob_capable, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    plt.ylim(0, 1.2)
    plt.axhline(y=0.5, color='gray', linestyle='--')
    
    # Add labels on bars
    for i, val in enumerate(prob_capable):
        label = "Yes" if val == 1 else "No" if val == 0 else "Partial"
        plt.text(i, val + 0.05, label, ha='center', fontweight='bold')
    
    plt.xlabel('Strategy')
    plt.ylabel('Probability Estimation Capability')
    plt.title('Native Probability Estimation Capability by Strategy')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'probability_estimation.png'), dpi=300)
    
    # Creating tables for Quiz answers
    # Define the quiz table data
    quiz_table = {
        'Property': [
            'Handles large number of classes efficiently',
            'Provides probability estimates naturally',
            'Most computationally efficient during training',
            'Most computationally efficient during prediction',
            'Most robust to class imbalance'
        ],
        'OVA': ['', 'X', '', '', ''],
        'OVO': ['', 'X', '', '', 'X'],
        'ECOC': ['X', '', '', '', 'X'],
        'Direct Multi-class': ['X', 'X', 'X', 'X', '']
    }
    
    quiz_df = pd.DataFrame(quiz_table)
    print("\nQuiz Answer Table:")
    print(quiz_df.to_markdown(index=False))
    
    # Write answers for questions 2-4
    answers = {
        "q2": "For a problem with 100 classes but limited training data, I would recommend the ECOC approach because it offers a good balance between handling many classes efficiently (logarithmic scaling) and being robust to limited training data through its error-correcting properties.",
        "q3": "For a problem with 3 classes and abundant training data, I would recommend the Direct Multi-class (Softmax) approach because it's the most computationally efficient and provides natural probability estimates without requiring multiple binary classifiers.",
        "q4": "The choice of base classifier affects multi-class strategy selection because weak classifiers benefit from error-correcting approaches like ECOC, while powerful classifiers like logistic regression can effectively utilize direct multi-class approaches. Additionally, base classifiers that natively provide probability estimates (like logistic regression) work better with OVA and direct multi-class than those that don't."
    }
    
    print("\n2. Recommendation for 100 classes with limited training data:")
    print(answers["q2"])
    
    print("\n3. Recommendation for 3 classes with abundant training data:")
    print(answers["q3"])
    
    print("\n4. Impact of base classifier choice:")
    print(answers["q4"])
    
    # Save answers to a text file
    with open(os.path.join(save_dir, 'quiz_answers.txt'), 'w') as f:
        f.write("Question 9 Answers:\n\n")
        f.write(f"2. {answers['q2']}\n\n")
        f.write(f"3. {answers['q3']}\n\n")
        f.write(f"4. {answers['q4']}\n")
    
    print(f"\nAll visualizations saved to: {save_dir}")
    
    return summary_df, quiz_df, answers

if __name__ == "__main__":
    summary_table, quiz_table, answers = main() 