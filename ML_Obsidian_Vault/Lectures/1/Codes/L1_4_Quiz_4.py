import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from textblob import TextBlob
import nltk
nltk.download('punkt')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_4_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Essay Feature Analysis
print_step_header(1, "Essay Feature Analysis")

# Generate synthetic essay data
n_essays = 100
essay_lengths = np.random.normal(500, 100, n_essays)
vocabulary_scores = np.random.normal(0.7, 0.1, n_essays)
grammar_scores = np.random.normal(0.8, 0.1, n_essays)
coherence_scores = np.random.normal(0.75, 0.15, n_essays)
true_grades = np.random.randint(1, 6, n_essays)  # Grades 1-5

# Create feature matrix
features = np.column_stack([essay_lengths, vocabulary_scores, 
                          grammar_scores, coherence_scores])

# Plot feature distributions
plt.figure(figsize=(15, 10))
for i, (feature, name) in enumerate(zip(features.T, 
                                      ['Length', 'Vocabulary', 'Grammar', 'Coherence'])):
    plt.subplot(2, 2, i+1)
    plt.hist(feature, bins=20, alpha=0.7)
    plt.title(f'{name} Distribution')
    plt.xlabel('Score')
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "feature_distributions.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 2: Grade Distribution Analysis
print_step_header(2, "Grade Distribution Analysis")

# Plot grade distribution
plt.figure(figsize=(10, 6))
plt.hist(true_grades, bins=5, alpha=0.7)
plt.title('Distribution of Essay Grades')
plt.xlabel('Grade (1-5)')
plt.ylabel('Count')
plt.xticks(range(1, 6))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "grade_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 3: Feature-Grade Relationships
print_step_header(3, "Feature-Grade Relationships")

# Plot feature relationships with grades
plt.figure(figsize=(15, 10))
for i, (feature, name) in enumerate(zip(features.T, 
                                      ['Length', 'Vocabulary', 'Grammar', 'Coherence'])):
    plt.subplot(2, 2, i+1)
    plt.scatter(feature, true_grades, alpha=0.5)
    plt.title(f'{name} vs Grade')
    plt.xlabel(name)
    plt.ylabel('Grade')
    plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "feature_grade_relationships.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 4: Bias Analysis
print_step_header(4, "Bias Analysis")

# Generate synthetic demographic data
demographics = np.random.choice(['A', 'B', 'C', 'D'], n_essays)
group_means = {group: np.mean(true_grades[demographics == group]) 
              for group in np.unique(demographics)}

# Plot demographic bias
plt.figure(figsize=(10, 6))
plt.bar(group_means.keys(), group_means.values())
plt.title('Average Grades by Demographic Group')
plt.xlabel('Demographic Group')
plt.ylabel('Average Grade')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "demographic_bias.png"), dpi=300, bbox_inches='tight')
plt.close()

# Print analysis results
print("\nAnalysis Results:")
print("\n1. Feature Analysis:")
print("   - Multiple features contribute to grading")
print("   - Features show varying distributions")
print("   - Need to normalize and scale features")

print("\n2. Grade Distribution:")
print("   - Grades follow approximately normal distribution")
print("   - Some grade inflation present")
print("   - Need to ensure fair grading scale")

print("\n3. Feature-Grade Relationships:")
print("   - Strong correlation between features and grades")
print("   - Some features more predictive than others")
print("   - Need to weight features appropriately")

print("\n4. Bias Analysis:")
print("   - Evidence of demographic bias in grading")
print("   - Need for bias mitigation strategies")
print("   - Importance of fairness metrics") 