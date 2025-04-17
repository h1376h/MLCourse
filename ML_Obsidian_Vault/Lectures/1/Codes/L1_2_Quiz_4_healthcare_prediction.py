import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_2_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Introduction to the Healthcare Prediction Problem
print_step_header(1, "Healthcare Readmission Prediction Problem")

print("Problem: Predicting whether a patient is likely to be readmitted to the hospital within 30 days after discharge.")
print("\nOur task is to:")
print("1. Formulate this as a specific type of machine learning problem")
print("2. Identify the input features to use")
print("3. Define the target variable precisely")
print("4. Discuss ethical considerations and potential biases")
print("5. Describe how to validate the model before deployment")

# Step 2: Problem Formulation
print_step_header(2, "Problem Formulation")

print("This is a binary classification problem where we need to predict one of two outcomes:")
print("- Class 0: Patient NOT readmitted within 30 days")
print("- Class 1: Patient IS readmitted within 30 days")
print("\nKey characteristics of this problem:")
print("- Time-sensitive prediction (30-day window)")
print("- Imbalanced classes (readmissions are typically much less frequent than non-readmissions)")
print("- High cost of errors (both false positives and false negatives)")
print("- Multiple data types and sources must be integrated")
print("- Need for model interpretability for clinical acceptance")

# Create a visualization of the problem formulation
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')

# Create a timeline visualization
timeline_y = 0.5
ax.axhline(y=timeline_y, xmin=0.05, xmax=0.95, color='black', linestyle='-', linewidth=2)

# Add key events
events = [
    {'position': 0.1, 'label': 'Initial Admission', 'color': '#1f77b4'},
    {'position': 0.35, 'label': 'Hospital Stay', 'color': '#ff7f0e'},
    {'position': 0.4, 'label': 'Discharge', 'color': '#2ca02c'},
    {'position': 0.7, 'label': '30-Day Window', 'color': '#d62728'},
]

for event in events:
    ax.scatter(event['position'], timeline_y, color=event['color'], s=100, zorder=5)
    ax.annotate(event['label'], xy=(event['position'], timeline_y), 
                xytext=(event['position'], timeline_y + 0.1), ha='center',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='gray'))

# Add prediction point
ax.scatter(0.4, timeline_y, marker='*', color='green', s=200, zorder=10)
ax.annotate('Prediction\nPoint', xy=(0.4, timeline_y), xytext=(0.4, timeline_y - 0.15), 
            ha='center', va='top', fontsize=10,
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='gray'))

# Add readmission event (possible outcome)
ax.scatter(0.6, timeline_y, marker='X', color='red', s=150, zorder=5)
ax.annotate('Potential\nReadmission', xy=(0.6, timeline_y), xytext=(0.6, timeline_y - 0.15),
           ha='center', va='top', fontsize=10,
           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.1', color='gray'))

# Add brackets for key periods
ax.annotate('', xy=(0.1, timeline_y - 0.05), xytext=(0.4, timeline_y - 0.05),
           arrowprops=dict(arrowstyle='|-|', color='blue', lw=2))
ax.text(0.25, timeline_y - 0.08, 'Hospital Stay', ha='center', color='blue')

ax.annotate('', xy=(0.4, timeline_y + 0.05), xytext=(0.7, timeline_y + 0.05),
           arrowprops=dict(arrowstyle='|-|', color='red', lw=2))
ax.text(0.55, timeline_y + 0.08, '30-Day Readmission Window', ha='center', color='red')

# Add problem formulation
formulation_text = """
Machine Learning Formulation:
- Task: Binary Classification
- Features: Patient demographics, medical history, diagnoses, 
  procedures, lab values, medications, length of stay, etc.
- Target: Readmission within 30 days (Yes/No)
- Model Output: Probability of readmission
- Key Metric: Area Under ROC Curve (AUC), Precision-Recall
"""

ax.text(0.05, 0.15, formulation_text, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8))

plt.title('Hospital Readmission Prediction: Problem Formulation', fontsize=14)
file_path = os.path.join(save_dir, "readmission_problem.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Input Features
print_step_header(3, "Input Features")

print("The following categories of features would be used in the readmission prediction model:")
print("\n1. Patient Demographics:")
print("   - Age, gender, race/ethnicity")
print("   - Socioeconomic indicators (income, education, insurance type)")
print("   - ZIP code and derived neighborhood characteristics")
print("\n2. Clinical History:")
print("   - Previous hospital admissions and emergency department visits")
print("   - Chronic conditions and comorbidities")
print("   - Previous surgeries and procedures")
print("   - Medication history and adherence")
print("\n3. Current Admission Details:")
print("   - Primary and secondary diagnoses (ICD codes)")
print("   - Procedures performed during hospitalization")
print("   - Length of stay")
print("   - Hospital ward/unit")
print("   - Admitting physician specialty")
print("\n4. Laboratory and Vital Signs:")
print("   - Abnormal lab values at admission and discharge")
print("   - Trends in vital signs during hospitalization")
print("   - Changes in key biomarkers")
print("\n5. Medications and Treatments:")
print("   - Discharge medications")
print("   - Changes to medication regimen during stay")
print("   - Number of medications (polypharmacy)")
print("\n6. Social and Behavioral Factors:")
print("   - Living situation (alone, with family, nursing facility)")
print("   - Substance use (smoking, alcohol, drugs)")
print("   - Mobility and functional status")
print("\n7. Post-Discharge Plan:")
print("   - Follow-up appointments scheduled")
print("   - Home health services arranged")
print("   - Discharge destination")

# Create a visualization of the feature categories
feature_categories = [
    'Demographics', 
    'Clinical History', 
    'Current Admission', 
    'Labs & Vitals', 
    'Medications', 
    'Social Factors',
    'Discharge Plan'
]

feature_examples = [
    ['Age', 'Gender', 'Race', 'SES', 'Insurance', 'Location'],
    ['Prior admissions', 'Comorbidities', 'Chronic conditions', 'Prior surgeries'],
    ['Diagnoses', 'Procedures', 'Length of stay', 'Specialty'],
    ['Lab values', 'Vital signs', 'Biomarkers', 'Trends'],
    ['Discharge meds', 'Med changes', 'Polypharmacy', 'High-risk meds'],
    ['Living situation', 'Substance use', 'Functional status', 'Support'],
    ['Follow-up appts', 'Home health', 'Destination']
]

importance_scores = [0.15, 0.22, 0.18, 0.17, 0.12, 0.08, 0.08]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1, 1.5]})

# Plot 1: Feature importance by category
y_pos = np.arange(len(feature_categories))
ax1.barh(y_pos, importance_scores, color=colors)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(feature_categories)
ax1.invert_yaxis()  # labels read top-to-bottom
ax1.set_xlabel('Relative Importance (example)')
ax1.set_title('Feature Category Importance', fontsize=12)

# Plot 2: Feature details
ax2.axis('off')
cell_height = 0.1
for i, (category, examples, color) in enumerate(zip(feature_categories, feature_examples, colors)):
    # Draw category box
    y_pos = 0.9 - i * (cell_height + 0.03)
    category_rect = plt.Rectangle((0.05, y_pos - cell_height), 0.2, cell_height, 
                                  facecolor=color, alpha=0.7, edgecolor='black')
    ax2.add_patch(category_rect)
    ax2.text(0.15, y_pos - cell_height/2, category, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw features box
    features_rect = plt.Rectangle((0.3, y_pos - cell_height), 0.65, cell_height, 
                                  facecolor='white', alpha=0.8, edgecolor=color)
    ax2.add_patch(features_rect)
    
    # Add feature examples
    feature_text = ', '.join(examples)
    ax2.text(0.33, y_pos - cell_height/2, feature_text, ha='left', va='center', fontsize=9)

ax2.set_title('Feature Details by Category', fontsize=12)

plt.tight_layout()
file_path = os.path.join(save_dir, "readmission_features.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Target Variable Definition
print_step_header(4, "Target Variable Definition")

print("The target variable needs precise definition to ensure consistent model training and evaluation:")
print("\nDefinition: Binary indicator of whether a patient was readmitted to the same hospital or any hospital")
print("in the healthcare system within 30 days of discharge from the index hospitalization.")
print("\nSpecific considerations in the target variable definition:")
print("1. Time Window: Exactly 30 days from discharge date/time")
print("2. Readmission Types:")
print("   - Include: Unplanned readmissions for any cause")
print("   - Exclude: Planned readmissions (e.g., scheduled chemotherapy)")
print("   - Exclude: Transfers to other facilities that are not readmissions")
print("3. Hospital Scope:")
print("   - Include: Readmissions to the same hospital")
print("   - Include: Readmissions to other hospitals in the same healthcare system")
print("   - Consider: Readmissions to any hospital if data is available")
print("4. Patient Constraints:")
print("   - Exclude: Patients who died during initial hospitalization")
print("   - Include: All adult patients (age ≥ 18)")
print("   - Consider separately: Specialty populations (psychiatric, obstetric)")

# Create a visualization for the target variable definition
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Create a decision flowchart for defining readmission
def draw_box(x, y, width, height, text, color='lightblue', text_color='black'):
    box = plt.Rectangle((x - width/2, y - height/2), width, height, 
                      facecolor=color, edgecolor='black', alpha=0.7)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', color=text_color, fontsize=9)
    return x, y

# Draw main boxes
start_x, start_y = draw_box(0.5, 0.9, 0.5, 0.1, "Patient discharged from hospital", "#d0e0ff")

decision1_x, decision1_y = draw_box(0.5, 0.75, 0.6, 0.1, "Was the patient admitted to any hospital\nwithin 30 days of discharge?", "#ffe0b0")

yes_box_x, yes_box_y = draw_box(0.3, 0.6, 0.4, 0.1, "Yes", "#ffffff")
no_box_x, no_box_y = draw_box(0.7, 0.6, 0.4, 0.1, "No", "#ffffff")

decision2_x, decision2_y = draw_box(0.3, 0.45, 0.4, 0.1, "Was the readmission planned?\n(e.g., scheduled chemotherapy)", "#ffe0b0")

yes2_box_x, yes2_box_y = draw_box(0.15, 0.3, 0.2, 0.1, "Yes", "#ffffff")
no2_box_x, no2_box_y = draw_box(0.45, 0.3, 0.2, 0.1, "No", "#ffffff")

target0_x, target0_y = draw_box(0.7, 0.3, 0.3, 0.1, "Target = 0\nNot readmitted", "#ffcccc")
target0_planned_x, target0_planned_y = draw_box(0.15, 0.15, 0.3, 0.1, "Target = 0\nExclude planned readmissions", "#ffcccc")
target1_x, target1_y = draw_box(0.45, 0.15, 0.3, 0.1, "Target = 1\nUnplanned readmission", "#ccffcc")

# Draw arrows
arrows = [
    ((start_x, start_y - 0.05), (decision1_x, decision1_y + 0.05)),
    ((decision1_x, decision1_y - 0.05), (yes_box_x, yes_box_y + 0.05)),
    ((decision1_x, decision1_y - 0.05), (no_box_x, no_box_y + 0.05)),
    ((yes_box_x, yes_box_y - 0.05), (decision2_x, decision2_y + 0.05)),
    ((no_box_x, no_box_y - 0.05), (target0_x, target0_y + 0.05)),
    ((decision2_x, decision2_y - 0.05), (yes2_box_x, yes2_box_y + 0.05)),
    ((decision2_x, decision2_y - 0.05), (no2_box_x, no2_box_y + 0.05)),
    ((yes2_box_x, yes2_box_y - 0.05), (target0_planned_x, target0_planned_y + 0.05)),
    ((no2_box_x, no2_box_y - 0.05), (target1_x, target1_y + 0.05))
]

for start, end in arrows:
    ax.annotate("", xy=end, xytext=start,
               arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

# Add note about exclusions
exclusion_text = """Exclusions from the analysis:
- Patients who died during initial hospitalization
- Patients under 18 years of age
- Patients discharged to hospice care
- Transfers to other acute care facilities"""

ax.text(0.85, 0.5, exclusion_text, ha='left', va='center', fontsize=9,
       bbox=dict(facecolor='#f5f5f5', edgecolor='gray', boxstyle='round,pad=0.5'))

plt.title('Precise Definition of 30-Day Readmission Target Variable', fontsize=14)
file_path = os.path.join(save_dir, "target_definition.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Ethical Considerations and Biases
print_step_header(5, "Ethical Considerations and Biases")

print("Developing and deploying a hospital readmission prediction model raises several ethical concerns:")
print("\n1. Algorithmic Bias:")
print("   - Potential for reinforcing existing healthcare disparities")
print("   - Underrepresentation of minority groups in training data")
print("   - Proxy variables that may encode socioeconomic or racial bias")
print("\n2. Data Privacy and Security:")
print("   - Handling sensitive protected health information (PHI)")
print("   - Compliance with HIPAA and other regulations")
print("   - Secure storage and transfer of patient data")
print("\n3. Explainability and Transparency:")
print("   - Clinicians need to understand why a prediction was made")
print("   - Patients have the right to know how their data is used")
print("   - 'Black box' models may face resistance in clinical settings")
print("\n4. Resource Allocation Concerns:")
print("   - How the model will influence clinical decision-making")
print("   - Potential for denying care based on algorithmic prediction")
print("   - Balancing cost-saving with equitable care delivery")
print("\n5. Informed Consent:")
print("   - Patient awareness that their data is used for prediction")
print("   - Opt-out options for algorithmic decision support")
print("   - Communication of model limitations to patients")

# Create a visualization for ethical considerations
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Define the ethical considerations and potential mitigation strategies
ethical_issues = [
    {
        'category': 'Algorithmic Bias',
        'issues': [
            'Racial/ethnic disparities in predictions',
            'Socioeconomic status as hidden factor',
            'Underrepresented groups in training data'
        ],
        'mitigation': [
            'Fairness metrics across demographic groups',
            'Remove or carefully control proxy variables',
            'Representative and balanced training data',
            'Regular bias audits and monitoring'
        ]
    },
    {
        'category': 'Privacy & Security',
        'issues': [
            'Protected health information exposure',
            'Re-identification risks',
            'Data sharing between organizations'
        ],
        'mitigation': [
            'HIPAA compliance and data encryption',
            'Differential privacy techniques',
            'Federated learning approaches',
            'Minimal data collection principles'
        ]
    },
    {
        'category': 'Explainability',
        'issues': [
            'Black box decision making',
            'Clinician distrust of opaque models',
            'Patient right to explanation'
        ],
        'mitigation': [
            'Interpretable models (e.g., GAMs)',
            'SHAP/LIME explanations for predictions',
            'Clinical validation of model logic',
            'Transparent documentation of model limitations'
        ]
    },
    {
        'category': 'Resource Allocation',
        'issues': [
            'Unequal distribution of interventions',
            'Cost-driven decision making',
            'Reinforcement of existing care gaps'
        ],
        'mitigation': [
            'Equity-aware intervention protocols',
            'Human oversight of algorithmic recommendations',
            'Regular assessment of outcome disparities',
            'Blend of risk and need in resource allocation'
        ]
    }
]

# Plot the issues and mitigation strategies
y_start = 0.9
cell_height = 0.15
row_spacing = 0.05

for i, issue in enumerate(ethical_issues):
    y_pos = y_start - i * (cell_height + row_spacing)
    
    # Draw category box
    category_rect = plt.Rectangle((0.05, y_pos - cell_height), 0.15, cell_height, 
                                  facecolor='#D8BFD8', edgecolor='#9370DB', alpha=0.8)
    ax.add_patch(category_rect)
    ax.text(0.125, y_pos - cell_height/2, issue['category'], 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw issues box
    issues_rect = plt.Rectangle((0.22, y_pos - cell_height), 0.3, cell_height, 
                               facecolor='#FFE4E1', edgecolor='#DB7093', alpha=0.8)
    ax.add_patch(issues_rect)
    
    # List issues
    for j, issue_text in enumerate(issue['issues']):
        y_text = y_pos - 0.03 - j * 0.035
        if y_text > y_pos - cell_height:  # Check if text is within the box
            ax.text(0.24, y_text, f"• {issue_text}", fontsize=8, va='center')
    
    # Draw arrow
    ax.annotate("", xy=(0.55, y_pos - cell_height/2), xytext=(0.52, y_pos - cell_height/2),
                arrowprops=dict(arrowstyle="->", color='#4682B4', lw=1.5))
    
    # Draw mitigation box
    mitigation_rect = plt.Rectangle((0.55, y_pos - cell_height), 0.4, cell_height, 
                                   facecolor='#E0FFFF', edgecolor='#5F9EA0', alpha=0.8)
    ax.add_patch(mitigation_rect)
    
    # List mitigation strategies
    for j, strategy in enumerate(issue['mitigation']):
        y_text = y_pos - 0.03 - j * 0.03
        if y_text > y_pos - cell_height:  # Check if text is within the box
            ax.text(0.57, y_text, f"• {strategy}", fontsize=8, va='center')

# Add headers
ax.text(0.125, y_start + 0.02, 'Ethical Issue', ha='center', fontsize=11, fontweight='bold')
ax.text(0.37, y_start + 0.02, 'Potential Problems', ha='center', fontsize=11, fontweight='bold')
ax.text(0.75, y_start + 0.02, 'Mitigation Strategies', ha='center', fontsize=11, fontweight='bold')

plt.title('Ethical Considerations in Hospital Readmission Prediction', fontsize=14)
file_path = os.path.join(save_dir, "ethical_considerations.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Model Validation
print_step_header(6, "Model Validation Before Deployment")

print("Before deploying a readmission prediction model in a clinical setting, rigorous validation is essential:")
print("\n1. Technical Validation:")
print("   - Cross-validation on historical data (temporal validation)")
print("   - External validation on data from different hospitals")
print("   - Regular retraining and performance monitoring")
print("   - Calibration assessment (predicted vs. actual probabilities)")
print("\n2. Clinical Validation:")
print("   - Prospective validation in real clinical settings")
print("   - Comparison with existing readmission risk scores")
print("   - Assessment by clinical experts")
print("   - Pilot studies before full deployment")
print("\n3. Impact Validation:")
print("   - Measure effect on readmission rates")
print("   - Evaluation of cost-effectiveness")
print("   - Assessment of workflow integration")
print("   - Patient and provider satisfaction surveys")
print("\n4. Fairness Validation:")
print("   - Evaluation of prediction disparities across demographic groups")
print("   - Impact assessment on vulnerable populations")
print("   - Review by ethics committee")
print("   - Community engagement and feedback")

# Create a visualization for model validation
# Generate some synthetic data for the validation plots
np.random.seed(42)

# Simulated model predictions and outcomes
n_samples = 1000
y_true = np.random.binomial(1, 0.15, n_samples)  # ~15% readmission rate
y_probs = np.zeros(n_samples)

# Create more realistic probability predictions
for i in range(n_samples):
    if y_true[i] == 1:
        y_probs[i] = np.random.beta(5, 3)  # Higher probabilities for true positives
    else:
        y_probs[i] = np.random.beta(1, 8)  # Lower probabilities for true negatives

# Apply a threshold for binary predictions
threshold = 0.3
y_pred = (y_probs >= threshold).astype(int)

# Create subgroups for fairness assessment
age_groups = np.random.choice(['18-44', '45-64', '65+'], n_samples, p=[0.3, 0.4, 0.3])
gender = np.random.choice(['Female', 'Male'], n_samples, p=[0.55, 0.45])
race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_samples, p=[0.65, 0.15, 0.1, 0.05, 0.05])

# Create a figure with multiple validation plots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: ROC curve
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

axs[0, 0].plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axs[0, 0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
axs[0, 0].set_xlim([0.0, 1.0])
axs[0, 0].set_ylim([0.0, 1.05])
axs[0, 0].set_xlabel('False Positive Rate')
axs[0, 0].set_ylabel('True Positive Rate')
axs[0, 0].set_title('Technical Validation: ROC Curve', fontsize=12)
axs[0, 0].legend(loc="lower right")
axs[0, 0].grid(True)

# Plot 2: Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_true, y_probs)
pr_auc = auc(recall, precision)

axs[0, 1].plot(recall, precision, color='#ff7f0e', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
axs[0, 1].axhline(y=sum(y_true)/len(y_true), color='gray', linestyle='--', label=f'Baseline (prevalence = {sum(y_true)/len(y_true):.2f})')
axs[0, 1].set_xlim([0.0, 1.0])
axs[0, 1].set_ylim([0.0, 1.05])
axs[0, 1].set_xlabel('Recall')
axs[0, 1].set_ylabel('Precision')
axs[0, 1].set_title('Technical Validation: Precision-Recall Curve', fontsize=12)
axs[0, 1].legend(loc="lower left")
axs[0, 1].grid(True)

# Plot 3: Calibration curve
prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)

axs[1, 0].plot(prob_pred, prob_true, 's-', color='#2ca02c', label='Calibration curve')
axs[1, 0].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
axs[1, 0].set_xlim([0.0, 1.0])
axs[1, 0].set_ylim([0.0, 1.0])
axs[1, 0].set_xlabel('Mean predicted probability')
axs[1, 0].set_ylabel('Fraction of positives')
axs[1, 0].set_title('Clinical Validation: Calibration Curve', fontsize=12)
axs[1, 0].legend(loc="lower right")
axs[1, 0].grid(True)

# Plot 4: Fairness metrics by demographic group
# Calculate false positive rates by race group
subgroups = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
fps_by_race = []
tps_by_race = []
fns_by_race = []
tns_by_race = []

for group in subgroups:
    group_idx = race == group
    y_true_group = y_true[group_idx]
    y_pred_group = y_pred[group_idx]
    
    # Calculate confusion matrix elements
    tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
    fp = np.sum((y_true_group == 0) & (y_pred_group == 1))
    fn = np.sum((y_true_group == 1) & (y_pred_group == 0))
    tn = np.sum((y_true_group == 0) & (y_pred_group == 0))
    
    # Calculate rates
    fps_by_race.append(fp / (fp + tn) if (fp + tn) > 0 else 0)  # FPR
    tps_by_race.append(tp / (tp + fn) if (tp + fn) > 0 else 0)  # TPR / Recall
    fns_by_race.append(fn / (fn + tp) if (fn + tp) > 0 else 0)  # FNR / Miss rate
    tns_by_race.append(tn / (tn + fp) if (tn + fp) > 0 else 0)  # TNR / Specificity

# Plot fairness metrics
bar_width = 0.2
index = np.arange(len(subgroups))

axs[1, 1].bar(index - bar_width*1.5, tps_by_race, bar_width, label='TPR (Recall)', color='#d62728')
axs[1, 1].bar(index - bar_width/2, fps_by_race, bar_width, label='FPR', color='#9467bd')
axs[1, 1].bar(index + bar_width/2, fns_by_race, bar_width, label='FNR (Miss rate)', color='#8c564b')
axs[1, 1].bar(index + bar_width*1.5, tns_by_race, bar_width, label='TNR (Specificity)', color='#e377c2')

axs[1, 1].set_xlabel('Racial/Ethnic Group')
axs[1, 1].set_ylabel('Rate')
axs[1, 1].set_title('Fairness Validation: Performance by Demographic Group', fontsize=12)
axs[1, 1].set_xticks(index)
axs[1, 1].set_xticklabels(subgroups)
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "model_validation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Summary
print_step_header(7, "Summary")

print("Hospital Readmission Prediction: A Comprehensive Approach")
print("\n1. Problem Formulation:")
print("   - Binary classification to predict 30-day readmission risk")
print("   - Critical for improving patient outcomes and reducing healthcare costs")
print("   - Requires balanced consideration of false positive and false negative risks")
print("\n2. Key Features:")
print("   - Diverse data types: demographics, clinical, medications, labs, social factors")
print("   - Time-dependent factors around admission and discharge")
print("   - Combination of structured and unstructured data sources")
print("\n3. Target Variable:")
print("   - Precisely defined as unplanned readmission within 30 days")
print("   - Exclusions for planned readmissions and transfers")
print("   - Consistent definition across healthcare system")
print("\n4. Ethical Framework:")
print("   - Bias monitoring and mitigation")
print("   - Privacy and security safeguards")
print("   - Transparency and explainability")
print("   - Equity in resource allocation")
print("\n5. Validation Strategy:")
print("   - Technical, clinical, impact, and fairness validation")
print("   - Prospective evaluation in real-world settings")
print("   - Continuous monitoring and improvement")
print("\nThis comprehensive approach ensures a model that is not only accurate but also")
print("ethically sound and clinically useful for reducing hospital readmissions.") 