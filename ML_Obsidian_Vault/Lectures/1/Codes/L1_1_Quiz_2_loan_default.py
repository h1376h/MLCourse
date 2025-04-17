import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_1_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Introduction to Loan Default Prediction
print_step_header(1, "Introduction to Loan Default Prediction")

print("In this problem, we're building a supervised learning model to predict whether a bank loan applicant will default on their loan.")
print("This is a classification problem, as we're predicting a discrete outcome (default or no default).")
print()

# Step 2: Generate synthetic loan data
print_step_header(2, "Generating Synthetic Loan Data")

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000

# Define features
income = np.random.normal(60000, 20000, n_samples)  # Annual income
credit_score = np.random.normal(700, 100, n_samples)  # Credit score
loan_amount = np.random.normal(150000, 75000, n_samples)  # Loan amount
debt_to_income = np.random.normal(0.3, 0.15, n_samples)  # Debt-to-income ratio
employment_length = np.random.normal(5, 3, n_samples)  # Years at current job

# Clip values to realistic ranges
credit_score = np.clip(credit_score, 300, 850)
debt_to_income = np.clip(debt_to_income, 0.05, 0.8)
employment_length = np.clip(employment_length, 0, 40)
income = np.clip(income, 20000, 200000)
loan_amount = np.clip(loan_amount, 10000, 500000)

# Calculate loan-to-income ratio
loan_to_income = loan_amount / income

# Define a function to generate default probability based on features
def default_probability(income, credit_score, loan_amount, debt_to_income, employment_length):
    # Higher credit score, income, and employment length decrease default probability
    # Higher loan amount and debt-to-income ratio increase default probability
    prob = (
        -0.00001 * income
        - 0.0015 * credit_score
        + 0.0000005 * loan_amount
        + 2 * debt_to_income
        - 0.05 * employment_length
        + 1.5  # base probability
    )
    # Apply sigmoid function to get probability between 0 and 1
    return 1 / (1 + np.exp(-prob))

# Generate default probabilities
default_prob = default_probability(income, credit_score, loan_amount, debt_to_income, employment_length)

# Generate binary outcome (default or not)
default = (np.random.random(n_samples) < default_prob).astype(int)

# Create a DataFrame
loan_data = pd.DataFrame({
    'Income': income,
    'Credit_Score': credit_score,
    'Loan_Amount': loan_amount,
    'Debt_to_Income': debt_to_income,
    'Employment_Length': employment_length,
    'Loan_to_Income': loan_to_income,
    'Default': default
})

# Display data summary
print("Generated synthetic loan data with the following features:")
print("1. Income: Annual income of the applicant")
print("2. Credit_Score: Credit score of the applicant")
print("3. Loan_Amount: Amount of loan requested")
print("4. Debt_to_Income: Ratio of existing debt to income")
print("5. Employment_Length: Years at current job")
print("6. Loan_to_Income: Ratio of loan amount to annual income")
print()
print("Summary statistics of the data:")
print(loan_data.describe())
print()
print(f"Default rate in the data: {loan_data['Default'].mean():.2%}")

# Step 3: Explore the relationship between features and loan default
print_step_header(3, "Exploratory Data Analysis")

# Create a pairplot to visualize feature relationships
features = ['Income', 'Credit_Score', 'Loan_Amount', 'Debt_to_Income', 'Employment_Length', 'Default']
sns.pairplot(loan_data[features], hue='Default', plot_kws={'alpha': 0.6})
plt.suptitle('Pairwise Relationships Between Features and Default Status', y=1.02, fontsize=16)
# Save the figure
file_path = os.path.join(save_dir, "loan_features_pairplot.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Feature distributions by default status
fig, axes = plt.subplots(3, 2, figsize=(16, 16))
axes = axes.flatten()

features_to_plot = ['Income', 'Credit_Score', 'Loan_Amount', 'Debt_to_Income', 'Employment_Length', 'Loan_to_Income']
titles = ['Annual Income', 'Credit Score', 'Loan Amount', 'Debt-to-Income Ratio', 'Employment Length (Years)', 'Loan-to-Income Ratio']

for i, (feature, title) in enumerate(zip(features_to_plot, titles)):
    # Plot histograms
    sns.histplot(data=loan_data, x=feature, hue='Default', bins=30, alpha=0.6, ax=axes[i])
    axes[i].set_title(f'Distribution of {title} by Default Status', fontsize=14)
    axes[i].set_xlabel(title, fontsize=12)
    axes[i].set_ylabel('Count', fontsize=12)
    axes[i].legend(['No Default', 'Default'])

plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "feature_distributions.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Calculate correlation matrix
correlation_matrix = loan_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Loan Features', fontsize=16)
# Save the figure
file_path = os.path.join(save_dir, "correlation_matrix.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Classification Problem vs. Regression Problem
print_step_header(4, "Classification vs. Regression")

# Create a figure to explain the difference
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
x = np.linspace(0, 1, 100)
y = 4*x*(1-x)
plt.plot(x, y, 'r-', linewidth=2)
plt.scatter(x[::10], y[::10] + np.random.normal(0, 0.05, 10), color='blue')
plt.title('Regression Problem', fontsize=14)
plt.xlabel('Feature X', fontsize=12)
plt.ylabel('Target Y (Continuous)', fontsize=12)
plt.grid(True)
plt.text(0.1, 0.8, "Prediction is a\ncontinuous value", fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.subplot(1, 2, 2)
x1 = np.random.normal(-1, 0.5, 50)
x2 = np.random.normal(1, 0.5, 50)
y1 = np.zeros(50)
y2 = np.ones(50)
plt.scatter(x1, np.random.normal(0.25, 0.05, 50), color='blue', label='Class 0')
plt.scatter(x2, np.random.normal(0.25, 0.05, 50), color='red', label='Class 1')
plt.title('Classification Problem', fontsize=14)
plt.xlabel('Feature X', fontsize=12)
plt.ylabel('Target Y (Binary)', fontsize=12)
plt.yticks([0, 1], ['No Default', 'Default'])
plt.grid(True)
plt.legend()
plt.text(0, 0.4, "Prediction is a\ndiscrete category", fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.suptitle('Loan Default Prediction is a Classification Problem', fontsize=16, y=1.05)
plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "classification_vs_regression.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("Why Loan Default Prediction is a Classification Problem:")
print("1. The target variable (default) is binary/categorical (yes/no)")
print("2. We want to predict which class/category a loan application belongs to")
print("3. The output is a discrete value (0 or 1) rather than a continuous value")
print("4. The evaluation metrics are specific to classification (accuracy, precision, recall, etc.)")
print()

# Step 5: Model Training and Evaluation
print_step_header(5, "Model Training and Evaluation")

# Prepare the data for modeling
X = loan_data[['Income', 'Credit_Score', 'Loan_Amount', 'Debt_to_Income', 'Employment_Length']]
y = loan_data['Default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred = log_reg.predict(X_test_scaled)
y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("Model Evaluation:")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
cr = classification_report(y_test, y_pred)
print(cr)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks([0.5, 1.5], ['No Default', 'Default'])
plt.yticks([0.5, 1.5], ['No Default', 'Default'])
# Save the figure
file_path = os.path.join(save_dir, "confusion_matrix.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True)
# Save the figure
file_path = os.path.join(save_dir, "roc_curve.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.grid(True)
# Save the figure
file_path = os.path.join(save_dir, "precision_recall_curve.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Visualize feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(log_reg.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Loan Default Prediction', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(True)
# Save the figure
file_path = os.path.join(save_dir, "feature_importance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Model Comparison (Simple vs. Complex)
print_step_header(6, "Model Comparison")

# Train a random forest as a more complex model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test_scaled)
y_pred_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

# Compare ROC curves
plt.figure(figsize=(10, 8))
# LogisticRegression ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')

# RandomForest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True)
# Save the figure
file_path = os.path.join(save_dir, "model_comparison_roc.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Visualize feature importance for Random Forest
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})
feature_importance_rf = feature_importance_rf.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_rf)
plt.title('Random Forest Feature Importance for Loan Default Prediction', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(True)
# Save the figure
file_path = os.path.join(save_dir, "rf_feature_importance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Visualize model decision boundaries (for simplified 2D data)
print_step_header(7, "Visualizing Decision Boundaries")

# Extract two most important features for visualization
if feature_importance.iloc[0]['Feature'] == 'Credit_Score':
    feature1 = 'Credit_Score'
    feature2 = feature_importance.iloc[1]['Feature']
else:
    feature1 = feature_importance.iloc[0]['Feature']
    feature2 = 'Credit_Score'

X_2d = loan_data[[feature1, feature2]]
y_2d = loan_data['Default']

# Split and scale the 2D data
X_2d_train, X_2d_test, y_2d_train, y_2d_test = train_test_split(X_2d, y_2d, test_size=0.3, random_state=42)
scaler_2d = StandardScaler()
X_2d_train_scaled = scaler_2d.fit_transform(X_2d_train)
X_2d_test_scaled = scaler_2d.transform(X_2d_test)

# Train models on 2D data
log_reg_2d = LogisticRegression(random_state=42)
log_reg_2d.fit(X_2d_train_scaled, y_2d_train)

rf_2d = RandomForestClassifier(n_estimators=100, random_state=42)
rf_2d.fit(X_2d_train_scaled, y_2d_train)

# Function to plot the decision boundary
def plot_decision_boundary(X, y, model, ax, title):
    # Create mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    
    # Plot the data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['red', 'blue']), edgecolor='k', s=50, alpha=0.7)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(feature1, fontsize=12)
    ax.set_ylabel(feature2, fontsize=12)
    
    return scatter

# Plot decision boundaries
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

scatter1 = plot_decision_boundary(X_2d_test_scaled, y_2d_test, log_reg_2d, ax1, 'Logistic Regression Decision Boundary')
scatter2 = plot_decision_boundary(X_2d_test_scaled, y_2d_test, rf_2d, ax2, 'Random Forest Decision Boundary')

# Create a common legend
handles, labels = scatter1.legend_elements()
labels = ['Default', 'No Default']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=12)

plt.suptitle(f'Decision Boundaries for Loan Default Prediction\nUsing {feature1} and {feature2}', fontsize=16, y=0.98)
plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "decision_boundaries.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Summary and Conclusion
print_step_header(8, "Summary and Conclusion")

print("Summary of Loan Default Prediction Analysis:")
print()
print("1. Problem Type: Classification")
print("   - Predicting whether a loan applicant will default (binary outcome)")
print()
print("2. Key Features for Prediction:")
for i, row in feature_importance.head().iterrows():
    print(f"   - {row['Feature']}")
print()
print("3. Training Data Requirements:")
print("   - Labeled historical data of past loans with known outcomes")
print("   - Features of loan applicants: income, credit score, loan amount, etc.")
print("   - Target variable: whether the applicant defaulted (1) or not (0)")
print()
print("4. Model Evaluation Metrics:")
print("   - Accuracy: Overall correctness of predictions")
print("   - Precision: Proportion of positive identifications that were actually correct")
print("   - Recall: Proportion of actual positives that were identified correctly")
print("   - F1 Score: Harmonic mean of precision and recall")
print("   - AUC-ROC: Area under the Receiver Operating Characteristic curve")
print()
print("5. Business Application:")
print("   - Risk assessment for loan approval decisions")
print("   - Setting appropriate interest rates based on risk")
print("   - Portfolio management and capital allocation")
print("   - Regulatory compliance and stress testing")
print()
print("Conclusion: Supervised learning classification models can effectively predict loan defaults")
print("using applicant features, helping banks make better lending decisions and manage risk.") 