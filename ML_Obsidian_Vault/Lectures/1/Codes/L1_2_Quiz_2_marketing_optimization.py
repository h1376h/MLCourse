import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import random

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_2_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Introduction to the Problem
print_step_header(1, "Marketing Optimization Problem")

print("Problem: A retail company wants to optimize its marketing strategy by targeting")
print("specific customer segments with personalized promotions.")
print("\nOur task is to:")
print("1. Formulate this as a machine learning problem")
print("2. Identify what type of learning problem this is")
print("3. Specify what data we would need")
print("4. Describe how to evaluate success")
print("5. Discuss potential challenges and limitations")

# Step 2: Problem Formulation
print_step_header(2, "Problem Formulation")

print("Marketing optimization can be broken down into two main machine learning tasks:")
print("\nTask 1: Customer Segmentation (Clustering)")
print("- Group customers into segments with similar characteristics and preferences")
print("- Apply unsupervised learning to discover natural customer groups")
print("\nTask 2: Response Prediction (Classification/Regression)")
print("- Predict which customers will respond positively to specific promotions")
print("- Apply supervised learning to predict response rates for each segment")
print("\nBy combining these tasks, we can create a targeted marketing strategy.")

# Create a diagram of the overall approach
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Define the boxes and their positions
boxes = [
    {'label': 'Customer Data Collection', 'position': (0.5, 0.9), 'width': 0.3, 'height': 0.1},
    {'label': 'Data Preprocessing', 'position': (0.5, 0.75), 'width': 0.3, 'height': 0.1},
    {'label': 'Customer Segmentation\n(Clustering)', 'position': (0.3, 0.55), 'width': 0.25, 'height': 0.15},
    {'label': 'Response Prediction\n(Classification/Regression)', 'position': (0.7, 0.55), 'width': 0.25, 'height': 0.15},
    {'label': 'Segment-Specific\nPromotion Strategies', 'position': (0.5, 0.3), 'width': 0.3, 'height': 0.15},
    {'label': 'Campaign Execution', 'position': (0.5, 0.1), 'width': 0.3, 'height': 0.1}
]

# Draw boxes
for box in boxes:
    rect = plt.Rectangle(
        (box['position'][0] - box['width']/2, box['position'][1] - box['height']/2),
        box['width'], box['height'], 
        facecolor='lightblue', 
        edgecolor='blue', 
        alpha=0.7
    )
    ax.add_patch(rect)
    ax.text(box['position'][0], box['position'][1], box['label'], ha='center', va='center', fontsize=11)

# Draw arrows
arrows = [
    ((0.5, 0.85), (0.5, 0.8)),
    ((0.5, 0.7), (0.35, 0.6)),
    ((0.5, 0.7), (0.65, 0.6)),
    ((0.3, 0.48), (0.5, 0.38)),
    ((0.7, 0.48), (0.5, 0.38)),
    ((0.5, 0.23), (0.5, 0.15))
]

for start, end in arrows:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color='blue', lw=1.5))

# Add evaluation feedback loop
ax.annotate("", xy=(0.2, 0.6), xytext=(0.2, 0.2), 
            arrowprops=dict(arrowstyle="->", color='green', lw=1.5, connectionstyle="arc3,rad=0.3"))
ax.text(0.15, 0.4, "Evaluation &\nRefinement", ha='center', va='center', fontsize=10, color='green', rotation=90)

plt.title('Marketing Optimization Framework', fontsize=16)
file_path = os.path.join(save_dir, "marketing_framework.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Type of Learning Problem
print_step_header(3, "Type of Learning Problem")

print("This is a multi-stage machine learning problem that combines:")
print("\n1. Unsupervised Learning (Clustering)")
print("   - Used for customer segmentation")
print("   - No predefined labels/categories")
print("   - Goal: Find natural groupings of customers with similar characteristics")
print("\n2. Supervised Learning (Classification)")
print("   - Used for response prediction")
print("   - Requires labeled data (past campaign results)")
print("   - Goal: Predict which customers will respond to specific promotions")

# Create a visualization for the learning types
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Unsupervised learning illustration
np.random.seed(42)

# Generate customer data for visualization
n_customers = 200
age = np.random.randint(18, 80, n_customers)
income = 20000 + age * 1000 + np.random.normal(0, 15000, n_customers)
spending = 500 + 0.05 * income + np.random.normal(0, 1000, n_customers)

# Create a dataframe for clustering
customer_data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Annual_Spending': spending
})

# Apply K-means clustering
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(customer_data_scaled)

# Plot the clusters
colors = ['blue', 'green', 'red']
for i in range(3):
    cluster_data = customer_data[clusters == i]
    axs[0].scatter(cluster_data['Age'], cluster_data['Annual_Spending'], 
                 color=colors[i], alpha=0.6, label=f'Segment {i+1}')

axs[0].set_title('Unsupervised Learning: Customer Segmentation', fontsize=12)
axs[0].set_xlabel('Age', fontsize=10)
axs[0].set_ylabel('Annual Spending ($)', fontsize=10)
axs[0].legend()
axs[0].grid(True)

# Supervised learning illustration
# Create labeled data for response prediction
np.random.seed(45)
response = np.zeros(n_customers, dtype=int)

# Create a rule for response: spending + random factor determines response
response_prob = (spending - spending.min()) / (spending.max() - spending.min())
response = np.random.random(n_customers) < (response_prob * 0.8 + 0.1)  # add some randomness

# Split data for visualization
X = customer_data.copy()
X['Cluster'] = clusters
y = response

# Create a simple classification model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Response', 'Response'],
           yticklabels=['No Response', 'Response'], ax=axs[1])
axs[1].set_title('Supervised Learning: Response Prediction', fontsize=12)
axs[1].set_xlabel('Predicted', fontsize=10)
axs[1].set_ylabel('Actual', fontsize=10)

# Add performance metrics as text
metrics_text = f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
axs[1].text(1.5, 0.1, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
file_path = os.path.join(save_dir, "learning_types.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Required Data
print_step_header(4, "Required Data")

print("To implement this marketing optimization strategy, we would need:")
print("\n1. Customer Demographics:")
print("   - Age, gender, location, income level, family size")
print("   - Customer tenure, loyalty program status")
print("\n2. Purchase History:")
print("   - Product categories purchased")
print("   - Purchase frequency, recency, and monetary value (RFM metrics)")
print("   - Seasonal buying patterns")
print("\n3. Browsing/Interaction Data:")
print("   - Website visit patterns")
print("   - App usage data")
print("   - Email open/click rates")
print("\n4. Past Campaign Results:")
print("   - Customer responses to previous promotions")
print("   - Channel preferences (email, SMS, app notifications)")
print("   - Offer type preferences (discounts, BOGO, free shipping)")

# Create a data schema visualization
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

# Define the tables and their contents
tables = {
    'Customers': ['customer_id (PK)', 'name', 'email', 'phone', 'age', 'gender', 'location_id (FK)', 
                 'join_date', 'loyalty_tier', 'income_bracket'],
    'Locations': ['location_id (PK)', 'city', 'state', 'country', 'zipcode'],
    'Purchases': ['purchase_id (PK)', 'customer_id (FK)', 'date', 'total_amount', 'store_id (FK)'],
    'Purchase_Items': ['item_id (PK)', 'purchase_id (FK)', 'product_id (FK)', 'quantity', 'price'],
    'Products': ['product_id (PK)', 'name', 'description', 'category_id (FK)', 'price'],
    'Categories': ['category_id (PK)', 'name', 'department'],
    'Campaigns': ['campaign_id (PK)', 'name', 'start_date', 'end_date', 'channel', 'offer_type', 'discount_amount'],
    'Campaign_Responses': ['response_id (PK)', 'campaign_id (FK)', 'customer_id (FK)', 'sent_date', 
                          'open_date', 'click_date', 'conversion_date', 'amount_spent']
}

# Define positions for tables (x, y coordinates in 0-1 range)
positions = {
    'Customers': (0.3, 0.7),
    'Locations': (0.1, 0.5),
    'Purchases': (0.6, 0.7),
    'Purchase_Items': (0.6, 0.5),
    'Products': (0.6, 0.3),
    'Categories': (0.6, 0.1),
    'Campaigns': (0.3, 0.3),
    'Campaign_Responses': (0.3, 0.1)
}

# Draw tables
for table_name, fields in tables.items():
    x, y = positions[table_name]
    
    # Calculate table height based on number of fields
    table_height = 0.05 + len(fields) * 0.02
    
    # Draw table rectangle
    rect = plt.Rectangle((x-0.15, y-table_height/2), 0.3, table_height, 
                         facecolor='lightblue', edgecolor='blue', alpha=0.7)
    ax.add_patch(rect)
    
    # Add table name at the top
    ax.text(x, y+table_height/2-0.02, table_name, ha='center', va='center', 
            fontsize=10, fontweight='bold', color='darkblue')
    
    # Add fields
    for i, field in enumerate(fields):
        y_pos = y + table_height/2 - 0.04 - i*0.02
        ax.text(x-0.14, y_pos, field, fontsize=8, va='center')
        
        # Highlight primary and foreign keys
        if '(PK)' in field:
            ax.text(x-0.145, y_pos, '🔑', fontsize=8, va='center')
        elif '(FK)' in field:
            ax.text(x-0.145, y_pos, '🔗', fontsize=8, va='center')

# Draw relationships between tables
relationships = [
    ('Customers', 'Locations', 'location_id'),
    ('Customers', 'Purchases', 'customer_id'),
    ('Purchases', 'Purchase_Items', 'purchase_id'),
    ('Products', 'Purchase_Items', 'product_id'),
    ('Products', 'Categories', 'category_id'),
    ('Customers', 'Campaign_Responses', 'customer_id'),
    ('Campaigns', 'Campaign_Responses', 'campaign_id')
]

for table1, table2, key in relationships:
    x1, y1 = positions[table1]
    x2, y2 = positions[table2]
    
    # Draw arrow for the relationship
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                arrowprops=dict(arrowstyle="-", color='gray', lw=1, connectionstyle="arc3,rad=0.1"))

plt.title('Marketing Data Schema for Machine Learning', fontsize=14)
file_path = os.path.join(save_dir, "data_schema.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Evaluation Methods
print_step_header(5, "Evaluation Methods")

print("To evaluate the success of our marketing optimization strategy, we would use:")
print("\n1. Business KPIs:")
print("   - Conversion rate: % of customers who make a purchase after receiving a promotion")
print("   - Revenue: Total sales generated from the campaign")
print("   - ROI: Return on investment for marketing spend")
print("   - Customer acquisition cost (CAC)")
print("   - Customer lifetime value (CLV)")
print("\n2. Model Performance Metrics:")
print("   - For clustering: Silhouette score, Davies-Bouldin index")
print("   - For classification: Precision, recall, F1-score, AUC-ROC")
print("\n3. A/B Testing:")
print("   - Compare optimized marketing strategies against control groups")
print("   - Statistical significance testing of results")

# Create a visualization for evaluation methods
# 1. ROI comparison chart
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# ROI comparison
campaign_types = ['Mass Marketing', 'Basic Segmentation', 'ML-Optimized']
roi_values = [120, 180, 240]  # percentages
colors = ['#CCCCCC', '#66CCEE', '#4477AA']

axs[0, 0].bar(campaign_types, roi_values, color=colors)
axs[0, 0].set_title('ROI Comparison by Campaign Type', fontsize=12)
axs[0, 0].set_ylabel('ROI (%)', fontsize=10)
axs[0, 0].grid(axis='y')

for i, v in enumerate(roi_values):
    axs[0, 0].text(i, v+5, f"{v}%", ha='center', fontsize=10)

# Conversion rate comparison
segment_names = ['Segment 1\n(High Value)', 'Segment 2\n(Moderate Value)', 'Segment 3\n(Low Value)']
conversion_generic = [5, 3, 2]  # percentages
conversion_targeted = [12, 8, 4]  # percentages

x = np.arange(len(segment_names))
width = 0.35

axs[0, 1].bar(x - width/2, conversion_generic, width, label='Generic Campaign', color='#DDAA33')
axs[0, 1].bar(x + width/2, conversion_targeted, width, label='Targeted Campaign', color='#BB5566')
axs[0, 1].set_title('Conversion Rate by Customer Segment', fontsize=12)
axs[0, 1].set_ylabel('Conversion Rate (%)', fontsize=10)
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(segment_names)
axs[0, 1].legend()
axs[0, 1].grid(axis='y')

for i, v in enumerate(conversion_generic):
    axs[0, 1].text(i - width/2, v + 0.3, f"{v}%", ha='center', fontsize=9)
for i, v in enumerate(conversion_targeted):
    axs[0, 1].text(i + width/2, v + 0.3, f"{v}%", ha='center', fontsize=9)

# Model performance metrics
model_metrics = {
    'Accuracy': 0.85,
    'Precision': 0.82,
    'Recall': 0.79,
    'F1 Score': 0.80,
    'AUC-ROC': 0.89
}

axs[1, 0].bar(model_metrics.keys(), model_metrics.values(), color='#117733')
axs[1, 0].set_ylim(0, 1.0)
axs[1, 0].set_title('Classification Model Performance Metrics', fontsize=12)
axs[1, 0].set_ylabel('Score', fontsize=10)
axs[1, 0].grid(axis='y')

for i, (k, v) in enumerate(model_metrics.items()):
    axs[1, 0].text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)

# A/B testing results
weeks = list(range(1, 9))
control_group = [100, 105, 110, 115, 118, 120, 122, 125]
test_group = [100, 110, 125, 145, 165, 182, 195, 210]

axs[1, 1].plot(weeks, control_group, 'o-', color='#AA4499', label='Control Group')
axs[1, 1].plot(weeks, test_group, 'o-', color='#332288', label='Test Group (ML-Optimized)')
axs[1, 1].axvline(x=2.5, color='gray', linestyle='--', label='Campaign Start')
axs[1, 1].set_title('A/B Testing: Revenue Over Time', fontsize=12)
axs[1, 1].set_xlabel('Week', fontsize=10)
axs[1, 1].set_ylabel('Revenue Index (Week 1 = 100)', fontsize=10)
axs[1, 1].legend()
axs[1, 1].grid(True)

# Add annotation for lift
axs[1, 1].annotate(f'68% Lift', xy=(8, 210), xytext=(6, 180),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.tight_layout()
file_path = os.path.join(save_dir, "evaluation_methods.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Challenges and Limitations
print_step_header(6, "Challenges and Limitations")

print("Implementing this machine learning approach to marketing optimization comes with several challenges:")
print("\n1. Data Quality and Integration:")
print("   - Missing or incomplete customer data")
print("   - Data silos across different business units")
print("   - Inconsistent data formats and collection methods")
print("\n2. Privacy and Regulatory Concerns:")
print("   - GDPR, CCPA, and other privacy regulations")
print("   - Need for customer consent for data usage")
print("   - Ethical considerations in targeting vulnerable groups")
print("\n3. Technical Challenges:")
print("   - Need for real-time processing capabilities")
print("   - Integration with existing marketing systems")
print("   - Model drift and need for regular retraining")
print("\n4. Business Challenges:")
print("   - Cross-department coordination")
print("   - Change management and adoption")
print("   - Balancing short-term gains vs. long-term customer relationships")

# Create a visualization for challenges and mitigation strategies
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Define challenges and mitigation strategies
challenges = [
    {
        'category': 'Data Challenges',
        'issues': [
            'Missing or incomplete customer data',
            'Data silos across departments',
            'Inconsistent data formats'
        ],
        'mitigation': [
            'Implement robust data quality processes',
            'Create unified customer data platform',
            'Use data imputation techniques'
        ]
    },
    {
        'category': 'Privacy & Regulatory',
        'issues': [
            'GDPR and CCPA compliance',
            'Consent management',
            'Ethical targeting concerns'
        ],
        'mitigation': [
            'Privacy by design implementation',
            'Transparent opt-in processes',
            'Regular privacy impact assessments'
        ]
    },
    {
        'category': 'Technical Challenges',
        'issues': [
            'Real-time processing needs',
            'System integration complexity',
            'Model drift over time'
        ],
        'mitigation': [
            'Invest in scalable infrastructure',
            'Use API-based architecture',
            'Implement model monitoring'
        ]
    },
    {
        'category': 'Business Challenges',
        'issues': [
            'Cross-department coordination',
            'ROI justification',
            'Balancing personalization vs. privacy'
        ],
        'mitigation': [
            'Create cross-functional teams',
            'Develop phased implementation',
            'Start with high-impact use cases'
        ]
    }
]

# Plot the challenges and mitigation strategies
y_start = 0.9
cell_height = 0.15
row_spacing = 0.03

for i, challenge in enumerate(challenges):
    y_pos = y_start - i * (cell_height + row_spacing)
    
    # Draw category box
    category_rect = plt.Rectangle((0.05, y_pos - cell_height), 0.2, cell_height, 
                                  facecolor='#BBDEFB', edgecolor='#2196F3', alpha=0.8)
    ax.add_patch(category_rect)
    ax.text(0.15, y_pos - cell_height/2, challenge['category'], 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw issues box
    issues_rect = plt.Rectangle((0.27, y_pos - cell_height), 0.3, cell_height, 
                               facecolor='#FFCCBC', edgecolor='#FF5722', alpha=0.8)
    ax.add_patch(issues_rect)
    
    # List issues
    for j, issue in enumerate(challenge['issues']):
        ax.text(0.3, y_pos - 0.03 - j*0.04, f"• {issue}", fontsize=9, va='center')
    
    # Draw arrow
    ax.annotate("", xy=(0.65, y_pos - cell_height/2), xytext=(0.58, y_pos - cell_height/2),
                arrowprops=dict(arrowstyle="->", color='#78909C', lw=1.5))
    
    # Draw mitigation box
    mitigation_rect = plt.Rectangle((0.65, y_pos - cell_height), 0.3, cell_height, 
                                   facecolor='#C8E6C9', edgecolor='#4CAF50', alpha=0.8)
    ax.add_patch(mitigation_rect)
    
    # List mitigation strategies
    for j, strategy in enumerate(challenge['mitigation']):
        ax.text(0.68, y_pos - 0.03 - j*0.04, f"• {strategy}", fontsize=9, va='center')

# Add headers
ax.text(0.15, y_start + 0.03, 'Challenge Category', ha='center', fontsize=12, fontweight='bold')
ax.text(0.42, y_start + 0.03, 'Key Issues', ha='center', fontsize=12, fontweight='bold')
ax.text(0.8, y_start + 0.03, 'Mitigation Strategies', ha='center', fontsize=12, fontweight='bold')

plt.title('Marketing Optimization: Challenges and Mitigation Strategies', fontsize=16)
file_path = os.path.join(save_dir, "challenges_mitigation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Summary
print_step_header(7, "Summary")

print("In summary, the retail marketing optimization problem can be formulated as a machine learning problem as follows:")
print("\n1. Problem Type: Multi-stage ML approach combining unsupervised learning (clustering) and supervised learning (classification)")
print("\n2. Data Required:")
print("   - Customer demographics and purchase history")
print("   - Browsing and interaction data")
print("   - Past campaign response data")
print("\n3. Evaluation Methods:")
print("   - Business KPIs: Conversion rates, revenue, ROI")
print("   - Model performance metrics: Precision, recall, F1-score")
print("   - A/B testing against control groups")
print("\n4. Implementation Challenges:")
print("   - Data quality and integration")
print("   - Privacy and regulatory concerns")
print("   - Technical and business coordination")
print("\nEffectively implemented, this approach can significantly improve marketing effectiveness,")
print("customer satisfaction, and business outcomes.") 