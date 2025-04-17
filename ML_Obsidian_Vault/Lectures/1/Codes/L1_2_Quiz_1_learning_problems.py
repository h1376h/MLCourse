import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_blobs, make_regression
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random
from matplotlib.colors import ListedColormap

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_2_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Introduction to Learning Problems
print_step_header(1, "Introduction to Learning Problems")

print("Machine learning problems can be categorized into four main types:")
print("1. Classification: Predicting a categorical label/class")
print("2. Regression: Predicting a continuous value")
print("3. Clustering: Finding natural groupings in data without predefined labels")
print("4. Dimensionality Reduction: Reducing the number of variables while preserving information")
print("\nLet's analyze each scenario from Question 1 and identify the type of learning problem.")

# Step 2: Regression - Aircraft Engine Remaining Useful Life
print_step_header(2, "Scenario 1: Aircraft Engine Remaining Useful Life")

print("Scenario: Predicting the remaining useful life of aircraft engines based on sensor readings")
print("\nAnalysis:")
print("- The target variable (remaining useful life) is a continuous value")
print("- We want to predict a numerical value based on input features")
print("- This is a REGRESSION problem")
print("\nFormulation:")
print("- Input Features (X): Sensor readings (temperature, pressure, vibration, etc.)")
print("- Output Variable (y): Remaining useful life in hours/cycles")
print("- Goal: Minimize the prediction error between actual and predicted remaining useful life")

# Create a visualization for regression problem
np.random.seed(42)
# Simulate engine cycles and remaining useful life with some noise
engine_cycles = np.sort(np.random.randint(0, 300, 100))
base_rul = 500 - 1.5 * engine_cycles
sensor_1 = 100 + 0.5 * engine_cycles + np.random.normal(0, 15, 100)
sensor_2 = 50 - 0.3 * engine_cycles + np.random.normal(0, 10, 100)
remaining_life = base_rul + np.random.normal(0, 30, 100)

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sensor readings over engine cycles
axs[0, 0].plot(engine_cycles, sensor_1, 'b-', alpha=0.7, label='Sensor 1 (Temperature)')
axs[0, 0].plot(engine_cycles, sensor_2, 'g-', alpha=0.7, label='Sensor 2 (Pressure)')
axs[0, 0].set_title('Sensor Readings vs. Engine Cycles', fontsize=12)
axs[0, 0].set_xlabel('Engine Cycles', fontsize=10)
axs[0, 0].set_ylabel('Sensor Values', fontsize=10)
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot 2: Remaining useful life vs. engine cycles (ground truth)
axs[0, 1].scatter(engine_cycles, remaining_life, color='red', alpha=0.6)
axs[0, 1].plot(engine_cycles, base_rul, 'k--', label='Ideal Degradation')
axs[0, 1].set_title('Remaining Useful Life vs. Engine Cycles', fontsize=12)
axs[0, 1].set_xlabel('Engine Cycles', fontsize=10)
axs[0, 1].set_ylabel('Remaining Useful Life (hours)', fontsize=10)
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot 3: Regression Model for RUL Prediction
# Create a simple regression model using sensor data
X = np.column_stack((engine_cycles, sensor_1, sensor_2))
X_train, X_test, y_train, y_test = train_test_split(X, remaining_life, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

axs[1, 0].scatter(y_test, y_pred, color='blue', alpha=0.6)
axs[1, 0].plot([0, 500], [0, 500], 'r--')  # Perfect prediction line
axs[1, 0].set_title('Regression Model: Actual vs. Predicted RUL', fontsize=12)
axs[1, 0].set_xlabel('Actual Remaining Useful Life', fontsize=10)
axs[1, 0].set_ylabel('Predicted Remaining Useful Life', fontsize=10)
axs[1, 0].grid(True)

# Plot 4: Feature Importance
feature_names = ['Engine Cycles', 'Temperature', 'Pressure']
coefficients = model.coef_
importance = np.abs(coefficients)
sorted_idx = np.argsort(importance)

axs[1, 1].barh([feature_names[i] for i in sorted_idx], importance[sorted_idx], color='teal')
axs[1, 1].set_title('Feature Importance for RUL Prediction', fontsize=12)
axs[1, 1].set_xlabel('Absolute Coefficient Value', fontsize=10)
axs[1, 1].grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "aircraft_engine_regression.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Classification - News Article Categorization
print_step_header(3, "Scenario 2: News Article Categorization")

print("Scenario: Categorizing news articles into topics like politics, sports, entertainment, etc.")
print("\nAnalysis:")
print("- The target variable is a category/class (politics, sports, entertainment)")
print("- We want to assign each article to one of several predefined categories")
print("- This is a CLASSIFICATION problem")
print("\nFormulation:")
print("- Input Features (X): Text content of news articles (processed into numerical features)")
print("- Output Variable (y): Category labels (politics, sports, entertainment, etc.)")
print("- Goal: Maximize the accuracy of category predictions")

# Create a visualization for classification problem
# Simulate some news articles data
np.random.seed(42)
categories = ['Politics', 'Sports', 'Entertainment', 'Technology']
category_colors = ['blue', 'green', 'red', 'purple']

# Sample headlines for each category
politics_headlines = [
    "President Signs New Bill Into Law",
    "Senate Debates Foreign Policy",
    "Election Campaign Enters Final Phase",
    "Government Announces Budget Plans",
    "Political Tensions Rise in Border Dispute"
]

sports_headlines = [
    "Local Team Wins Championship",
    "Olympic Gold for National Swimmer",
    "Basketball Star Signs Million Dollar Contract",
    "World Cup Final Set for Sunday",
    "Tennis Player Advances to Semi-finals"
]

entertainment_headlines = [
    "New Movie Breaks Box Office Records",
    "Celebrity Couple Announces Engagement",
    "Music Awards Ceremony Highlights",
    "TV Series Renewed for Another Season",
    "Famous Actor Joins Upcoming Film Project"
]

technology_headlines = [
    "Tech Company Launches New Smartphone",
    "AI Innovation Revolutionizes Industry",
    "Security Breach at Social Media Giant",
    "Quantum Computing Breakthrough Announced",
    "Electric Car Maker Expands Production"
]

all_headlines = politics_headlines + sports_headlines + entertainment_headlines + technology_headlines
all_labels = ([0] * len(politics_headlines) + [1] * len(sports_headlines) + 
             [2] * len(entertainment_headlines) + [3] * len(technology_headlines))

# Create document term matrix
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(all_headlines)
words = vectorizer.get_feature_names_out()

# Reduce to 2D for visualization
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X.toarray())

# Create figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Articles in 2D feature space
for i, category in enumerate(categories):
    indices = [j for j, label in enumerate(all_labels) if label == i]
    axs[0, 0].scatter(X_2d[indices, 0], X_2d[indices, 1], color=category_colors[i], 
                     label=category, alpha=0.7)
axs[0, 0].set_title('News Articles in 2D Feature Space', fontsize=12)
axs[0, 0].set_xlabel('Principal Component 1', fontsize=10)
axs[0, 0].set_ylabel('Principal Component 2', fontsize=10)
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot 2: Decision boundaries (simplified)
# Create a mesh grid
h = 0.02
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Just for visualization, use a simple decision boundary
Z = np.zeros(xx.shape)
for i in range(len(X_2d)):
    Z += np.exp(-((xx - X_2d[i, 0])**2 + (yy - X_2d[i, 1])**2) / 0.5) * all_labels[i]
Z = np.round(Z / np.max(Z) * 3)

# Plot the decision boundary
axs[0, 1].contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(category_colors))
for i, category in enumerate(categories):
    indices = [j for j, label in enumerate(all_labels) if label == i]
    axs[0, 1].scatter(X_2d[indices, 0], X_2d[indices, 1], color=category_colors[i], 
                     label=category, alpha=0.7)
axs[0, 1].set_title('Classification Decision Boundaries', fontsize=12)
axs[0, 1].set_xlabel('Principal Component 1', fontsize=10)
axs[0, 1].set_ylabel('Principal Component 2', fontsize=10)
axs[0, 1].grid(True)

# Plot 3: Word cloud representation (simplified)
top_words_per_category = []
for category_idx in range(len(categories)):
    # Filter documents in this category
    category_docs = [i for i, label in enumerate(all_labels) if label == category_idx]
    
    # Sum the word frequencies across documents in this category
    category_word_freq = np.sum(X[category_docs].toarray(), axis=0)
    
    # Get the top 5 words
    top_indices = np.argsort(category_word_freq)[-5:]
    top_words = [(words[i], category_word_freq[i]) for i in top_indices if category_word_freq[i] > 0]
    top_words_per_category.append(top_words)

# Create a bar chart of top words for each category
for i, category in enumerate(categories):
    y_pos = np.arange(len(top_words_per_category[i]))
    word_freq = [freq for _, freq in top_words_per_category[i]]
    word_labels = [word for word, _ in top_words_per_category[i]]
    
    axs[1, 0].barh(y_pos + i*0.25, word_freq, height=0.2, color=category_colors[i], label=category)
    for j, (word, freq) in enumerate(top_words_per_category[i]):
        axs[1, 0].text(freq + 0.05, y_pos[j] + i*0.25, word, fontsize=8)

axs[1, 0].set_title('Top Words per Category', fontsize=12)
axs[1, 0].set_xlabel('Word Frequency', fontsize=10)
axs[1, 0].set_yticks([])
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot 4: Confusion matrix simulation
# Simulate a confusion matrix for classification results
confusion = np.array([
    [4, 0, 1, 0],  # Politics
    [0, 5, 0, 0],  # Sports
    [1, 0, 4, 0],  # Entertainment
    [0, 0, 0, 5]   # Technology
])

sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=categories, 
           yticklabels=categories, ax=axs[1, 1])
axs[1, 1].set_title('Confusion Matrix (Simulated)', fontsize=12)
axs[1, 1].set_xlabel('Predicted Category', fontsize=10)
axs[1, 1].set_ylabel('True Category', fontsize=10)

plt.tight_layout()
file_path = os.path.join(save_dir, "news_classification.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Clustering - Customer Profiles
print_step_header(4, "Scenario 3: Customer Profile Clustering")

print("Scenario: Identifying groups of similar customer profiles from demographic and purchasing data")
print("\nAnalysis:")
print("- There are no predefined classes or labels")
print("- We want to discover natural groupings in the data")
print("- This is a CLUSTERING problem")
print("\nFormulation:")
print("- Input Features (X): Customer demographic data and purchasing behavior")
print("- Output: Cluster assignments (customer segments)")
print("- Goal: Find meaningful customer segments with high intra-cluster similarity and low inter-cluster similarity")

# Create a visualization for clustering problem
np.random.seed(42)

# Simulate customer data
n_customers = 200
age = np.random.randint(18, 80, n_customers)
income = 20000 + age * 1000 + np.random.normal(0, 15000, n_customers)
purchase_frequency = np.random.randint(1, 50, n_customers)
avg_purchase_value = 10 + 0.01 * income + np.random.normal(0, 50, n_customers)

# Create customer dataframe
customer_data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Purchase_Frequency': purchase_frequency,
    'Avg_Purchase_Value': avg_purchase_value
})

# Standardize the data
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)

# Apply K-means clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(customer_data_scaled)
customer_data['Cluster'] = clusters

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Income vs. Age with clusters
colors = ['blue', 'green', 'red', 'purple']
for i in range(n_clusters):
    cluster_data = customer_data[customer_data['Cluster'] == i]
    axs[0, 0].scatter(cluster_data['Age'], cluster_data['Income'], 
                     color=colors[i], alpha=0.6, label=f'Cluster {i+1}')

axs[0, 0].set_title('Customer Clusters: Income vs. Age', fontsize=12)
axs[0, 0].set_xlabel('Age', fontsize=10)
axs[0, 0].set_ylabel('Income ($)', fontsize=10)
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot 2: Purchase Frequency vs. Average Purchase Value with clusters
for i in range(n_clusters):
    cluster_data = customer_data[customer_data['Cluster'] == i]
    axs[0, 1].scatter(cluster_data['Purchase_Frequency'], cluster_data['Avg_Purchase_Value'], 
                     color=colors[i], alpha=0.6, label=f'Cluster {i+1}')

axs[0, 1].set_title('Customer Clusters: Purchase Behavior', fontsize=12)
axs[0, 1].set_xlabel('Purchase Frequency (per year)', fontsize=10)
axs[0, 1].set_ylabel('Average Purchase Value ($)', fontsize=10)
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot 3: Parallel coordinates for all features
from pandas.plotting import parallel_coordinates

# Sample subset for parallel coordinates
sample_data = customer_data.sample(50, random_state=42)
parallel_coordinates(sample_data, 'Cluster', color=colors, ax=axs[1, 0])
axs[1, 0].set_title('Parallel Coordinates Plot of Customer Features', fontsize=12)
axs[1, 0].grid(True)
axs[1, 0].legend_.remove()  # Remove the default legend which is too large

# Plot 4: Profile of cluster centers
cluster_centers = kmeans.cluster_centers_
cluster_centers_df = pd.DataFrame(
    scaler.inverse_transform(cluster_centers),
    columns=['Age', 'Income', 'Purchase_Frequency', 'Avg_Purchase_Value']
)

# Round the values for display
cluster_centers_df = cluster_centers_df.round(0)

# Plot 4: Bar chart of cluster centers
cluster_centers_df_melted = pd.melt(
    cluster_centers_df.reset_index(), 
    id_vars='index', 
    value_vars=['Age', 'Income', 'Purchase_Frequency', 'Avg_Purchase_Value'],
    var_name='Feature', 
    value_name='Value'
)

# Create a grouped bar chart
for i, feature in enumerate(['Age', 'Income', 'Purchase_Frequency', 'Avg_Purchase_Value']):
    feature_data = cluster_centers_df_melted[cluster_centers_df_melted['Feature'] == feature]
    x_positions = np.arange(n_clusters) + i * 0.2
    axs[1, 1].bar(x_positions, feature_data['Value'] / feature_data['Value'].max(), width=0.15, 
                label=feature, alpha=0.7)

axs[1, 1].set_title('Normalized Cluster Center Values', fontsize=12)
axs[1, 1].set_xlabel('Cluster', fontsize=10)
axs[1, 1].set_ylabel('Normalized Value', fontsize=10)
axs[1, 1].set_xticks(np.arange(n_clusters) + 0.3)
axs[1, 1].set_xticklabels([f'Cluster {i+1}' for i in range(n_clusters)])
axs[1, 1].legend(title='Feature')
axs[1, 1].grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "customer_clustering.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Dimensionality Reduction - Gene Expression Data
print_step_header(5, "Scenario 4: Gene Expression Dimensionality Reduction")

print("Scenario: Reducing the number of features in a high-dimensional gene expression dataset while preserving the most important patterns")
print("\nAnalysis:")
print("- We want to reduce the dimensionality of the data")
print("- The goal is to find a lower-dimensional representation that preserves the important information")
print("- This is a DIMENSIONALITY REDUCTION problem")
print("\nFormulation:")
print("- Input Features (X): High-dimensional gene expression data (thousands of genes)")
print("- Output: Lower-dimensional representation of the data")
print("- Goal: Preserve the most important patterns/variance in the data with fewer dimensions")

# Create a visualization for dimensionality reduction problem
np.random.seed(42)

# Simulate gene expression data
n_samples = 100  # Number of patient samples
n_genes = 500     # Number of genes (features)
n_components = 2  # Number of underlying components (for simulation)

# Create 2 underlying components that represent real biological processes
component1 = np.random.normal(0, 1, n_samples)  # e.g., immune response
component2 = np.random.normal(0, 1, n_samples)  # e.g., cell cycle activity

# Create gene expression matrix by combining these components
gene_expr = np.zeros((n_samples, n_genes))
for i in range(n_genes):
    # Each gene is affected by the underlying components to varying degrees
    w1 = np.random.normal(0, 1)
    w2 = np.random.normal(0, 1)
    noise = np.random.normal(0, 0.5, n_samples)
    gene_expr[:, i] = w1 * component1 + w2 * component2 + noise

# Add disease labels for visualization (simulating cancer vs control)
is_cancer = (component1 > 0.5) | (component2 > 1)

# Apply PCA for dimensionality reduction
pca = PCA()
pca_result = pca.fit_transform(gene_expr)

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Heatmap of gene expression data (subset)
subset_expr = gene_expr[:10, :10]
sns.heatmap(subset_expr, cmap="YlGnBu", ax=axs[0, 0])
axs[0, 0].set_title('Gene Expression Data (Subset)', fontsize=12)
axs[0, 0].set_xlabel('Genes', fontsize=10)
axs[0, 0].set_ylabel('Samples', fontsize=10)

# Plot 2: PCA Scatterplot
axs[0, 1].scatter(pca_result[~is_cancer, 0], pca_result[~is_cancer, 1], color='blue', alpha=0.6, label='Control')
axs[0, 1].scatter(pca_result[is_cancer, 0], pca_result[is_cancer, 1], color='red', alpha=0.6, label='Cancer')
axs[0, 1].set_title('PCA: First Two Principal Components', fontsize=12)
axs[0, 1].set_xlabel('Principal Component 1', fontsize=10)
axs[0, 1].set_ylabel('Principal Component 2', fontsize=10)
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot 3: Explained variance ratio
explained_var = pca.explained_variance_ratio_
cum_explained_var = np.cumsum(explained_var)

axs[1, 0].bar(range(1, 11), explained_var[:10], alpha=0.7, label='Individual')
axs[1, 0].step(range(1, 11), cum_explained_var[:10], where='mid', color='red', alpha=0.7, label='Cumulative')
axs[1, 0].axhline(y=0.8, color='black', linestyle='--', alpha=0.5, label='80% Threshold')
axs[1, 0].set_title('Explained Variance by Principal Components', fontsize=12)
axs[1, 0].set_xlabel('Principal Components', fontsize=10)
axs[1, 0].set_ylabel('Explained Variance Ratio', fontsize=10)
axs[1, 0].set_xticks(range(1, 11))
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot 4: Feature loadings (how genes contribute to PCs)
loadings = pca.components_
n_top_genes = 10
top_genes_pc1 = np.abs(loadings[0]).argsort()[-n_top_genes:][::-1]
top_genes_pc2 = np.abs(loadings[1]).argsort()[-n_top_genes:][::-1]

axs[1, 1].bar(range(n_top_genes), loadings[0][top_genes_pc1], color='blue', alpha=0.7)
axs[1, 1].set_title('Top Genes Contributing to PC1', fontsize=12)
axs[1, 1].set_xlabel('Top Genes (Feature Index)', fontsize=10)
axs[1, 1].set_ylabel('Loading Value', fontsize=10)
axs[1, 1].set_xticks(range(n_top_genes))
axs[1, 1].set_xticklabels([f'Gene {i}' for i in top_genes_pc1], rotation=45)
axs[1, 1].grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "gene_expression_dimensionality_reduction.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Summary
print_step_header(6, "Summary of Learning Problem Types")

print("1. Regression Problem (Aircraft Engine RUL):")
print("   - Input Features: Sensor readings (temperature, pressure, vibration, etc.)")
print("   - Output Variable: Remaining useful life (continuous value)")
print("   - Examples: Linear regression, Random Forest Regressor, Neural Networks")
print()
print("2. Classification Problem (News Article Categorization):")
print("   - Input Features: Text content (processed into numerical features)")
print("   - Output Variable: Category labels (politics, sports, entertainment, etc.)")
print("   - Examples: Logistic Regression, Decision Trees, Naive Bayes, SVM")
print()
print("3. Clustering Problem (Customer Profiles):")
print("   - Input Features: Customer demographic and purchasing data")
print("   - Output: Cluster assignments (customer segments)")
print("   - Examples: K-means, Hierarchical Clustering, DBSCAN")
print()
print("4. Dimensionality Reduction Problem (Gene Expression Data):")
print("   - Input Features: High-dimensional gene expression data")
print("   - Output: Lower-dimensional representation")
print("   - Examples: PCA, t-SNE, Autoencoders")
print()
print("Each type of learning problem requires different approaches, algorithms, and evaluation metrics.")
print("The formulation of the problem guides the choice of methods and dictates how we measure success.") 