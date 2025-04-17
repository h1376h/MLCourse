import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
import matplotlib.cm as cm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_1_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Introduction to Customer Clustering
print_step_header(1, "Introduction to Customer Clustering")

print("In this problem, we're using unsupervised learning to cluster customer transaction data")
print("to identify patterns of purchasing behavior without predefined categories.")
print()
print("We'll explore two main clustering approaches:")
print("1. K-means clustering: Partitions data into k clusters, each observation belongs to the cluster with the nearest mean")
print("2. Hierarchical clustering: Builds a tree of clusters, either by starting with individual items and")
print("   merging them (agglomerative) or by starting with all items and recursively dividing them (divisive)")
print()

# Step 2: Generate synthetic customer data
print_step_header(2, "Generating Synthetic Customer Data")

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers
n_customers = 800

# Create customer segments with different purchasing patterns
# Segment 1: High frequency, low value (200 customers)
freq_1 = np.random.normal(12, 2, 200)  # Purchases per year
amount_1 = np.random.normal(30, 10, 200)  # Average purchase amount
recency_1 = np.random.normal(15, 5, 200)  # Days since last purchase
category_diversity_1 = np.random.normal(3, 1, 200)  # Number of different categories
online_ratio_1 = np.random.beta(2, 8, 200)  # Ratio of online purchases

# Segment 2: Medium frequency, medium value (300 customers)
freq_2 = np.random.normal(6, 2, 300)
amount_2 = np.random.normal(80, 20, 300)
recency_2 = np.random.normal(30, 10, 300)
category_diversity_2 = np.random.normal(5, 1.5, 300)
online_ratio_2 = np.random.beta(5, 5, 300)

# Segment 3: Low frequency, high value (200 customers)
freq_3 = np.random.normal(2, 1, 200)
amount_3 = np.random.normal(200, 50, 200)
recency_3 = np.random.normal(60, 20, 200)
category_diversity_3 = np.random.normal(8, 2, 200)
online_ratio_3 = np.random.beta(8, 2, 200)

# Segment 4: Inactive/rare shoppers (100 customers)
freq_4 = np.random.normal(0.5, 0.3, 100)
amount_4 = np.random.normal(50, 30, 100)
recency_4 = np.random.normal(120, 30, 100)
category_diversity_4 = np.random.normal(1.5, 0.5, 100)
online_ratio_4 = np.random.beta(1, 9, 100)

# Combine all segments
purchase_frequency = np.concatenate([freq_1, freq_2, freq_3, freq_4])
average_purchase = np.concatenate([amount_1, amount_2, amount_3, amount_4])
recency = np.concatenate([recency_1, recency_2, recency_3, recency_4])
category_diversity = np.concatenate([category_diversity_1, category_diversity_2, category_diversity_3, category_diversity_4])
online_purchase_ratio = np.concatenate([online_ratio_1, online_ratio_2, online_ratio_3, online_ratio_4])

# Calculate annual spend (derived feature)
annual_spend = purchase_frequency * average_purchase

# Clean up any negative values
purchase_frequency = np.maximum(purchase_frequency, 0.1)
average_purchase = np.maximum(average_purchase, 5)
recency = np.maximum(recency, 1)
category_diversity = np.maximum(category_diversity, 1)
annual_spend = np.maximum(annual_spend, 5)

# Create a DataFrame
customer_data = pd.DataFrame({
    'PurchaseFrequency': purchase_frequency,
    'AveragePurchase': average_purchase,
    'Recency': recency,
    'CategoryDiversity': category_diversity,
    'OnlinePurchaseRatio': online_purchase_ratio,
    'AnnualSpend': annual_spend
})

# Display data summary
print("Generated synthetic customer transaction data with the following features:")
print("1. PurchaseFrequency: Number of purchases per year")
print("2. AveragePurchase: Average amount spent per purchase")
print("3. Recency: Days since last purchase")
print("4. CategoryDiversity: Number of different product categories purchased")
print("5. OnlinePurchaseRatio: Proportion of purchases made online vs. in-store")
print("6. AnnualSpend: Total annual spending (PurchaseFrequency * AveragePurchase)")
print()
print("Summary statistics of the data:")
print(customer_data.describe())
print()

# Step 3: Explore and visualize the customer data
print_step_header(3, "Exploratory Data Analysis")

# Visualize feature distributions
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

# Plot histograms for each feature
for i, feature in enumerate(customer_data.columns):
    sns.histplot(customer_data[feature], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {feature}', fontsize=14)
    axes[i].set_xlabel(feature, fontsize=12)
    axes[i].set_ylabel('Count', fontsize=12)

plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "feature_distributions.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Pairplot to visualize feature relationships
features_to_plot = ['PurchaseFrequency', 'AveragePurchase', 'Recency', 'CategoryDiversity', 'AnnualSpend']
sns.pairplot(customer_data[features_to_plot], diag_kind='kde')
plt.suptitle('Pairwise Relationships Between Customer Features', y=1.02, fontsize=16)
# Save the figure
file_path = os.path.join(save_dir, "feature_relationships.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Correlation matrix
correlation_matrix = customer_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Customer Features', fontsize=16)
# Save the figure
file_path = os.path.join(save_dir, "correlation_matrix.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Preprocessing - Scaling the data
print_step_header(4, "Data Preprocessing")

print("Since clustering algorithms are often distance-based, we need to scale the features")
print("to ensure that features with larger scales don't dominate the clustering.")
print()

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)
scaled_df = pd.DataFrame(scaled_data, columns=customer_data.columns)

print("Data after scaling (mean=0, std=1):")
print(scaled_df.describe().round(2))

# Step 5: K-means Clustering
print_step_header(5, "K-means Clustering")

# Determine the optimal number of clusters (Elbow Method)
inertia = []
silhouette_scores = []
calinski_scores = []
davies_bouldin_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    
    # Silhouette score (higher is better)
    silhouette = silhouette_score(scaled_data, kmeans.labels_)
    silhouette_scores.append(silhouette)
    
    # Calinski-Harabasz index (higher is better)
    calinski = calinski_harabasz_score(scaled_data, kmeans.labels_)
    calinski_scores.append(calinski)
    
    # Davies-Bouldin index (lower is better)
    davies = davies_bouldin_score(scaled_data, kmeans.labels_)
    davies_bouldin_scores.append(davies)
    
    print(f"For k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette:.2f}, "
          f"Calinski-Harabasz={calinski:.2f}, Davies-Bouldin={davies:.2f}")

# Plot the elbow method
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Method for Optimal k', fontsize=14)
plt.grid(True)
plt.xticks(k_range)

plt.subplot(2, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Score for Optimal k', fontsize=14)
plt.grid(True)
plt.xticks(k_range)

plt.subplot(2, 2, 3)
plt.plot(k_range, calinski_scores, 'go-')
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Calinski-Harabasz Score', fontsize=12)
plt.title('Calinski-Harabasz Score for Optimal k', fontsize=14)
plt.grid(True)
plt.xticks(k_range)

plt.subplot(2, 2, 4)
plt.plot(k_range, davies_bouldin_scores, 'mo-')
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Davies-Bouldin Score', fontsize=12)
plt.title('Davies-Bouldin Score for Optimal k', fontsize=14)
plt.grid(True)
plt.xticks(k_range)

plt.tight_layout()
plt.suptitle('Determining the Optimal Number of Clusters', fontsize=16, y=1.02)
# Save the figure
file_path = os.path.join(save_dir, "optimal_k_metrics.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Select the optimal number of clusters based on the metrics
optimal_k = 4  # Based on the elbow method and other metrics

# Apply k-means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original data
customer_data['KMeansCluster'] = kmeans_labels

# Visualize the clusters in 2D using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Create a scatter plot colored by cluster
plt.figure(figsize=(12, 10))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(scatter, label='Cluster')

# Add cluster centers (transformed to PCA space)
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, label='Cluster Centers')

plt.title('K-means Clustering Results (PCA Visualization)', fontsize=16)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=14)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "kmeans_clusters_pca.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Visualize cluster profiles
plt.figure(figsize=(14, 10))

# Get cluster means for each feature
cluster_means = customer_data.groupby('KMeansCluster').mean()

# Create a heatmap of cluster characteristics
sns.heatmap(cluster_means, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
plt.title('K-means Cluster Profiles', fontsize=16)
plt.ylabel('Cluster', fontsize=14)
plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "kmeans_cluster_profiles.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Create a radar chart to visualize cluster characteristics
def radar_chart(cluster_means, title):
    # Normalize the values for radar chart
    normalized_means = cluster_means.copy()
    for feature in normalized_means.columns:
        normalized_means[feature] = (normalized_means[feature] - normalized_means[feature].min()) / \
                                 (normalized_means[feature].max() - normalized_means[feature].min())
    
    # Set up the radar chart
    categories = normalized_means.columns
    N = len(categories)
    
    # Create angle for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    # Draw the cluster profiles
    for cluster in normalized_means.index:
        values = normalized_means.loc[cluster].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    plt.title(title, fontsize=16, y=1.08)
    
    return fig, ax

# Create radar chart for k-means clusters
fig, ax = radar_chart(cluster_means, 'K-means Cluster Characteristics')
# Save the figure
file_path = os.path.join(save_dir, "kmeans_radar_chart.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Hierarchical Clustering
print_step_header(6, "Hierarchical Clustering")

# Create a dendrogram for a subset of the data (for visualization purposes)
sample_indices = np.random.choice(len(scaled_data), 100, replace=False)
sample_data = scaled_data[sample_indices]

# Compute the linkage matrix
linkage_matrix = linkage(sample_data, method='ward')

# Plot the dendrogram
plt.figure(figsize=(18, 10))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram (Sample of 100 Customers)', fontsize=16)
plt.xlabel('Customer Index', fontsize=14)
plt.ylabel('Distance', fontsize=14)
plt.axhline(y=6, color='r', linestyle='--', label='Cut for 4 clusters')  # Example cut line
plt.legend()
plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "hierarchical_dendrogram.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Apply hierarchical clustering to the full dataset
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(scaled_data)

# Add hierarchical clustering labels to the data
customer_data['HierarchicalCluster'] = hierarchical_labels

# Visualize hierarchical clusters using PCA
plt.figure(figsize=(12, 10))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=hierarchical_labels, cmap='plasma', s=50, alpha=0.7)
plt.colorbar(scatter, label='Cluster')

plt.title('Hierarchical Clustering Results (PCA Visualization)', fontsize=16)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=14)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=14)
plt.grid(True)
plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "hierarchical_clusters_pca.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Visualize hierarchical cluster profiles
plt.figure(figsize=(14, 10))

# Get cluster means for each feature
hierarchical_cluster_means = customer_data.groupby('HierarchicalCluster').mean()

# Create a heatmap of cluster characteristics
sns.heatmap(hierarchical_cluster_means.drop(['KMeansCluster'], axis=1), annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
plt.title('Hierarchical Cluster Profiles', fontsize=16)
plt.ylabel('Cluster', fontsize=14)
plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "hierarchical_cluster_profiles.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Create radar chart for hierarchical clusters
fig, ax = radar_chart(hierarchical_cluster_means.drop(['KMeansCluster'], axis=1), 'Hierarchical Cluster Characteristics')
# Save the figure
file_path = os.path.join(save_dir, "hierarchical_radar_chart.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Compare K-means and Hierarchical Clustering
print_step_header(7, "Comparing K-means and Hierarchical Clustering")

# Create a confusion matrix to compare the cluster assignments
confusion_matrix = pd.crosstab(customer_data['KMeansCluster'], customer_data['HierarchicalCluster'])
confusion_matrix.columns.name = 'Hierarchical Clusters'
confusion_matrix.index.name = 'K-means Clusters'

print("Confusion Matrix between K-means and Hierarchical Clustering:")
print(confusion_matrix)
print()

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Comparison of K-means and Hierarchical Clustering Assignments', fontsize=16)
plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "clustering_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Create a more advanced 2D visualization using t-SNE
print("Applying t-SNE for better visualization of high-dimensional data...")
tsne = TSNE(n_components=2, random_state=42, perplexity=40)
tsne_result = tsne.fit_transform(scaled_data)

# Create a DataFrame with t-SNE results and cluster labels
tsne_df = pd.DataFrame({
    'tsne_1': tsne_result[:, 0],
    'tsne_2': tsne_result[:, 1],
    'kmeans_cluster': kmeans_labels,
    'hierarchical_cluster': hierarchical_labels
})

# Visualize both clustering results with t-SNE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# K-means
scatter1 = ax1.scatter(tsne_df['tsne_1'], tsne_df['tsne_2'], c=tsne_df['kmeans_cluster'], cmap='viridis', s=50, alpha=0.7)
ax1.set_title('K-means Clusters (t-SNE Visualization)', fontsize=16)
ax1.set_xlabel('t-SNE Dimension 1', fontsize=14)
ax1.set_ylabel('t-SNE Dimension 2', fontsize=14)
ax1.grid(True)
fig.colorbar(scatter1, ax=ax1, label='Cluster')

# Hierarchical
scatter2 = ax2.scatter(tsne_df['tsne_1'], tsne_df['tsne_2'], c=tsne_df['hierarchical_cluster'], cmap='plasma', s=50, alpha=0.7)
ax2.set_title('Hierarchical Clusters (t-SNE Visualization)', fontsize=16)
ax2.set_xlabel('t-SNE Dimension 1', fontsize=14)
ax2.set_ylabel('t-SNE Dimension 2', fontsize=14)
ax2.grid(True)
fig.colorbar(scatter2, ax=ax2, label='Cluster')

plt.suptitle('Comparison of Clustering Methods using t-SNE', fontsize=20, y=0.95)
plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "tsne_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Interpreting the Clusters and Business Applications
print_step_header(8, "Interpreting Clusters and Business Applications")

# Analyze and name the clusters based on their characteristics
cluster_descriptions = {
    0: "High-Frequency Low-Value Shoppers",
    1: "Medium-Frequency Medium-Value Shoppers",
    2: "Low-Frequency High-Value Shoppers",
    3: "Inactive/Rare Shoppers"
}

print("K-means Cluster Interpretations:")
for cluster_id, description in cluster_descriptions.items():
    cluster_data = customer_data[customer_data['KMeansCluster'] == cluster_id]
    cluster_size = len(cluster_data)
    cluster_percentage = (cluster_size / len(customer_data)) * 100
    print(f"Cluster {cluster_id} - {description}:")
    print(f"  Size: {cluster_size} customers ({cluster_percentage:.1f}% of total)")
    print(f"  Average Purchase Frequency: {cluster_data['PurchaseFrequency'].mean():.2f} purchases/year")
    print(f"  Average Purchase Amount: ${cluster_data['AveragePurchase'].mean():.2f}")
    print(f"  Average Annual Spend: ${cluster_data['AnnualSpend'].mean():.2f}")
    print(f"  Average Recency: {cluster_data['Recency'].mean():.1f} days")
    print(f"  Average Category Diversity: {cluster_data['CategoryDiversity'].mean():.1f} categories")
    print(f"  Average Online Purchase Ratio: {cluster_data['OnlinePurchaseRatio'].mean():.2f}")
    print()

# Create a visualization for business applications
business_actions = {
    "High-Frequency Low-Value Shoppers": [
        "Upsell to higher-value products",
        "Create bundle offers",
        "Loyalty program rewards",
        "Encourage category exploration"
    ],
    "Medium-Frequency Medium-Value Shoppers": [
        "Increase purchase frequency",
        "Personalized product recommendations",
        "Targeted promotions",
        "Channel-specific campaigns"
    ],
    "Low-Frequency High-Value Shoppers": [
        "Increase shopping frequency",
        "VIP treatment and services",
        "Early access to new products",
        "Premium customer service"
    ],
    "Inactive/Rare Shoppers": [
        "Re-engagement campaigns",
        "Win-back offers",
        "Surveys to understand churn reasons",
        "Special discounts for returning"
    ]
}

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for i, (cluster_name, actions) in enumerate(business_actions.items()):
    axes[i].axis('off')
    axes[i].text(0.5, 0.9, cluster_name, ha='center', fontsize=16, weight='bold')
    
    for j, action in enumerate(actions):
        axes[i].text(0.1, 0.8 - j*0.15, f"• {action}", fontsize=14)
    
    # Add a small chart representing the cluster profile
    cluster_id = list(cluster_descriptions.keys())[list(cluster_descriptions.values()).index(cluster_name)]
    cluster_data = customer_data[customer_data['KMeansCluster'] == cluster_id]
    
    # Add a small visualization inside each quadrant
    inner_ax = fig.add_axes([0.15 + (i%2)*0.5, 0.12 + (1 - i//2)*0.5, 0.2, 0.2])
    
    # Create a small bar chart of key metrics
    metrics = ['PurchaseFrequency', 'AveragePurchase', 'Recency']
    values = [
        cluster_data['PurchaseFrequency'].mean() / customer_data['PurchaseFrequency'].max(),
        cluster_data['AveragePurchase'].mean() / customer_data['AveragePurchase'].max(),
        1 - (cluster_data['Recency'].mean() / customer_data['Recency'].max())  # Inverted for recency
    ]
    
    inner_ax.bar(metrics, values, color=['blue', 'green', 'red'], alpha=0.7)
    inner_ax.set_ylim(0, 1)
    inner_ax.set_title('Relative Metrics', fontsize=10)
    inner_ax.tick_params(axis='x', labelsize=8, rotation=45)
    inner_ax.tick_params(axis='y', labelsize=8)

plt.suptitle('Business Applications for Customer Segments', fontsize=20)
# Save the figure
file_path = os.path.join(save_dir, "business_applications.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 9: Summary and Key Points
print_step_header(9, "Summary and Key Points")

print("Key Differences Between K-means and Hierarchical Clustering:")
print()
print("K-means Clustering:")
print("- Pros:")
print("  • Scalable to large datasets")
print("  • Simple to understand and implement")
print("  • Works well with globular clusters")
print("- Cons:")
print("  • Requires pre-specifying the number of clusters (k)")
print("  • Sensitive to initial centroid placement")
print("  • Struggles with non-globular cluster shapes")
print("  • Not deterministic (results can vary across runs)")
print()
print("Hierarchical Clustering:")
print("- Pros:")
print("  • No need to specify the number of clusters beforehand")
print("  • Produces a dendrogram that shows the hierarchical structure")
print("  • Deterministic results (same result each time)")
print("  • Can handle various cluster shapes")
print("- Cons:")
print("  • Computationally intensive for large datasets (O(n²) or O(n³))")
print("  • Can be difficult to determine where to 'cut' the dendrogram")
print("  • Sensitive to noise and outliers")
print()
print("Challenges in Determining the Optimal Number of Clusters:")
print("1. Subjective interpretation of clustering metrics (elbow method, silhouette score)")
print("2. Balance between simplicity (fewer clusters) and detail (more clusters)")
print("3. Business interpretability vs. statistical optimization")
print("4. Different metrics may suggest different optimal k values")
print("5. Domain knowledge often needed for final determination")
print()
print("Methods for Evaluating Cluster Quality Without Ground Truth Labels:")
print("1. Silhouette Score: Measures how similar an object is to its own cluster vs. other clusters")
print("2. Calinski-Harabasz Index: Ratio of between-cluster variance to within-cluster variance")
print("3. Davies-Bouldin Index: Average similarity between clusters (lower is better)")
print("4. Inertia (Within-cluster sum of squares): Measures compactness of clusters")
print("5. Visual inspection using dimension reduction techniques (PCA, t-SNE)")
print("6. Business interpretability and actionability of resulting segments")
print()
print("Real-World Business Applications of Customer Clustering:")
print("1. Personalized Marketing: Tailoring campaigns to specific customer segments")
print("2. Product Recommendations: Suggesting products based on segment preferences")
print("3. Pricing Strategy: Differentiating pricing based on customer value segments")
print("4. Customer Retention: Targeting at-risk segments with retention campaigns")
print("5. Service Differentiation: Providing different service levels to different segments")
print("6. Inventory Management: Stocking products preferred by dominant customer segments")
print("7. Store Layout: Arranging products based on purchasing patterns of key segments")
print("8. New Product Development: Identifying unmet needs in specific segments")
print()
print("Conclusion: Unsupervised clustering provides valuable insights into customer purchasing")
print("behavior, allowing businesses to develop more targeted marketing strategies and improve")
print("customer satisfaction through personalized approaches.") 