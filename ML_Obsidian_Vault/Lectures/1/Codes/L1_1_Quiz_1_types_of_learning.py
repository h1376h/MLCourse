import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from matplotlib.gridspec import GridSpec
import networkx as nx
from scipy.stats import norm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_1_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Introduction to Types of Machine Learning
print_step_header(1, "Introduction to Types of Machine Learning")

print("Machine Learning can be broadly categorized into four main types:")
print("1. Supervised Learning")
print("2. Unsupervised Learning")
print("3. Reinforcement Learning")
print("4. Semi-Supervised Learning")
print()

# Step 2: Create a figure showing the four types of ML
print_step_header(2, "Visual Representation of ML Types")

# Create a figure showing the four types of ML
plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

# 1. Supervised Learning - Visualization with labeled data points
ax1 = plt.subplot(gs[0, 0])
# Generate data for supervised learning visualization
np.random.seed(42)
X_supervised = np.random.randn(40, 2)
y_supervised = (X_supervised[:, 0] + X_supervised[:, 1] > 0).astype(int)

# Plot supervised learning data
ax1.scatter(X_supervised[y_supervised==0, 0], X_supervised[y_supervised==0, 1], color='blue', label='Class 0')
ax1.scatter(X_supervised[y_supervised==1, 0], X_supervised[y_supervised==1, 1], color='red', label='Class 1')

# Add decision boundary
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = (xx + yy > 0).astype(int)
ax1.contour(xx, yy, Z, colors='green', levels=[0.5], alpha=0.5, linestyles='--')

ax1.set_title('Supervised Learning', fontsize=14)
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
ax1.legend()
ax1.text(-2.8, 2.5, "Features have labels\nModel learns the mapping", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# 2. Unsupervised Learning - Clustering visualization
ax2 = plt.subplot(gs[0, 1])
# Generate data for unsupervised learning visualization
X_unsupervised, _ = make_blobs(n_samples=100, centers=3, random_state=42)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_unsupervised)

# Plot unsupervised learning data
ax2.scatter(X_unsupervised[clusters==0, 0], X_unsupervised[clusters==0, 1], color='purple', label='Cluster 1')
ax2.scatter(X_unsupervised[clusters==1, 0], X_unsupervised[clusters==1, 1], color='orange', label='Cluster 2')
ax2.scatter(X_unsupervised[clusters==2, 0], X_unsupervised[clusters==2, 1], color='green', label='Cluster 3')

# Add cluster centers
ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker='X', c='black', label='Centroids')

ax2.set_title('Unsupervised Learning (Clustering)', fontsize=14)
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.legend()
ax2.text(-7, 5, "No labels provided\nModel finds patterns", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# 3. Reinforcement Learning - Agent environment interaction
ax3 = plt.subplot(gs[1, 0])

# Create a simple grid world
G = nx.grid_2d_graph(5, 5)
pos = {(x, y): (y, -x) for x, y in G.nodes()}

# Define start, goal, and obstacle states
start_node = (0, 0)
goal_node = (4, 4)
obstacles = [(1, 1), (1, 2), (2, 1), (3, 3)]

# Remove obstacle nodes
for obstacle in obstacles:
    if obstacle in G:
        G.remove_node(obstacle)

# Define a simple policy (just for visualization)
policy = {}
for node in G.nodes():
    x, y = node
    if node == goal_node:
        policy[node] = "G"  # Goal
    elif x < 4 and (x+1, y) in G.nodes():
        policy[node] = "→"
    elif y < 4 and (x, y+1) in G.nodes():
        policy[node] = "↑"
    else:
        policy[node] = "?"

# Draw the grid world
nx.draw(G, pos, node_color='lightblue', with_labels=False, node_size=700, ax=ax3)

# Add labels for nodes
node_labels = {node: policy[node] for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=14, ax=ax3)

# Mark start, goal, and obstacles
nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='green', node_size=700, ax=ax3)
nx.draw_networkx_nodes(G, pos, nodelist=[goal_node], node_color='red', node_size=700, ax=ax3)

ax3.set_title('Reinforcement Learning', fontsize=14)
ax3.text(2, -5.5, "Agent learns through\ninteraction with environment", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
ax3.text(-0.5, 0, "S", fontsize=14, ha='center', va='center')
ax3.text(4, -4, "G", fontsize=14, ha='center', va='center')

# Add legend-like annotations
ax3.text(-1, -1.5, "Start", fontsize=10, color='green', ha='center')
ax3.text(-1, -2, "Goal", fontsize=10, color='red', ha='center')
ax3.text(-1, -2.5, "Policy", fontsize=10, color='blue', ha='center')

# 4. Semi-Supervised Learning - Mix of labeled and unlabeled data
ax4 = plt.subplot(gs[1, 1])

# Generate data for semi-supervised learning visualization
np.random.seed(42)
X_semi = np.random.randn(100, 2)
# Create true labels (just for generating the data)
true_labels = (X_semi[:, 0] + X_semi[:, 1] > 0).astype(int)

# Only a small portion is labeled (10%)
labeled_indices = np.random.choice(range(100), 10, replace=False)
unlabeled_indices = list(set(range(100)) - set(labeled_indices))

# Plot data points
ax4.scatter(X_semi[unlabeled_indices, 0], X_semi[unlabeled_indices, 1], color='gray', alpha=0.5, label='Unlabeled')
ax4.scatter(X_semi[labeled_indices][true_labels[labeled_indices]==0, 0], 
            X_semi[labeled_indices][true_labels[labeled_indices]==0, 1], 
            color='blue', label='Labeled Class 0')
ax4.scatter(X_semi[labeled_indices][true_labels[labeled_indices]==1, 0], 
            X_semi[labeled_indices][true_labels[labeled_indices]==1, 1], 
            color='red', label='Labeled Class 1')

ax4.set_title('Semi-Supervised Learning', fontsize=14)
ax4.set_xlabel('Feature 1', fontsize=12)
ax4.set_ylabel('Feature 2', fontsize=12)
ax4.legend()
ax4.text(-2.8, 2.5, "Few labeled examples\nMany unlabeled examples", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "ml_types_overview.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Analyze each scenario from Question 1
print_step_header(3, "Analysis of Machine Learning Scenarios")

# List the scenarios from Question 1
scenarios = [
    {
        'id': 1,
        'description': 'A robot learning to navigate through a maze by receiving rewards when it makes progress toward the exit and penalties when it hits walls',
        'learning_type': 'Reinforcement Learning',
        'explanation': 'This is a classic reinforcement learning scenario where the agent (robot) takes actions (movements) in an environment (maze) and receives feedback in the form of rewards and penalties. The robot learns to maximize rewards over time by exploring the environment and adjusting its policy.'
    },
    {
        'id': 2,
        'description': 'Grouping customers into different segments based on their purchasing behavior without any predefined categories',
        'learning_type': 'Unsupervised Learning',
        'explanation': 'This is unsupervised learning because we\'re finding patterns in data without labeled examples. Specifically, this is a clustering task where we group similar customers together based on their purchasing behavior without having predefined categories or labels.'
    },
    {
        'id': 3,
        'description': 'Training a system to predict housing prices based on features like square footage, number of bedrooms, and location using historical sales data',
        'learning_type': 'Supervised Learning',
        'explanation': 'This is supervised learning because we have labeled training data (historical sales with known prices) and a clear target variable (price) to predict. The algorithm learns the relationship between input features and the target variable.'
    },
    {
        'id': 4,
        'description': 'Training a spam detection system using a small set of labeled emails (spam/not spam) and a large set of unlabeled emails',
        'learning_type': 'Semi-Supervised Learning',
        'explanation': 'This is semi-supervised learning because we have a combination of labeled data (the small set of emails marked as spam/not spam) and unlabeled data (the large set of unlabeled emails). The algorithm can use both to improve its performance.'
    }
]

# Display the analysis of each scenario
for scenario in scenarios:
    print(f"Scenario {scenario['id']}:")
    print(f"Description: {scenario['description']}")
    print(f"Learning Type: {scenario['learning_type']}")
    print(f"Explanation: {scenario['explanation']}")
    print()

# Step 4: Create individual visualizations for each scenario
print_step_header(4, "Detailed Visualizations for Each Scenario")

# Scenario 1: Reinforcement Learning - Robot in a maze
plt.figure(figsize=(10, 8))

# Create a maze grid
maze_size = 10
maze = np.ones((maze_size, maze_size))

# Set start and goal positions
start_pos = (0, 0)
goal_pos = (maze_size-1, maze_size-1)

# Create some walls/obstacles in the maze
obstacles = [
    (1, 1), (1, 2), (1, 3), 
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
    (5, 5), (5, 6), (5, 7), (5, 8),
    (7, 1), (7, 2), (7, 3), (7, 4)
]

# Mark the paths and obstacles
maze.fill(0)  # All cells are paths (0)
for obstacle in obstacles:
    maze[obstacle] = 1  # Mark obstacles as 1

# Create a Q-table (state-action values) for visualization
# This is simplified and would be learned through reinforcement learning
q_values = np.zeros((maze_size, maze_size, 4))  # 4 actions: up, right, down, left

# Define a simple policy (just for visualization)
# Higher Q-values for actions leading toward the goal
for i in range(maze_size):
    for j in range(maze_size):
        if maze[i, j] == 1:  # If obstacle, skip
            continue
        
        # Value increases as we get closer to the goal
        distance_to_goal = abs(i - (maze_size-1)) + abs(j - (maze_size-1))
        value = max(0, 10 - distance_to_goal)
        
        # Set Q-values according to distance to goal
        if i > 0 and maze[i-1, j] == 0:  # Up is valid
            q_values[i, j, 0] = value if i > (maze_size-1) else value * 0.8
        if j < maze_size-1 and maze[i, j+1] == 0:  # Right is valid
            q_values[i, j, 1] = value
        if i < maze_size-1 and maze[i+1, j] == 0:  # Down is valid
            q_values[i, j, 2] = value if i < (maze_size-1) else value * 0.8
        if j > 0 and maze[i, j-1] == 0:  # Left is valid
            q_values[i, j, 3] = value if j > (maze_size-1) else value * 0.8

# Plot the maze and policy
plt.imshow(maze, cmap='binary')

# Add arrows to show the policy (direction with highest Q-value)
for i in range(maze_size):
    for j in range(maze_size):
        if maze[i, j] == 1:  # Skip obstacles
            continue
        
        # Get the action with the highest Q-value
        best_action = np.argmax(q_values[i, j])
        
        # Draw arrows based on the best action
        if q_values[i, j, best_action] > 0:
            dx, dy = 0, 0
            if best_action == 0:  # Up
                dx, dy = 0, -0.3
            elif best_action == 1:  # Right
                dx, dy = 0.3, 0
            elif best_action == 2:  # Down
                dx, dy = 0, 0.3
            elif best_action == 3:  # Left
                dx, dy = -0.3, 0
            
            plt.arrow(j, i, dy, dx, head_width=0.2, head_length=0.2, fc='blue', ec='blue')

# Mark the start and goal positions
plt.scatter(start_pos[1], start_pos[0], color='green', s=100, marker='o', label='Start')
plt.scatter(goal_pos[1], goal_pos[0], color='red', s=100, marker='*', label='Goal')

# Add labels and title
plt.title('Scenario 1: Reinforcement Learning - Robot in a Maze', fontsize=14)
plt.xlabel('Column', fontsize=12)
plt.ylabel('Row', fontsize=12)
plt.xticks(range(maze_size))
plt.yticks(range(maze_size))
plt.grid(False)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

# Add explanatory text
text_x = 1
text_y = -1.5
plt.figtext(0.5, 0.01, "The robot (agent) learns to navigate through the maze by receiving rewards\n"
                      "when it makes progress toward the exit (goal) and penalties when it hits walls.\n"
                      "Through exploration and exploitation, it determines the optimal policy (arrows).", 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "scenario1_reinforcement_learning.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Scenario 2: Unsupervised Learning - Customer Segmentation
plt.figure(figsize=(12, 8))

# Generate synthetic customer data
np.random.seed(42)
n_customers = 200

# Create features: purchase frequency, average order value, and tenure
purchase_frequency = np.concatenate([
    np.random.normal(10, 3, 50),  # High frequency cluster
    np.random.normal(5, 2, 100),   # Medium frequency cluster
    np.random.normal(2, 1, 50)     # Low frequency cluster
])

avg_order_value = np.concatenate([
    np.random.normal(100, 20, 50),  # High value cluster
    np.random.normal(50, 15, 100),   # Medium value cluster
    np.random.normal(30, 10, 50)     # Low value cluster
])

# Clean up any negative values
purchase_frequency = np.maximum(purchase_frequency, 0.5)
avg_order_value = np.maximum(avg_order_value, 10)

# Create a DataFrame for visualization
customer_data = pd.DataFrame({
    'Purchase Frequency': purchase_frequency,
    'Average Order Value': avg_order_value
})

# Apply k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(customer_data)

# Add cluster labels to the DataFrame
customer_data['Cluster'] = clusters

# Create descriptive cluster names
cluster_names = {
    0: 'High Value Shoppers',
    1: 'Regular Shoppers',
    2: 'Occasional Shoppers'
}

# Map cluster numbers to names
customer_data['Cluster Name'] = customer_data['Cluster'].map(lambda x: cluster_names.get(x, f'Cluster {x}'))

# Get cluster centers
centers = kmeans.cluster_centers_

# Plot the clusters
colors = ['purple', 'orange', 'green']
for i, cluster_name in cluster_names.items():
    cluster_data = customer_data[customer_data['Cluster'] == i]
    plt.scatter(
        cluster_data['Purchase Frequency'], 
        cluster_data['Average Order Value'],
        c=colors[i], 
        label=f'{cluster_name} (n={len(cluster_data)})',
        alpha=0.7
    )

# Plot the cluster centers
plt.scatter(
    centers[:, 0], centers[:, 1],
    c='black',
    s=200,
    marker='X',
    label='Cluster Centers'
)

# Add labels and title
plt.title('Scenario 2: Unsupervised Learning - Customer Segmentation', fontsize=14)
plt.xlabel('Purchase Frequency (orders per year)', fontsize=12)
plt.ylabel('Average Order Value ($)', fontsize=12)
plt.grid(True)
plt.legend()

# Add explanatory text
plt.figtext(0.5, 0.01, "Customers are segmented based on their purchasing behavior without predefined categories.\n"
                      "The algorithm identifies natural clusters in the data, which can then be used for targeted marketing.", 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "scenario2_unsupervised_learning.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Scenario 3: Supervised Learning - Housing Price Prediction
plt.figure(figsize=(12, 8))

# Generate synthetic housing data
np.random.seed(42)
n_houses = 100

# Create features: square footage and number of bedrooms
square_footage = np.random.uniform(1000, 4000, n_houses)
bedrooms = np.random.randint(1, 6, n_houses)

# Create target: housing prices with some noise
# Price formula: base price + (sq_ft * price_per_sq_ft) + (bedrooms * price_per_bedroom) + location_factor + noise
base_price = 50000
price_per_sq_ft = 100
price_per_bedroom = 25000
noise = np.random.normal(0, 25000, n_houses)

# Location factor (categorical feature)
locations = np.random.choice(['Urban', 'Suburban', 'Rural'], n_houses)
location_factor = np.zeros(n_houses)
location_factor[locations == 'Urban'] = 50000
location_factor[locations == 'Suburban'] = 25000

# Calculate prices
prices = base_price + (square_footage * price_per_sq_ft / 1000) + (bedrooms * price_per_bedroom) + location_factor + noise

# Create a DataFrame for visualization
housing_data = pd.DataFrame({
    'Square Footage': square_footage,
    'Bedrooms': bedrooms,
    'Location': locations,
    'Price': prices
})

# Create a scatter plot with multiple dimensions
plt.subplot(1, 2, 1)

# Color map for locations
location_colors = {'Urban': 'red', 'Suburban': 'blue', 'Rural': 'green'}
colors = [location_colors[loc] for loc in housing_data['Location']]

# Size based on number of bedrooms
sizes = housing_data['Bedrooms'] * 30

plt.scatter(housing_data['Square Footage'], housing_data['Price'], 
           c=colors, s=sizes, alpha=0.6)

plt.title('Housing Price vs. Square Footage', fontsize=14)
plt.xlabel('Square Footage', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.grid(True)

# Add a legend for locations
for location, color in location_colors.items():
    plt.scatter([], [], c=color, label=f'Location: {location}')

# Add a legend for bedrooms
for i in range(1, 6):
    plt.scatter([], [], c='black', s=i*30, alpha=0.6, label=f'{i} Bedrooms')

plt.legend(loc='upper left', title='Features')

# Plot a simple linear regression for visualization
plt.subplot(1, 2, 2)

# Simple linear regression on square footage only for visualization
from sklearn.linear_model import LinearRegression
X = housing_data[['Square Footage']]
y = housing_data['Price']

model = LinearRegression()
model.fit(X, y)

# Generate predictions for a line
x_line = np.linspace(1000, 4000, 100).reshape(-1, 1)
y_pred = model.predict(x_line)

# Plot the data and regression line
plt.scatter(housing_data['Square Footage'], housing_data['Price'], alpha=0.6)
plt.plot(x_line, y_pred, 'r-', linewidth=2, label=f'Regression Line: Price = {model.intercept_:.0f} + {model.coef_[0]:.2f} × SqFt')

plt.title('Simple Linear Regression for Housing Prices', fontsize=14)
plt.xlabel('Square Footage', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.grid(True)
plt.legend()

# Add explanatory text
plt.figtext(0.5, 0.01, "A supervised learning model is trained to predict housing prices based on features\n"
                      "like square footage, number of bedrooms, and location using historical sales data.\n"
                      "The model learns the relationship between these features and the target variable (price).", 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "scenario3_supervised_learning.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Scenario 4: Semi-Supervised Learning - Spam Detection
plt.figure(figsize=(12, 8))

# Create a grid of email features
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
xx, yy = np.meshgrid(x, y)

# Generate a simple decision boundary for visualization
# Let's say emails with high frequency of certain words are more likely to be spam
z = xx + 0.5 * yy  # This creates a simple linear boundary

# Generate synthetic email data
np.random.seed(42)
n_emails = 200

# Generate feature values
feature1 = np.random.randn(n_emails)  # e.g., frequency of certain words
feature2 = np.random.randn(n_emails)  # e.g., email length or structure features

# True labels (0: not spam, 1: spam) based on our simple boundary
true_labels = (feature1 + 0.5 * feature2 > 0).astype(int)

# Create a DataFrame for emails
emails = pd.DataFrame({
    'Feature 1': feature1,
    'Feature 2': feature2,
    'Is Spam': true_labels
})

# Select a small subset (10%) as labeled
labeled_indices = np.random.choice(range(n_emails), int(0.1 * n_emails), replace=False)
unlabeled_indices = list(set(range(n_emails)) - set(labeled_indices))

# Plot the decision boundary
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, (z > 0).astype(int), alpha=0.3, levels=[0, 0.5, 1], colors=['lightblue', 'lightsalmon'])
plt.contour(xx, yy, z, levels=[0], colors='red', linewidths=2)

# Plot labeled emails
plt.scatter(
    emails.loc[labeled_indices][emails.loc[labeled_indices]['Is Spam'] == 0]['Feature 1'],
    emails.loc[labeled_indices][emails.loc[labeled_indices]['Is Spam'] == 0]['Feature 2'],
    color='blue', marker='o', s=100, label='Labeled: Not Spam', edgecolors='black'
)
plt.scatter(
    emails.loc[labeled_indices][emails.loc[labeled_indices]['Is Spam'] == 1]['Feature 1'],
    emails.loc[labeled_indices][emails.loc[labeled_indices]['Is Spam'] == 1]['Feature 2'],
    color='red', marker='o', s=100, label='Labeled: Spam', edgecolors='black'
)

# Plot unlabeled emails
plt.scatter(
    emails.loc[unlabeled_indices]['Feature 1'],
    emails.loc[unlabeled_indices]['Feature 2'],
    color='gray', alpha=0.5, s=50, label='Unlabeled Emails'
)

plt.title('Initial State with Few Labeled Emails', fontsize=14)
plt.xlabel('Feature 1 (e.g., word frequency)', fontsize=12)
plt.ylabel('Feature 2 (e.g., email structure)', fontsize=12)
plt.legend()
plt.grid(True)

# Subplot 2: After semi-supervised learning
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, (z > 0).astype(int), alpha=0.3, levels=[0, 0.5, 1], colors=['lightblue', 'lightsalmon'])
plt.contour(xx, yy, z, levels=[0], colors='red', linewidths=2)

# Plot all emails, now "classified" by semi-supervised learning
# For visualization, we'll use the true labels as if they were predicted
plt.scatter(
    emails[emails['Is Spam'] == 0]['Feature 1'],
    emails[emails['Is Spam'] == 0]['Feature 2'],
    color='blue', alpha=0.6, s=50, label='Classified: Not Spam'
)
plt.scatter(
    emails[emails['Is Spam'] == 1]['Feature 1'],
    emails[emails['Is Spam'] == 1]['Feature 2'],
    color='red', alpha=0.6, s=50, label='Classified: Spam'
)

# Highlight the original labeled emails
plt.scatter(
    emails.loc[labeled_indices][emails.loc[labeled_indices]['Is Spam'] == 0]['Feature 1'],
    emails.loc[labeled_indices][emails.loc[labeled_indices]['Is Spam'] == 0]['Feature 2'],
    facecolors='none', edgecolors='blue', s=100, linewidth=2, label='Original Labeled: Not Spam'
)
plt.scatter(
    emails.loc[labeled_indices][emails.loc[labeled_indices]['Is Spam'] == 1]['Feature 1'],
    emails.loc[labeled_indices][emails.loc[labeled_indices]['Is Spam'] == 1]['Feature 2'],
    facecolors='none', edgecolors='red', s=100, linewidth=2, label='Original Labeled: Spam'
)

plt.title('After Semi-Supervised Learning', fontsize=14)
plt.xlabel('Feature 1 (e.g., word frequency)', fontsize=12)
plt.ylabel('Feature 2 (e.g., email structure)', fontsize=12)
plt.legend()
plt.grid(True)

# Add explanatory text
plt.figtext(0.5, 0.01, "A spam detection system is trained using a small set of labeled emails (spam/not spam)\n"
                      "and a large set of unlabeled emails. The semi-supervised learning algorithm leverages\n"
                      "both labeled and unlabeled data to improve classification performance.", 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
# Save the figure
file_path = os.path.join(save_dir, "scenario4_semi_supervised_learning.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Summary and Conclusion
print_step_header(5, "Summary and Conclusion")

print("Summary of Machine Learning Types for Each Scenario:")
print()
print("Scenario 1: Robot Navigation in a Maze")
print("Type: Reinforcement Learning")
print("Key Components: Agent (robot), Environment (maze), Actions (movements), Rewards (progress, penalties)")
print("Learning Process: The robot learns through trial and error, receiving rewards for progress and penalties for mistakes")
print()
print("Scenario 2: Customer Segmentation")
print("Type: Unsupervised Learning")
print("Key Components: Unlabeled data (customer behaviors), Clustering algorithm (K-means)")
print("Learning Process: The algorithm identifies natural patterns in customer behavior without predefined categories")
print()
print("Scenario 3: Housing Price Prediction")
print("Type: Supervised Learning")
print("Key Components: Labeled data (features with known prices), Regression algorithm")
print("Learning Process: The model learns the relationship between house features and prices from historical data")
print()
print("Scenario 4: Spam Detection with Limited Labels")
print("Type: Semi-Supervised Learning")
print("Key Components: Small set of labeled emails, Large set of unlabeled emails")
print("Learning Process: The algorithm uses both labeled and unlabeled data to improve classification performance")
print()
print("Conclusion: Different machine learning approaches are suitable for different types of problems,")
print("depending on the nature of the data, the presence of labels, and the learning objective.") 