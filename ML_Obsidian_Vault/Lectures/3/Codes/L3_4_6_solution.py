import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
images_dir = os.path.join(parent_dir, "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Set random seed for reproducibility
np.random.seed(42)

# Define the station data
stations = {
    'A': {'location': np.array([1, 1]), 'temperature': 22},
    'B': {'location': np.array([4, 2]), 'temperature': 25},
    'C': {'location': np.array([2, 5]), 'temperature': 20},
    'D': {'location': np.array([5, 5]), 'temperature': 23}
}

print("Step 1: Define a Radial Basis Function (RBF)")
print("=" * 50)
print("A radial basis function (RBF) is a real-valued function whose value depends only on the distance")
print("from the input to some fixed point, called the center. In our case, the centers are the weather stations.")
print()
print("The most common form of RBF is the Gaussian RBF, defined as:")
print("φ(x) = exp(-||x - center||²/(2σ²))")
print("where:")
print("  - x is the input location")
print("  - center is the fixed center point (weather station location)")
print("  - ||x - center|| is the Euclidean distance between x and the center")
print("  - σ is a parameter that controls the width of the Gaussian")
print()
print("RBFs are useful for spatial prediction problems because:")
print("1. They naturally handle the concept of 'distance' - closer points have more influence")
print("2. They provide smooth interpolation between known data points")
print("3. They can model complex, non-linear relationships in the data")
print("4. They respect the locality principle: the influence of a point decreases with distance")
print()

# Define the Gaussian RBF function
def gaussian_rbf(x, center, sigma):
    """
    Compute the Gaussian RBF value.
    
    Parameters:
    x (np.array): The input location
    center (np.array): The center location
    sigma (float): The width parameter
    
    Returns:
    float: The value of the Gaussian RBF
    """
    distance_squared = np.sum((x - center) ** 2)
    return np.exp(-distance_squared / (2 * sigma ** 2))

# Demo: Create a visualization of Gaussian RBF
def visualize_rbf(sigma=1.5):
    """Visualize a Gaussian RBF in 3D"""
    x = np.linspace(0, 6, 100)
    y = np.linspace(0, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = gaussian_rbf(point, stations['A']['location'], sigma)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    ax.set_xlabel('x₁ coordinate')
    ax.set_ylabel('x₂ coordinate')
    ax.set_zlabel('RBF value')
    ax.set_title(f'Gaussian RBF centered at Station A with σ = {sigma}')
    
    # Add a colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Mark the center
    ax.scatter(
        stations['A']['location'][0], 
        stations['A']['location'][1], 
        1.0, 
        color='red', s=100, label='Station A'
    )
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f"rbf_visualization_sigma_{sigma}.png"))
    plt.close()

# Create RBF visualizations with different sigma values
visualize_rbf(sigma=1.5)  # The sigma value specified in the problem
visualize_rbf(sigma=0.8)  # A smaller sigma for comparison
visualize_rbf(sigma=3.0)  # A larger sigma for comparison

print("Step 2: Calculate the RBF value for a specific location")
print("=" * 50)
# Calculate the RBF value for the location (3, 3) using station A
test_location = np.array([3, 3])
sigma = 1.5
rbf_value_A = gaussian_rbf(test_location, stations['A']['location'], sigma)

print(f"Given information:")
print(f"  - Test location: ({test_location[0]}, {test_location[1]})")
print(f"  - Station A location: ({stations['A']['location'][0]}, {stations['A']['location'][1]})")
print(f"  - Gaussian RBF with σ = {sigma}")
print()
print("Calculation:")
print(f"1. Compute the squared Euclidean distance:")
distance_squared = np.sum((test_location - stations['A']['location']) ** 2)
print(f"   ||x - center||² = ||({test_location[0]}, {test_location[1]}) - ({stations['A']['location'][0]}, {stations['A']['location'][1]})||²")
print(f"   = ({test_location[0]} - {stations['A']['location'][0]})² + ({test_location[1]} - {stations['A']['location'][1]})²")
print(f"   = ({test_location[0] - stations['A']['location'][0]})² + ({test_location[1] - stations['A']['location'][1]})²")
print(f"   = {(test_location[0] - stations['A']['location'][0])}² + {(test_location[1] - stations['A']['location'][1])}²")
print(f"   = {(test_location[0] - stations['A']['location'][0])**2} + {(test_location[1] - stations['A']['location'][1])**2}")
print(f"   = {distance_squared}")
print()
print(f"2. Apply the Gaussian RBF formula:")
print(f"   φ_A(x) = exp(-||x - center||²/(2σ²))")
print(f"   = exp(-{distance_squared}/(2 × {sigma}²))")
print(f"   = exp(-{distance_squared}/(2 × {sigma**2}))")
print(f"   = exp(-{distance_squared}/{2 * sigma**2})")
print(f"   = exp(-{distance_squared / (2 * sigma**2)})")
print(f"   = {rbf_value_A}")
print()

# Create a visual representation of the RBF calculation
def visualize_rbf_calculation(test_location):
    """Visualize the RBF calculation for a specific test location"""
    # Create a grid for plotting
    x = np.linspace(0, 6, 100)
    y = np.linspace(0, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Calculate RBF values for the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = gaussian_rbf(point, stations['A']['location'], sigma)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the contour of the RBF
    contour = ax.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.7)
    ax.contour(X, Y, Z, 20, colors='white', alpha=0.3, linewidths=0.5)
    
    # Mark the stations
    for station_id, station_data in stations.items():
        ax.scatter(
            station_data['location'][0], 
            station_data['location'][1], 
            color='red', s=100, 
            label=f'Station {station_id} ({station_data["temperature"]}°C)'
        )
    
    # Mark the test location
    ax.scatter(
        test_location[0], 
        test_location[1], 
        color='blue', s=100, 
        label=f'Test Location ({test_location[0]}, {test_location[1]})'
    )
    
    # Draw a line from station A to the test location
    ax.plot(
        [stations['A']['location'][0], test_location[0]],
        [stations['A']['location'][1], test_location[1]],
        'k--', label=f'Distance: {np.sqrt(distance_squared):.2f} units'
    )
    
    # Add RBF value annotation
    ax.annotate(
        f'φ_A(x) = {rbf_value_A:.4f}',
        xy=(test_location[0], test_location[1]),
        xytext=(test_location[0] + 0.5, test_location[1] + 0.5),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7)
    )
    
    # Add a colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('RBF Value')
    
    # Set title and labels
    ax.set_title(f'Calculation of Gaussian RBF for Location ({test_location[0]}, {test_location[1]}) with Station A as Center')
    ax.set_xlabel('x₁ coordinate')
    ax.set_ylabel('x₂ coordinate')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.grid(True)
    ax.legend(loc='upper right')
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, "rbf_calculation_visualization.png"))
    plt.close()

# Visualize the RBF calculation for the test location
visualize_rbf_calculation(test_location)

print("Step 3: Write the complete model equation")
print("=" * 50)
print("The complete RBF model for predicting temperature at any location would be:")
print("T(x) = w_A × φ_A(x) + w_B × φ_B(x) + w_C × φ_C(x) + w_D × φ_D(x)")
print("where:")
print("  - T(x) is the predicted temperature at location x")
print("  - φ_i(x) is the Gaussian RBF value for station i at location x")
print("  - w_i is the weight for station i")
print()
print("Using the Gaussian RBF with σ = 1.5:")
print("T(x) = w_A × exp(-||x - x_A||²/(2σ²)) + w_B × exp(-||x - x_B||²/(2σ²)) + ")
print("      w_C × exp(-||x - x_C||²/(2σ²)) + w_D × exp(-||x - x_D||²/(2σ²))")
print()
print("Where x_A = (1,1), x_B = (4,2), x_C = (2,5), and x_D = (5,5)")
print()
print("To find the weights w_i, we can solve the system of equations that ensures the model")
print("exactly predicts the known temperatures at the station locations.")
print()

# Calculate the weights for the RBF model
def calculate_weights():
    """Calculate the weights for the RBF model using the known temperatures"""
    # Create the design matrix
    n_stations = len(stations)
    Phi = np.zeros((n_stations, n_stations))
    
    # Fill the design matrix
    stations_list = list(stations.values())
    for i in range(n_stations):
        for j in range(n_stations):
            Phi[i, j] = gaussian_rbf(
                stations_list[i]['location'], 
                stations_list[j]['location'], 
                sigma
            )
    
    # Create the target vector
    T = np.array([station['temperature'] for station in stations_list])
    
    # Solve for the weights
    weights = np.linalg.solve(Phi, T)
    
    return weights, stations_list

weights, stations_list = calculate_weights()

print("Solving for the weights:")
print("We need to find weights such that the model exactly predicts the known temperatures at each station.")
print("This leads to a system of linear equations:")
print()
for i, (station_id, station_data) in enumerate(stations.items()):
    print(f"For Station {station_id} at {tuple(station_data['location'])} with temperature {station_data['temperature']}°C:")
    equation_terms = []
    for j, (other_id, _) in enumerate(stations.items()):
        if i == j:  # When i=j, the RBF value is 1 (distance = 0)
            equation_terms.append(f"w_{other_id} × 1.0")
        else:
            rbf_val = gaussian_rbf(station_data['location'], stations_list[j]['location'], sigma)
            equation_terms.append(f"w_{other_id} × {rbf_val:.4f}")
    print(" + ".join(equation_terms) + f" = {station_data['temperature']}")
print()

station_ids = list(stations.keys())
print("In matrix form, this is:")
print("Φw = T")
print()
print("where Φ is the design matrix of RBF values, w is the weight vector, and T is the temperature vector.")
print()
print("Solving this system, we get the following weights:")
for i, station_id in enumerate(station_ids):
    print(f"w_{station_id} = {weights[i]:.4f}")

# Define the RBF prediction function
def predict_temperature(x, weights, station_locations, sigma):
    """Predict temperature at location x using the RBF model"""
    prediction = 0
    for i, location in enumerate(station_locations):
        prediction += weights[i] * gaussian_rbf(x, location, sigma)
    return prediction

# Create a temperature prediction map
def create_temperature_map(sigma_value=1.5):
    """Create a temperature prediction map using the RBF model"""
    # Create a grid for plotting
    x = np.linspace(0, 6, 100)
    y = np.linspace(0, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Calculate weights with the given sigma
    weights, stations_list = calculate_weights()
    station_locations = [station['location'] for station in stations_list]
    
    # Calculate temperature predictions for the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = predict_temperature(point, weights, station_locations, sigma_value)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the temperature contour
    contour = ax.contourf(X, Y, Z, 20, cmap='RdBu_r', alpha=0.8)
    ax.contour(X, Y, Z, 20, colors='white', alpha=0.3, linewidths=0.5)
    
    # Mark the stations
    for station_id, station_data in stations.items():
        ax.scatter(
            station_data['location'][0], 
            station_data['location'][1], 
            color='black', s=100, edgecolor='white',
            label=f'Station {station_id} ({station_data["temperature"]}°C)'
        )
    
    # Add a colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Temperature (°C)')
    
    # Set title and labels
    ax.set_title(f'Temperature Prediction Map using Gaussian RBF (σ = {sigma_value})')
    ax.set_xlabel('x₁ coordinate')
    ax.set_ylabel('x₂ coordinate')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.grid(True)
    ax.legend(loc='upper right')
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f"temperature_map_sigma_{sigma_value}.png"))
    plt.close()

# Create temperature maps with different sigma values
create_temperature_map(sigma_value=1.5)  # The sigma value specified in the problem
create_temperature_map(sigma_value=0.8)  # A smaller sigma for comparison
create_temperature_map(sigma_value=3.0)  # A larger sigma for comparison

print("Step 4: Explain the role of the parameter σ in the Gaussian RBF")
print("=" * 50)
print("The parameter σ in the Gaussian RBF controls the width or spread of the basis function.")
print("It determines how quickly the influence of a station diminishes with distance.")
print()
print("Effects of changing σ:")
print("1. Smaller σ (e.g., σ = 0.8):")
print("   - Creates narrower, more peaked basis functions")
print("   - Each station has stronger influence in its immediate vicinity")
print("   - Influence diminishes more rapidly with distance")
print("   - Results in more localized predictions with sharper transitions")
print("   - May lead to more 'bullseye' patterns around stations")
print()
print("2. Larger σ (e.g., σ = 3.0):")
print("   - Creates wider, more spread-out basis functions")
print("   - Each station's influence extends further")
print("   - Influence diminishes more gradually with distance")
print("   - Results in smoother predictions with more gradual transitions")
print("   - May lead to more averaged or blended predictions")
print()
print("Choosing an appropriate σ value is important for the model's performance:")
print("- Too small: May lead to overfitting, where predictions match known points well")
print("  but fluctuate unrealistically between them")
print("- Too large: May lead to underfitting, where predictions are too smooth and don't")
print("  capture local variations adequately")
print()
print("The ideal σ value depends on various factors:")
print("- The physical properties of the phenomenon being modeled (e.g., how temperature")
print("  typically varies across a city)")
print("- The density of measurement stations (sparser networks may benefit from larger σ)")
print("- The scale of the area being modeled")
print()
print("In practice, σ is often determined through cross-validation or based on domain knowledge.")

# Create influence map to visualize the effect of sigma
def create_influence_map():
    """Create a map showing the influence of each station with different sigma values"""
    # Create a figure with subplots for different sigma values
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sigma_values = [0.8, 1.5, 3.0]
    
    # Define colors for each station
    colors = ['red', 'green', 'blue', 'purple']
    
    for ax_idx, sigma_value in enumerate(sigma_values):
        ax = axes[ax_idx]
        
        # Create a grid for plotting
        x = np.linspace(0, 6, 100)
        y = np.linspace(0, 6, 100)
        X, Y = np.meshgrid(x, y)
        
        # Calculate the RBF values for each station
        Z_stations = []
        for station_data in stations.values():
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    point = np.array([X[i, j], Y[i, j]])
                    Z[i, j] = gaussian_rbf(point, station_data['location'], sigma_value)
            Z_stations.append(Z)
        
        # Find the station with maximum influence at each point
        Z_max = np.zeros_like(X)
        Z_max_idx = np.zeros_like(X, dtype=int)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                rbf_values = [Z[i, j] for Z in Z_stations]
                Z_max[i, j] = max(rbf_values)
                Z_max_idx[i, j] = np.argmax(rbf_values)
        
        # Create a mask for each station (where it has the highest influence)
        station_masks = []
        for idx in range(len(stations)):
            mask = (Z_max_idx == idx)
            station_masks.append(mask)
        
        # Plot the influence regions
        cmaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues, plt.cm.Purples]
        for idx, (mask, cmap) in enumerate(zip(station_masks, cmaps)):
            # Mask areas where this station doesn't have the highest influence
            masked_Z = np.ma.masked_where(~mask, Z_stations[idx])
            ax.contourf(X, Y, masked_Z, 20, cmap=cmap, alpha=0.5)
        
        # Mark the stations
        for idx, (station_id, station_data) in enumerate(stations.items()):
            ax.scatter(
                station_data['location'][0], 
                station_data['location'][1], 
                color=colors[idx], s=100, edgecolor='white',
                label=f'Station {station_id}'
            )
        
        # Set title and labels
        ax.set_title(f'Station Influence with σ = {sigma_value}')
        ax.set_xlabel('x₁ coordinate')
        ax.set_ylabel('x₂ coordinate')
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        ax.grid(True)
        
        # Only add legend to the first subplot to avoid repetition
        if ax_idx == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "station_influence_comparison.png"))
    plt.close()

# Create visualization of station influence with different sigma values
create_influence_map()

# Create a visualization to demonstrate the effect of sigma on a 1D example
def visualize_sigma_effect_1d():
    """Visualize the effect of sigma on RBF shape in 1D"""
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define the x range
    x = np.linspace(-5, 5, 1000)
    
    # Calculate RBF values for different sigma values
    sigma_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    
    for sigma_value, color in zip(sigma_values, colors):
        # Calculate the 1D Gaussian RBF
        y = np.exp(-x**2 / (2 * sigma_value**2))
        
        # Plot the RBF
        ax.plot(x, y, color=color, linewidth=2, label=f'σ = {sigma_value}')
    
    # Mark the center
    ax.scatter(0, 1, color='black', s=100, zorder=5, label='Center')
    
    # Set title and labels
    ax.set_title('Effect of σ on Gaussian RBF Shape (1D)', fontsize=14)
    ax.set_xlabel('Distance from Center', fontsize=12)
    ax.set_ylabel('RBF Value', fontsize=12)
    ax.grid(True)
    ax.legend(fontsize=12)
    
    # Add annotations
    ax.annotate('Narrower peak\n(more localized)', xy=(-1.5, 0.7), xytext=(-3.5, 0.8),
               arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    ax.annotate('Wider spread\n(more influence\nat distance)', xy=(3, 0.3), xytext=(3, 0.6),
               arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sigma_effect_1d.png"))
    plt.close()

# Create visualization of sigma effect in 1D
visualize_sigma_effect_1d()

print("\nAll visualizations have been saved to:", save_dir)

# Calculate RBF values for all stations at the test location
print("\nRBF values for all stations at the test location (3, 3):")
for station_id, station_data in stations.items():
    rbf_value = gaussian_rbf(test_location, station_data['location'], sigma)
    print(f"φ_{station_id}({test_location[0]}, {test_location[1]}) = {rbf_value:.6f}")

# Calculate the predicted temperature at the test location
test_prediction = predict_temperature(
    test_location, 
    weights, 
    [station_data['location'] for station_data in stations.values()], 
    sigma
)
print(f"\nPredicted temperature at location ({test_location[0]}, {test_location[1]}): {test_prediction:.2f}°C")

# Summary
print("\n-------------------------------------")
print("Summary of Results for Question 6:")
print("-------------------------------------")
print(f"1. Gaussian RBF value for Station A at location (3, 3) with σ = 1.5: {rbf_value_A:.6f}")
print(f"2. Complete model equation: T(x) = {weights[0]:.4f}×φ_A(x) + {weights[1]:.4f}×φ_B(x) + {weights[2]:.4f}×φ_C(x) + {weights[3]:.4f}×φ_D(x)")
print("3. The σ parameter controls the width of the Gaussian RBF, affecting how quickly station influence diminishes with distance")
print("   - Smaller σ: More localized influence with sharper transitions")
print("   - Larger σ: More widespread influence with smoother transitions") 