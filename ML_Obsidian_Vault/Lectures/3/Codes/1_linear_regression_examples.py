import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images/Linear_Regression_Examples relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Linear_Regression_Examples")

# Create images directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

def plot_simple_linear_regression(X, y, title, filename, show_prediction=True):
    """
    Plots a simple linear regression with one feature and the target variable.
    
    Parameters:
    X : array-like, shape (n_samples, 1)
        Feature values
    y : array-like, shape (n_samples,)
        Target values
    title : str
        Plot title
    filename : str
        Filename to save the plot
    show_prediction : bool
        Whether to show the regression line
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data points')
    
    if show_prediction:
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        
        # Generate points for the regression line
        X_line = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
        y_pred = model.predict(X_line)
        
        # Plot regression line
        plt.plot(X_line, y_pred, color='red', linewidth=2, label=f'y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x')
        
        # Add equation to the plot
        equation = f'y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x'
        plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    plt.savefig(os.path.join(images_dir, f'{filename}.png'))
    plt.close()

def plot_design_matrix(X, title, filename):
    """
    Visualizes a design matrix as a heatmap.
    
    Parameters:
    X : array-like
        Design matrix
    title : str
        Plot title
    filename : str
        Filename to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(X, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    
    # Add annotations
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            plt.text(j, i, f'{X[i, j]:.1f}', ha='center', va='center', 
                     color='white' if X[i, j] > np.mean(X) else 'black')
    
    # Add labels and title
    plt.title(title)
    plt.xlabel('Feature index')
    plt.ylabel('Sample index')
    
    # For X with intercept, label the first column as intercept
    if X.shape[1] > 1:
        plt.xticks(np.arange(X.shape[1]), ['intercept'] + [f'x{i}' for i in range(1, X.shape[1])])
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{filename}.png'))
    plt.close()

def plot_multiple_regression_3d(X, y, title, filename):
    """
    Creates a 3D plot for multiple regression with two features.
    
    Parameters:
    X : array-like, shape (n_samples, 2)
        Feature values (without intercept term)
    y : array-like, shape (n_samples,)
        Target values
    title : str
        Plot title
    filename : str
        Filename to save the plot
    """
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Create a meshgrid for the regression plane
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                          np.linspace(y_min, y_max, 20))
    
    # Predict z values for the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the regression plane
    surf = ax.plot_surface(xx, yy, Z, alpha=0.5, cmap='viridis', linewidth=0, antialiased=True)
    
    # Plot the data points
    scatter = ax.scatter(X[:, 0], X[:, 1], y, c='blue', marker='o', s=30, label='Data points')
    
    # Add equation information
    equation = f'y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x₁ + {model.coef_[1]:.2f}x₂'
    ax.text2D(0.05, 0.95, equation, transform=ax.transAxes, fontsize=12, 
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add labels and title
    ax.set_xlabel('Feature 1 (x₁)')
    ax.set_ylabel('Feature 2 (x₂)')
    ax.set_zlabel('Target (y)')
    ax.set_title(title)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.7, aspect=10, label='Predicted value')
    
    # Save the figure
    plt.savefig(os.path.join(images_dir, f'{filename}.png'))
    plt.close()

def example_1_house_prices():
    """Example 1: House Prices vs Size (simple linear regression)"""
    print("\n=== Example 1: House Prices vs Size ===\n")
    
    # Data
    house_sizes = np.array([1500, 2000, 1800, 2200, 1600])  # square feet
    house_prices = np.array([300, 400, 350, 480, 310])  # thousand dollars
    
    # Step 1: Plot the data
    plot_simple_linear_regression(
        house_sizes, house_prices,
        "House Prices vs Size",
        "house_prices_vs_size_scatter"
    )
    
    # Step 2: Create and visualize the design matrix
    X = np.column_stack((np.ones(len(house_sizes)), house_sizes))
    print("Design Matrix:")
    print(X)
    plot_design_matrix(X, "Design Matrix for House Price Example", "house_prices_design_matrix")
    
    # Step 3: Calculate the linear regression coefficients manually
    X_transpose = X.T
    beta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(house_prices)
    intercept, slope = beta
    print(f"Manually calculated coefficients: β₀ = {intercept:.4f}, β₁ = {slope:.4f}")
    
    # Step 4: Calculate using sklearn for comparison
    model = LinearRegression()
    model.fit(house_sizes.reshape(-1, 1), house_prices)
    print(f"Sklearn calculated coefficients: β₀ = {model.intercept_:.4f}, β₁ = {model.coef_[0]:.4f}")
    
    # Step 5: Plot with regression line
    plot_simple_linear_regression(
        house_sizes, house_prices,
        "House Prices vs Size with Regression Line",
        "house_prices_vs_size_regression",
        show_prediction=True
    )
    
    # Regression equation in matrix notation
    print("\nMatrix equation: y = Xβ")
    print(f"Where y = {house_prices}")
    print(f"X = {X}")
    print(f"β = {beta}")
    
    # Make predictions for a few new house sizes
    new_sizes = np.array([1700, 2500, 3000])
    new_X = np.column_stack((np.ones(len(new_sizes)), new_sizes))
    predictions = new_X.dot(beta)
    print("\nPredictions for new house sizes:")
    for size, price in zip(new_sizes, predictions):
        print(f"A {size} sq ft house would cost approximately ${price:.2f}k")
    
    return X, house_prices, beta

def example_2_multiple_features():
    """Example 2: House Prices with Multiple Features"""
    print("\n=== Example 2: House Prices with Multiple Features ===\n")
    
    # Data with multiple features
    # [house_size, bedrooms, age]
    X_data = np.array([
        [1500, 3, 10],
        [1800, 4, 5],
        [1200, 2, 15],
        [2200, 4, 7]
    ])
    
    # House prices in $1000s
    y_data = np.array([320, 380, 270, 460])
    
    # Step 1: Create and visualize the design matrix
    X = np.column_stack((np.ones(len(X_data)), X_data))
    print("Design Matrix (with intercept):")
    print(X)
    plot_design_matrix(X, "Design Matrix for Multiple Features", "multiple_features_design_matrix")
    
    # Step 2: Calculate coefficients manually
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_data)
    print("\nManually calculated coefficients:")
    print(f"β₀ (intercept) = {beta[0]:.4f}")
    print(f"β₁ (house size) = {beta[1]:.4f}")
    print(f"β₂ (bedrooms) = {beta[2]:.4f}")
    print(f"β₃ (age) = {beta[3]:.4f}")
    
    # Step 3: Calculate using sklearn for comparison
    model = LinearRegression()
    model.fit(X_data, y_data)
    print("\nSklearn calculated coefficients:")
    print(f"β₀ (intercept) = {model.intercept_:.4f}")
    print(f"β₁ (house size) = {model.coef_[0]:.4f}")
    print(f"β₂ (bedrooms) = {model.coef_[1]:.4f}")
    print(f"β₃ (age) = {model.coef_[2]:.4f}")
    
    # Step 4: Visualize 3D projection (using first two features only)
    plot_multiple_regression_3d(
        X_data[:, 0:2], y_data,
        "House Prices vs Size and Bedrooms (partial visualization)",
        "house_prices_multiple_features_3d"
    )
    
    # Regression equation
    equation = f"Price = {beta[0]:.2f} + {beta[1]:.4f}×Size + {beta[2]:.2f}×Bedrooms + {beta[3]:.2f}×Age"
    print(f"\nRegression equation: {equation}")
    
    # Predictions
    new_house = np.array([1, 1600, 3, 5])  # [intercept, size, bedrooms, age]
    predicted_price = new_house.dot(beta)
    print(f"\nPredicted price for a 1600 sq ft, 3 bedroom, 5 year old house: ${predicted_price:.2f}k")
    
    return X, y_data, beta

def example_3_ice_cream_sales():
    """Example 3: Ice Cream Sales and Temperature"""
    print("\n=== Example 3: Ice Cream Sales vs Temperature ===\n")
    
    # Data
    temperatures = np.array([75, 82, 68, 90, 73])  # Temperature in °F
    sales = np.array([120, 150, 95, 180, 110])  # Number of ice creams sold
    
    # Step 1: Plot the data
    plot_simple_linear_regression(
        temperatures, sales,
        "Ice Cream Sales vs Temperature",
        "ice_cream_sales_scatter"
    )
    
    # Step 2: Create and visualize the design matrix
    X = np.column_stack((np.ones(len(temperatures)), temperatures))
    print("Design Matrix:")
    print(X)
    plot_design_matrix(X, "Design Matrix for Ice Cream Sales", "ice_cream_design_matrix")
    
    # Step 3: Calculate the linear regression coefficients
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(sales)
    intercept, slope = beta
    print(f"Coefficients: β₀ = {intercept:.4f}, β₁ = {slope:.4f}")
    
    # Step 4: Plot with regression line
    plot_simple_linear_regression(
        temperatures, sales,
        "Ice Cream Sales vs Temperature with Regression Line",
        "ice_cream_sales_regression",
        show_prediction=True
    )
    
    # Step 5: Interpret the results
    print(f"\nRegression equation: Sales = {intercept:.2f} + {slope:.2f} × Temperature")
    print(f"Interpretation: For each 1°F increase in temperature, ice cream sales increase by approximately {slope:.2f} units")
    
    # Make predictions for specific temperatures
    new_temps = np.array([60, 70, 80, 95])
    predictions = intercept + slope * new_temps
    
    print("\nPredictions for specific temperatures:")
    for temp, pred_sales in zip(new_temps, predictions):
        print(f"At {temp}°F, predicted sales: {pred_sales:.1f} ice creams")
    
    return X, sales, beta

def example_4_student_scores():
    """Example 4: Study Hours and Exam Scores"""
    print("\n=== Example 4: Study Hours and Exam Scores ===\n")
    
    # Data
    study_hours = np.array([2, 5, 1, 3.5, 0, 4])
    exam_scores = np.array([65, 85, 55, 75, 30, 80])
    
    # Step 1: Plot the data
    plot_simple_linear_regression(
        study_hours, exam_scores,
        "Exam Scores vs Study Hours",
        "exam_scores_scatter"
    )
    
    # Step 2: Create and visualize the design matrix
    X = np.column_stack((np.ones(len(study_hours)), study_hours))
    print("Design Matrix:")
    print(X)
    plot_design_matrix(X, "Design Matrix for Exam Scores", "exam_scores_design_matrix")
    
    # Step 3: Calculate the linear regression coefficients
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(exam_scores)
    intercept, slope = beta
    print(f"Coefficients: β₀ = {intercept:.4f}, β₁ = {slope:.4f}")
    
    # Step 4: Plot with regression line
    plot_simple_linear_regression(
        study_hours, exam_scores,
        "Exam Scores vs Study Hours with Regression Line",
        "exam_scores_regression",
        show_prediction=True
    )
    
    # Step 5: Interpret the results
    print(f"\nRegression equation: Score = {intercept:.2f} + {slope:.2f} × StudyHours")
    print(f"Interpretation: For each additional hour of study, exam scores increase by approximately {slope:.2f} points")
    print(f"The intercept ({intercept:.2f}) represents the predicted score with zero hours of studying")
    
    return X, exam_scores, beta

def example_5_plant_growth():
    """Example 5: Plant Growth and Sunlight"""
    print("\n=== Example 5: Plant Growth and Sunlight ===\n")
    
    # Data
    sunlight_hours = np.array([3, 6, 8, 4])
    plant_height = np.array([10, 18, 24, 14])
    
    # Step 1: Plot the data
    plot_simple_linear_regression(
        sunlight_hours, plant_height,
        "Plant Height vs Sunlight Hours",
        "plant_height_scatter"
    )
    
    # Step 2: Create and visualize the design matrix
    X = np.column_stack((np.ones(len(sunlight_hours)), sunlight_hours))
    print("Design Matrix:")
    print(X)
    plot_design_matrix(X, "Design Matrix for Plant Growth", "plant_growth_design_matrix")
    
    # Step 3: Calculate the linear regression coefficients
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(plant_height)
    intercept, slope = beta
    print(f"Coefficients: β₀ = {intercept:.4f}, β₁ = {slope:.4f}")
    
    # Step 4: Plot with regression line
    plot_simple_linear_regression(
        sunlight_hours, plant_height,
        "Plant Height vs Sunlight Hours with Regression Line",
        "plant_height_regression",
        show_prediction=True
    )
    
    # Step 5: Matrix equation in detail
    print("\nMatrix form of the linear regression:")
    print(f"y = [10, 18, 24, 14]")
    print("X = [")
    for row in X:
        print(f"  {row}")
    print("]")
    print(f"β = [{intercept:.2f}, {slope:.2f}]")
    
    # Step 6: Make predictions
    new_hours = np.array([5, 7, 10])
    predictions = intercept + slope * new_hours
    
    print("\nPredictions for specific sunlight hours:")
    for hours, height in zip(new_hours, predictions):
        print(f"With {hours} hours of sunlight, predicted plant height: {height:.1f} cm")
    
    return X, plant_height, beta

def example_6_coffee_typing():
    """Example 6: Coffee and Typing Speed"""
    print("\n=== Example 6: Coffee and Typing Speed ===\n")
    
    # Data
    coffee_cups = np.array([0, 1, 2, 3, 4])
    typing_speed = np.array([45, 60, 75, 90, 82])
    
    # Step 1: Plot the data
    plot_simple_linear_regression(
        coffee_cups, typing_speed,
        "Typing Speed vs Coffee Consumption",
        "typing_speed_scatter"
    )
    
    # Step 2: Create and visualize the design matrix
    X = np.column_stack((np.ones(len(coffee_cups)), coffee_cups))
    print("Design Matrix:")
    print(X)
    plot_design_matrix(X, "Design Matrix for Typing Speed", "typing_speed_design_matrix")
    
    # Step 3: Calculate the linear regression coefficients
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(typing_speed)
    intercept, slope = beta
    print(f"Coefficients: β₀ = {intercept:.4f}, β₁ = {slope:.4f}")
    
    # Step 4: Plot with regression line
    plot_simple_linear_regression(
        coffee_cups, typing_speed,
        "Typing Speed vs Coffee Consumption with Regression Line",
        "typing_speed_regression",
        show_prediction=True
    )
    
    # Step 5: Interpret the results
    print(f"\nRegression equation: TypingSpeed = {intercept:.2f} + {slope:.2f} × CoffeeCups")
    print(f"Interpretation: β₀ ({intercept:.2f}) represents the baseline typing speed with no coffee")
    print(f"β₁ ({slope:.2f}) represents the increase in typing speed per cup of coffee consumed")
    
    # Step 6: Discuss diminishing returns or non-linearity
    print("\nNote: The linear model assumes a constant effect per cup of coffee. In reality,")
    print("there might be diminishing returns or even negative effects with excessive consumption,")
    print("which would require a non-linear model to capture accurately.")
    
    return X, typing_speed, beta

def compare_all_models():
    """Compare coefficients and statistics across all examples"""
    print("\n=== Comparison of Linear Regression Models ===\n")
    
    examples = [
        ("House Prices vs Size", example_1_house_prices),
        ("Multiple Features House Prices", example_2_multiple_features),
        ("Ice Cream Sales vs Temperature", example_3_ice_cream_sales),
        ("Exam Scores vs Study Hours", example_4_student_scores),
        ("Plant Height vs Sunlight", example_5_plant_growth),
        ("Typing Speed vs Coffee", example_6_coffee_typing)
    ]
    
    print("Example             | Intercept (β₀) | Slope (β₁)   | R² Score   | Model Formula")
    print("-" * 90)
    
    for name, func in examples:
        if name == "Multiple Features House Prices":
            X, y, beta = func()
            X_data = X[:, 1:]  # Remove intercept column
            model = LinearRegression()
            model.fit(X_data, y)
            r2 = model.score(X_data, y)
            print(f"{name.ljust(20)} | {beta[0]:13.2f} | Multiple    | {r2:10.4f} | y = {beta[0]:.2f} + {beta[1]:.2f}x₁ + {beta[2]:.2f}x₂ + ...")
        else:
            X, y, beta = func()
            intercept, slope = beta
            X_data = X[:, 1:]  # Remove intercept column
            model = LinearRegression()
            model.fit(X_data, y)
            r2 = model.score(X_data, y)
            print(f"{name.ljust(20)} | {intercept:13.2f} | {slope:11.2f} | {r2:10.4f} | y = {intercept:.2f} + {slope:.2f}x")
    
    # Create a visual comparison of all simple linear regression examples
    plt.figure(figsize=(15, 10))
    
    # Simple linear regression examples
    simple_examples = [
        (example_1_house_prices, "House Prices vs Size", "size (sq ft)", "price ($k)"),
        (example_3_ice_cream_sales, "Ice Cream Sales vs Temperature", "temperature (°F)", "sales"),
        (example_4_student_scores, "Exam Scores vs Study Hours", "study hours", "score"),
        (example_5_plant_growth, "Plant Height vs Sunlight", "sunlight (hours)", "height (cm)"),
        (example_6_coffee_typing, "Typing Speed vs Coffee", "coffee cups", "WPM")
    ]
    
    for i, (func, title, xlabel, ylabel) in enumerate(simple_examples):
        plt.subplot(2, 3, i+1)
        X, y, beta = func()
        x_values = X[:, 1]  # Get the feature column
        
        # Plot data points
        plt.scatter(x_values, y, color='blue', alpha=0.7)
        
        # Plot regression line
        x_line = np.linspace(min(x_values), max(x_values), 100)
        y_line = beta[0] + beta[1] * x_line
        plt.plot(x_line, y_line, color='red', linewidth=2)
        
        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "all_linear_models_comparison.png"))
    plt.close()

def main():
    """Main function to run all examples"""
    print("\n=== Linear Regression Formulation Examples ===\n")
    print("Running examples demonstrating linear regression formulation...")
    
    # Run all examples
    example_1_house_prices()
    example_2_multiple_features()
    example_3_ice_cream_sales()
    example_4_student_scores()
    example_5_plant_growth()
    example_6_coffee_typing()
    
    # Compare all models
    compare_all_models()
    
    print(f"\nAll examples completed. Visualizations saved to {images_dir}")

if __name__ == "__main__":
    main() 