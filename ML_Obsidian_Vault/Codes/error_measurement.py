import matplotlib.pyplot as plt
import numpy as np

def generate_error_measurement():
    """
    Generate a visualization of how the error is measured in linear regression.
    """
    # Create sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y_true = 2 * x + 1
    y = y_true + np.random.normal(0, 1, size=x.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scatter points
    ax.scatter(x, y, c='red', marker='x', s=50, label='Training examples')
    
    # Plot true function
    ax.plot(x, y_true, 'b-', linewidth=2, label='Regression line')
    
    # Highlight the error for a specific point
    point_idx = 10
    x_point = x[point_idx]
    y_point = y[point_idx]
    y_pred = y_true[point_idx]
    
    # Draw vertical line showing error
    ax.plot([x_point, x_point], [y_point, y_pred], 'r--', linewidth=1.5)
    
    # Annotate the error
    ax.annotate(r'$y^{(i)} - f(x^{(i)}; \boldsymbol{w})$', 
                xy=(x_point + 0.2, (y_point + y_pred)/2), 
                fontsize=12)
    
    # Add square to represent squared error
    ax.annotate('', 
                xy=(x_point + 0.5, y_point), 
                xytext=(x_point + 0.5, y_pred),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    
    ax.text(x_point + 0.7, (y_point + y_pred)/2, 
            r'Squared: $(y^{(i)} - f(x^{(i)}; \boldsymbol{w}))^2$', 
            fontsize=12, color='green')
    
    # Set labels
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title('How to Measure the Error', fontsize=16)
    
    # Add legend and grid
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add text about squared error
    ax.text(0.5, 0.05, 
            r'Squared error: $(y^{(i)} - f(x^{(i)}; \boldsymbol{w}))^2$', 
            transform=ax.transAxes,
            fontsize=14, ha='center',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('plots/error_measurement.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_error_measurement()
    print("Error measurement visualization generated successfully.") 