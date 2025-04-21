import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def main():
    # Create output directory
    output_dir = os.path.join("..", "Images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 9: Rainfall Measurements (Censored Data)
    create_rainfall_censored_visual(output_dir)
    
    # Example 10: Plant Height Study (Grouped Data)
    create_plant_height_grouped_visual(output_dir)
    
    # Example 11: Student Heights (Simple Sample Mean)
    create_student_heights_visual(output_dir)
    
    # Example 12: Weight Measurements (Known Mean, Unknown Variance)
    create_weight_measurements_visual(output_dir)
    
    print("All visualizations created successfully!")

def create_rainfall_censored_visual(output_dir):
    """Create visualization for Example 9: Rainfall Measurements (Censored Data)"""
    # Parameters (from the MLE estimation)
    mle_mean = 17.53
    mle_std = 8.19
    naive_mean = 16.35
    naive_std = 6.24
    censoring_point = 25.0
    
    # Data (simplified version of the dataset)
    uncensored_data = [12.3, 8.7, 15.2, 10.8, 18.4, 7.2, 14.9, 20.1, 11.5, 16.8, 9.3]
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot range for normal distributions
    x = np.linspace(0, 45, 1000)
    
    # Plot the uncensored data histogram
    ax.hist(uncensored_data, bins=10, density=True, alpha=0.5, color='skyblue', 
            label='Uncensored Data')
    
    # Plot censoring point
    ax.axvline(x=censoring_point, color='red', linestyle='--', 
               label=f'Censoring Point (25 mm)')
    
    # Plot the fitted normal distributions
    ax.plot(x, norm.pdf(x, mle_mean, mle_std), 'g-', lw=2, 
            label=f'MLE with Censoring: μ={mle_mean:.2f}, σ={mle_std:.2f}')
    ax.plot(x, norm.pdf(x, naive_mean, naive_std), 'r:', lw=2, 
            label=f'Naive MLE: μ={naive_mean:.2f}, σ={naive_std:.2f}')
    
    # Add annotation for the censored region
    ax.fill_between(x[x > censoring_point], 0, 
                    norm.pdf(x[x > censoring_point], mle_mean, mle_std), 
                    color='green', alpha=0.3)
    ax.text(30, 0.02, "Censored Region\n(4 observations)", 
            ha='center', va='center', color='darkgreen')
    
    # Add title and labels
    ax.set_title('Example 9: Rainfall Measurements (Censored Data)')
    ax.set_xlabel('Rainfall (mm)')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rainfall_censored_separate.png"), dpi=300)
    plt.close()

def create_plant_height_grouped_visual(output_dir):
    """Create visualization for Example 10: Plant Height Study (Grouped Data)"""
    # Parameters (from the MLE estimation)
    mle_mean = 16.35
    mle_std = 2.48
    
    # Bin data
    bins = [(10.0, 12.0), (12.1, 14.0), (14.1, 16.0), 
            (16.1, 18.0), (18.1, 20.0), (20.1, 22.0)]
    bin_counts = [6, 12, 25, 32, 18, 7]
    expected_counts = [3.4, 12.8, 26.2, 28.7, 17.0, 5.4]
    total_plants = sum(bin_counts)
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bar chart comparing observed vs expected
    bin_centers = [(b[0] + b[1]) / 2 for b in bins]
    bin_widths = [b[1] - b[0] for b in bins]
    
    # Plot observed counts
    ax1.bar(bin_centers, bin_counts, width=bin_widths, alpha=0.7, 
            label='Observed Counts', color='skyblue', edgecolor='black')
    
    # Plot expected counts
    ax1.bar(bin_centers, expected_counts, width=bin_widths, alpha=0.5, 
            label='Expected Counts (MLE)', color='green', edgecolor='black')
    
    ax1.set_title('Observed vs Expected Frequencies')
    ax1.set_xlabel('Plant Height (cm)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Normal distribution with highlighted bins
    x = np.linspace(8, 24, 1000)
    ax2.plot(x, norm.pdf(x, mle_mean, mle_std) * total_plants, 'g-', lw=2,
             label=f'Fitted Normal: μ={mle_mean:.2f}, σ={mle_std:.2f}')
    
    # Add colored regions for each bin
    colors = ['skyblue', 'lightgreen', 'salmon', 'khaki', 'lightblue', 'lightpink']
    for i, ((low, high), count) in enumerate(zip(bins, bin_counts)):
        x_bin = np.linspace(low, high, 100)
        y_bin = norm.pdf(x_bin, mle_mean, mle_std) * total_plants
        ax2.fill_between(x_bin, 0, y_bin, alpha=0.4, color=colors[i % len(colors)])
        
        # Add annotations
        ax2.text((low + high) / 2, norm.pdf((low + high) / 2, mle_mean, mle_std) * total_plants / 2,
                f"{count}", ha='center', va='center', fontweight='bold')
    
    ax2.set_title('Fitted Normal Distribution with Grouped Data')
    ax2.set_xlabel('Plant Height (cm)')
    ax2.set_ylabel('Frequency Density')
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Example 10: Plant Height Study (Grouped Data)', fontsize=16)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(os.path.join(output_dir, "plant_height_grouped_separate.png"), dpi=300)
    plt.close()

def create_student_heights_visual(output_dir):
    """Create visualization for Example 11: Student Heights (Simple Sample Mean)"""
    # Data and parameters
    heights = [165, 172, 168, 175, 170]
    mle_mean = 170.0
    mle_std = 3.41
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Visualization of individual measurements and mean
    ax1.scatter(range(len(heights)), heights, s=100, color='blue', zorder=5,
                label='Individual Heights')
    ax1.axhline(y=mle_mean, color='red', linestyle='-', linewidth=2,
                label=f'MLE Mean: {mle_mean} cm')
    
    # Add labels for each point
    for i, height in enumerate(heights):
        ax1.annotate(f"{height} cm", (i, height), xytext=(0, 10), 
                    textcoords='offset points', ha='center')
    
    # Customize plot
    ax1.set_title('Student Height Measurements')
    ax1.set_xlabel('Student')
    ax1.set_ylabel('Height (cm)')
    ax1.set_xticks(range(len(heights)))
    ax1.set_xticklabels([f"Student {i+1}" for i in range(len(heights))])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Fitted normal distribution
    x = np.linspace(160, 180, 1000)
    ax2.plot(x, norm.pdf(x, mle_mean, mle_std), 'r-', lw=2,
             label=f'Fitted Normal: μ={mle_mean:.1f}, σ={mle_std:.2f}')
    
    # Add markers for actual data points
    for height in heights:
        ax2.axvline(x=height, color='blue', linestyle='--', alpha=0.5)
    
    # Highlight confidence intervals
    ax2.axvline(x=mle_mean - mle_std, color='green', linestyle=':', linewidth=2,
               label='68% Confidence Interval')
    ax2.axvline(x=mle_mean + mle_std, color='green', linestyle=':', linewidth=2)
    ax2.axvspan(mle_mean - mle_std, mle_mean + mle_std, alpha=0.2, color='green')
    
    # Customize plot
    ax2.set_title('Fitted Normal Distribution for Student Heights')
    ax2.set_xlabel('Height (cm)')
    ax2.set_ylabel('Probability Density')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add overall title
    fig.suptitle('Example 11: Student Heights (Simple Sample Mean)', fontsize=16)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(os.path.join(output_dir, "student_heights_separate.png"), dpi=300)
    plt.close()

def create_weight_measurements_visual(output_dir):
    """Create visualization for Example 12: Weight Measurements (Known Mean, Unknown Variance)"""
    # Data and parameters
    weights = [65, 70, 67, 71, 66, 69]
    known_mean = 68
    mle_std = 2.16
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Visualization of deviations from known mean
    deviations = [w - known_mean for w in weights]
    ax1.stem(range(len(weights)), deviations, linefmt='b-', markerfmt='bo', basefmt='r-',
            label='Deviation from Known Mean')
    
    # Add labels
    for i, (weight, dev) in enumerate(zip(weights, deviations)):
        ax1.annotate(f"{weight} kg\n({dev:+d} kg)", (i, dev), xytext=(0, 10 if dev > 0 else -20), 
                    textcoords='offset points', ha='center')
    
    # Customize plot
    ax1.set_title('Weight Deviations from Known Mean')
    ax1.set_xlabel('Observation')
    ax1.set_ylabel('Deviation from Mean (kg)')
    ax1.set_xticks(range(len(weights)))
    ax1.set_xticklabels([f"Obs {i+1}" for i in range(len(weights))])
    ax1.axhline(y=0, color='red', linestyle='-', linewidth=2,
                label=f'Known Mean: {known_mean} kg')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Fitted normal distribution with known mean
    x = np.linspace(60, 76, 1000)
    ax2.plot(x, norm.pdf(x, known_mean, mle_std), 'r-', lw=2,
             label=f'Normal Distribution: μ={known_mean} (known), σ={mle_std:.2f} (MLE)')
    
    # Add markers for actual data points
    for weight in weights:
        ax2.axvline(x=weight, color='blue', linestyle='--', alpha=0.5)
        ax2.plot(weight, norm.pdf(weight, known_mean, mle_std), 'bo')
    
    # Highlight confidence intervals
    ax2.axvline(x=known_mean - mle_std, color='green', linestyle=':', linewidth=2,
               label='68% Confidence Interval')
    ax2.axvline(x=known_mean + mle_std, color='green', linestyle=':', linewidth=2)
    ax2.axvspan(known_mean - mle_std, known_mean + mle_std, alpha=0.2, color='green')
    
    # Customize plot
    ax2.set_title('Normal Distribution with Known Mean and MLE for Variance')
    ax2.set_xlabel('Weight (kg)')
    ax2.set_ylabel('Probability Density')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add overall title
    fig.suptitle('Example 12: Weight Measurements (Known Mean, Unknown Variance)', fontsize=16)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(os.path.join(output_dir, "weight_measurements_separate.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main() 