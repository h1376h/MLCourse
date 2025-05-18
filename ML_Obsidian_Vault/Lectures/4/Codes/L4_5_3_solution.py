import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_5_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'

# Problem parameters
num_examples = 10000
batch_gd_time_per_epoch = 2  # seconds
sgd_time_per_example = 0.0005  # seconds
mini_batch_size = 100
mini_batch_time = 0.03  # seconds per mini-batch
batch_gd_epochs = 100
sgd_epochs = 5

# Task 1: Calculate time per epoch for each method
sgd_time_per_epoch = num_examples * sgd_time_per_example
mini_batch_time_per_epoch = (num_examples / mini_batch_size) * mini_batch_time

print("Task 1: Time per Epoch")
print(f"Batch Gradient Descent: {batch_gd_time_per_epoch} seconds")
print(f"Stochastic Gradient Descent: {sgd_time_per_epoch} seconds")
print(f"Mini-Batch Gradient Descent (batch size {mini_batch_size}): {mini_batch_time_per_epoch} seconds")

# Task 2: Calculate total time to convergence
batch_gd_total_time = batch_gd_time_per_epoch * batch_gd_epochs
sgd_total_time = sgd_time_per_epoch * sgd_epochs

print("\nTask 2: Total Time to Convergence")
print(f"Batch Gradient Descent: {batch_gd_total_time} seconds")
print(f"Stochastic Gradient Descent: {sgd_total_time} seconds")

# Task 3: Already calculated mini-batch time per epoch in Task 1

# Additional analysis: Calculate time for various batch sizes
batch_sizes = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
time_per_epoch = []

# For simplicity, we'll use a model that assumes:
# - Single example processing time is constant (SGD time)
# - As batch size increases, there's efficiency gain but diminishing returns
# - At full batch size, it matches the given batch GD time

for size in batch_sizes:
    if size == 1:  # SGD
        time_per_epoch.append(sgd_time_per_epoch)
    elif size == 10000:  # Full batch
        time_per_epoch.append(batch_gd_time_per_epoch)
    else:
        # Interpolate between SGD and full batch time with a logarithmic scale
        # to represent diminishing efficiency gains
        ratio = np.log(size) / np.log(num_examples)
        interpolated_time = sgd_time_per_epoch * (1 - ratio) + batch_gd_time_per_epoch * ratio
        time_per_epoch.append(interpolated_time)

# Additional analysis: Model convergence speed vs batch size
# Simplified model: larger batch sizes need more epochs to converge
convergence_epochs = []
for size in batch_sizes:
    if size == 1:  # SGD
        convergence_epochs.append(sgd_epochs)
    elif size == 10000:  # Full batch
        convergence_epochs.append(batch_gd_epochs)
    else:
        # Interpolate between SGD and full batch epochs with a power function
        # to represent the rapid increase in required epochs as batch size grows
        ratio = (size / num_examples) ** 0.5
        interpolated_epochs = sgd_epochs * (1 - ratio) + batch_gd_epochs * ratio
        convergence_epochs.append(interpolated_epochs)

total_time = [t * e for t, e in zip(time_per_epoch, convergence_epochs)]

# Create visualization 1: Time per epoch for different methods
plt.figure(figsize=(10, 6))
methods = ['Batch GD', 'SGD', 'Mini-Batch GD']
times = [batch_gd_time_per_epoch, sgd_time_per_epoch, mini_batch_time_per_epoch]
colors = ['#3498db', '#e74c3c', '#2ecc71']

plt.bar(methods, times, color=colors)
plt.ylabel('Time (seconds)')
plt.title('Time per Epoch for Different Optimization Methods')
for i, v in enumerate(times):
    plt.text(i, v + 0.1, f"{v:.3f}s", ha='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'time_per_epoch.png'), dpi=300, bbox_inches='tight')

# Create visualization 2: Total time to convergence
plt.figure(figsize=(10, 6))
methods = ['Batch GD', 'SGD']
times = [batch_gd_total_time, sgd_total_time]
colors = ['#3498db', '#e74c3c']

plt.bar(methods, times, color=colors)
plt.ylabel('Time (seconds)')
plt.title('Total Time to Convergence')
for i, v in enumerate(times):
    plt.text(i, v + 2, f"{v:.2f}s", ha='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'total_convergence_time.png'), dpi=300, bbox_inches='tight')

# Create visualization 3: Effect of batch size on epoch time
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(batch_sizes, time_per_epoch, 'o-', color='#3498db', linewidth=2)
plt.xscale('log')
plt.xlabel('Batch Size (log scale)')
plt.ylabel('Time per Epoch (seconds)')
plt.title('Effect of Batch Size on Epoch Time')
plt.grid(True)

# Create visualization 4: Effect of batch size on convergence epochs
plt.subplot(1, 2, 2)
plt.plot(batch_sizes, convergence_epochs, 'o-', color='#e74c3c', linewidth=2)
plt.xscale('log')
plt.xlabel('Batch Size (log scale)')
plt.ylabel('Epochs to Converge')
plt.title('Effect of Batch Size on Convergence Speed')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'batch_size_effects.png'), dpi=300, bbox_inches='tight')

# Create visualization 5: Combined effect - total time to converge vs batch size
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, total_time, 'o-', color='#9b59b6', linewidth=2)
plt.xscale('log')
plt.xlabel('Batch Size (log scale)')
plt.ylabel('Total Time to Converge (seconds)')
plt.title('Total Training Time vs. Batch Size')
plt.grid(True)

# Mark the optimal batch size
optimal_idx = np.argmin(total_time)
optimal_batch = batch_sizes[optimal_idx]
optimal_time = total_time[optimal_idx]

plt.scatter([optimal_batch], [optimal_time], color='red', s=100, zorder=10)
plt.annotate(f'Optimal: {optimal_batch}',
             xy=(optimal_batch, optimal_time),
             xytext=(optimal_batch*3, optimal_time*1.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'total_time_vs_batch.png'), dpi=300, bbox_inches='tight')

# Summary of findings for the trade-off
print("\nTrade-off between batch size and convergence speed:")
print("1. Smaller batch sizes (like in SGD) have higher variance in updates")
print("   but require fewer epochs to converge.")
print("2. Larger batch sizes (like in batch GD) provide more stable and accurate")
print("   gradient estimates but require more epochs to reach convergence.")
print("3. Mini-batch GD strikes a balance, providing a compromise between")
print("   per-epoch computation time and number of epochs needed.")
print(f"4. In our model, the optimal batch size is around {optimal_batch},")
print("   which minimizes the total time to convergence.")

print("\nAll visualizations have been saved to:", save_dir)