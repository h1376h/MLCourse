import os

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
    print("Created plots directory")
else:
    print("Plots directory already exists") 