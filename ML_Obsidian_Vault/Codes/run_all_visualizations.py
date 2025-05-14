import os
import subprocess
import sys

def run_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], 
                            capture_output=True, 
                            text=True)
    if result.returncode == 0:
        print(f"✓ {script_name} completed successfully")
        print(result.stdout)
    else:
        print(f"✗ {script_name} failed")
        print(result.stderr)
    print("-" * 50)

# First make sure the plots directory exists
run_script("ensure_plots_directory.py")

# List of all visualization scripts
scripts = [
    "mse_vs_training_size.py",
    "linear_regression_training_examples.py",
    # Add other scripts as they are created
    # "learning_algorithm_diagram.py",
    # "error_measurement.py",
    # "cost_function_visualization.py",
    # "geometric_interpretation.py",
    # "regularization_comparison.py",
    # "gradient_descent.py",
]

# Run each script
for script in scripts:
    if os.path.exists(script):
        run_script(script)
    else:
        print(f"✗ {script} not found")
        print("-" * 50)

print("All visualizations completed!") 