import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy.special import comb
import time

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 6: FEATURE SELECTION SEARCH SPACE COMPLEXITY")
print("=" * 80)

# Question 1: Total possible feature subsets for 10 features
print("\n1. TOTAL POSSIBLE FEATURE SUBSETS FOR 10 FEATURES")
print("-" * 50)

n = 10
total_subsets = 2**n
print(f"Number of features (n) = {n}")
print(f"Total possible subsets = 2^{n} = {total_subsets:,}")
print(f"Formula: For n features, total subsets = 2^n")
print(f"Explanation: Each feature can either be included (1) or excluded (0) from a subset")
print(f"          This gives us 2 choices for each of the {n} features")
print(f"          Total combinations = 2 × 2 × ... × 2 ({n} times) = 2^{n}")

# Question 2: Subsets with exactly 5 features
print("\n2. SUBSETS WITH EXACTLY 5 FEATURES")
print("-" * 50)

k = 5
subsets_k = comb(n, k, exact=True)
print(f"Number of features (n) = {n}")
print(f"Subset size (k) = {k}")
print(f"Number of subsets with exactly {k} features = C({n}, {k}) = {subsets_k:,}")
print(f"Formula: C(n,k) = n! / (k! × (n-k)!)")
print(f"Calculation: C({n},{k}) = {n}! / ({k}! × ({n}-{k})!)")
print(f"          = {math.factorial(n)} / ({math.factorial(k)} × {math.factorial(n-k)})")
print(f"          = {math.factorial(n)} / ({math.factorial(k) * math.factorial(n-k)}) = {subsets_k:,}")

# Question 3: Growth rate of search space
print("\n3. GROWTH RATE OF SEARCH SPACE")
print("-" * 50)

print(f"Growth rate as a function of n:")
print(f"Total subsets = 2^n")
print(f"This is exponential growth")
print(f"Examples:")
for i in range(5, 26, 5):
    subsets_i = 2**i
    print(f"  n = {i:2d}: 2^{i} = {subsets_i:,} subsets")

# Question 4: Time for exhaustive search with 20 features
print("\n4. EXHAUSTIVE SEARCH TIME FOR 20 FEATURES")
print("-" * 50)

n_20 = 20
total_subsets_20 = 2**n_20
evaluation_time = 1  # seconds per subset
total_time_seconds = total_subsets_20 * evaluation_time

# Convert to different time units
minutes = total_time_seconds / 60
hours = total_time_seconds / 3600
days = total_time_seconds / (24 * 3600)
years = total_time_seconds / (365.25 * 24 * 3600)

print(f"Number of features (n) = {n_20}")
print(f"Total subsets = 2^{n_20} = {total_subsets_20:,}")
print(f"Time per evaluation = {evaluation_time} second")
print(f"Total time = {total_subsets_20:,} × {evaluation_time} second = {total_time_seconds:,} seconds")
print(f"Time in different units:")
print(f"  Minutes: {minutes:,.2f}")
print(f"  Hours: {hours:,.2f}")
print(f"  Days: {days:,.2f}")
print(f"  Years: {years:,.2f}")

# Question 5: Subsets with 3-7 features from 20 total
print("\n5. SUBSETS WITH 3-7 FEATURES FROM 20 TOTAL FEATURES")
print("-" * 50)

n_total = 20
k_min, k_max = 3, 7
total_subsets_range = 0

print(f"Number of features (n) = {n_total}")
print(f"Subset size range: {k_min} to {k_max} features")
print(f"Calculating C({n_total}, k) for k = {k_min} to {k_max}:")
print()

for k in range(k_min, k_max + 1):
    subsets_k = comb(n_total, k, exact=True)
    total_subsets_range += subsets_k
    print(f"  C({n_total}, {k}) = {subsets_k:,} subsets")
    
print(f"Total subsets with {k_min}-{k_max} features = {total_subsets_range:,}")
print(f"Percentage of total search space: {total_subsets_range / 2**n_total * 100:.2f}%")

# Question 6: Greedy vs exhaustive search comparison
print("\n6. GREEDY VS EXHAUSTIVE SEARCH COMPARISON")
print("-" * 50)

n_greedy = 50
evaluation_time_greedy = 0.1  # seconds per evaluation

# Greedy forward selection: evaluates n features sequentially
greedy_evaluations = n_greedy
greedy_time = greedy_evaluations * evaluation_time_greedy

# Exhaustive search: evaluates all 2^n subsets
exhaustive_evaluations = 2**n_greedy
exhaustive_time = exhaustive_evaluations * evaluation_time_greedy

# Convert to different time units
greedy_minutes = greedy_time / 60
exhaustive_years = exhaustive_time / (365.25 * 24 * 3600)
speedup_factor = exhaustive_time / greedy_time

print(f"Number of features (n) = {n_greedy}")
print(f"Evaluation time per feature/subset = {evaluation_time_greedy} seconds")
print()

print("GREEDY FORWARD SELECTION:")
print(f"  Evaluations = {n_greedy} (one per feature)")
print(f"  Total time = {n_greedy} × {evaluation_time_greedy} = {greedy_time} seconds")
print(f"  Time in minutes: {greedy_minutes:.2f}")
print()

print("EXHAUSTIVE SEARCH:")
print(f"  Evaluations = 2^{n_greedy} = {exhaustive_evaluations:,}")
print(f"  Total time = {exhaustive_evaluations:,} × {evaluation_time_greedy} = {exhaustive_time:,.0f} seconds")
print(f"  Time in years: {exhaustive_years:,.2f}")
print()

print("COMPARISON:")
print(f"  Speedup factor = Exhaustive time / Greedy time")
print(f"  Speedup factor = {exhaustive_time:,.0f} / {greedy_time} = {speedup_factor:,.0f}")
print(f"  Greedy is {speedup_factor:,.0f} times faster than exhaustive search!")

# Create visualizations
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)

# Visualization 1: Growth of search space
plt.figure(figsize=(12, 8))
n_values = np.arange(1, 21)
subset_counts = 2**n_values

plt.subplot(2, 2, 1)
plt.semilogy(n_values, subset_counts, 'b-o', linewidth=2, markersize=6)
plt.xlabel('Number of Features ($n$)')
plt.ylabel('Number of Subsets (log scale)')
plt.title('Exponential Growth of Feature Selection Search Space')
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, 21, 5))

# Add annotations for specific values
for i, (n_val, count) in enumerate(zip(n_values[::4], subset_counts[::4])):
    plt.annotate(f'$2^{{{n_val}}} = {count:,}$', 
                 (n_val, count), 
                 xytext=(10, 10), 
                 textcoords='offset points',
                 fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))

# Visualization 2: Subset size distribution for n=20
plt.subplot(2, 2, 2)
n_plot = 20
k_values = np.arange(0, n_plot + 1)
subset_counts_k = np.array([comb(n_plot, k, exact=True) for k in k_values])

plt.bar(k_values, subset_counts_k, alpha=0.7, color='green')
plt.xlabel('Subset Size ($k$)')
plt.ylabel('Number of Subsets')
plt.title(f'Distribution of Subset Sizes for {n_plot} Features')
plt.grid(True, alpha=0.3)

# Highlight the range 3-7
highlight_mask = (k_values >= 3) & (k_values <= 7)
plt.bar(k_values[highlight_mask], subset_counts_k[highlight_mask], 
        alpha=0.9, color='red', label='Range 3-7')

plt.legend()
plt.xticks(np.arange(0, 21, 2))

# Visualization 3: Time comparison for different n values
plt.subplot(2, 2, 3)
n_time = np.arange(5, 26, 5)
exhaustive_times = 2**n_time * 1  # 1 second per evaluation
greedy_times = n_time * 0.1       # 0.1 seconds per evaluation

plt.semilogy(n_time, exhaustive_times, 'r-o', linewidth=2, markersize=6, label='Exhaustive Search')
plt.semilogy(n_time, greedy_times, 'g-s', linewidth=2, markersize=6, label='Greedy Selection')
plt.xlabel('Number of Features ($n$)')
plt.ylabel('Time (seconds, log scale)')
plt.title('Search Time Comparison: Exhaustive vs Greedy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(n_time)

# Add time annotations
for i, (n_val, ex_time, gr_time) in enumerate(zip(n_time, exhaustive_times, greedy_times)):
    if n_val == 20:
        plt.annotate(f'${ex_time:.0e}$s', (n_val, ex_time), 
                     xytext=(10, 10), textcoords='offset points', fontsize=8)
        plt.annotate(f'${gr_time:.1f}$s', (n_val, gr_time), 
                     xytext=(10, -10), textcoords='offset points', fontsize=8)

# Visualization 4: Speedup factor
plt.subplot(2, 2, 4)
n_speedup = np.arange(5, 26, 5)
speedup_factors = (2**n_speedup * 0.1) / (n_speedup * 0.1)

plt.semilogy(n_speedup, speedup_factors, 'purple', linewidth=2, marker='o', markersize=6)
plt.xlabel('Number of Features ($n$)')
plt.ylabel('Speedup Factor (log scale)')
plt.title('Greedy vs Exhaustive: Speedup Factor')
plt.grid(True, alpha=0.3)
plt.xticks(n_speedup)

# Add speedup annotations
for i, (n_val, speedup) in enumerate(zip(n_speedup, speedup_factors)):
    if n_val in [10, 15, 20, 25]:
        plt.annotate(f'${speedup:.0e}\\times$', (n_val, speedup), 
                     xytext=(10, 10), textcoords='offset points', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_complexity_analysis.png'), 
            dpi=300, bbox_inches='tight')

# Create detailed breakdown table
print("\nDETAILED BREAKDOWN TABLE")
print("=" * 80)
print(f"{'n':>3} | {'2^n':>10} | {'C(n,5)':>10} | {'Time (s)':>12} | {'Time (years)':>15}")
print("-" * 80)

for n_val in [5, 10, 15, 20, 25]:
    total_subsets = 2**n_val
    subsets_5 = comb(n_val, 5, exact=True) if n_val >= 5 else 0
    time_seconds = total_subsets * 1
    time_years = time_seconds / (365.25 * 24 * 3600)
    
    print(f"{n_val:3d} | {total_subsets:10,} | {subsets_5:10,} | {time_seconds:12,.0f} | {time_years:15.2f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print(f"Visualizations saved to: {save_dir}")
print("=" * 80)
