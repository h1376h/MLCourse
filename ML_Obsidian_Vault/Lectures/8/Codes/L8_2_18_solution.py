import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize_scalar
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("="*80)
print("Question 18: Domain-Specific Requirements for Feature Selection")
print("="*80)

# 1. Key considerations for medical diagnosis features
print("\n1. Key Considerations for Medical Diagnosis Features:")
print("-" * 50)
medical_considerations = [
    "High reliability and accuracy requirements",
    "Interpretability for clinical decision making",
    "Ethical constraints on sensitive data",
    "Regulatory compliance (FDA, HIPAA)",
    "Feature stability over time",
    "Minimal false positive/negative rates",
    "Cost considerations for feature acquisition"
]

for i, consideration in enumerate(medical_considerations, 1):
    print(f"{i}. {consideration}")

# 2. Comparison of financial vs image recognition feature selection
print("\n2. Financial vs Image Recognition Feature Selection:")
print("-" * 55)

financial_needs = [
    "Real-time processing requirements",
    "High-frequency trading constraints",
    "Market volatility adaptation",
    "Risk management compliance",
    "Feature correlation with market indicators",
    "Cost of delayed decisions"
]

image_needs = [
    "High-dimensional data processing",
    "Spatial feature relationships",
    "Computational complexity tolerance",
    "Feature redundancy handling",
    "Scale and rotation invariance",
    "Real-time processing for applications"
]

print("Financial Applications:")
for i, need in enumerate(financial_needs, 1):
    print(f"  {i}. {need}")

print("\nImage Recognition:")
for i, need in enumerate(image_needs, 1):
    print(f"  {i}. {need}")

# 3. Sample size calculations for confidence levels
print("\n3. Sample Size Calculations for Feature Relevance Testing:")
print("-" * 60)

def binomial_sample_size(confidence_level, p=0.5, precision=0.05):
    """
    Calculate sample size for binomial proportion confidence interval
    n = (Z² * p * (1-p)) / E²
    where E is the precision/margin of error
    """
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    n = (z_score**2 * p * (1-p)) / (precision**2)
    return int(np.ceil(n))

def confidence_from_samples(n, p=0.5, precision=0.05):
    """
    Calculate confidence level achievable with given sample size
    """
    z_score = np.sqrt(n * precision**2 / (p * (1-p)))
    confidence_level = 2 * (stats.norm.cdf(z_score)) - 1
    return confidence_level * 100

# Medical diagnosis: 99.9% confidence
medical_confidence = 0.999
medical_precision = 0.001  # Very high precision needed
medical_sample_size = binomial_sample_size(medical_confidence, p=0.5, precision=medical_precision)

# Financial applications: 95% confidence
financial_confidence = 0.95
financial_precision = 0.05  # More tolerance for error
financial_sample_size = binomial_sample_size(financial_confidence, p=0.5, precision=financial_precision)

# With 1000 samples, what confidence can we achieve?
given_samples = 1000
achievable_confidence_medical = confidence_from_samples(given_samples, p=0.5, precision=medical_precision)
achievable_confidence_financial = confidence_from_samples(given_samples, p=0.5, precision=financial_precision)

print(f"Medical Diagnosis (99.9% confidence, precision={medical_precision*100}%):")
print(f"  Required sample size: {medical_sample_size:,}")

print(f"\nFinancial Applications (95% confidence, precision={financial_precision*100}%):")
print(f"  Required sample size: {financial_sample_size:,}")

print(f"\nWith {given_samples:,} samples:")
print(f"  For medical precision ({medical_precision*100}%): {achievable_confidence_medical:.3f}% confidence")
print(f"  For financial precision ({financial_precision*100}%): {achievable_confidence_financial:.3f}% confidence")

# 4. Regulatory compliance impact
print("\n4. Regulatory Compliance Impact on Feature Selection:")
print("-" * 55)

regulatory_impacts = [
    "Mandatory feature documentation and audit trails",
    "Restricted use of sensitive personal data",
    "Required validation procedures for high-stakes decisions",
    "Limitations on feature combinations",
    "Mandatory bias and fairness assessments",
    "Data retention and deletion requirements",
    "Transparency requirements for model decisions"
]

for i, impact in enumerate(regulatory_impacts, 1):
    print(f"{i}. {impact}")

# Define variables for radar chart
domains = ['Medical', 'Financial', 'Image Recognition']
categories = ['Reliability', 'Speed', 'Interpretability', 'Cost', 'Regulatory\nCompliance', 'Data\nDimensionality']

# Create domain-specific scores (0-10 scale)
medical_scores = [10, 3, 9, 8, 10, 4]
financial_scores = [8, 10, 6, 9, 8, 5]
image_scores = [7, 7, 5, 6, 4, 10]

# Create visualizations
print(f"\nGenerating visualizations in: {save_dir}")

# Create a separate figure for the polar plot first to avoid deprecation warnings
plt.figure(figsize=(8, 6))
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

plt.polar(angles, medical_scores + [medical_scores[0]], 'o-', linewidth=2, label='Medical', color='red')
plt.polar(angles, financial_scores + [financial_scores[0]], 's-', linewidth=2, label='Financial', color='green')
plt.polar(angles, image_scores + [image_scores[0]], '^-', linewidth=2, label='Image Recognition', color='blue')
plt.xticks(angles[:-1], categories)
plt.title('Domain Comparison: Feature Selection Requirements')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'domain_radar_chart.png'), dpi=300, bbox_inches='tight')
plt.close()

# Sample size vs confidence level visualization
confidence_levels = np.linspace(0.8, 0.999, 100)
sample_sizes = [binomial_sample_size(conf, p=0.5, precision=0.05) for conf in confidence_levels]

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(confidence_levels * 100, sample_sizes, 'b-', linewidth=2)
plt.axhline(y=medical_sample_size, color='r', linestyle='--', alpha=0.7,
            label=f'Medical (99.9%): {medical_sample_size:,} samples')
plt.axhline(y=financial_sample_size, color='g', linestyle='--', alpha=0.7,
            label=f'Financial (95%): {financial_sample_size:,} samples')
plt.axvline(x=medical_confidence * 100, color='r', linestyle=':', alpha=0.7)
plt.axvline(x=financial_confidence * 100, color='g', linestyle=':', alpha=0.7)
plt.xlabel('Confidence Level (%)')
plt.ylabel('Required Sample Size')
plt.title('Sample Size vs Confidence Level\n(Precision = 5%)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.yscale('log')

# Precision vs sample size for fixed confidence
precisions = np.linspace(0.001, 0.1, 100)
sample_sizes_medical = [binomial_sample_size(medical_confidence, p=0.5, precision=p) for p in precisions]
sample_sizes_financial = [binomial_sample_size(financial_confidence, p=0.5, precision=p) for p in precisions]

plt.subplot(2, 2, 2)
plt.plot(precisions * 100, sample_sizes_medical, 'r-', linewidth=2, label='Medical (99.9% confidence)')
plt.plot(precisions * 100, sample_sizes_financial, 'g-', linewidth=2, label='Financial (95% confidence)')
plt.axvline(x=medical_precision * 100, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=financial_precision * 100, color='g', linestyle='--', alpha=0.7)
plt.xlabel('Precision (%)')
plt.ylabel('Required Sample Size')
plt.title('Precision vs Sample Size')
plt.grid(True, alpha=0.3)
plt.legend()
plt.yscale('log')

# Achievable confidence with fixed sample size
precisions_fixed_n = np.linspace(0.001, 0.1, 100)
achievable_conf_medical = [confidence_from_samples(given_samples, p=0.5, precision=p) for p in precisions_fixed_n]
achievable_conf_financial = [confidence_from_samples(given_samples, p=0.5, precision=p) for p in precisions_fixed_n]

plt.subplot(2, 2, 3)
plt.plot(precisions_fixed_n * 100, achievable_conf_medical, 'r-', linewidth=2, label='With 1,000 samples')
plt.axhline(y=99.9, color='r', linestyle='--', alpha=0.7, label='Medical target (99.9%)')
plt.axhline(y=95, color='g', linestyle='--', alpha=0.7, label='Financial target (95%)')
plt.axvline(x=medical_precision * 100, color='r', linestyle=':', alpha=0.5)
plt.axvline(x=financial_precision * 100, color='g', linestyle=':', alpha=0.5)
plt.xlabel('Precision (%)')
plt.ylabel('Achievable Confidence (%)')
plt.title('Achievable Confidence with 1,000 Samples')
plt.grid(True, alpha=0.3)
plt.legend()

# Add a new informative visualization: 3D surface plot showing the relationship
plt.subplot(2, 2, 4)
# Create a grid of confidence and precision values
conf_grid = np.linspace(0.8, 0.999, 20)
prec_grid = np.linspace(0.01, 0.1, 20)
CONF, PREC = np.meshgrid(conf_grid, prec_grid)
SAMPLE_SIZE = np.zeros_like(CONF)

for i in range(len(prec_grid)):
    for j in range(len(conf_grid)):
        SAMPLE_SIZE[i, j] = binomial_sample_size(conf_grid[j], p=0.5, precision=prec_grid[i])

# Use log scale for better visualization
SAMPLE_SIZE_LOG = np.log10(SAMPLE_SIZE)
contour = plt.contourf(CONF * 100, PREC * 100, SAMPLE_SIZE_LOG, levels=20, cmap='viridis')
plt.colorbar(contour, label=r'$\log_{10}$(Sample Size)')
plt.xlabel('Confidence Level (%)')
plt.ylabel('Precision (%)')
plt.title('Sample Size as Function of\nConfidence and Precision')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'domain_feature_selection_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Sample size comparison bar chart
plt.figure(figsize=(10, 6))
domains_comparison = ['Medical\n(99.9%, 0.1% precision)', 'Financial\n(95%, 5% precision)']
sample_sizes_comparison = [medical_sample_size, financial_sample_size]

bars = plt.bar(domains_comparison, sample_sizes_comparison, color=['red', 'green'], alpha=0.7)
plt.ylabel('Required Sample Size')
plt.title('Sample Size Requirements by Domain')
plt.grid(True, alpha=0.3, axis='y')

for bar, size in zip(bars, sample_sizes_comparison):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
             f'{size:,}', ha='center', va='bottom', fontweight='bold')

plt.savefig(os.path.join(save_dir, 'sample_size_requirements.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a new separate informative visualization: Trade-off analysis
plt.figure(figsize=(14, 10))

# Create a comprehensive trade-off visualization
plt.subplot(2, 3, 1)
# Cost vs Accuracy trade-off
accuracy_levels = [0.90, 0.95, 0.99, 0.999]
medical_costs = [1000, 5000, 50000, 500000]  # Relative costs
financial_costs = [100, 500, 2000, 10000]
image_costs = [200, 800, 3000, 15000]

plt.plot(accuracy_levels, medical_costs, 'ro-', linewidth=2, markersize=8, label='Medical')
plt.plot(accuracy_levels, financial_costs, 'gs-', linewidth=2, markersize=8, label='Financial')
plt.plot(accuracy_levels, image_costs, 'b^-', linewidth=2, markersize=8, label='Image Recognition')
plt.xlabel('Accuracy Level')
plt.ylabel('Relative Cost')
plt.title('Cost vs Accuracy Trade-off')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 3, 2)
# Speed vs Reliability trade-off
reliability_levels = [0.8, 0.9, 0.95, 0.99, 0.999]
medical_speed = [1, 0.8, 0.5, 0.2, 0.05]  # Relative speed (1 = fastest)
financial_speed = [1, 0.9, 0.7, 0.4, 0.1]
image_speed = [1, 0.95, 0.8, 0.6, 0.3]

plt.plot(reliability_levels, medical_speed, 'ro-', linewidth=2, markersize=8, label='Medical')
plt.plot(reliability_levels, financial_speed, 'gs-', linewidth=2, markersize=8, label='Financial')
plt.plot(reliability_levels, image_speed, 'b^-', linewidth=2, markersize=8, label='Image Recognition')
plt.xlabel('Reliability Level')
plt.ylabel('Relative Speed')
plt.title('Speed vs Reliability Trade-off')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 3, 3)
# Regulatory Compliance vs Flexibility
compliance_levels = ['Low', 'Medium', 'High', 'Very High']
medical_flexibility = [0.9, 0.6, 0.3, 0.1]  # Flexibility decreases with compliance
financial_flexibility = [0.8, 0.5, 0.2, 0.05]
image_flexibility = [0.95, 0.8, 0.6, 0.4]

x_pos = np.arange(len(compliance_levels))
width = 0.25
plt.bar(x_pos - width, medical_flexibility, width, label='Medical', color='red', alpha=0.7)
plt.bar(x_pos, financial_flexibility, width, label='Financial', color='green', alpha=0.7)
plt.bar(x_pos + width, image_flexibility, width, label='Image Recognition', color='blue', alpha=0.7)
plt.xlabel('Regulatory Compliance Level')
plt.ylabel('Flexibility')
plt.title('Compliance vs Flexibility')
plt.xticks(x_pos, compliance_levels)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 4)
# Data Requirements vs Model Complexity
complexity_levels = ['Simple', 'Moderate', 'Complex', 'Very Complex']
medical_data = [1000, 10000, 100000, 1000000]  # Data requirements
financial_data = [100, 1000, 10000, 100000]
image_data = [10000, 100000, 1000000, 10000000]

plt.plot(complexity_levels, medical_data, 'ro-', linewidth=2, markersize=8, label='Medical')
plt.plot(complexity_levels, financial_data, 'gs-', linewidth=2, markersize=8, label='Financial')
plt.plot(complexity_levels, image_data, 'b^-', linewidth=2, markersize=8, label='Image Recognition')
plt.xlabel('Model Complexity')
plt.ylabel('Data Requirements')
plt.title('Data vs Complexity Trade-off')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 3, 5)
# Feature Interpretability vs Performance
interpretability_levels = ['High', 'Medium', 'Low', 'Very Low']
medical_performance = [0.7, 0.8, 0.9, 0.95]  # Performance increases with complexity
financial_performance = [0.8, 0.85, 0.9, 0.92]
image_performance = [0.6, 0.75, 0.85, 0.95]

plt.plot(interpretability_levels, medical_performance, 'ro-', linewidth=2, markersize=8, label='Medical')
plt.plot(interpretability_levels, financial_performance, 'gs-', linewidth=2, markersize=8, label='Financial')
plt.plot(interpretability_levels, image_performance, 'b^-', linewidth=2, markersize=8, label='Image Recognition')
plt.xlabel('Feature Interpretability')
plt.ylabel('Model Performance')
plt.title('Interpretability vs Performance')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 3, 6)
# Risk Tolerance vs Validation Requirements
risk_levels = ['Very Low', 'Low', 'Medium', 'High']
validation_effort = [100, 70, 40, 20]  # Validation effort (100 = maximum effort)
medical_risk = [100, 80, 50, 20]
financial_risk = [80, 60, 40, 30]
image_risk = [60, 40, 25, 15]

plt.plot(risk_levels, medical_risk, 'ro-', linewidth=2, markersize=8, label='Medical')
plt.plot(risk_levels, financial_risk, 'gs-', linewidth=2, markersize=8, label='Financial')
plt.plot(risk_levels, image_risk, 'b^-', linewidth=2, markersize=8, label='Image Recognition')
plt.xlabel('Risk Tolerance')
plt.ylabel('Validation Effort Required')
plt.title('Risk vs Validation Effort')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'comprehensive_trade_off_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a comprehensive summary table
print("\n" + "="*80)
print("SUMMARY TABLE: Domain-Specific Feature Selection Requirements")
print("="*80)

print(f"{'Domain':<20} {'Confidence':<12} {'Precision':<12} {'Sample Size':<15} {'Key Focus':<25}")
print("-" * 85)
print(f"{'Medical':<20} {medical_confidence*100:<12.1f} {medical_precision*100:<12.2f} {medical_sample_size:<15,} {'Reliability, Ethics':<25}")
print(f"{'Financial':<20} {financial_confidence*100:<12.1f} {financial_precision*100:<12.2f} {financial_sample_size:<15,} {'Speed, Compliance':<25}")
print(f"{'Image Recognition':<20} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'High Dimensions':<25}")

print(f"\nWith {given_samples:,} samples:")
print(f"{'Medical precision':<20} {'N/A':<12} {medical_precision*100:<12.2f} {given_samples:<15,} {achievable_confidence_medical:<25.3f}% confidence")
print(f"{'Financial precision':<20} {'N/A':<12} {financial_precision*100:<12.2f} {given_samples:<15,} {achievable_confidence_financial:<25.3f}% confidence")

print(f"\nPlots saved to: {save_dir}")
print("Files created:")
print("1. domain_feature_selection_comparison.png - Multi-panel comparison")
print("2. domain_radar_comparison.png - Domain radar chart comparison")
print("3. sample_size_requirements.png - Bar chart of sample size requirements")
print("4. confidence_precision_heatmap.png - Confidence vs Precision heatmap")
