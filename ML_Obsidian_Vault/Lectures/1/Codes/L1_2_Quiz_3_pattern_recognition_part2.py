import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.dates as mdates
import scipy.stats as stats
from datetime import datetime, timedelta
from matplotlib.colors import ListedColormap

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_2_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 4: Emotion Recognition
print_step_header(4, "Emotion Recognition in Facial Expressions")

print("Approach for emotion recognition in facial expressions:")
print("\n1. Problem Type: Image Classification (Multi-class)")
print("2. Key Challenges:")
print("   - Subtle differences between expressions")
print("   - Variations in lighting, pose, occlusions")
print("   - Cultural differences in expression")
print("   - Ambiguity and mixed emotions")
print("\n3. Suitable Algorithms:")
print("   - Deep Convolutional Neural Networks (CNNs)")
print("   - Transfer learning with pre-trained models")
print("   - Ensemble methods")
print("   - Sequence models for dynamic expressions (LSTM, GRU)")
print("\n4. Feature Engineering:")
print("   - Facial landmarks (eyes, mouth, etc.)")
print("   - Action Units based on FACS (Facial Action Coding System)")
print("   - Texture features (LBP, HOG)")
print("   - Deep features from pre-trained networks")
print("\n5. Implementation Steps:")
print("   - Face detection and alignment")
print("   - Feature extraction")
print("   - Emotion classification")
print("   - Temporal smoothing for video")

# Visualize emotion recognition
# Simulate facial expressions using simplified data
emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral', 'Fearful', 'Disgusted']
emotion_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']

# Create a figure with multiple plots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Facial landmarks (simplified representation)
ax1 = axs[0, 0]
ax1.axis('off')

# Draw a face outline
circle = plt.Circle((0.5, 0.5), 0.4, fill=False, color='black')
ax1.add_patch(circle)

# Draw landmarks for different features
landmarks = {
    'Eyes': [(0.35, 0.6), (0.65, 0.6)],
    'Eyebrows': [(0.35, 0.7), (0.65, 0.7)],
    'Nose': [(0.5, 0.5)],
    'Mouth': [(0.5, 0.3)],
    'Jaw': [(0.5, 0.15)]
}

landmark_colors = {
    'Eyes': '#1f77b4',
    'Eyebrows': '#ff7f0e',
    'Nose': '#2ca02c',
    'Mouth': '#d62728',
    'Jaw': '#9467bd'
}

for feature, points in landmarks.items():
    for x, y in points:
        ax1.scatter(x, y, color=landmark_colors[feature], s=100, label=feature)

# Remove duplicate labels
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys(), title="Facial Features", 
           loc='upper right', bbox_to_anchor=(1.1, 1))

ax1.set_title('Facial Landmarks for Emotion Recognition', fontsize=12)

# Plot 2: Emotion classification accuracy
ax2 = axs[0, 1]

# Simulated accuracy data for different emotions
accuracies = {
    'Happy': 0.92,
    'Sad': 0.78,
    'Angry': 0.75,
    'Surprised': 0.88,
    'Neutral': 0.85,
    'Fearful': 0.71,
    'Disgusted': 0.69
}

# Sort emotions by accuracy for better visualization
sorted_emotions = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
emotion_names = [item[0] for item in sorted_emotions]
acc_values = [item[1] for item in sorted_emotions]

bars = ax2.bar(emotion_names, acc_values, color=emotion_colors)
ax2.set_title('Classification Accuracy by Emotion', fontsize=12)
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y')

# Add accuracy values on top of bars
for i, v in enumerate(acc_values):
    ax2.text(i, v + 0.02, f"{v:.2f}", ha='center')

# Plot 3: Confusion matrix for emotion recognition
ax3 = axs[1, 0]

# Create a simulated confusion matrix
np.random.seed(42)
conf_matrix = np.zeros((7, 7))

# Diagonal elements (correct predictions) should be higher
for i in range(7):
    conf_matrix[i, i] = accuracies[emotions[i]] * 100  # convert to percentage

# Off-diagonal elements (misclassifications)
for i in range(7):
    for j in range(7):
        if i != j:
            # More confusion between related emotions
            if (i, j) in [(1, 2), (2, 1), (0, 4), (4, 0), (3, 5), (5, 3), (2, 6), (6, 2)]:
                conf_matrix[i, j] = np.random.uniform(5, 15)
            else:
                conf_matrix[i, j] = np.random.uniform(1, 5)

# Normalize rows to sum to 100%
for i in range(7):
    row_sum = np.sum(conf_matrix[i, :])
    conf_matrix[i, :] = conf_matrix[i, :] / row_sum * 100

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues',
           xticklabels=emotions, yticklabels=emotions, ax=ax3)
ax3.set_title('Emotion Recognition Confusion Matrix (%)', fontsize=12)
ax3.set_xlabel('Predicted Emotion')
ax3.set_ylabel('True Emotion')

# Plot 4: Deep Learning Architecture for Emotion Recognition
ax4 = axs[1, 1]
ax4.axis('off')

# Function to draw network components
def draw_network_component(ax, x, y, width, height, color, text):
    rect = plt.Rectangle((x, y), width, height, facecolor=color, alpha=0.7, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', color='black')

# Simplified emotion recognition architecture
# Input
draw_network_component(ax4, 0.1, 0.8, 0.15, 0.1, '#a1dab4', 'Input Image\n224x224x3')

# Feature Extraction (Transfer Learning)
draw_network_component(ax4, 0.1, 0.65, 0.15, 0.1, '#41b6c4', 'VGG/ResNet\nFeature Extraction')
draw_network_component(ax4, 0.1, 0.5, 0.15, 0.1, '#2c7fb8', 'Facial Landmark\nDetection')

# Feature Fusion
draw_network_component(ax4, 0.4, 0.6, 0.2, 0.15, '#253494', 'Feature Fusion')

# Classification
draw_network_component(ax4, 0.7, 0.6, 0.15, 0.15, '#081d58', 'Emotion\nClassification')

# Output
draw_network_component(ax4, 0.7, 0.3, 0.15, 0.2, '#f46d43', 'Output\n7 Emotions')

# Add arrows
arrows = [
    (0.25, 0.85, 0.4, 0.7),  # Input to Feature Fusion
    (0.25, 0.7, 0.4, 0.65),   # Feature Extraction to Fusion
    (0.25, 0.55, 0.4, 0.62),  # Landmark Detection to Fusion
    (0.6, 0.675, 0.7, 0.675),  # Fusion to Classification
    (0.775, 0.6, 0.775, 0.5)  # Classification to Output
]

for start_x, start_y, end_x, end_y in arrows:
    ax4.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

ax4.set_title('Deep Learning Architecture for Emotion Recognition', fontsize=12)

plt.tight_layout()
file_path = os.path.join(save_dir, "emotion_recognition.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Seasonal Patterns in Retail Sales
print_step_header(5, "Seasonal Patterns in Retail Sales Data")

print("Approach for identifying seasonal patterns in retail sales data:")
print("\n1. Problem Type: Time Series Analysis and Pattern Detection")
print("2. Key Challenges:")
print("   - Distinguishing seasonal patterns from trends and noise")
print("   - Multiple overlapping seasonal patterns (daily, weekly, monthly, yearly)")
print("   - Impact of holidays, promotions, and external factors")
print("   - Evolving seasonal patterns over time")
print("\n3. Suitable Algorithms:")
print("   - Seasonal ARIMA (SARIMA)")
print("   - Fourier Analysis")
print("   - Prophet (Facebook's time series forecasting library)")
print("   - Recurrent Neural Networks (LSTM, GRU)")
print("\n4. Analysis Techniques:")
print("   - Seasonal decomposition (trend, seasonal, residual)")
print("   - Autocorrelation analysis")
print("   - Frequency domain analysis")
print("   - Anomaly detection for identifying unusual sales periods")
print("\n5. Implementation Steps:")
print("   - Data preprocessing and aggregation")
print("   - Exploratory time series analysis")
print("   - Seasonal pattern extraction")
print("   - Model building and forecasting")
print("   - Evaluation and interpretation")

# Visualize seasonal patterns in retail sales
# Create a simulated retail sales dataset
np.random.seed(42)

# Generate dates for 3 years of daily data
start_date = datetime(2020, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365 * 3)]
dates = np.array(dates)

# Generate synthetic sales data with various seasonal patterns
t = np.arange(len(dates))

# Yearly seasonality (higher in summer and December, lower in January-February)
yearly_seasonality = 20 * np.sin(2 * np.pi * t / 365 + 150)

# Weekly seasonality (weekends have higher sales)
weekly_seasonality = 10 * np.sin(2 * np.pi * t / 7)

# Monthly seasonality (higher at beginning of month)
monthly_seasonality = 5 * np.sin(2 * np.pi * t / 30)

# Special events (holidays with higher sales)
special_events = np.zeros(len(dates))
for year in [2020, 2021, 2022, 2023]:
    # Black Friday (day after Thanksgiving, 4th Thursday in November)
    thanksgiving_idx = np.where((dates >= datetime(year, 11, 22)) & (dates <= datetime(year, 11, 28)) & 
                               (np.array([d.weekday() for d in dates]) == 3))[0]
    if len(thanksgiving_idx) > 0:
        black_friday_idx = thanksgiving_idx[0] + 1
        if black_friday_idx < len(special_events):
            special_events[black_friday_idx-3:black_friday_idx+5] = np.array([10, 20, 40, 100, 60, 30, 20, 10])
    
    # Christmas shopping season
    christmas_idx = np.where(dates == datetime(year, 12, 25))[0]
    if len(christmas_idx) > 0:
        special_events[christmas_idx[0]-14:christmas_idx[0]+1] = 50 * np.exp(-0.1 * np.arange(15)[::-1])

# Overall trend (slight growth)
trend = 100 + 0.05 * t

# Combine all components
sales = trend + yearly_seasonality + weekly_seasonality + monthly_seasonality + special_events
sales = sales + np.random.normal(0, 10, len(sales))  # Add noise

# Convert to dataframe
sales_data = pd.DataFrame({
    'Date': dates,
    'Sales': sales
})

# Create a figure with multiple plots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Full time series
axs[0, 0].plot(sales_data['Date'], sales_data['Sales'], color='#1f77b4')
axs[0, 0].set_title('Retail Sales Time Series (3 Years)', fontsize=12)
axs[0, 0].set_xlabel('Date')
axs[0, 0].set_ylabel('Sales ($)')
axs[0, 0].grid(True)

# Format x-axis to show years nicely
years = mdates.YearLocator()
years_fmt = mdates.DateFormatter('%Y')
axs[0, 0].xaxis.set_major_locator(years)
axs[0, 0].xaxis.set_major_formatter(years_fmt)

# Plot 2: Seasonal decomposition (manually performed)
# Extract components for visualization
smoothed = np.convolve(sales, np.ones(30)/30, mode='same')  # 30-day moving average for trend
seasonal = sales - smoothed  # Simplified seasonality extraction

axs[0, 1].plot(sales_data['Date'], trend, color='#ff7f0e', label='Trend')
axs[0, 1].plot(sales_data['Date'][29:-29], smoothed[29:-29], color='#2ca02c', label='Smoothed')
axs[0, 1].set_title('Trend Component', fontsize=12)
axs[0, 1].set_xlabel('Date')
axs[0, 1].set_ylabel('Sales ($)')
axs[0, 1].legend()
axs[0, 1].grid(True)
axs[0, 1].xaxis.set_major_locator(years)
axs[0, 1].xaxis.set_major_formatter(years_fmt)

# Plot 3: Yearly seasonal pattern
# Focus on a single year (2022)
year_2022 = np.where((dates >= datetime(2022, 1, 1)) & (dates < datetime(2023, 1, 1)))[0]
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b')

axs[1, 0].plot(sales_data['Date'][year_2022], sales_data['Sales'][year_2022], color='#d62728')
axs[1, 0].set_title('Seasonal Pattern: 2022', fontsize=12)
axs[1, 0].set_xlabel('Month')
axs[1, 0].set_ylabel('Sales ($)')
axs[1, 0].grid(True)
axs[1, 0].xaxis.set_major_locator(months)
axs[1, 0].xaxis.set_major_formatter(months_fmt)

# Annotate special events
black_friday_idx = np.where(dates == datetime(2022, 11, 25))[0]
if len(black_friday_idx) > 0:
    axs[1, 0].annotate('Black Friday', xy=(dates[black_friday_idx[0]], sales[black_friday_idx[0]]),
                     xytext=(dates[black_friday_idx[0]-10], sales[black_friday_idx[0]]+30),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

christmas_idx = np.where(dates == datetime(2022, 12, 25))[0]
if len(christmas_idx) > 0:
    axs[1, 0].annotate('Christmas', xy=(dates[christmas_idx[0]], sales[christmas_idx[0]]),
                     xytext=(dates[christmas_idx[0]-15], sales[christmas_idx[0]]+30),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Plot 4: Autocorrelation analysis
lags = 60
acf = np.zeros(lags)
mean = np.mean(sales)
var = np.var(sales)

for lag in range(lags):
    acf[lag] = np.sum((sales[lag:] - mean) * (sales[:-lag if lag > 0 else None] - mean)) / (len(sales) - lag) / var

axs[1, 1].stem(range(lags), acf, linefmt='b-', markerfmt='bo', basefmt='r-')
axs[1, 1].set_title('Autocorrelation Function', fontsize=12)
axs[1, 1].set_xlabel('Lag (days)')
axs[1, 1].set_ylabel('Autocorrelation')
axs[1, 1].grid(True)

# Highlight significant lags
axs[1, 1].axhline(y=1.96/np.sqrt(len(sales)), linestyle='--', color='gray')
axs[1, 1].axhline(y=-1.96/np.sqrt(len(sales)), linestyle='--', color='gray')

# Add annotations for weekly and yearly patterns
axs[1, 1].annotate('Weekly Pattern', xy=(7, acf[7]), xytext=(7, acf[7]+0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
if lags >= 30:
    axs[1, 1].annotate('Monthly Pattern', xy=(30, acf[30]), xytext=(30, acf[30]+0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.tight_layout()
file_path = os.path.join(save_dir, "seasonal_patterns.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Summary
print_step_header(6, "Summary of Pattern Recognition Approaches")

print("This analysis has demonstrated different approaches to pattern recognition across various domains:")
print("\n1. Handwritten Digit Recognition (Image Classification):")
print("   - Uses convolutional neural networks to identify digits from raw pixel data")
print("   - Leverages spatial hierarchies of features in images")
print("   - Applications: OCR, document processing, postal services")
print("\n2. Fraud Detection (Anomaly Detection & Classification):")
print("   - Combines statistical methods with machine learning to identify unusual patterns")
print("   - Handles class imbalance and requires real-time processing")
print("   - Applications: Financial security, insurance claims, online purchases")
print("\n3. Emotion Recognition (Computer Vision & Psychology):")
print("   - Analyzes facial features and expressions to infer emotional states")
print("   - Combines domain knowledge (FACS) with deep learning")
print("   - Applications: HCI, market research, mental health monitoring")
print("\n4. Seasonal Pattern Analysis (Time Series Analysis):")
print("   - Decomposes time series data to extract recurring patterns at different scales")
print("   - Combines statistical methods with domain expertise")
print("   - Applications: Demand forecasting, inventory management, resource planning")
print("\nEach pattern recognition task requires a tailored approach based on the data characteristics")
print("and problem domain, but all share the common goal of extracting meaningful patterns from data.") 