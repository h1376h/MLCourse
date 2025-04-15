import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

print("\n=== PROBABILITY DISTRIBUTIONS IN MACHINE LEARNING: STEP-BY-STEP EXAMPLES ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the Lectures/2 directory
parent_dir = os.path.dirname(current_dir)
# Use Images directory relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "L2_1_ML")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Helper function to create confidence ellipses for normal distribution
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from the square root of the variance
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # Calculating the standard deviation of y from the square root of the variance
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Example 1: Binary Classification with Bernoulli Distribution
print("Example 1: Binary Classification with Bernoulli Distribution")
print("===========================================================\n")

print("Problem: Building a simple spam classifier based on keyword presence")
print("We'll use the Bernoulli distribution to model whether an email contains a suspicious keyword\n")

# Create synthetic data for spam classification
np.random.seed(42)

# Number of emails of each type
n_spam = 100
n_ham = 200

# True probability of keyword presence
p_keyword_given_spam = 0.8
p_keyword_given_ham = 0.1

# Generate synthetic data
spam_data = np.random.binomial(1, p_keyword_given_spam, n_spam)
ham_data = np.random.binomial(1, p_keyword_given_ham, n_ham)

# Create a DataFrame
df = pd.DataFrame({
    'has_keyword': np.concatenate([spam_data, ham_data]),
    'is_spam': np.concatenate([np.ones(n_spam), np.zeros(n_ham)])
})

print("Step 1: Define the model")
print("We use Bernoulli distribution to model keyword presence:")
print(f"For spam emails: X|spam ~ Bernoulli({p_keyword_given_spam})")
print(f"For non-spam emails: X|not spam ~ Bernoulli({p_keyword_given_ham})")
print()

print("Step 2: Estimate parameters from data")
# Calculate parameter estimates from our "observed" data
p1_hat = np.mean(df[df['is_spam'] == 1]['has_keyword'])
p0_hat = np.mean(df[df['is_spam'] == 0]['has_keyword'])

print(f"From our data of {n_spam} spam emails and {n_ham} non-spam emails:")
print(f"Estimated p₁ = {p1_hat:.4f} (true value: {p_keyword_given_spam})")
print(f"Estimated p₀ = {p0_hat:.4f} (true value: {p_keyword_given_ham})")
print()

print("Step 3: Apply Bayes' theorem for classification")
# Equal prior probabilities
p_spam = 0.5
p_ham = 0.5

# Calculate posterior probabilities
p_spam_given_keyword = (p1_hat * p_spam) / ((p1_hat * p_spam) + (p0_hat * p_ham))
p_ham_given_keyword = (p0_hat * p_ham) / ((p0_hat * p_ham) + (p1_hat * p_spam))

p_spam_given_no_keyword = ((1 - p1_hat) * p_spam) / (((1 - p1_hat) * p_spam) + ((1 - p0_hat) * p_ham))
p_ham_given_no_keyword = ((1 - p0_hat) * p_ham) / (((1 - p0_hat) * p_ham) + ((1 - p1_hat) * p_spam))

print("Calculating posterior probabilities using Bayes' theorem:")
print(f"P(spam|keyword) = ({p1_hat:.4f} × {p_spam}) / (({p1_hat:.4f} × {p_spam}) + ({p0_hat:.4f} × {p_ham}))")
print(f"P(spam|keyword) = {p_spam_given_keyword:.4f}")
print()
print(f"P(spam|no keyword) = ({(1-p1_hat):.4f} × {p_spam}) / (({(1-p1_hat):.4f} × {p_spam}) + ({(1-p0_hat):.4f} × {p_ham}))")
print(f"P(spam|no keyword) = {p_spam_given_no_keyword:.4f}")
print()

# Visualization
plt.figure(figsize=(15, 5))

# Plot 1: Distribution of keyword presence for each class
plt.subplot(1, 3, 1)
labels = ['No Keyword', 'Has Keyword']
spam_counts = [n_spam - sum(spam_data), sum(spam_data)]
ham_counts = [n_ham - sum(ham_data), sum(ham_data)]

x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, spam_counts, width, label='Spam')
plt.bar(x + width/2, ham_counts, width, label='Not Spam')
plt.xlabel('Feature')
plt.ylabel('Count')
plt.title('Distribution of Keyword Presence')
plt.xticks(x, labels)
plt.legend()

# Plot 2: Bernoulli PMF for each class
plt.subplot(1, 3, 2)
x_values = [0, 1]
plt.stem(x_values, [1-p1_hat, p1_hat], 'r-', label='Spam', basefmt='r-')
plt.stem(x_values, [1-p0_hat, p0_hat], 'b-', label='Not Spam', basefmt='b-')
plt.xlabel('Keyword Present')
plt.ylabel('Probability')
plt.title('Bernoulli PMF for Each Class')
plt.xticks([0, 1], ['No', 'Yes'])
plt.ylim(0, 1)
plt.legend()

# Plot 3: Posterior probabilities
plt.subplot(1, 3, 3)
posterior_values = [[p_ham_given_no_keyword, p_spam_given_no_keyword], 
                     [p_ham_given_keyword, p_spam_given_keyword]]

plt.imshow(posterior_values, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Probability')
plt.xlabel('Class')
plt.ylabel('Feature')
plt.title('Posterior Probabilities')
plt.xticks([0, 1], ['Not Spam', 'Spam'])
plt.yticks([0, 1], ['No Keyword', 'Has Keyword'])

# Add probability values as text
for i in range(2):
    for j in range(2):
        plt.text(j, i, f'{posterior_values[i][j]:.2f}', 
                 ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'bernoulli_spam_classification.png'), dpi=300)
plt.close()

# Train a Bernoulli Naive Bayes classifier on our synthetic data
print("Training a Bernoulli Naive Bayes classifier:")
X = df[['has_keyword']]
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = BernoulliNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print model parameters
print("\nLearned Model Parameters:")
print(f"Class priors: {model.class_log_prior_}")
print(f"Feature log probabilities:")
print(model.feature_log_prob_)

# Calculate feature probabilities from log probabilities
feature_prob = np.exp(model.feature_log_prob_)
print(f"\nFeature probabilities P(X=1|class):")
print(f"P(keyword=1|not spam) = {feature_prob[0][0]:.4f}")
print(f"P(keyword=1|spam) = {feature_prob[1][0]:.4f}")
print("\n")

# Example 2: Multi-class Classification with Multinomial Distribution
print("Example 2: Multi-class Classification with Multinomial Distribution")
print("================================================================\n")

print("Problem: Classifying text documents into categories based on word frequencies")
print("We'll use the Multinomial distribution to model the distribution of words in documents\n")

# Create synthetic data for text classification
np.random.seed(42)

# Number of documents in each category
n_sports = 100
n_politics = 100
n_entertainment = 100

# Define a vocabulary with 10 words
vocabulary_size = 10
word_names = ["goal", "team", "vote", "election", "actor", "movie", "win", "party", "play", "star"]

# Word probability distributions for each category
# Each row represents probabilities for the words in our vocabulary
word_probs_sports = np.array([0.20, 0.15, 0.02, 0.01, 0.05, 0.05, 0.25, 0.06, 0.18, 0.03])
word_probs_politics = np.array([0.03, 0.02, 0.25, 0.20, 0.05, 0.04, 0.10, 0.25, 0.03, 0.03])
word_probs_entertainment = np.array([0.03, 0.04, 0.02, 0.01, 0.25, 0.25, 0.10, 0.05, 0.05, 0.20])

print("Step 1: Define the model")
print("We use the Multinomial distribution to model word frequencies:")
print("Word probabilities for each category:")
print("\nSports category word probabilities:")
for i, word in enumerate(word_names):
    print(f"  P('{word}'|sports) = {word_probs_sports[i]:.2f}")
    
print("\nPolitics category word probabilities:")
for i, word in enumerate(word_names):
    print(f"  P('{word}'|politics) = {word_probs_politics[i]:.2f}")
    
print("\nEntertainment category word probabilities:")
for i, word in enumerate(word_names):
    print(f"  P('{word}'|entertainment) = {word_probs_entertainment[i]:.2f}")
print()

print("Step 2: Generate synthetic documents based on these probability distributions")
# Function to generate a document as word counts
def generate_document(category, doc_length=100):
    if category == "sports":
        word_probs = word_probs_sports
    elif category == "politics":
        word_probs = word_probs_politics
    else:  # entertainment
        word_probs = word_probs_entertainment
        
    # Generate a document as a multinomial distribution of word counts
    word_counts = np.random.multinomial(doc_length, word_probs)
    return word_counts

# Generate documents for each category
sports_docs = np.array([generate_document("sports") for _ in range(n_sports)])
politics_docs = np.array([generate_document("politics") for _ in range(n_politics)])
entertainment_docs = np.array([generate_document("entertainment") for _ in range(n_entertainment)])

# Combine all documents
all_docs = np.vstack([sports_docs, politics_docs, entertainment_docs])
all_labels = np.array(["sports"] * n_sports + ["politics"] * n_politics + ["entertainment"] * n_entertainment)

print(f"Generated {n_sports + n_politics + n_entertainment} documents:")
print(f"- {n_sports} sports documents")
print(f"- {n_politics} politics documents")
print(f"- {n_entertainment} entertainment documents")
print(f"Each document has {all_docs[0].sum()} words distributed across {vocabulary_size} vocabulary terms")
print()

# Display a sample document from each category
print("Sample documents:")
for category in ["sports", "politics", "entertainment"]:
    sample_doc = all_docs[all_labels == category][0]
    print(f"\n{category.capitalize()} document word counts:")
    for i, word in enumerate(word_names):
        print(f"  '{word}': {sample_doc[i]}")
print()

print("Step 3: Train a Multinomial Naive Bayes classifier")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_docs, all_labels, test_size=0.3, random_state=42)

# Train a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB(alpha=1.0)  # alpha=1.0 for Laplace smoothing
nb_classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print model parameters
print("\nLearned Model Parameters:")
print(f"Class priors: {np.exp(nb_classifier.class_log_prior_)}")

print("\nEstimated word probabilities (with Laplace smoothing):")
feature_log_prob = nb_classifier.feature_log_prob_
categories = nb_classifier.classes_

for i, category in enumerate(categories):
    print(f"\n{category.capitalize()} category word probabilities:")
    word_probs = np.exp(feature_log_prob[i])
    for j, word in enumerate(word_names):
        print(f"  P('{word}'|{category}) = {word_probs[j]:.4f}")

# Step 4: Classify a new document
print("\nStep 4: Classify a new document using Bayes' theorem")
new_doc = np.array([10, 8, 2, 1, 3, 2, 12, 3, 9, 5])  # Sports-like document

print("New document word counts:")
for i, word in enumerate(word_names):
    print(f"  '{word}': {new_doc[i]}")

# Predict probabilities for the new document
probs = nb_classifier.predict_proba([new_doc])[0]

print("\nClassification probabilities:")
for i, category in enumerate(categories):
    print(f"  P({category}|document) = {probs[i]:.4f}")

predicted_category = nb_classifier.predict([new_doc])[0]
print(f"\nThe document is classified as: {predicted_category}")

# Visualization: Word distributions for each category
plt.figure(figsize=(15, 10))

# Plot 1: True word probabilities
plt.subplot(2, 2, 1)
x = np.arange(len(word_names))
width = 0.25

plt.bar(x - width, word_probs_sports, width, label='Sports')
plt.bar(x, word_probs_politics, width, label='Politics')
plt.bar(x + width, word_probs_entertainment, width, label='Entertainment')

plt.xlabel('Words')
plt.ylabel('Probability')
plt.title('True Word Probabilities by Category')
plt.xticks(x, word_names, rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Plot 2: Estimated word probabilities
plt.subplot(2, 2, 2)
estimated_probs = np.exp(feature_log_prob)

plt.bar(x - width, estimated_probs[0], width, label=categories[0])
plt.bar(x, estimated_probs[1], width, label=categories[1])
plt.bar(x + width, estimated_probs[2], width, label=categories[2])

plt.xlabel('Words')
plt.ylabel('Probability')
plt.title('Estimated Word Probabilities by Category')
plt.xticks(x, word_names, rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Plot 3: Sample document word distributions
plt.subplot(2, 2, 3)
sports_sample = all_docs[all_labels == "sports"][0]
politics_sample = all_docs[all_labels == "politics"][0]
entertainment_sample = all_docs[all_labels == "entertainment"][0]

plt.bar(x - width, sports_sample, width, label='Sports Sample')
plt.bar(x, politics_sample, width, label='Politics Sample')
plt.bar(x + width, entertainment_sample, width, label='Entertainment Sample')

plt.xlabel('Words')
plt.ylabel('Word Count')
plt.title('Sample Document Word Counts')
plt.xticks(x, word_names, rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Plot 4: Classification probabilities for new document
plt.subplot(2, 2, 4)
plt.bar(categories, probs)
plt.xlabel('Category')
plt.ylabel('Probability')
plt.title('Classification Probabilities for New Document')
plt.xticks(rotation=45)
plt.ylim(0, 1)

# Add the text of the probabilities
for i, v in enumerate(probs):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'multinomial_text_classification.png'), dpi=300)
plt.close()

print("\n")

# Example 3: Count Data Modeling with Poisson Distribution
print("Example 3: Count Data Modeling with Poisson Distribution")
print("=======================================================\n")

print("Problem: Modeling the number of visitors to a website per minute for capacity planning")
print("We'll use the Poisson distribution to model the arrival process\n")

# Create synthetic website traffic data
np.random.seed(42)

# Parameters
lambda_rate = 5  # Average visitors per minute
num_minutes = 1000  # Number of minutes to simulate

# Generate Poisson-distributed visitor counts
visitor_counts = np.random.poisson(lambda_rate, num_minutes)

print(f"Step 1: Define the model")
print(f"We model website visitors as a Poisson process:")
print(f"X ~ Poisson(λ), where λ = {lambda_rate} visitors per minute")
print()

print(f"Step 2: Analyze the synthetic visitor data")
mean_visitors = np.mean(visitor_counts)
var_visitors = np.var(visitor_counts)
max_visitors = np.max(visitor_counts)

print(f"Visitor Statistics:")
print(f"- Total minutes analyzed: {num_minutes}")
print(f"- Total visitors: {np.sum(visitor_counts)}")
print(f"- Average (mean) visitors per minute: {mean_visitors:.2f}")
print(f"- Variance of visitors per minute: {var_visitors:.2f}")
print(f"- Maximum visitors in any minute: {max_visitors}")
print()

print(f"Note that for a Poisson distribution, the mean and variance should be approximately equal.")
print(f"In this data: mean = {mean_visitors:.2f}, variance = {var_visitors:.2f}")
print()

print(f"Step 3: Calculate probabilities for capacity planning")
# Calculate probabilities of different visitor counts using the Poisson PMF
k_values = np.arange(0, 20)
pmf_values = stats.poisson.pmf(k_values, lambda_rate)

# Find high load probabilities
prob_more_than_10 = 1 - stats.poisson.cdf(10, lambda_rate)
prob_more_than_15 = 1 - stats.poisson.cdf(15, lambda_rate)

print(f"Probability of getting exactly 10 visitors in a minute:")
print(f"P(X = 10) = e^(-λ) × λ^10 / 10!")
print(f"P(X = 10) = e^(-{lambda_rate}) × {lambda_rate}^10 / 10!")
print(f"P(X = 10) = {stats.poisson.pmf(10, lambda_rate):.6f}")
print()

print(f"Probability of getting more than 10 visitors in a minute:")
print(f"P(X > 10) = 1 - P(X ≤ 10)")
print(f"P(X > 10) = 1 - {stats.poisson.cdf(10, lambda_rate):.6f}")
print(f"P(X > 10) = {prob_more_than_10:.6f}")
print()

print(f"Probability of getting more than 15 visitors in a minute:")
print(f"P(X > 15) = 1 - P(X ≤ 15)")
print(f"P(X > 15) = 1 - {stats.poisson.cdf(15, lambda_rate):.6f}")
print(f"P(X > 15) = {prob_more_than_15:.6f}")
print()

# Calculate the server capacity needed for 99% of cases
capacity_99 = stats.poisson.ppf(0.99, lambda_rate)
print(f"To handle 99% of all traffic scenarios, the server should be dimensioned")
print(f"to handle at least {int(capacity_99)} visitors per minute.")
print()

# Visualization: Poisson distribution and empirical data
plt.figure(figsize=(15, 5))

# Plot 1: PMF of Poisson distribution
plt.subplot(1, 3, 1)
plt.bar(k_values, pmf_values, alpha=0.7)
plt.axvline(x=lambda_rate, color='r', linestyle='--', 
            label=f'λ = {lambda_rate}')
plt.xlabel('Number of Visitors per Minute')
plt.ylabel('Probability')
plt.title('Poisson PMF for Website Traffic')
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Histogram of empirical data
plt.subplot(1, 3, 2)
plt.hist(visitor_counts, bins=range(0, max_visitors+2), density=True, 
         alpha=0.7, label='Observed Data')
plt.bar(k_values, pmf_values, alpha=0.4, color='red', 
        label='Poisson PMF')
plt.xlabel('Number of Visitors per Minute')
plt.ylabel('Frequency (Normalized)')
plt.title('Observed Visitor Counts vs. Poisson PMF')
plt.legend()
plt.grid(alpha=0.3)

# Plot 3: Capacity planning visualization
plt.subplot(1, 3, 3)
cumulative_prob = np.cumsum(pmf_values)
plt.plot(k_values, cumulative_prob, 'o-', label='Cumulative Probability')
plt.axhline(y=0.99, color='r', linestyle='--', 
            label='99% Capacity Threshold')
plt.axvline(x=capacity_99, color='g', linestyle='-', 
            label=f'Required Capacity: {int(capacity_99)}')
plt.xlabel('Server Capacity (Visitors per Minute)')
plt.ylabel('Cumulative Probability')
plt.title('Server Capacity Planning')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'poisson_website_traffic.png'), dpi=300)
plt.close()

# Time series visualization
plt.figure(figsize=(15, 6))

# Plot time series of visitor counts for the first 100 minutes
time_window = 100
plt.subplot(2, 1, 1)
plt.step(range(time_window), visitor_counts[:time_window], where='mid')
plt.axhline(y=lambda_rate, color='r', linestyle='--', 
            label=f'Expected Value (λ = {lambda_rate})')
plt.axhline(y=capacity_99, color='g', linestyle='-', 
            label=f'99% Capacity Threshold')
plt.xlabel('Time (minutes)')
plt.ylabel('Visitors')
plt.title('Website Traffic Time Series (First 100 Minutes)')
plt.legend()
plt.grid(alpha=0.3)

# Plot smoothed time series with moving average
plt.subplot(2, 1, 2)
window_size = 10
moving_avg = np.convolve(visitor_counts, np.ones(window_size)/window_size, mode='valid')
plt.plot(range(len(moving_avg)), moving_avg)
plt.axhline(y=lambda_rate, color='r', linestyle='--', 
            label=f'Expected Value (λ = {lambda_rate})')
plt.xlabel('Time (minutes)')
plt.ylabel(f'Visitors ({window_size}-minute Moving Average)')
plt.title(f'Smoothed Website Traffic with {window_size}-minute Moving Average')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'poisson_website_traffic_timeseries.png'), dpi=300)
plt.close()

print("For all traffic minutes in our simulation, these are the counts:")
minute_counts = np.bincount(visitor_counts)
for i, count in enumerate(minute_counts):
    if count > 0:
        print(f"- {i} visitors: {count} minutes ({count/num_minutes:.1%} of time)")

print("\n")

# Example 4: Feature Modeling with Normal Distribution
print("Example 4: Feature Modeling with Normal Distribution")
print("===================================================\n")

print("Problem: Modeling house sizes for a real estate price prediction model")
print("We'll use the Normal distribution to model the distribution of house sizes\n")

# Create synthetic house size data
np.random.seed(42)

# Parameters
mean_size = 2000  # Mean house size in square feet
std_size = 300   # Standard deviation of house sizes
num_houses = 1000  # Number of houses to simulate

# Generate normally-distributed house sizes
house_sizes = np.random.normal(mean_size, std_size, num_houses)

print(f"Step 1: Define the model")
print(f"We model house sizes as a Normal distribution:")
print(f"X ~ N(μ, σ²), where μ = {mean_size} and σ = {std_size}")
print()

print(f"Step 2: Analyze the synthetic house size data")
observed_mean = np.mean(house_sizes)
observed_std = np.std(house_sizes)
min_size = np.min(house_sizes)
max_size = np.max(house_sizes)

print(f"House Size Statistics:")
print(f"- Number of houses analyzed: {num_houses}")
print(f"- Mean house size: {observed_mean:.2f} sq ft")
print(f"- Standard deviation: {observed_std:.2f} sq ft")
print(f"- Minimum size: {min_size:.2f} sq ft")
print(f"- Maximum size: {max_size:.2f} sq ft")
print()

print(f"Step 3: Apply the model for preprocessing and analysis")

# Calculate standardized values (z-scores)
z_scores = (house_sizes - observed_mean) / observed_std

# Count outliers (houses with sizes more than 3 standard deviations from the mean)
outlier_threshold = 3
outliers = np.abs(z_scores) > outlier_threshold
num_outliers = np.sum(outliers)

print(f"Feature standardization converts house sizes to z-scores:")
print(f"Z = (X - μ) / σ = (X - {observed_mean:.2f}) / {observed_std:.2f}")
print()

print(f"Outlier detection using the 3-sigma rule:")
print(f"Houses with sizes more than {outlier_threshold} standard deviations from the mean")
print(f"are considered outliers (< {observed_mean - 3*observed_std:.2f} or > {observed_mean + 3*observed_std:.2f} sq ft)")
print(f"Found {num_outliers} outliers out of {num_houses} houses ({num_outliers/num_houses:.1%})")
print()

# Calculate probability for a specific range
lower_bound = 1800
upper_bound = 2200
prob_in_range = stats.norm.cdf(upper_bound, observed_mean, observed_std) - \
                stats.norm.cdf(lower_bound, observed_mean, observed_std)
z_lower = (lower_bound - observed_mean) / observed_std
z_upper = (upper_bound - observed_mean) / observed_std

print(f"Probability calculations using the normal distribution:")
print(f"P({lower_bound} < X < {upper_bound}) = P({z_lower:.2f} < Z < {z_upper:.2f})")
print(f"= {stats.norm.cdf(z_upper)} - {stats.norm.cdf(z_lower)}")
print(f"= {prob_in_range:.4f}")
print()
print(f"This means about {prob_in_range:.1%} of houses have sizes between {lower_bound} and {upper_bound} sq ft.")
print()

# Visualization of the normal distribution
plt.figure(figsize=(15, 10))

# Plot 1: Histogram with normal PDF overlay
plt.subplot(2, 2, 1)
plt.hist(house_sizes, bins=30, density=True, alpha=0.6, color='skyblue')

# Plot the PDF of the fitted normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, observed_mean, observed_std)
plt.plot(x, p, 'k', linewidth=2)

plt.xlabel('House Size (sq ft)')
plt.ylabel('Probability Density')
plt.title('Distribution of House Sizes with Normal PDF')
plt.grid(alpha=0.3)

# Plot 2: Q-Q plot to assess normality
plt.subplot(2, 2, 2)
stats.probplot(house_sizes, dist="norm", plot=plt)
plt.title('Q-Q Plot to Assess Normality')
plt.grid(alpha=0.3)

# Plot 3: Standardized values (z-scores)
plt.subplot(2, 2, 3)
plt.hist(z_scores, bins=30, color='lightgreen', alpha=0.6)
plt.axvline(x=-outlier_threshold, color='r', linestyle='--', 
            label=f'-{outlier_threshold}σ Threshold')
plt.axvline(x=outlier_threshold, color='r', linestyle='--', 
            label=f'+{outlier_threshold}σ Threshold')
plt.xlabel('Z-score')
plt.ylabel('Frequency')
plt.title('Standardized House Sizes (Z-scores)')
plt.legend()
plt.grid(alpha=0.3)

# Plot 4: Probability visualization
plt.subplot(2, 2, 4)
x = np.linspace(mean_size - 4*std_size, mean_size + 4*std_size, 1000)
y = stats.norm.pdf(x, observed_mean, observed_std)
plt.plot(x, y, 'b-', linewidth=2)

# Shade the area for the given range
x_range = np.linspace(lower_bound, upper_bound, 100)
y_range = stats.norm.pdf(x_range, observed_mean, observed_std)
plt.fill_between(x_range, y_range, alpha=0.3, color='green', 
                 label=f'{lower_bound} to {upper_bound} sq ft')

plt.axvline(x=observed_mean, color='r', linestyle='-', label=f'Mean = {observed_mean:.0f}')
plt.axvline(x=observed_mean - observed_std, color='k', linestyle='--', 
            label=f'Mean ± σ')
plt.axvline(x=observed_mean + observed_std, color='k', linestyle='--')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Probability Density')
plt.title(f'Normal Distribution: P({lower_bound} < X < {upper_bound}) = {prob_in_range:.4f}')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_house_sizes.png'), dpi=300)
plt.close()

# Generate additional visualizations showing feature engineering applications
plt.figure(figsize=(15, 6))

# Create some missing values in a copy of our data
sizes_with_missing = house_sizes.copy()
missing_indices = np.random.choice(np.arange(num_houses), size=int(num_houses*0.1), replace=False)
sizes_with_missing[missing_indices] = np.nan

# Plot 1: Missing value imputation
plt.subplot(1, 2, 1)
imputed_sizes = sizes_with_missing.copy()
imputed_sizes[np.isnan(imputed_sizes)] = observed_mean


plt.hist([house_sizes, imputed_sizes], bins=30, alpha=0.6, 
         label=['Original Data', 'After Imputation'])
plt.axvline(x=observed_mean, color='r', linestyle='-', linewidth=2,
            label=f'Mean = {observed_mean:.0f}')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Frequency')
plt.title('Missing Value Imputation with Mean')
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Feature standardization for model training
plt.subplot(1, 2, 2)
plt.hist(z_scores, bins=30, alpha=0.6, color='green')
plt.axvline(x=0, color='r', linestyle='-', linewidth=2,
            label='Mean = 0')
plt.axvline(x=-1, color='k', linestyle='--',
            label='±1 std dev')
plt.axvline(x=1, color='k', linestyle='--')
plt.xlabel('Standardized Size (Z-score)')
plt.ylabel('Frequency')
plt.title('Standardized Feature for Model Training')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_feature_engineering.png'), dpi=300)
plt.close()

print("\n")

# Example 5: Neural Network Initialization with Normal Distribution
print("Example 5: Neural Network Initialization with Normal Distribution")
print("==============================================================\n")

print("Problem: Initializing weights for a neural network layer")
print("We'll use the Normal distribution with appropriate variance for weight initialization\n")

np.random.seed(42)

# Define a simple neural network layer dimensions
n_in = 784  # Input size (e.g., MNIST images: 28x28 = 784)
n_out = 100  # Output size (e.g., hidden layer with 100 neurons)

print(f"Step 1: Choose the distribution for weight initialization")
print(f"For a layer with {n_in} input neurons and {n_out} output neurons:")
print(f"We'll use Xavier/Glorot initialization with a normal distribution:")
print(f"W ~ N(0, σ²) where σ² = 2 / (n_in + n_out)")
print()

# Calculate variance according to Xavier/Glorot initialization
xavier_variance = 2.0 / (n_in + n_out)
xavier_stddev = np.sqrt(xavier_variance)

# Also calculate He initialization for comparison
he_variance = 2.0 / n_in
he_stddev = np.sqrt(he_variance)

# And standard normal initialization for comparison
std_normal_stddev = 1.0

print(f"Step 2: Calculate the initialization parameters")
print(f"Xavier/Glorot variance: σ² = 2 / ({n_in} + {n_out}) = {xavier_variance:.6f}")
print(f"Xavier/Glorot standard deviation: σ = {xavier_stddev:.6f}")
print()
print(f"For comparison:")
print(f"He initialization (better for ReLU): σ = √(2 / {n_in}) = {he_stddev:.6f}")
print(f"Standard normal initialization: σ = 1.0")
print()

# Generate weights using different initialization methods
weights_xavier = np.random.normal(0, xavier_stddev, size=(n_in, n_out))
weights_he = np.random.normal(0, he_stddev, size=(n_in, n_out))
weights_std = np.random.normal(0, std_normal_stddev, size=(n_in, n_out))

print(f"Step 3: Analyze the initialized weights")
print(f"Xavier/Glorot initialization statistics:")
print(f"- Mean: {np.mean(weights_xavier):.6f}")
print(f"- Standard deviation: {np.std(weights_xavier):.6f}")
print(f"- Min value: {np.min(weights_xavier):.6f}")
print(f"- Max value: {np.max(weights_xavier):.6f}")
print()

# Visualizations of different weight initializations
plt.figure(figsize=(15, 10))

# Plot 1: Histograms of weights for different initialization methods
plt.subplot(2, 2, 1)
plt.hist(weights_xavier.flatten(), bins=50, alpha=0.6, label='Xavier/Glorot')
plt.hist(weights_he.flatten(), bins=50, alpha=0.6, label='He')
plt.hist(weights_std.flatten(), bins=50, alpha=0.6, label='Standard Normal')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Comparison of Weight Initialization Methods')
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Analyze activation outputs for a random input
random_input = np.random.normal(0, 1, size=(100, n_in))  # 100 random examples

# Forward propagation for linear activations (no activation function)
linear_activation_xavier = np.dot(random_input, weights_xavier)
linear_activation_he = np.dot(random_input, weights_he)
linear_activation_std = np.dot(random_input, weights_std)

# Plot histograms of pre-activation values
plt.subplot(2, 2, 2)
plt.hist(linear_activation_xavier.flatten(), bins=50, alpha=0.6, label='Xavier/Glorot')
plt.hist(linear_activation_he.flatten(), bins=50, alpha=0.6, label='He')
plt.hist(linear_activation_std.flatten(), bins=50, alpha=0.6, label='Standard Normal')
plt.xlabel('Pre-activation Value')
plt.ylabel('Frequency')
plt.title('Distribution of Pre-activation Values')
plt.legend()
plt.grid(alpha=0.3)

# Plot 3: Standard deviation of activations across neurons
plt.subplot(2, 2, 3)
# Calculate standard deviation for each neuron's output
std_per_neuron_xavier = np.std(linear_activation_xavier, axis=0)
std_per_neuron_he = np.std(linear_activation_he, axis=0)
std_per_neuron_std = np.std(linear_activation_std, axis=0)

# Plot distribution of standard deviations
plt.hist(std_per_neuron_xavier, bins=20, alpha=0.6, label=f'Xavier/Glorot: μ={np.mean(std_per_neuron_xavier):.2f}')
plt.hist(std_per_neuron_he, bins=20, alpha=0.6, label=f'He: μ={np.mean(std_per_neuron_he):.2f}')
plt.hist(std_per_neuron_std, bins=20, alpha=0.6, label=f'Standard: μ={np.mean(std_per_neuron_std):.2f}')
plt.xlabel('Standard Deviation per Neuron')
plt.ylabel('Frequency')
plt.title('Variability Across Neurons')
plt.legend()
plt.grid(alpha=0.3)

# Plot 4: Visualize weight matrix patterns (subset of weights)
plt.subplot(2, 2, 4)
subset_size = 50  # Show a subset of the weight matrix for visibility
subset_weights = weights_xavier[:subset_size, :subset_size]
plt.imshow(subset_weights, cmap='viridis')
plt.colorbar(label='Weight Value')
plt.title(f'Xavier/Glorot Weight Matrix Visualization\n(First {subset_size}x{subset_size} weights)')
plt.xlabel('Output Neuron Index')
plt.ylabel('Input Neuron Index')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'neural_network_initialization.png'), dpi=300)
plt.close()

# Additional visualization of neural network initialization impact
plt.figure(figsize=(15, 6))

# Simple sigmoid activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Apply activation functions
sigmoid_xavier = sigmoid(linear_activation_xavier)
sigmoid_std = sigmoid(linear_activation_std)

# Plot 1: Sigmoid activations with different initializations
plt.subplot(1, 2, 1)
plt.hist(sigmoid_xavier.flatten(), bins=50, alpha=0.6, label='Xavier/Glorot')
plt.hist(sigmoid_std.flatten(), bins=50, alpha=0.6, label='Standard Normal')
plt.xlabel('Sigmoid Activation Value')
plt.ylabel('Frequency')
plt.title('Impact of Initialization on Sigmoid Activations')
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Network training with different initializations (simulated)
plt.subplot(1, 2, 2)
# Simulated loss curves
epochs = np.arange(1, 51)
loss_xavier = 2 * np.exp(-0.1 * epochs) + 0.5 * np.random.random(50)
loss_he = 2 * np.exp(-0.15 * epochs) + 0.3 * np.random.random(50)
loss_std = 3 - 2.5 * np.exp(-0.05 * epochs) + 0.8 * np.random.random(50)

plt.plot(epochs, loss_xavier, 'b-', label='Xavier/Glorot')
plt.plot(epochs, loss_he, 'g-', label='He')
plt.plot(epochs, loss_std, 'r-', label='Standard Normal')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Simulated Training Progress with Different Initializations')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'nn_init_performance.png'), dpi=300)
plt.close()

print(f"Step 4: Understand the benefits of proper initialization")
print(f"Proper weight initialization with variance scaled to layer dimensions:")
print(f"- Keeps activations and gradients from vanishing or exploding")
print(f"- Helps the network converge faster during training")
print(f"- Reduces the chance of neurons saturating (especially with sigmoid/tanh)")
print(f"- Improves the overall training stability")
print("\n")

# Example 6: Bayesian Inference with Beta Distribution
print("Example 6: Bayesian Inference with Beta Distribution")
print("==================================================\n")

print("Problem: Estimating the efficacy rate of a new drug using Bayesian inference")
print("We'll use the Beta distribution as a prior and update it with observed data\n")

np.random.seed(42)

print(f"Step 1: Define the prior distribution")
# Prior belief: the efficacy rate is around 70%, but with uncertainty
prior_alpha = 7  # Prior successes
prior_beta = 3   # Prior failures

# Plot the beta prior at various x values
x = np.linspace(0, 1, 1000)
prior_dist = stats.beta(prior_alpha, prior_beta)
prior_pdf = prior_dist.pdf(x)

# Calculate prior mean and 95% credible interval
prior_mean = prior_alpha / (prior_alpha + prior_beta)
prior_mode = (prior_alpha - 1) / (prior_alpha + prior_beta - 2) if prior_alpha > 1 and prior_beta > 1 else "N/A"
prior_var = (prior_alpha * prior_beta) / ((prior_alpha + prior_beta)**2 * (prior_alpha + prior_beta + 1))
prior_std = np.sqrt(prior_var)
prior_ci = prior_dist.ppf([0.025, 0.975])

print(f"We use a Beta({prior_alpha}, {prior_beta}) prior, which represents:")
print(f"- Prior mean: {prior_mean:.4f} (expected efficacy rate)")
print(f"- Prior mode: {prior_mode if isinstance(prior_mode, str) else prior_mode:.4f} (most likely value)")
print(f"- Prior standard deviation: {prior_std:.4f}")
print(f"- 95% prior credible interval: [{prior_ci[0]:.4f}, {prior_ci[1]:.4f}]")
print()

print(f"Step 2: Collect data and define the likelihood")
# Simulated clinical trial data
n_patients = 20   # Total number of patients in trial
n_success = 15    # Number of successful treatments

print(f"Clinical trial results:")
print(f"- Total patients: {n_patients}")
print(f"- Successful treatments: {n_success}")
print(f"- Sample proportion: {n_success/n_patients:.4f}")
print()
print(f"This follows a Binomial({n_patients}, θ) distribution, where θ is the unknown efficacy rate.")
print()

print(f"Step 3: Apply Bayes' theorem to get the posterior")
# Calculate posterior parameters
posterior_alpha = prior_alpha + n_success
posterior_beta = prior_beta + (n_patients - n_success)
posterior_dist = stats.beta(posterior_alpha, posterior_beta)
posterior_pdf = posterior_dist.pdf(x)

# Calculate posterior mean and 95% credible interval
posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
posterior_mode = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2) if posterior_alpha > 1 and posterior_beta > 1 else "N/A"
posterior_var = (posterior_alpha * posterior_beta) / ((posterior_alpha + posterior_beta)**2 * (posterior_alpha + posterior_beta + 1))
posterior_std = np.sqrt(posterior_var)
posterior_ci = posterior_dist.ppf([0.025, 0.975])

print(f"For the Beta-Binomial model, the posterior is Beta(α + successes, β + failures):")
print(f"Posterior = Beta({prior_alpha} + {n_success}, {prior_beta} + {n_patients - n_success})")
print(f"= Beta({posterior_alpha}, {posterior_beta})")
print()
print(f"Posterior distribution statistics:")
print(f"- Posterior mean: {posterior_mean:.4f} (updated estimate of efficacy rate)")
print(f"- Posterior mode: {posterior_mode if isinstance(posterior_mode, str) else posterior_mode:.4f} (most likely value)")
print(f"- Posterior standard deviation: {posterior_std:.4f}")
print(f"- 95% posterior credible interval: [{posterior_ci[0]:.4f}, {posterior_ci[1]:.4f}]")
print()

print(f"Step 4: Make decisions using the posterior")
# Calculate probabilities of interest
prob_greater_than_half = 1 - posterior_dist.cdf(0.5)
prob_greater_than_0_8 = 1 - posterior_dist.cdf(0.8)
prob_less_than_0_6 = posterior_dist.cdf(0.6)

print(f"Using the posterior distribution, we can calculate:")
print(f"- P(θ > 0.5) = {prob_greater_than_half:.4f}")
print(f"  (Probability that the drug is better than a coin flip)")
print(f"- P(θ > 0.8) = {prob_greater_than_0_8:.4f}")
print(f"  (Probability that the drug is highly effective)")
print(f"- P(θ < 0.6) = {prob_less_than_0_6:.4f}")
print(f"  (Probability that the drug efficacy is less than 60%)")
print()

# Visualizations of Bayesian inference with Beta distribution
plt.figure(figsize=(15, 10))

# Plot 1: Prior, likelihood, and posterior
plt.subplot(2, 2, 1)
plt.plot(x, prior_pdf, 'b-', label=f'Prior: Beta({prior_alpha}, {prior_beta})', linewidth=2)
plt.plot(x, posterior_pdf, 'r-', label=f'Posterior: Beta({posterior_alpha}, {posterior_beta})', linewidth=2)

# Add a scaled binomial likelihood
x_discrete = np.linspace(0.01, 0.99, 99)
likelihood = [stats.binom.pmf(n_success, n_patients, p) for p in x_discrete]
likelihood_scaled = np.array(likelihood) / max(likelihood) * max(posterior_pdf) * 0.8  # Scale for visibility
plt.plot(x_discrete, likelihood_scaled, 'g--', label='Scaled Likelihood (Binomial)', linewidth=2)

plt.axvline(x=prior_mean, color='b', linestyle='--', label=f'Prior Mean: {prior_mean:.2f}')
plt.axvline(x=posterior_mean, color='r', linestyle='--', label=f'Posterior Mean: {posterior_mean:.2f}')
plt.axvline(x=n_success/n_patients, color='g', linestyle=':', label=f'Sample Proportion: {n_success/n_patients:.2f}')

plt.xlabel('Efficacy Rate (θ)')
plt.ylabel('Probability Density')
plt.title('Bayesian Inference for Drug Efficacy')
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Credible intervals
plt.subplot(2, 2, 2)
plt.plot(x, posterior_pdf, 'r-', linewidth=2)

# Shade the 95% credible interval
x_ci = np.linspace(posterior_ci[0], posterior_ci[1], 100)
y_ci = posterior_dist.pdf(x_ci)
plt.fill_between(x_ci, y_ci, alpha=0.3, color='red', 
                 label=f'95% Credible Interval: [{posterior_ci[0]:.2f}, {posterior_ci[1]:.2f}]')

plt.axvline(x=posterior_mean, color='r', linestyle='--', label=f'Posterior Mean: {posterior_mean:.2f}')
if not isinstance(posterior_mode, str):
    plt.axvline(x=posterior_mode, color='darkred', linestyle='-.', label=f'Posterior Mode: {posterior_mode:.2f}')

plt.xlabel('Efficacy Rate (θ)')
plt.ylabel('Probability Density')
plt.title('Posterior Distribution with 95% Credible Interval')
plt.legend()
plt.grid(alpha=0.3)

# Plot 3: Visualize probability thresholds
plt.subplot(2, 2, 3)
plt.plot(x, posterior_pdf, 'r-', linewidth=2)

# Shade areas corresponding to various probability statements
x_gt_half = np.linspace(0.5, 1, 100)
y_gt_half = posterior_dist.pdf(x_gt_half)
plt.fill_between(x_gt_half, y_gt_half, alpha=0.3, color='green', 
                 label=f'P(θ > 0.5) = {prob_greater_than_half:.3f}')

x_gt_08 = np.linspace(0.8, 1, 100)
y_gt_08 = posterior_dist.pdf(x_gt_08)
plt.fill_between(x_gt_08, y_gt_08, alpha=0.5, color='blue', 
                 label=f'P(θ > 0.8) = {prob_greater_than_0_8:.3f}')

plt.axvline(x=0.5, color='green', linestyle='--')
plt.axvline(x=0.8, color='blue', linestyle='--')
plt.xlabel('Efficacy Rate (θ)')
plt.ylabel('Probability Density')
plt.title('Posterior Probabilities of Interest')
plt.legend()
plt.grid(alpha=0.3)

# Plot 4: Bayesian updating with different sample sizes
plt.subplot(2, 2, 4)

# Calculate posteriors for different sample sizes (keeping the same proportion)
sample_sizes = [0, 5, 20, 50, 100]  # 0 means prior only
proportions = [0.75] * len(sample_sizes)  # Same proportion for all
alphas = [prior_alpha + proportions[i] * size for i, size in enumerate(sample_sizes)]
betas = [prior_beta + (1 - proportions[i]) * size for i, size in enumerate(sample_sizes)]

for i, size in enumerate(sample_sizes):
    if size == 0:
        label = f'Prior'
    else:
        label = f'n = {size}'
    
    current_dist = stats.beta(alphas[i], betas[i])
    current_pdf = current_dist.pdf(x)
    plt.plot(x, current_pdf, linewidth=2, label=label)

plt.xlabel('Efficacy Rate (θ)')
plt.ylabel('Probability Density')
plt.title('Bayesian Updating with Increasing Sample Size')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'bayesian_beta_inference.png'), dpi=300)
plt.close()

# Additional visualization of Bayesian sequential updating
plt.figure(figsize=(15, 6))

# Generate some sequential data (patients one by one)
np.random.seed(42)
true_rate = 0.75
sequential_outcomes = np.random.binomial(1, true_rate, 20)  # 20 patients
cumulative_successes = np.cumsum(sequential_outcomes)
cumulative_trials = np.arange(1, len(sequential_outcomes) + 1)
cumulative_proportions = cumulative_successes / cumulative_trials

# Plot 1: Sequential updating of the Beta distribution
plt.subplot(1, 2, 1)
alpha_seq = prior_alpha
beta_seq = prior_beta

selected_steps = [0, 1, 5, 10, 20]  # Prior and after 1, 5, 10, 20 patients
colors = ['blue', 'green', 'orange', 'red', 'purple']

for i, step in enumerate(selected_steps):
    if step == 0:
        # Just plot the prior
        current_alpha = alpha_seq
        current_beta = beta_seq
        label = 'Prior'
    else:
        # Update with data up to this step
        successes = cumulative_successes[step-1]
        trials = cumulative_trials[step-1]
        current_alpha = alpha_seq + successes
        current_beta = beta_seq + (trials - successes)
        label = f'After {step} patients'
    
    current_dist = stats.beta(current_alpha, current_beta)
    current_pdf = current_dist.pdf(x)
    plt.plot(x, current_pdf, color=colors[i], linewidth=2, label=label)

plt.xlabel('Efficacy Rate (θ)')
plt.ylabel('Probability Density')
plt.title('Sequential Bayesian Updating')
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Evolution of the posterior mean and credible intervals
plt.subplot(1, 2, 2)

# Calculate posterior means and CIs for each step
posterior_means = []
lower_cis = []
upper_cis = []

for t in range(len(sequential_outcomes)):
    current_alpha = alpha_seq + cumulative_successes[t]
    current_beta = beta_seq + (t + 1 - cumulative_successes[t])
    
    mean = current_alpha / (current_alpha + current_beta)
    ci = stats.beta(current_alpha, current_beta).ppf([0.025, 0.975])
    
    posterior_means.append(mean)
    lower_cis.append(ci[0])
    upper_cis.append(ci[1])

# Plot the evolution of the posterior mean and credible intervals
plt.plot(cumulative_trials, posterior_means, 'r-', label='Posterior Mean')
plt.fill_between(cumulative_trials, lower_cis, upper_cis, alpha=0.2, color='red',
                label='95% Credible Interval')
plt.plot(cumulative_trials, cumulative_proportions, 'ko--', label='Observed Proportion')
plt.axhline(y=true_rate, color='blue', linestyle='--', label=f'True Rate: {true_rate}')

plt.xlabel('Number of Patients')
plt.ylabel('Efficacy Rate (θ)')
plt.title('Evolution of Posterior Mean and Credible Intervals')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'bayesian_sequential_updating.png'), dpi=300)
plt.close()

print("The Beta distribution is ideal for Bayesian inference about probabilities:")
print("- It provides a full probability distribution for our parameter of interest")
print("- It is conjugate to the Binomial likelihood, making updates analytically tractable")
print("- It allows us to incorporate prior knowledge and update beliefs based on evidence")
print("- It quantifies uncertainty in our estimates with credible intervals")
print("\n") 