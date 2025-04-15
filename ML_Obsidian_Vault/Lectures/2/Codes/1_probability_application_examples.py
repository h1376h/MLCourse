import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

print("\n=== PROBABILITY APPLICATION EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the lecture directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Naive Bayes Text Classification
print("Example 1: Naive Bayes Text Classification")
print("A simple text classifier uses the Naive Bayes approach to categorize news articles")
print("as either sports (S), politics (P), or entertainment (E).")

# Prior probabilities
p_sports = 0.25
p_politics = 0.45
p_entertainment = 0.30

# Conditional probabilities
p_election_given_sports = 0.05
p_election_given_politics = 0.60
p_election_given_entertainment = 0.10

p_game_given_sports = 0.70
p_game_given_politics = 0.08
p_game_given_entertainment = 0.20

print("\nPrior probabilities:")
print(f"P(Sports) = {p_sports}")
print(f"P(Politics) = {p_politics}")
print(f"P(Entertainment) = {p_entertainment}")

print("\nConditional probabilities:")
print(f"P(election|Sports) = {p_election_given_sports}")
print(f"P(election|Politics) = {p_election_given_politics}")
print(f"P(election|Entertainment) = {p_election_given_entertainment}")
print(f"P(game|Sports) = {p_game_given_sports}")
print(f"P(game|Politics) = {p_game_given_politics}")
print(f"P(game|Entertainment) = {p_game_given_entertainment}")

# Step 3: Applying Naive Bayes to compute posterior probabilities
print("\nComputing posterior probabilities using Naive Bayes:")

p_sports_given_words = p_election_given_sports * p_game_given_sports * p_sports
p_politics_given_words = p_election_given_politics * p_game_given_politics * p_politics
p_entertainment_given_words = p_election_given_entertainment * p_game_given_entertainment * p_entertainment

print("\nStep-by-step calculation for P(Sports|election,game):")
print(f"  P(election|Sports) × P(game|Sports) × P(Sports)")
print(f"  {p_election_given_sports} × {p_game_given_sports} × {p_sports} = {p_sports_given_words:.5f}")

print("\nStep-by-step calculation for P(Politics|election,game):")
print(f"  P(election|Politics) × P(game|Politics) × P(Politics)")
print(f"  {p_election_given_politics} × {p_game_given_politics} × {p_politics} = {p_politics_given_words:.5f}")

print("\nStep-by-step calculation for P(Entertainment|election,game):")
print(f"  P(election|Entertainment) × P(game|Entertainment) × P(Entertainment)")
print(f"  {p_election_given_entertainment} × {p_game_given_entertainment} × {p_entertainment} = {p_entertainment_given_words:.5f}")

# Normalizing the probabilities (optional)
sum_probs = p_sports_given_words + p_politics_given_words + p_entertainment_given_words
p_sports_given_words_norm = p_sports_given_words / sum_probs
p_politics_given_words_norm = p_politics_given_words / sum_probs
p_entertainment_given_words_norm = p_entertainment_given_words / sum_probs

print("\nAfter normalization:")
print(f"P(Sports|election,game) = {p_sports_given_words_norm:.5f}")
print(f"P(Politics|election,game) = {p_politics_given_words_norm:.5f}")
print(f"P(Entertainment|election,game) = {p_entertainment_given_words_norm:.5f}")

# Determining the class with highest probability
max_prob = max(p_sports_given_words, p_politics_given_words, p_entertainment_given_words)
if max_prob == p_sports_given_words:
    prediction = "Sports"
elif max_prob == p_politics_given_words:
    prediction = "Politics"
else:
    prediction = "Entertainment"

print(f"\nThe article is classified as: {prediction}")

# Create a bar plot for the results
plt.figure(figsize=(10, 6))
categories = ['Sports', 'Politics', 'Entertainment']
posterior_probs = [p_sports_given_words, p_politics_given_words, p_entertainment_given_words]

# Creating the bar chart
colors = ['skyblue', 'darkred', 'lightgreen']
plt.bar(categories, posterior_probs, color=colors, alpha=0.7)

# Customize the chart
plt.grid(True, alpha=0.3)
plt.ylabel('Unnormalized Posterior Probability', fontsize=12)
plt.title('Naive Bayes Classification Results', fontsize=14)
plt.xticks(fontsize=12)
plt.ylim(0, 0.025)

# Add probability values on top of each bar
for i, prob in enumerate(posterior_probs):
    plt.text(i, prob + 0.001, f'{prob:.5f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'naive_bayes_classification.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Weather Prediction with Total Probability
print("\n\nExample 2: Weather Prediction with Total Probability")
print("Weather forecasting uses conditional probabilities to predict future conditions.")

# Define probabilities
p_rain_given_cloudy = 0.70
p_rain_given_clear = 0.20
p_cloudy = 0.40
p_clear = 0.60  # = 1 - p_cloudy

print("\nGiven information:")
print(f"P(Rain|Cloudy) = {p_rain_given_cloudy}")
print(f"P(Rain|Clear) = {p_rain_given_clear}")
print(f"P(Cloudy) = {p_cloudy}")
print(f"P(Clear) = {p_clear}")

# Apply the law of total probability
print("\nApplying the law of total probability:")
print("P(Rain) = P(Rain|Cloudy) × P(Cloudy) + P(Rain|Clear) × P(Clear)")

term1 = p_rain_given_cloudy * p_cloudy
term2 = p_rain_given_clear * p_clear
p_rain = term1 + term2

print(f"P(Rain) = {p_rain_given_cloudy} × {p_cloudy} + {p_rain_given_clear} × {p_clear}")
print(f"P(Rain) = {term1} + {term2} = {p_rain}")
print(f"\nTherefore, the probability of rain tomorrow is {p_rain*100:.0f}%")

# Create a visualization for the total probability calculation
plt.figure(figsize=(12, 6))

# Create a tree diagram
ax = plt.subplot(1, 2, 1)
plt.axis('off')

# Draw the tree
plt.plot([0, 1], [0, 1], 'k-', lw=2)
plt.plot([0, 1], [0, -1], 'k-', lw=2)

# Add labels
plt.text(0, 0, "Today", ha='right', va='center', fontsize=12, fontweight='bold')
plt.text(1, 1, "Cloudy\n40%", ha='left', va='center', fontsize=12)
plt.text(1, -1, "Clear\n60%", ha='left', va='center', fontsize=12)

# Add second level of the tree
plt.plot([1, 2], [1, 1.5], 'k-', lw=2)
plt.plot([1, 2], [1, 0.5], 'k-', lw=2)
plt.plot([1, 2], [-1, -0.5], 'k-', lw=2)
plt.plot([1, 2], [-1, -1.5], 'k-', lw=2)

# Add second level labels
plt.text(2, 1.5, "Rain\n70%", ha='left', va='center', fontsize=12, color='blue')
plt.text(2, 0.5, "No Rain\n30%", ha='left', va='center', fontsize=12)
plt.text(2, -0.5, "Rain\n20%", ha='left', va='center', fontsize=12, color='blue')
plt.text(2, -1.5, "No Rain\n80%", ha='left', va='center', fontsize=12)

# Add title
plt.text(1, 2, "Weather Probability Tree", ha='center', va='center', fontsize=14, fontweight='bold')

# Create a bar chart for the total probability
ax = plt.subplot(1, 2, 2)
outcomes = ['Rain', 'No Rain']
probabilities = [p_rain, 1-p_rain]

bars = plt.bar(outcomes, probabilities, color=['steelblue', 'lightgray'], alpha=0.7)
plt.grid(True, alpha=0.3)
plt.ylabel('Probability', fontsize=12)
plt.title('Tomorrow\'s Weather Prediction', fontsize=14)
plt.ylim(0, 1.0)

# Add probability values on top of each bar
for i, prob in enumerate(probabilities):
    plt.text(i, prob + 0.02, f'{prob:.2f}', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'total_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Natural Language Processing with Chain Rule
print("\n\nExample 3: Natural Language Processing with Chain Rule")
print("A language model estimates the probability of a sentence by applying the chain rule of probability.")

# Define probabilities
p_the = 0.05
p_cat_given_the = 0.03
p_sat_given_the_cat = 0.2

print("\nGiven information:")
print(f"P(\"The\") = {p_the}")
print(f"P(\"cat\"|\"The\") = {p_cat_given_the}")
print(f"P(\"sat\"|\"The cat\") = {p_sat_given_the_cat}")

# Apply the chain rule
print("\nApplying the chain rule of probability:")
print("P(\"The cat sat\") = P(\"The\") × P(\"cat\"|\"The\") × P(\"sat\"|\"The cat\")")

p_sentence = p_the * p_cat_given_the * p_sat_given_the_cat
print(f"P(\"The cat sat\") = {p_the} × {p_cat_given_the} × {p_sat_given_the_cat} = {p_sentence}")
print(f"P(\"The cat sat\") = {p_sentence:.5f} = {p_sentence:.1e}")

# Create a visualization for the chain rule
plt.figure(figsize=(10, 6))

# Create a table to show the calculation
data = {
    'Word': ['The', 'cat', 'sat'],
    'Conditional Probability': [f'P("The") = {p_the}', 
                                f'P("cat"|"The") = {p_cat_given_the}', 
                                f'P("sat"|"The cat") = {p_sat_given_the_cat}'],
    'Value': [p_the, p_cat_given_the, p_sat_given_the_cat]
}

# Create the table
table = plt.table(cellText=[[d[0], d[1], f'{d[2]:.5f}'] for d in zip(data['Word'], data['Conditional Probability'], data['Value'])],
                  colLabels=['Word', 'Conditional Probability', 'Value'],
                  loc='center',
                  cellLoc='center',
                  bbox=[0.15, 0.6, 0.7, 0.3])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Add title
plt.text(0.5, 0.95, 'Language Model: Chain Rule of Probability', 
         ha='center', va='center', fontsize=14, fontweight='bold')

# Add a visual representation of the sentence
plt.text(0.5, 0.15, 'The cat sat.', 
         ha='center', va='center', fontsize=20, fontweight='bold', style='italic',
         bbox=dict(boxstyle="round4,pad=0.6", facecolor="lightyellow", alpha=0.8))

plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'language_model_chain_rule.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: A/B Testing with Bayesian Statistics
print("\n\nExample 4: A/B Testing with Bayesian Statistics")
print("A technology company is running an A/B test for a new website design")

# Define parameters
alpha_a, beta_a = 50, 950  # Prior for design A
alpha_b, beta_b = 1, 1     # Prior for design B
conversions_b = 60         # Conversions for design B
trials_b = 1000            # Total trials for design B

# Update posterior for design B
posterior_alpha_b = alpha_b + conversions_b
posterior_beta_b = beta_b + (trials_b - conversions_b)

print("\nPrior distributions:")
print(f"Design A prior: Beta({alpha_a}, {beta_a}) centered at {alpha_a/(alpha_a+beta_a):.3f}")
print(f"Design B prior: Beta({alpha_b}, {beta_b}) (uniform prior)")

print("\nObserved data for Design B:")
print(f"Conversions: {conversions_b} out of {trials_b} trials")

print("\nPosterior distribution for Design B:")
print(f"Design B posterior: Beta({posterior_alpha_b}, {posterior_beta_b})")
print(f"Design B posterior mean: {posterior_alpha_b/(posterior_alpha_b+posterior_beta_b):.5f}")

# Run Monte Carlo simulation to compute P(p_B > p_A)
np.random.seed(42)  # For reproducibility
n_samples = 10000
samples_a = np.random.beta(alpha_a, beta_a, n_samples)
samples_b = np.random.beta(posterior_alpha_b, posterior_beta_b, n_samples)
prob_b_better = (samples_b > samples_a).mean()

print("\nMonte Carlo simulation with 10,000 samples:")
print(f"P(Design B better than Design A) = {prob_b_better:.5f} = {prob_b_better*100:.1f}%")

# Create visualization for the A/B test
plt.figure(figsize=(10, 7))

# Plot the distributions
x = np.linspace(0, 0.15, 1000)
y_a = stats.beta.pdf(x, alpha_a, beta_a)
y_b = stats.beta.pdf(x, posterior_alpha_b, posterior_beta_b)

plt.subplot(2, 1, 1)
plt.plot(x, y_a, 'b-', lw=2, label=f'Design A: Beta({alpha_a}, {beta_a})')
plt.plot(x, y_b, 'r-', lw=2, label=f'Design B: Beta({posterior_alpha_b}, {posterior_beta_b})')
plt.axvline(alpha_a/(alpha_a+beta_a), color='blue', linestyle='--', 
            label=f'A Mean: {alpha_a/(alpha_a+beta_a):.3f}')
plt.axvline(posterior_alpha_b/(posterior_alpha_b+posterior_beta_b), color='red', linestyle='--',
            label=f'B Mean: {posterior_alpha_b/(posterior_alpha_b+posterior_beta_b):.3f}')
plt.grid(True, alpha=0.3)
plt.title('A/B Test: Posterior Distributions of Conversion Rates', fontsize=14)
plt.xlabel('Conversion Rate', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()

# Create a second subplot for the samples and probability
plt.subplot(2, 1, 2)
plt.hist(samples_a, bins=30, alpha=0.3, color='blue', label='Design A samples')
plt.hist(samples_b, bins=30, alpha=0.3, color='red', label='Design B samples')
plt.grid(True, alpha=0.3)
plt.xlabel('Conversion Rate', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Monte Carlo Samples of Conversion Rates', fontsize=14)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'ab_testing_bayesian.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll probability application example images created successfully.") 