import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\n=== CONDITIONAL PROBABILITY & BAYES' THEOREM IN ML: STEP-BY-STEP EXAMPLES ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the Lectures/2 directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "L2_1_ML")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Email Spam Classification with Naive Bayes
print("Example 1: Email Spam Classification with Naive Bayes")
print("====================================================\n")

# Simplified email dataset
emails = [
    {"text": "Get discount now buy limited offer", "label": "spam"},
    {"text": "Meeting scheduled for tomorrow", "label": "ham"},
    {"text": "Free money claim your prize now", "label": "spam"},
    {"text": "Project report deadline next week", "label": "ham"},
    {"text": "Buy now with special discount", "label": "spam"},
    {"text": "Reminder for the conference call", "label": "ham"},
    {"text": "Win a free vacation instant claim", "label": "spam"},
    {"text": "Please review the attached document", "label": "ham"},
    {"text": "Amazing deals don't miss out", "label": "spam"},
    {"text": "Notes from yesterday's meeting", "label": "ham"}
]

# Create DataFrame
df = pd.DataFrame(emails)
print("Email Dataset:")
print(df)
print("\n")

# Convert to lists for easier processing
texts = df['text'].tolist()
labels = df['label'].tolist()

# Step 1: Feature extraction - convert text to word counts
print("Step 1: Feature Extraction (Bag of Words)")
print("----------------------------------------\n")

# Use CountVectorizer to convert text to word counts
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

# Show the vocabulary and document-term matrix
print(f"Vocabulary (features): {feature_names}")
print("\nDocument-Term Matrix:")
dtm = pd.DataFrame(X.toarray(), columns=feature_names)
dtm['label'] = labels
print(dtm)
print("\n")

# Step 2: Calculate Prior Probabilities
print("Step 2: Calculate Prior Probabilities P(Class)")
print("--------------------------------------------\n")

# Calculate P(spam) and P(ham)
num_docs = len(labels)
spam_count = labels.count('spam')
ham_count = labels.count('ham')

p_spam = spam_count / num_docs
p_ham = ham_count / num_docs

print(f"P(spam) = {spam_count} / {num_docs} = {p_spam:.2f}")
print(f"P(ham) = {ham_count} / {num_docs} = {p_ham:.2f}")
print("\n")

# Step 3: Calculate Likelihoods
print("Step 3: Calculate Likelihoods P(Word|Class)")
print("------------------------------------------\n")

# Get spam and ham document indices
spam_indices = [i for i, label in enumerate(labels) if label == 'spam']
ham_indices = [i for i, label in enumerate(labels) if label == 'ham']

# Count word occurrences in each class
word_counts = {}
for word in feature_names:
    # Get column index for the word
    word_idx = list(feature_names).index(word)
    
    # Count occurrences in spam and ham
    spam_word_count = sum(X[i, word_idx] for i in spam_indices)
    ham_word_count = sum(X[i, word_idx] for i in ham_indices)
    
    # Total words in each class (with Laplace smoothing)
    total_spam_words = X[spam_indices].sum() + len(feature_names)
    total_ham_words = X[ham_indices].sum() + len(feature_names)
    
    # Calculate P(word|spam) and P(word|ham) with Laplace smoothing
    p_word_given_spam = (spam_word_count + 1) / total_spam_words
    p_word_given_ham = (ham_word_count + 1) / total_ham_words
    
    word_counts[word] = {
        'count_spam': spam_word_count,
        'count_ham': ham_word_count,
        'p_word_given_spam': p_word_given_spam,
        'p_word_given_ham': p_word_given_ham
    }

# Display word probabilities for selected words
selected_words = ['discount', 'free', 'meeting', 'report']
print("Word Likelihoods with Laplace Smoothing:")
print("| Word     | Count in Spam | Count in Ham | P(Word|Spam) | P(Word|Ham) |")
print("|----------|---------------|--------------|-------------|-------------|")
for word in selected_words:
    if word in word_counts:
        stats = word_counts[word]
        print(f"| {word:<8} | {stats['count_spam']:<13} | {stats['count_ham']:<12} | {stats['p_word_given_spam']:.4f} | {stats['p_word_given_ham']:.4f} |")
print("\n")

# Step 4: Naive Bayes Classification (manual calculation for a new email)
print("Step 4: Manual Naive Bayes Classification")
print("---------------------------------------\n")

# New email to classify
new_email = "Free discount offer for you"
print(f"New email: \"{new_email}\"")

# Tokenize the new email and match to our vocabulary
new_email_words = [word.lower() for word in new_email.split() if word.lower() in feature_names]
print(f"Words in our vocabulary: {new_email_words}")

# Calculate P(spam|email) using Bayes' theorem
log_prob_spam = np.log(p_spam)
log_prob_ham = np.log(p_ham)

print("\nCalculating posterior probabilities using log probabilities to avoid underflow:")
print(f"Starting with log(P(spam)) = log({p_spam:.2f}) = {log_prob_spam:.4f}")
print(f"Starting with log(P(ham)) = log({p_ham:.2f}) = {log_prob_ham:.4f}")

for word in new_email_words:
    if word in word_counts:
        p_word_spam = word_counts[word]['p_word_given_spam']
        p_word_ham = word_counts[word]['p_word_given_ham']
        
        log_prob_spam += np.log(p_word_spam)
        log_prob_ham += np.log(p_word_ham)
        
        print(f"  After '{word}': log_prob_spam += log({p_word_spam:.4f}) = {log_prob_spam:.4f}")
        print(f"  After '{word}': log_prob_ham += log({p_word_ham:.4f}) = {log_prob_ham:.4f}")

# Convert back from logs and normalize
prob_spam = np.exp(log_prob_spam)
prob_ham = np.exp(log_prob_ham)
total = prob_spam + prob_ham
prob_spam_normalized = prob_spam / total
prob_ham_normalized = prob_ham / total

print(f"\nFinal posterior probabilities (normalized):")
print(f"P(spam|email) = {prob_spam_normalized:.4f}")
print(f"P(ham|email) = {prob_ham_normalized:.4f}")
print(f"Conclusion: The email is classified as {'spam' if prob_spam_normalized > prob_ham_normalized else 'ham'}")
print("\n")

# Step 5: Using scikit-learn's Naive Bayes
print("Step 5: Using scikit-learn's Naive Bayes")
print("---------------------------------------\n")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train the model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Get predictions
y_pred = nb_classifier.predict(X_test)

# Print results
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Process the new email
new_email_vec = vectorizer.transform([new_email])
prediction = nb_classifier.predict(new_email_vec)[0]
proba = nb_classifier.predict_proba(new_email_vec)[0]

print(f"scikit-learn classification for '{new_email}':")
print(f"Prediction: {prediction}")
print(f"Probability of spam: {proba[list(nb_classifier.classes_).index('spam')]:.4f}")
print(f"Probability of ham: {proba[list(nb_classifier.classes_).index('ham')]:.4f}")
print("\n")

# Example 2: Medical Diagnosis with Bayes' Theorem
print("Example 2: Medical Diagnosis with Bayes' Theorem")
print("===============================================\n")

# Medical diagnosis problem
print("Problem: Using Bayes' theorem for medical diagnosis")
print("A disease affects 1% of the population. There is a test that is 90% sensitive (P(+|D) = 0.9)")
print("and 80% specific (P(-|~D) = 0.8).")
print("If a person tests positive, what is the probability they have the disease?")
print("\n")

# Define the probabilities
p_disease = 0.01  # P(D)
p_no_disease = 0.99  # P(~D)
p_pos_given_disease = 0.9  # P(+|D)
p_neg_given_no_disease = 0.8  # P(-|~D)
p_pos_given_no_disease = 0.2  # P(+|~D)

# Step 1: Calculate P(+) using Law of Total Probability
print("Step 1: Calculate P(+) using Law of Total Probability")
p_pos = p_pos_given_disease * p_disease + p_pos_given_no_disease * p_no_disease
print(f"P(+) = P(+|D) × P(D) + P(+|~D) × P(~D)")
print(f"    = {p_pos_given_disease} × {p_disease} + {p_pos_given_no_disease} × {p_no_disease}")
print(f"    = {p_pos_given_disease * p_disease} + {p_pos_given_no_disease * p_no_disease}")
print(f"    = {p_pos}")
print()

# Step 2: Apply Bayes' theorem
print("Step 2: Apply Bayes' Theorem to find P(D|+)")
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos
print(f"P(D|+) = P(+|D) × P(D) / P(+)")
print(f"       = ({p_pos_given_disease} × {p_disease}) / {p_pos}")
print(f"       = {p_pos_given_disease * p_disease} / {p_pos}")
print(f"       = {p_disease_given_pos:.4f}")
print()

# Visualization for the medical example
plt.figure(figsize=(12, 6))

# Bar chart for comparing probabilities
plt.subplot(1, 2, 1)
probs = [p_disease, p_disease_given_pos]
labels = ['P(Disease)', 'P(Disease|Positive Test)']
plt.bar(labels, probs, color=['blue', 'red'])
plt.title('Disease Probability')
plt.ylabel('Probability')
plt.ylim(0, 0.1) # Adjusted to show the small probabilities better
for i, v in enumerate(probs):
    plt.text(i, v + 0.005, f'{v:.4f}', ha='center')

# Create a visual representation of Bayes' rule for medical diagnosis
plt.subplot(1, 2, 2)

# Total population
total = 1000
diseased = total * p_disease
non_diseased = total * p_no_disease

# Test results
true_positives = diseased * p_pos_given_disease
false_negatives = diseased * (1 - p_pos_given_disease)
false_positives = non_diseased * p_pos_given_no_disease
true_negatives = non_diseased * p_neg_given_no_disease

# Create a 2x2 table visualization
table_data = [
    [f"True Positives\n{true_positives:.0f}", f"False Positives\n{false_positives:.0f}"],
    [f"False Negatives\n{false_negatives:.0f}", f"True Negatives\n{true_negatives:.0f}"]
]

table = plt.table(cellText=table_data,
                 rowLabels=['Disease', 'No Disease'],
                 colLabels=['Positive Test', 'Negative Test'],
                 loc='center',
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)
plt.axis('off')
plt.title('Confusion Matrix (per 1000 people)')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'medical_diagnosis_bayes.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Chain Rule Application in Sequential Data (Natural Language Processing)
print("Example 3: Chain Rule Application in Sequential Data (NLP)")
print("=========================================================\n")

print("Problem: Calculate the probability of a sentence using the chain rule")
print("Sentence: 'machine learning is fun'")
print("\n")

# Define a simple language model with conditional probabilities
# P(word_i | word_{i-1})
conditional_probs = {
    '<start>': {'machine': 0.1, 'learning': 0.05, 'is': 0.2, 'fun': 0.05},
    'machine': {'learning': 0.8, 'is': 0.1, 'fun': 0.01},
    'learning': {'machine': 0.01, 'is': 0.4, 'fun': 0.1},
    'is': {'machine': 0.05, 'learning': 0.1, 'fun': 0.5},
    'fun': {'machine': 0.05, 'learning': 0.1, 'is': 0.1}
}

# Sentence to analyze
sentence = "machine learning is fun"
words = sentence.split()

# Calculate probability using the chain rule
p_sentence = 1.0
prev_word = '<start>'

print("Step 1: Apply the Chain Rule: P(w1,w2,...,wn) = P(w1) × P(w2|w1) × P(w3|w1,w2) × ... × P(wn|w1,...,wn-1)")
print("Using a simple Markov model (first-order): P(w1,w2,...,wn) ≈ P(w1) × P(w2|w1) × P(w3|w2) × ... × P(wn|wn-1)")
print("\nCalculation:")

# First word probability P(machine)
p_first_word = conditional_probs['<start>'][words[0]]
p_sentence *= p_first_word
print(f"P({words[0]}) = P({words[0]}|<start>) = {p_first_word}")

# Subsequent word probabilities P(word_i|word_{i-1})
for i in range(1, len(words)):
    if prev_word in conditional_probs and words[i] in conditional_probs[prev_word]:
        p_current = conditional_probs[prev_word][words[i]]
        print(f"P({words[i]}|{prev_word}) = {p_current}")
        p_sentence *= p_current
    else:
        p_current = 0.001  # small probability for unknown transitions
        print(f"P({words[i]}|{prev_word}) = {p_current} (unknown transition)")
        p_sentence *= p_current
    prev_word = words[i]

print(f"\nFinal probability of sentence: P('{sentence}') = {p_sentence:.8f}")

# Visualization for language model probabilities
plt.figure(figsize=(10, 6))

# Create a simple graph representation
nodes = ['<start>'] + words
plt.scatter([0, 1, 2, 3, 4], [0, 0, 0, 0, 0], s=1000, c='lightblue', zorder=1)

# Add node labels
for i, node in enumerate(nodes):
    plt.text(i, 0, node, ha='center', va='center', fontsize=12)

# Add directed edges with transition probabilities
for i in range(len(nodes)-1):
    if nodes[i] in conditional_probs and nodes[i+1] in conditional_probs[nodes[i]]:
        prob = conditional_probs[nodes[i]][nodes[i+1]]
        plt.annotate(f"{prob:.2f}", 
                     xy=(i+1, 0.05), 
                     xytext=(i, 0.05), 
                     arrowprops=dict(arrowstyle="->", color='blue'),
                     ha='center', va='bottom', fontsize=10)

plt.title('Markov Chain Representation of Sentence Probability')
plt.xlim(-0.5, len(nodes)-0.5)
plt.ylim(-0.5, 0.5)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'nlp_chain_rule.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Marginalization and Hidden Variables (Latent Variable Models)
print("Example 4: Marginalization and Hidden Variables")
print("=============================================\n")

print("Problem: Customer purchase data analysis with hidden customer types")
print("We have customer data and want to model the probability of purchase behavior,")
print("taking into account hidden (latent) customer types.")
print("\n")

# Define model parameters
# Two customer types: price-sensitive (0) and feature-driven (1)
# Products: basic, standard, premium

# Prior probabilities of customer types
p_type = [0.6, 0.4]  # P(Type=price-sensitive), P(Type=feature-driven)

# Conditional probabilities of purchase given customer type
# P(Product | Type)
p_product_given_type = {
    'basic': [0.7, 0.2],     # P(basic | type0), P(basic | type1)
    'standard': [0.25, 0.3],  # P(standard | type0), P(standard | type1)
    'premium': [0.05, 0.5]   # P(premium | type0), P(premium | type1)
}

# Step 1: Marginal probabilities using Total Probability Law
print("Step 1: Calculate marginal product probabilities using the Law of Total Probability")
print("P(Product) = ∑_type P(Product|Type) × P(Type)")

p_product = {}
for product in p_product_given_type:
    p_product[product] = sum(p_product_given_type[product][t] * p_type[t] for t in range(len(p_type)))
    
    # Show calculation
    calculation = " + ".join([f"P({product}|Type={t}) × P(Type={t}) = {p_product_given_type[product][t]} × {p_type[t]} = {p_product_given_type[product][t] * p_type[t]:.4f}" for t in range(len(p_type))])
    print(f"P({product}) = {calculation} = {p_product[product]:.4f}")

print()

# Step 2: Calculate posterior probabilities of customer types given purchases
print("Step 2: Calculate posterior probabilities using Bayes' Theorem")
print("P(Type|Product) = P(Product|Type) × P(Type) / P(Product)")

p_type_given_product = {}
for product in p_product_given_type:
    p_type_given_product[product] = []
    
    for t in range(len(p_type)):
        # Bayes' theorem
        p_posterior = (p_product_given_type[product][t] * p_type[t]) / p_product[product]
        p_type_given_product[product].append(p_posterior)
        
        # Show calculation
        print(f"P(Type={t}|{product}) = P({product}|Type={t}) × P(Type={t}) / P({product})")
        print(f"                     = {p_product_given_type[product][t]} × {p_type[t]} / {p_product[product]:.4f}")
        print(f"                     = {p_product_given_type[product][t] * p_type[t]:.4f} / {p_product[product]:.4f}")
        print(f"                     = {p_posterior:.4f}")
        print()

# Visualization for the latent variable model
products = list(p_product_given_type.keys())
types = ["Price-sensitive", "Feature-driven"]

plt.figure(figsize=(15, 6))

# Plot 1: Prior and marginal probabilities
plt.subplot(1, 3, 1)
plt.bar(types, p_type, color='lightblue')
plt.title('Prior Probabilities of Customer Types')
plt.ylabel('Probability')
plt.ylim(0, 1)
for i, v in enumerate(p_type):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')

# Plot 2: Conditional probabilities of products given types
plt.subplot(1, 3, 2)
bar_width = 0.35
index = np.arange(len(products))

# For each customer type, plot the probability of each product
for t in range(len(types)):
    # Get probabilities for this customer type across all products
    probs = [p_product_given_type[product][t] for product in products]
    plt.bar(index + t * bar_width, probs, bar_width, label=types[t])

plt.xlabel('Product')
plt.ylabel('Conditional Probability')
plt.title('P(Product|Type)')
plt.xticks(index + bar_width/2, products)
plt.legend()
plt.ylim(0, 1)

# Plot 3: Posterior probabilities of types given products
plt.subplot(1, 3, 3)
index = np.arange(len(types))

# For each product, plot the posterior probability of each customer type
for p, product in enumerate(products):
    # Get posterior probabilities for this product across all types
    posterior_probs = p_type_given_product[product]
    plt.bar(index + p * bar_width, posterior_probs, bar_width, label=product)

plt.xlabel('Customer Type')
plt.ylabel('Posterior Probability')
plt.title('P(Type|Product)')
plt.xticks(index + bar_width, types)
plt.legend()
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'latent_variable_model.png'), dpi=100, bbox_inches='tight')
plt.close()

print("All conditional probability and Bayes' theorem example images created successfully.") 