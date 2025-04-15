import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib_venn import venn2
import os
from matplotlib.ticker import PercentFormatter

print("\n=== CONDITIONAL PROBABILITY EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the parent directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Simple Conditional Probability (Students studying ML and DS)
print("Example 1: Simple Conditional Probability (Students studying ML and DS)")
total_students = 50
ml_students = 30
ds_students = 25
both_students = 15

print(f"Total students: {total_students}")
print(f"Students studying ML: {ml_students}")
print(f"Students studying DS: {ds_students}")
print(f"Students studying both ML and DS: {both_students}")

# Calculate probabilities
p_ml = ml_students / total_students
p_ds = ds_students / total_students
p_both = both_students / total_students

print("\nStep 1: Identify events and their probabilities")
print(f"P(M) = Students studying ML = {ml_students}/{total_students} = {p_ml:.3f}")
print(f"P(D) = Students studying DS = {ds_students}/{total_students} = {p_ds:.3f}")
print(f"P(M ∩ D) = Students studying both ML and DS = {both_students}/{total_students} = {p_both:.3f}")

print("\nStep 2: Calculate the conditional probability P(M|D)")
p_ml_given_ds = p_both / p_ds
print(f"P(M|D) = P(M ∩ D) / P(D) = {p_both:.3f} / {p_ds:.3f} = {p_ml_given_ds:.3f}")

print("\nStep 3: Calculate the conditional probability P(D|M)")
p_ds_given_ml = p_both / p_ml
print(f"P(D|M) = P(M ∩ D) / P(M) = {p_both:.3f} / {p_ml:.3f} = {p_ds_given_ml:.3f}")

print("\nTherefore:")
print(f"a) The probability that a student studies ML given that they study DS is {p_ml_given_ds:.3f} or {p_ml_given_ds*100:.1f}%.")
print(f"b) The probability that a student studies DS given that they study ML is {p_ds_given_ml:.3f} or {p_ds_given_ml*100:.1f}%.")

# Create a clean Venn diagram for the student example
plt.figure(figsize=(8, 5))
# Using the values for the diagram
ml_only = ml_students - both_students
ds_only = ds_students - both_students

# Create the Venn diagram - the sizes here are proportional to the actual values
v = venn2(subsets=(ml_only, ds_only, both_students), 
          set_labels=('Machine Learning', 'Data Science'))

# Add counts to the diagram
v.get_label_by_id('10').set_text(f'{ml_only}')
v.get_label_by_id('01').set_text(f'{ds_only}')
v.get_label_by_id('11').set_text(f'{both_students}')

# Set colors
v.get_patch_by_id('10').set_color('lightblue')
v.get_patch_by_id('01').set_color('lightgreen')
v.get_patch_by_id('11').set_color('lightyellow')

# Add minimal title
plt.title('Students Studying ML and DS', fontsize=12)

# Save the clean visualization
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'students_conditional_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Card Drawing
print("\n\nExample 2: Card Drawing")
total_cards = 52
hearts = 13
red_cards = 26
spades = 13
face_cards_per_suit = 3
face_cards_in_spades = 3

print(f"Total cards in a deck: {total_cards}")
print(f"Red cards (hearts and diamonds): {red_cards}")
print(f"Hearts: {hearts}")
print(f"Spades: {spades}")
print(f"Face cards per suit (Jack, Queen, King): {face_cards_per_suit}")
print(f"Face cards in spades: {face_cards_in_spades}")

# Calculate probabilities
p_heart = hearts / total_cards
p_red = red_cards / total_cards
p_heart_given_red = hearts / red_cards
p_spade = spades / total_cards
p_face_card_in_spade = face_cards_in_spades / total_cards
p_face_card_given_spade = face_cards_in_spades / spades

print("\nStep 1: Identify the relevant sets")
print(f"P(heart) = {hearts}/{total_cards} = {p_heart:.3f}")
print(f"P(red) = {red_cards}/{total_cards} = {p_red:.3f}")
print(f"P(spade) = {spades}/{total_cards} = {p_spade:.3f}")
print(f"P(face card in spade) = {face_cards_in_spades}/{total_cards} = {p_face_card_in_spade:.3f}")

print("\nStep 2: Calculate conditional probability P(heart|red)")
print(f"P(heart|red) = P(heart ∩ red) / P(red) = {p_heart:.3f} / {p_red:.3f} = {p_heart_given_red:.3f}")

print("\nStep 3: Calculate conditional probability P(face card|spade)")
print(f"P(face card|spade) = P(face card ∩ spade) / P(spade) = {p_face_card_in_spade:.3f} / {p_spade:.3f} = {p_face_card_given_spade:.3f}")

print("\nTherefore:")
print(f"a) The probability that the card is a heart given that it's red is {p_heart_given_red:.3f} or {p_heart_given_red*100:.1f}%.")
print(f"b) The probability that the card is a face card given that it's a spade is {p_face_card_given_spade:.3f} or {p_face_card_given_spade*100:.1f}%.")

# Create a clean visual representation of a deck of cards
plt.figure(figsize=(10, 4))

# Scenario 1: Hearts given Red
plt.subplot(1, 2, 1)
labels = ['Hearts', 'Diamonds']
counts = [hearts, red_cards - hearts]
colors = ['red', 'darkred']
bars = plt.bar(labels, counts, color=colors, alpha=0.7)

plt.title('Red Cards in a Deck', fontsize=12)
plt.ylabel('Number of Cards', fontsize=10)
plt.ylim(0, 15)

# Scenario 2: Face cards given Spade
plt.subplot(1, 2, 2)
labels = ['Face Cards', 'Number Cards']
counts = [face_cards_in_spades, spades - face_cards_in_spades]
colors = ['black', 'gray']
bars = plt.bar(labels, counts, color=colors, alpha=0.7)

plt.title('Spades in a Deck', fontsize=12)
plt.ylim(0, 15)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'cards_conditional_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Disease Testing
print("\n\nExample 3: Disease Testing")
prevalence = 0.02  # P(D) = 0.02 (changed from 0.01 to make visualization clearer)
sensitivity = 0.95  # P(T|D) = 0.95
specificity = 0.90  # P(T'|D') = 0.90

print(f"Disease prevalence: {prevalence:.2f} ({prevalence*100:.0f}% of population)")
print(f"Test sensitivity: {sensitivity:.2f} ({sensitivity*100:.0f}% of diseased patients test positive)")
print(f"Test specificity: {specificity:.2f} ({specificity*100:.0f}% of healthy patients test negative)")

# Calculate derived probabilities
p_disease = prevalence
p_no_disease = 1 - prevalence
p_positive_given_disease = sensitivity
p_negative_given_no_disease = specificity
p_positive_given_no_disease = 1 - specificity

print("\nStep 1: Define events and given probabilities")
print(f"P(D) = {p_disease:.4f} (prevalence)")
print(f"P(D') = 1 - P(D) = {p_no_disease:.4f}")
print(f"P(T|D) = {p_positive_given_disease:.4f} (sensitivity)")
print(f"P(T'|D') = {p_negative_given_no_disease:.4f} (specificity)")
print(f"P(T|D') = 1 - P(T'|D') = {p_positive_given_no_disease:.4f}")

# Calculate total probability of a positive test
p_positive = (p_positive_given_disease * p_disease) + (p_positive_given_no_disease * p_no_disease)

print("\nStep 2: Use Bayes' theorem")
print("We need to find P(D|T), which we can calculate using Bayes' theorem:")
print("P(D|T) = [P(T|D) × P(D)] / P(T)")
print("\nTo find P(T), we use the law of total probability:")
print(f"P(T) = P(T|D) × P(D) + P(T|D') × P(D')")
print(f"     = {p_positive_given_disease:.4f} × {p_disease:.4f} + {p_positive_given_no_disease:.4f} × {p_no_disease:.4f}")
print(f"     = {p_positive_given_disease * p_disease:.6f} + {p_positive_given_no_disease * p_no_disease:.6f}")
print(f"     = {p_positive:.6f}")

# Calculate posterior probability (the answer)
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print("\nStep 3: Calculate the final probability")
print(f"P(D|T) = [P(T|D) × P(D)] / P(T)")
print(f"       = [{p_positive_given_disease:.4f} × {p_disease:.4f}] / {p_positive:.6f}")
print(f"       = {p_positive_given_disease * p_disease:.6f} / {p_positive:.6f}")
print(f"       = {p_disease_given_positive:.6f} or {p_disease_given_positive*100:.2f}%")

print(f"\nTherefore, the probability that a person who tests positive actually has the disease is {p_disease_given_positive:.6f} or {p_disease_given_positive*100:.2f}%.")

# Create a visual for the medical test scenario - minimalist design
plt.figure(figsize=(10, 6))

# Create a population diagram to show the concept
population_size = 1000
diseased = int(population_size * prevalence)
non_diseased = population_size - diseased
true_positives = int(diseased * sensitivity)
false_negatives = diseased - true_positives
false_positives = int(non_diseased * (1-specificity))
true_negatives = non_diseased - false_positives
all_positives = true_positives + false_positives

# Top panel: Population distribution - minimal text
ax1 = plt.subplot(211)
plt.barh(['Population'], [population_size], color='lightgray', height=0.5)
plt.barh(['Population'], [diseased], color='#FF6666', height=0.5)

plt.title('Population Distribution', fontsize=14)
plt.xlim(0, population_size)
plt.xticks([0, 250, 500, 750, 1000])
plt.grid(axis='x', alpha=0.3)

# Bottom panel: Test results - minimal text
ax2 = plt.subplot(212)

# Three bars: all negative tests, false positives, and true positives
plt.barh(['Test Results'], [population_size], color='lightgray', height=0.5)
plt.barh(['Test Results'], [all_positives], color='#FFCC99', height=0.5)
plt.barh(['Test Results'], [true_positives], color='#FF6666', height=0.5)

plt.title('Test Results', fontsize=14)
plt.xlim(0, population_size)
plt.xticks([0, 250, 500, 750, 1000])
plt.xlabel('Number of People', fontsize=12)
plt.grid(axis='x', alpha=0.3)

# Save the clean visualization without annotations
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'disease_testing_conditional_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Create a minimalist 2×2 contingency table visualization
plt.figure(figsize=(8, 6))

# Set up the axes
ax = plt.subplot(111)
ax.axis('off')  # Hide the axes

# Draw the table with a clean, minimal style
table_bg = patches.Rectangle((0.1, 0.25), 0.8, 0.5, linewidth=1, 
                           edgecolor='black', facecolor='#F8F8F8', alpha=0.3)
ax.add_patch(table_bg)

# Draw cell borders with thin, clean lines
plt.axvline(x=0.5, ymin=0.25, ymax=0.75, color='black', linestyle='-', linewidth=1)
plt.axhline(y=0.5, xmin=0.1, xmax=0.9, color='black', linestyle='-', linewidth=1)

# Create subtle colors for the cells
true_positive_color = '#FFCCCC'
false_positive_color = '#FFFFCC'
false_negative_color = '#CCFFCC'
true_negative_color = '#CCCCFF'

# Add colored backgrounds to cells
tp_bg = patches.Rectangle((0.1, 0.5), 0.4, 0.25, linewidth=0, facecolor=true_positive_color, alpha=0.7)
fp_bg = patches.Rectangle((0.5, 0.5), 0.4, 0.25, linewidth=0, facecolor=false_positive_color, alpha=0.7)
fn_bg = patches.Rectangle((0.1, 0.25), 0.4, 0.25, linewidth=0, facecolor=false_negative_color, alpha=0.7)
tn_bg = patches.Rectangle((0.5, 0.25), 0.4, 0.25, linewidth=0, facecolor=true_negative_color, alpha=0.7)
ax.add_patch(tp_bg)
ax.add_patch(fp_bg)
ax.add_patch(fn_bg)
ax.add_patch(tn_bg)

# Add minimal headers
plt.text(0.3, 0.8, 'Disease +', ha='center', va='center', fontsize=12)
plt.text(0.7, 0.8, 'Disease -', ha='center', va='center', fontsize=12)
plt.text(0.05, 0.625, 'Test +', ha='right', va='center', fontsize=12)
plt.text(0.05, 0.375, 'Test -', ha='right', va='center', fontsize=12)

# Add only the numbers to the cells
plt.text(0.3, 0.625, f"{true_positives}", ha='center', va='center', fontsize=12)
plt.text(0.7, 0.625, f"{false_positives}", ha='center', va='center', fontsize=12)
plt.text(0.3, 0.375, f"{false_negatives}", ha='center', va='center', fontsize=12)
plt.text(0.7, 0.375, f"{true_negatives}", ha='center', va='center', fontsize=12)

# Add simple row and column totals
plt.text(0.3, 0.2, f"{diseased}", ha='center', va='center', fontsize=11)
plt.text(0.7, 0.2, f"{non_diseased}", ha='center', va='center', fontsize=11)
plt.text(0.95, 0.625, f"{all_positives}", ha='center', va='center', fontsize=11)
plt.text(0.95, 0.375, f"{population_size-all_positives}", ha='center', va='center', fontsize=11)

# Add a simple title
plt.title('Medical Test Results', fontsize=14)

# Save the clean visualization
plt.savefig(os.path.join(images_dir, 'bayes_theorem_disease_testing.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll conditional probability example images created successfully.")

# Example 4: Naive Bayes Text Classification (Spam Detection)
print("\n\nExample 4: Naive Bayes Text Classification (Spam Detection)")

# Define our training data
emails = [
    {"text": "free offer limited time discount", "class": "spam"},
    {"text": "click now free prize winner", "class": "spam"},
    {"text": "urgent reply needed money transfer", "class": "spam"},
    {"text": "meeting tomorrow bring presentation", "class": "not spam"},
    {"text": "project update deadline next week", "class": "not spam"},
    {"text": "lunch plan tuesday restaurant", "class": "not spam"},
    {"text": "report data analysis results ready", "class": "not spam"}
]

# Count class frequencies
class_counts = {"spam": 0, "not spam": 0}
for email in emails:
    class_counts[email["class"]] += 1

# Calculate class probabilities
total_emails = len(emails)
p_spam = class_counts["spam"] / total_emails
p_not_spam = class_counts["not spam"] / total_emails

print(f"Training data: {len(emails)} emails")
print(f"Spam emails: {class_counts['spam']}")
print(f"Not spam emails: {class_counts['not spam']}")
print(f"P(spam) = {p_spam:.3f}")
print(f"P(not spam) = {p_not_spam:.3f}")

# Create a vocabulary and count word occurrences by class
vocabulary = set()
word_counts = {"spam": {}, "not spam": {}}
total_words = {"spam": 0, "not spam": 0}

# Process the emails
for email in emails:
    words = email["text"].lower().split()
    for word in words:
        vocabulary.add(word)
        word_counts[email["class"]][word] = word_counts[email["class"]].get(word, 0) + 1
        total_words[email["class"]] += 1

# Print vocabulary statistics
print(f"\nVocabulary size: {len(vocabulary)}")
print(f"Total words in spam emails: {total_words['spam']}")
print(f"Total words in non-spam emails: {total_words['not spam']}")

print("\nStep 1: Calculate word probabilities by class (with Laplace smoothing)")
# Calculate P(word|class) with Laplace smoothing
word_probs = {"spam": {}, "not spam": {}}
alpha = 1  # Smoothing parameter
vocab_size = len(vocabulary)

for word in vocabulary:
    # P(word|spam) with Laplace smoothing
    count_in_spam = word_counts["spam"].get(word, 0)
    word_probs["spam"][word] = (count_in_spam + alpha) / (total_words["spam"] + alpha * vocab_size)
    
    # P(word|not spam) with Laplace smoothing
    count_in_not_spam = word_counts["not spam"].get(word, 0)
    word_probs["not spam"][word] = (count_in_not_spam + alpha) / (total_words["not spam"] + alpha * vocab_size)

# Print some example word probabilities
print("Example word probabilities:")
example_words = ["free", "meeting", "project", "urgent"]
for word in example_words:
    print(f"P('{word}'|spam) = {word_probs['spam'][word]:.4f}")
    print(f"P('{word}'|not spam) = {word_probs['not spam'][word]:.4f}")
    
# Define a new email to classify
new_email = "free discount offer today"
words_in_new_email = new_email.lower().split()

print(f"\nStep 2: Classify a new email: '{new_email}'")

# Calculate P(spam|words) using Bayes' theorem and the Naive Bayes assumption
# P(spam|words) ∝ P(spam) * P(word_1|spam) * P(word_2|spam) * ... * P(word_n|spam)

# Log probabilities to avoid numerical underflow
log_prob_spam = np.log(p_spam)
log_prob_not_spam = np.log(p_not_spam)

# Multiply the probabilities of each word given the class (or add the log probabilities)
for word in words_in_new_email:
    if word in vocabulary:
        log_prob_spam += np.log(word_probs["spam"][word])
        log_prob_not_spam += np.log(word_probs["not spam"][word])
    else:
        # Handle out-of-vocabulary words with Laplace smoothing
        log_prob_spam += np.log(alpha / (total_words["spam"] + alpha * vocab_size))
        log_prob_not_spam += np.log(alpha / (total_words["not spam"] + alpha * vocab_size))

print(f"Log P(words|spam) + Log P(spam) = {log_prob_spam:.4f}")
print(f"Log P(words|not spam) + Log P(not spam) = {log_prob_not_spam:.4f}")

# Convert from log probabilities back to probabilities
prob_spam = np.exp(log_prob_spam)
prob_not_spam = np.exp(log_prob_not_spam)

# Normalize to get probabilities
total_prob = prob_spam + prob_not_spam
posterior_prob_spam = prob_spam / total_prob
posterior_prob_not_spam = prob_not_spam / total_prob

print(f"\nStep 3: Calculate normalized posterior probabilities")
print(f"P(spam|words) = {posterior_prob_spam:.6f} or {posterior_prob_spam*100:.2f}%")
print(f"P(not spam|words) = {posterior_prob_not_spam:.6f} or {posterior_prob_not_spam*100:.2f}%")

prediction = "spam" if posterior_prob_spam > posterior_prob_not_spam else "not spam"
print(f"\nClassification result: This email is predicted to be {prediction}.")

# Create a clean, minimal visualization for Naive Bayes classification
plt.figure(figsize=(10, 6))

# Create a simple visualization of the email classes
plt.subplot(1, 2, 1)
class_labels = ['Spam', 'Not Spam']
class_values = [class_counts['spam'], class_counts['not spam']]
colors = ['#FF9999', '#66B2FF']
bars = plt.bar(class_labels, class_values, color=colors, alpha=0.7)

plt.title('Email Classes', fontsize=12)
plt.ylabel('Number of Emails', fontsize=10)
plt.ylim(0, max(class_values) + 1)

# Create a clean heatmap of word probabilities
plt.subplot(1, 2, 2)
key_words = ['free', 'discount', 'offer', 'winner', 'urgent', 'money', 'meeting', 'project', 'deadline']

# Prepare the data for the heatmap
word_probs_data = []
for word in key_words:
    spam_prob = word_probs['spam'].get(word, 0)
    not_spam_prob = word_probs['not spam'].get(word, 0)
    word_probs_data.append([spam_prob, not_spam_prob])

# Create the heatmap with minimal styling
heatmap = plt.imshow(word_probs_data, cmap='YlOrRd', aspect='auto')
plt.colorbar(heatmap, label='Probability')
plt.title('Word Probabilities', fontsize=12)
plt.yticks(range(len(key_words)), key_words)
plt.xticks([0, 1], ['Spam', 'Not Spam'])

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'naive_bayes_spam_detection.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll conditional probability example images created successfully.") 