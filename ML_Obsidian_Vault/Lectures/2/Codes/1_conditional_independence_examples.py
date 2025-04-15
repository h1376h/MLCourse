import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images/Conditional_Independence relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Conditional_Independence")

# Create images directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

def print_step(step_num, title, content):
    print(f"\nStep {step_num}: {title}")
    print("-" * 50)
    print(content)
    print("-" * 50)

# Example 1: Medical Diagnosis Bayesian Network
print("\n=== Example 1: Medical Diagnosis Bayesian Network ===")

# Step 1: Define Network Structure
print_step(1, "Define Network Structure", 
"""We create a directed graph where:
- D (Disease) is the parent node
- S (Symptoms) and T (Test) are child nodes
This structure implies that symptoms and test results are conditionally independent given the disease status.""")

G = nx.DiGraph()
G.add_edges_from([('D', 'S'), ('D', 'T')])

# Step 2: Define Probabilities
print_step(2, "Define Probabilities",
"""We define the following probabilities:
- P(D) = 0.010 (base rate of disease)
- P(S|D) = 0.900 (sensitivity of symptoms)
- P(S|¬D) = 0.100 (false positive rate)
- P(T|D) = 0.950 (test sensitivity)
- P(T|¬D) = 0.050 (test false positive rate)""")

P_D = 0.01
P_S_given_D = 0.9
P_S_given_not_D = 0.1
P_T_given_D = 0.95
P_T_given_not_D = 0.05

# Step 3: Calculate Joint Probability
print_step(3, "Calculate Joint Probability",
f"""Using the chain rule of probability:
P(D,S,T) = P(D) × P(S|D) × P(T|D)
= {P_D} × {P_S_given_D} × {P_T_given_D}
= {P_D * P_S_given_D * P_T_given_D:.6f}""")

joint_prob = P_D * P_S_given_D * P_T_given_D

# Step 4: Verify Conditional Independence
print_step(4, "Verify Conditional Independence",
f"""We verify that P(S|D,T) = P(S|D):
P(S|D,T) = P(S|D) = {P_S_given_D}
This shows that symptoms and test results are conditionally independent given the disease status.""")

# Visualize the network
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=2000, font_size=16, font_weight='bold')
plt.title('Medical Diagnosis Bayesian Network')
plt.savefig(os.path.join(images_dir, 'medical_bayesian_network.png'))
plt.close()

# Example 2: Naive Bayes Classifier
print("\n=== Example 2: Naive Bayes Classifier ===")

# Step 1: Define Class and Word Probabilities
print_step(1, "Define Class and Word Probabilities",
"""We define:
- P(spam) = 0.3
- P(not_spam) = 0.7
And word probabilities given class:
Spam:
- P(money|spam) = 0.80
- P(lottery|spam) = 0.60
- P(viagra|spam) = 0.70
Not Spam:
- P(money|not_spam) = 0.10
- P(lottery|not_spam) = 0.05
- P(viagra|not_spam) = 0.01""")

P_spam = 0.3
P_not_spam = 0.7

word_probs_spam = {
    'money': 0.8,
    'lottery': 0.6,
    'viagra': 0.7
}

word_probs_not_spam = {
    'money': 0.1,
    'lottery': 0.05,
    'viagra': 0.01
}

# Step 2: Calculate Joint Probability for Specific Email
print_step(2, "Calculate Joint Probability",
f"""For an email containing 'money' and 'lottery':
P(email,spam) = P(spam) × P(money|spam) × P(lottery|spam)
= {P_spam} × {word_probs_spam['money']} × {word_probs_spam['lottery']}
= {P_spam * word_probs_spam['money'] * word_probs_spam['lottery']:.6f}

P(email,not_spam) = P(not_spam) × P(money|not_spam) × P(lottery|not_spam)
= {P_not_spam} × {word_probs_not_spam['money']} × {word_probs_not_spam['lottery']}
= {P_not_spam * word_probs_not_spam['money'] * word_probs_not_spam['lottery']:.6f}""")

# Step 3: Verify Conditional Independence
print_step(3, "Verify Conditional Independence",
"""The Naive Bayes assumption states that words are conditionally independent given the class.
This is evident from the factorization of the joint probability, where we multiply individual word probabilities.""")

# Visualize the network
G = nx.DiGraph()
G.add_edges_from([('Class', 'money'), ('Class', 'lottery'), ('Class', 'viagra')])
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', 
        node_size=2000, font_size=16, font_weight='bold')
plt.title('Naive Bayes Structure')
plt.savefig(os.path.join(images_dir, 'naive_bayes_structure.png'))
plt.close()

# Example 3: Hidden Markov Model
print("\n=== Example 3: Hidden Markov Model ===")

# Step 1: Define States and Observations
print_step(1, "Define States and Observations",
"""We define:
- States: A, B, C
- Observations: 1, 2, 3
The HMM makes two key assumptions:
1. Current state depends only on previous state
2. Current observation depends only on current state""")

# Step 2: Define Transition Probabilities
print_step(2, "Define Transition Probabilities",
"""Transition probabilities from each state:
From A: [0.7 0.2 0.1]
From B: [0.3 0.5 0.2]
From C: [0.1 0.3 0.6]""")

transition_probs = {
    'A': [0.7, 0.2, 0.1],
    'B': [0.3, 0.5, 0.2],
    'C': [0.1, 0.3, 0.6]
}

# Step 3: Define Emission Probabilities
print_step(3, "Define Emission Probabilities",
"""Emission probabilities for each state:
State A: [0.6 0.3 0.1]
State B: [0.2 0.5 0.3]
State C: [0.1 0.2 0.7]""")

emission_probs = {
    'A': [0.6, 0.3, 0.1],
    'B': [0.2, 0.5, 0.3],
    'C': [0.1, 0.2, 0.7]
}

# Step 4: Demonstrate Conditional Independence
print_step(4, "Demonstrate Conditional Independence",
f"""1. Current state depends only on previous state:
P(X_2|X_1,X_0) = P(X_2|X_1) = {transition_probs['A'][1]:.2f}

2. Current observation depends only on current state:
P(Y_2|X_2,Y_1,X_1) = P(Y_2|X_2) = {emission_probs['B'][1]:.2f}""")

# Visualize the HMM
G = nx.DiGraph()
G.add_edges_from([('X1', 'X2'), ('X2', 'X3'), 
                  ('X1', 'Y1'), ('X2', 'Y2'), ('X3', 'Y3')])
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightyellow', 
        node_size=2000, font_size=16, font_weight='bold')
plt.title('Hidden Markov Model Structure')
plt.savefig(os.path.join(images_dir, 'hmm_structure.png'))
plt.close()

# Example 4: Weather Prediction
print("\n=== Example 4: Weather Prediction ===")

# Step 1: Define Variables and Structure
print_step(1, "Define Variables and Structure",
"""We define three variables:
- W: Weather (sunny/rainy)
- T: Temperature (high/low)
- H: Humidity (high/low)
The structure shows that temperature and humidity are conditionally independent given the weather.""")

# Step 2: Define Weather Probabilities
print_step(2, "Define Weather Probabilities",
f"""Base probabilities:
- P(W=sunny) = 0.70
- P(W=rainy) = 0.30""")

P_sunny = 0.7
P_rainy = 0.3

# Step 3: Define Conditional Probabilities
print_step(3, "Define Conditional Probabilities",
f"""Temperature given weather:
- P(T=high|W=sunny) = 0.80
- P(T=high|W=rainy) = 0.20

Humidity given weather:
- P(H=high|W=sunny) = 0.30
- P(H=high|W=rainy) = 0.90""")

P_T_high_sunny = 0.8
P_T_high_rainy = 0.2
P_H_high_sunny = 0.3
P_H_high_rainy = 0.9

# Step 4: Calculate Joint Probability
print_step(4, "Calculate Joint Probability",
f"""For sunny weather with high temperature and high humidity:
P(W=sunny,T=high,H=high) = P(W=sunny) × P(T=high|W=sunny) × P(H=high|W=sunny)
= {P_sunny} × {P_T_high_sunny} × {P_H_high_sunny}
= {P_sunny * P_T_high_sunny * P_H_high_sunny:.4f}""")

# Step 5: Verify Conditional Independence
print_step(5, "Verify Conditional Independence",
f"""We verify that P(T=high|W=sunny,H=high) = P(T=high|W=sunny):
P(T=high|W=sunny,H=high) = P(T=high|W=sunny) = {P_T_high_sunny}""")

# Visualize the network
G = nx.DiGraph()
G.add_edges_from([('W', 'T'), ('W', 'H')])
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightyellow', 
        node_size=2000, font_size=16, font_weight='bold')
plt.title('Weather Prediction Network')
plt.savefig(os.path.join(images_dir, 'weather_network.png'))
plt.close()

# Example 5: Sensor Network
print("\n=== Example 5: Sensor Network ===")

# Step 1: Define Network Structure
print_step(1, "Define Network Structure",
"""We define a sensor network with:
- S: Signal source (high/low)
- R1, R2: Receivers (high/low)
- N1, N2: Noise at each receiver (high/low)
The structure shows that receivers are conditionally independent given the signal and their respective noise sources.""")

# Step 2: Define Signal and Noise Probabilities
print_step(2, "Define Signal and Noise Probabilities",
f"""Base probabilities:
- P(S=high) = 0.50
- P(N1=high) = 0.20
- P(N2=high) = 0.20""")

P_S_high = 0.5
P_N1_high = 0.2
P_N2_high = 0.2

# Step 3: Define Receiver Probabilities
print_step(3, "Define Receiver Probabilities",
f"""Receiver readings given signal and noise:
- P(R1=high|S=high,N1=low) = 0.90
- P(R1=high|S=low,N1=high) = 0.80
- P(R2=high|S=high,N2=low) = 0.90
- P(R2=high|S=low,N2=high) = 0.80""")

P_R1_high_S_high_N1_low = 0.9
P_R1_high_S_low_N1_high = 0.8
P_R2_high_S_high_N2_low = 0.9
P_R2_high_S_low_N2_high = 0.8

# Step 4: Calculate Joint Probability
print_step(4, "Calculate Joint Probability",
f"""For high signal and both receivers reading high:
P(S=high,R1=high,R2=high) = P(S=high) × P(R1=high|S=high) × P(R2=high|S=high)
= {P_S_high} × {P_R1_high_S_high_N1_low} × {P_R2_high_S_high_N2_low}
= {P_S_high * P_R1_high_S_high_N1_low * P_R2_high_S_high_N2_low:.4f}""")

# Step 5: Verify Conditional Independence
print_step(5, "Verify Conditional Independence",
"""R1 and R2 are conditionally independent given S and their respective noise sources.
This is evident from the network structure where R1 and R2 have no direct connection.""")

# Visualize the network
G = nx.DiGraph()
G.add_edges_from([('S', 'R1'), ('S', 'R2'), 
                  ('N1', 'R1'), ('N2', 'R2')])
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', 
        node_size=2000, font_size=16, font_weight='bold')
plt.title('Sensor Network')
plt.savefig(os.path.join(images_dir, 'sensor_network.png'))
plt.close()

print(f"All example visualizations saved to {images_dir}/") 