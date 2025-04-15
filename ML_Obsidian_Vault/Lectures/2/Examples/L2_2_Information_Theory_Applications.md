# Information Theory Applications - Examples

## 1. Decision Tree Information Gain

### Example: Calculating Information Gain

Consider a dataset for classifying whether someone will play tennis based on weather conditions:

| Weather   | Play Tennis |
|-----------|-------------|
| Sunny     | No          |
| Sunny     | No          |
| Overcast  | Yes         |
| Rainy     | Yes         |
| Rainy     | Yes         |
| Rainy     | No          |
| Overcast  | Yes         |
| Sunny     | No          |
| Sunny     | Yes         |
| Rainy     | Yes         |
| Sunny     | Yes         |
| Overcast  | Yes         |
| Overcast  | Yes         |
| Rainy     | No          |

**Step 1: Calculate Entropy of Target Variable**

P(Play=Yes) = 9/14 = 0.643
P(Play=No) = 5/14 = 0.357

H(Play) = -0.643 × log₂(0.643) - 0.357 × log₂(0.357) = 0.940 bits

**Step 2: Calculate Conditional Entropy for each Feature**

For Weather:
- P(Weather=Sunny) = 5/14 = 0.357
  - P(Play=Yes|Weather=Sunny) = 2/5 = 0.4
  - P(Play=No|Weather=Sunny) = 3/5 = 0.6
  - H(Play|Weather=Sunny) = -0.4 × log₂(0.4) - 0.6 × log₂(0.6) = 0.971 bits
- P(Weather=Overcast) = 4/14 = 0.286
  - P(Play=Yes|Weather=Overcast) = 4/4 = 1
  - P(Play=No|Weather=Overcast) = 0/4 = 0
  - H(Play|Weather=Overcast) = -1 × log₂(1) - 0 × log₂(0) = 0 bits
- P(Weather=Rainy) = 5/14 = 0.357
  - P(Play=Yes|Weather=Rainy) = 3/5 = 0.6
  - P(Play=No|Weather=Rainy) = 2/5 = 0.4
  - H(Play|Weather=Rainy) = -0.6 × log₂(0.6) - 0.4 × log₂(0.4) = 0.971 bits

H(Play|Weather) = 0.357 × 0.971 + 0.286 × 0 + 0.357 × 0.971 = 0.693 bits

**Step 3: Calculate Information Gain**

Information Gain = H(Play) - H(Play|Weather) = 0.940 - 0.693 = 0.247 bits

## 2. Cross-Entropy Loss in Neural Networks

### Example: Binary Classification with Cross-Entropy Loss

Consider a neural network model predicting whether an email is spam (1) or not spam (0):

| Email | True Label | Predicted Probability |
|-------|------------|------------------------|
| 1     | 1          | 0.8                    |
| 2     | 0          | 0.2                    |
| 3     | 1          | 0.6                    |
| 4     | 0          | 0.3                    |

The cross-entropy loss for each example is:
- Email 1: -(1 × log(0.8) + 0 × log(1-0.8)) = -log(0.8) = 0.223
- Email 2: -(0 × log(0.2) + 1 × log(1-0.2)) = -log(0.8) = 0.223
- Email 3: -(1 × log(0.6) + 0 × log(1-0.6)) = -log(0.6) = 0.511
- Email 4: -(0 × log(0.3) + 1 × log(1-0.3)) = -log(0.7) = 0.357

Average cross-entropy loss = (0.223 + 0.223 + 0.511 + 0.357) / 4 = 0.328

This loss can be minimized through gradient descent to improve the model.

## 3. Mutual Information for Feature Selection

### Example: Feature Selection in a Dataset

Consider a dataset with features X₁, X₂, X₃ and target variable Y. We've calculated the mutual information between each feature and the target:

- I(X₁; Y) = 0.8 bits
- I(X₂; Y) = 0.3 bits
- I(X₃; Y) = 0.5 bits

Since X₁ has the highest mutual information with Y, it's the most informative feature for predicting Y. However, if we're selecting two features, we need to consider redundant information between features. If X₁ and X₃ share significant mutual information (are highly correlated), then X₁ and X₂ might be a better combination despite X₂ having lower individual mutual information.

## 4. KL Divergence in Variational Autoencoders

### Example: VAE Training

In a Variational Autoencoder, we optimize:

L(θ, φ) = E_q_φ(z|x)[log p_θ(x|z)] - D_KL(q_φ(z|x) || p(z))

Where:
- q_φ(z|x) is the encoder that approximates the posterior
- p_θ(x|z) is the decoder that reconstructs the input
- p(z) is the prior distribution (typically N(0,1))
- D_KL term regularizes the latent space

Assuming q_φ(z|x) = N(μ, σ²) and p(z) = N(0, 1), the KL divergence has closed form:

D_KL = 0.5 × (μ² + σ² - log(σ²) - 1)

This term encourages the latent space to be close to the prior distribution while still capturing meaningful features from the data. 