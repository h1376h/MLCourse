# Well-Posed Learning Problem Examples

Tom Mitchell's definition: "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E."

## Example 1: Email Spam Classification

### Problem Statement
Every day, people receive unwanted spam emails mixed with important messages. A machine learning system can analyze incoming emails and separate the legitimate ones from spam, saving users time and protecting them from potential threats.

### Solution

- **Task (T)**: Classifying emails as spam or not spam
- **Experience (E)**: Database of emails with spam/not spam labels
- **Performance (P)**: Accuracy of spam detection, or F1 score balancing precision and recall

## Example 2: Chess Playing Program

### Problem Statement
Chess is a complex game with many possible moves at each turn. A machine learning system can learn to play chess by studying games and developing strategies to defeat human or computer opponents.

### Solution

- **Task (T)**: Playing chess
- **Experience (E)**: Playing games against itself or human opponents
- **Performance (P)**: Percentage of games won against test opponents, or Elo rating

## Example 3: Image Recognition

### Problem Statement
Computers traditionally couldn't "see" and understand images the way humans do. A machine learning system can be trained to recognize objects, people, and scenes in photos and videos, enabling applications like photo organization and visual search.

### Solution

- **Task (T)**: Identifying objects in images
- **Experience (E)**: Database of labeled images showing various objects
- **Performance (P)**: Percentage of objects correctly identified in new images

## Example 4: Recommendation System

### Problem Statement
With millions of products, movies, or songs available, finding items a person will like can be overwhelming. A machine learning system can suggest items based on what a user has liked before and what similar users have enjoyed.

### Solution

- **Task (T)**: Recommending products to users
- **Experience (E)**: Historical data of user purchases and ratings
- **Performance (P)**: Click-through rate, conversion rate, or user engagement

## Example 5: Self-Driving Car

### Problem Statement
Driving requires making many complex decisions based on visual information and knowledge of traffic rules. A machine learning system can process sensor data from a car to navigate roads safely without human intervention.

### Solution

- **Task (T)**: Driving safely on roads
- **Experience (E)**: Sensor data paired with correct driving decisions (from human demonstrations or simulations)
- **Performance (P)**: Distance traveled without human intervention, or rate of traffic rule violations

## Example 6: Natural Language Translation

### Problem Statement
Translating between languages requires understanding grammar, context, and cultural nuances. A machine learning system can convert text from one language to another, helping people communicate across language barriers.

### Solution

- **Task (T)**: Translating text between languages
- **Experience (E)**: Corpus of texts with translations in target languages
- **Performance (P)**: BLEU score measuring translation quality compared to human translations

## Example 7: Medical Diagnosis

### Problem Statement
Doctors use patient symptoms, test results, and medical images to diagnose diseases. A machine learning system can assist by analyzing this data to identify patterns associated with different conditions, potentially catching issues human doctors might miss.

### Solution

- **Task (T)**: Diagnosing diseases from medical images/data
- **Experience (E)**: Database of patient data with confirmed diagnoses
- **Performance (P)**: Accuracy, sensitivity, and specificity of diagnostic predictions

## Example 8: Stock Price Prediction

### Problem Statement
Stock prices change based on countless factors including company performance, market trends, and world events. A machine learning system can analyze historical data to predict how prices might change in the future, helping investors make decisions.

### Solution

- **Task (T)**: Predicting future stock prices
- **Experience (E)**: Historical stock price data and relevant market indicators
- **Performance (P)**: Mean squared error between predicted and actual prices, or trading profit using the predictions

## Example 9: Customer Churn Prediction

### Problem Statement
Businesses lose money when customers cancel subscriptions or stop using their services. A machine learning system can identify which customers are likely to leave based on their behavior patterns, allowing companies to take steps to retain them.

### Solution

- **Task (T)**: Predicting which customers will leave a service
- **Experience (E)**: Historical customer data including who churned
- **Performance (P)**: Area under the ROC curve (AUC) or precision-recall curve

## Example 10: Speech Recognition

### Problem Statement
Converting spoken words to text enables voice assistants, transcription services, and accessibility features. A machine learning system can analyze audio recordings of speech and determine which words were spoken.

### Solution

- **Task (T)**: Converting spoken language to text
- **Experience (E)**: Audio recordings paired with correct transcriptions
- **Performance (P)**: Word error rate in transcribing new audio 

## Quiz Example

### Problem Statement
You are designing a machine learning system to help a streaming music service create personalized playlists for users. For each of the following descriptions, identify the Task (T), Experience (E), and Performance Measure (P) according to Tom Mitchell's definition of a well-posed learning problem.

### Solution

**Scenario**: A streaming music service wants to automatically generate customized playlists that match each user's musical taste.

**Task (T)**: Generating personalized music playlists for users that match their preferences.

**Experience (E)**: 
- Historical data of songs users have listened to
- Skip/complete rates for songs
- Explicit ratings users have given to songs
- Playlists users have created themselves
- Listening patterns (time of day, sequence of songs, etc.)

**Performance (P)**: 
- Average listening time before skipping a recommended song
- Percentage of recommended songs that users add to their own playlists
- User ratings of automatically generated playlists
- User retention rate when using personalized playlists vs. generic ones

**Explanation**:
The system learns from users' listening history and ratings (Experience) to perform the task of creating personalized playlists (Task). Its success is measured by how engaged users are with the recommendations, shown through metrics like listening time and explicit feedback (Performance). 