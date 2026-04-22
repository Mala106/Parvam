import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load the dataset
def load_data(filepath):
    """Load data from train.txt file"""
    texts = []
    emotions = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split by semicolon to separate text and emotion
                parts = line.rsplit(';', 1)
                if len(parts) == 2:
                    text, emotion = parts
                    texts.append(text)
                    emotions.append(emotion)
    
    return texts, emotions

# Load data
print("Loading dataset...")
texts, emotions = load_data('train.txt')
print(f"Total records: {len(texts)}")
print(f"\nEmotion distribution:")
print(pd.Series(emotions).value_counts())

# Create DataFrame
df = pd.DataFrame({
    'text': texts,
    'emotion': emotions
})

# Split into train and test sets
print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    texts, emotions, test_size=0.2, random_state=42, stratify=emotions
)

# Vectorize text using TF-IDF
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes classifier
print("Training model...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate model
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test with sample text
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)
sample_texts = [
    "i feel so happy and excited",
    "i am very sad and depressed",
    "i am so angry right now",
    "i am scared and worried",
    "i love this so much",
    "what a surprise this is"
]

for sample in sample_texts:
    sample_vec = vectorizer.transform([sample])
    prediction = model.predict(sample_vec)[0]
    probability = model.predict_proba(sample_vec)[0]
    print(f"\nText: '{sample}'")
    print(f"Predicted emotion: {prediction}")
    print(f"Confidence: {max(probability):.4f}")
