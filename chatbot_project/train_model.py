import json
import pickle
import string
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset from intents.json (Kaggle-style)
# -----------------------------
DATASET_PATH = os.path.join(os.path.dirname(__file__), "intents.json")

print(f"Loading dataset from: {DATASET_PATH}")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

intents = data["intents"]

# -----------------------------
# Build training data and response map
# -----------------------------
def clean_text(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

questions = []
labels = []
intent_responses = {}

for intent in intents:
    tag = intent["tag"]
    patterns = intent["patterns"]
    responses = intent["responses"]

    # Map tag -> list of responses
    intent_responses[tag] = responses

    for pattern in patterns:
        questions.append(clean_text(pattern))
        labels.append(tag)

print(f"Total intents: {len(intents)}")
print(f"Total training samples: {len(questions)}")

# -----------------------------
# Vectorize with TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # Use unigrams + bigrams for better accuracy
    max_features=5000
)
X = vectorizer.fit_transform(questions)
y = labels

# Split dataset (use 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

# -----------------------------
# Train Naive Bayes Model
# -----------------------------
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate Accuracy
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------
# Save Model, Vectorizer, and Responses
# -----------------------------
MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), "chatbot", "model.pkl")
pickle.dump((model, vectorizer, intent_responses), open(MODEL_OUTPUT, "wb"))
print(f"\nModel saved successfully to: {MODEL_OUTPUT}")

# Also save to root dir for compatibility
ROOT_MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), "model.pkl")
pickle.dump((model, vectorizer, intent_responses), open(ROOT_MODEL_OUTPUT, "wb"))
print(f"Model also saved to: {ROOT_MODEL_OUTPUT}")