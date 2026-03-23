import pickle
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Training Data
# -----------------------------
training_data = {
    "greeting": [
        "hello", "hi", "hey", "good morning", "good evening",
        "hiya", "hey there", "morning", "hi there", "hello there"
    ],
    "goodbye": [
        "bye", "goodbye", "see you later", "talk to you soon",
        "catch you later", "see ya", "farewell"
    ],
    "name": [
        "what is your name", "who are you", "tell me your name",
        "your name?", "may I know your name?"
    ],
    "ability": [
        "what can you do", "how can you help", "what are your abilities",
        "help me", "what do you know", "what are your skills"
    ]
}

# Actual responses
intent_responses = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! How's it going?",
        "Hey! Nice to see you."
    ],
    "goodbye": [
        "Goodbye! Have a nice day.",
        "See you later! Take care.",
        "Bye! Talk to you soon."
    ],
    "name": [
        "I am your friendly chatbot.",
        "You can call me Chatbot.",
        "I am an AI assistant here to help you."
    ],
    "ability": [
        "I can answer your questions and help you with tasks.",
        "I can chat with you and provide useful information.",
        "I can assist you in various ways, just ask me."
    ]
}

# -----------------------------
# Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

questions = []
labels = []

for intent, phrases in training_data.items():
    for phrase in phrases:
        questions.append(clean_text(phrase))
        labels.append(intent)

# -----------------------------
# Convert text to numbers
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)
y = labels

# Split dataset (use 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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
print("Model Accuracy:", accuracy * 100, "%")

# -----------------------------
# Save Model and Vectorizer
# -----------------------------
pickle.dump((model, vectorizer, intent_responses), open("model.pkl", "wb"))
print("Model saved successfully!")