import json
import pickle
import string
import random
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# -----------------------------
# Configuration & Setup
# -----------------------------
DATASET_PATH = os.path.join(os.path.dirname(__file__), "intents.json")
MODEL_OUTPUT_CHATBOT = os.path.join(os.path.dirname(__file__), "chatbot", "model.pkl")
MODEL_OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "model.pkl")

# Download necessary NLTK data (Default path)
def setup_nltk():
    print("Setting up NLTK resources...")
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for res in resources:
        nltk.download(res, quiet=True)

setup_nltk()

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Utility Functions
# -----------------------------
def clean_text(text):
    """
    Advanced cleaning: lowercase, remove punctuation, 
    tokenize, remove stopwords, and lemmatize.
    """
    # 1. Lowercase and strip
    text = text.lower().strip()
    
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tokenize
    words = nltk.word_tokenize(text)
    
    # 4. Lemmatize and remove stopwords
    # Note: 'english' stop_words from sklearn or nltk can sometimes be too aggressive for chatbots
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return " ".join(words)

def load_data(filepath):
    """Load intents dataset from JSON file."""
    print(f"Loading dataset from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]

def prepare_data(intents):
    """Extract patterns and tags from intents and build response map."""
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
            cleaned_pattern = clean_text(pattern)
            if cleaned_pattern: # Only add if not empty after cleaning
                questions.append(cleaned_pattern)
                labels.append(tag)
    
    return questions, labels, intent_responses

# -----------------------------
# Main Training Logic
# -----------------------------
def train():
    # 1. Load and Prepare Data
    intents_data = load_data(DATASET_PATH)
    questions, labels, intent_responses = prepare_data(intents_data)

    print(f"Total intents: {len(intents_data)}")
    print(f"Total training samples: {len(questions)}")

    # 2. Vectorize with TF-IDF (Improved parameters)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),   # Use unigrams, bigrams, and trigrams
        max_features=10000,
        stop_words='english'   # Also use sklearn's internal stop words
    )
    X = vectorizer.fit_transform(questions)
    y = np.array(labels)

    # 3. Split dataset (Stratified split for better evaluation)
    # Note: Some classes might have very few samples, so we handle that
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback to non-stratified if some classes have only 1 member
        print("Warning: Stratified split failed (some classes have too few samples). Falling back to normal split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # 4. Define Models to Compare (Fine-tuned parameters)
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.01), # Lower alpha for better fit
        "Logistic Regression": LogisticRegression(max_iter=2000, solver='lbfgs', C=10) # Higher C for less regularization
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    print("\n" + "="*30)
    print("Advanced Model Comparison Results")
    print("="*30)

    # 5. Train and Evaluate Each Model
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate performance
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel: {name}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        # Keep track of the best performing model
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    print("\n" + "="*30)
    print(f"Best Model Found: {best_model_name} ({best_accuracy * 100:.2f}%)")
    print("="*30)

    # Final Evaluation for the best model
    y_pred_best = best_model.predict(X_test)
    print("\nFull Classification Report for Best Model:")
    print(classification_report(y_test, y_pred_best, zero_division=0))

    # 6. Save Best Model, Vectorizer, and Responses
    print(f"\nSaving Best Model ({best_model_name})...")
    
    # Save to chatbot directory
    with open(MODEL_OUTPUT_CHATBOT, "wb") as f:
        pickle.dump((best_model, vectorizer, intent_responses), f)
    print(f"Model saved successfully to: {MODEL_OUTPUT_CHATBOT}")

    # Also save to root directory for compatibility
    with open(MODEL_OUTPUT_ROOT, "wb") as f:
        pickle.dump((best_model, vectorizer, intent_responses), f)
    print(f"Model also saved to: {MODEL_OUTPUT_ROOT}")

if __name__ == "__main__":
    train()