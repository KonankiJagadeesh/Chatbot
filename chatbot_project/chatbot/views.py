import pickle
import random
import string
import os
from django.shortcuts import render
from django.http import JsonResponse

# -----------------------------
# Load ML model, vectorizer, and responses
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    model, vectorizer, intent_responses = pickle.load(open(MODEL_PATH, "rb"))
except FileNotFoundError:
    model = None
    vectorizer = None
    intent_responses = {}
    print("model.pkl not found. Run train_model.py first.")

# -----------------------------
# Function to clean input and predict response
# -----------------------------
def get_response(text):
    if not model or not vectorizer:
        return "Model is not loaded yet."
    # clean text
    text_clean = text.lower().translate(str.maketrans('', '', string.punctuation))
    vect_text = vectorizer.transform([text_clean])
    # predict intent
    intent = model.predict(vect_text)[0]
    # choose random response
    return random.choice(intent_responses[intent])

# -----------------------------
# View for index page
# -----------------------------
def index(request):
    return render(request, "chatbot/index.html")  # app-level template

# -----------------------------
# API view for AJAX requests
# -----------------------------
def chat_api(request):
    user_message = request.GET.get("message", "")
    if user_message.strip() == "":
        return JsonResponse({"response": "Please type something!"})
    bot_response = get_response(user_message)
    return JsonResponse({"response": bot_response})