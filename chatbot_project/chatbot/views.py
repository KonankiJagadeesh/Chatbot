import pickle
import random
import string
import os
import re
from datetime import datetime
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
# Confidence threshold — below this, use fallback
# -----------------------------
CONFIDENCE_THRESHOLD = 0.30

# -----------------------------
# Calendar: notable events per (month, day)
# -----------------------------
NOTABLE_EVENTS = {
    (1, 1):  "🎉 New Year's Day! A fresh start for everyone.",
    (1, 26): "🇮🇳 Republic Day of India!",
    (2, 14): "💕 Valentine's Day! A day of love and affection.",
    (3, 8):  "👩 International Women's Day! Celebrating women worldwide.",
    (3, 21): "🌿 World Poetry Day & First Day of Spring!",
    (3, 22): "💧 World Water Day!",
    (3, 26): "🌟 It's a regular Thursday — but every day is special when you make it count!",
    (4, 1):  "😄 April Fool's Day! Watch out for pranks.",
    (4, 22): "🌍 Earth Day! Celebrate our planet.",
    (5, 1):  "👷 International Labour Day / May Day!",
    (6, 5):  "🌿 World Environment Day!",
    (6, 21): "🧘 International Day of Yoga!",
    (8, 15): "🇮🇳 Indian Independence Day!",
    (9, 5):  "📚 Teachers' Day (India)!",
    (10, 2): "🕊️ Gandhi Jayanti — Birthday of Mahatma Gandhi!",
    (10, 31): "🎃 Halloween! A spooky day.",
    (11, 14): "🧒 Children's Day (India)!",
    (12, 25): "🎄 Christmas Day! Merry Christmas!",
    (12, 31): "🥂 New Year's Eve! The year is ending.",
}

def get_calendar_info():
    """Returns real-time date info and any notable event for today."""
    now = datetime.now()
    day_name   = now.strftime("%A")        # e.g. Wednesday
    month_name = now.strftime("%B")        # e.g. March
    day        = now.day                   # e.g. 26
    year       = now.year                  # e.g. 2026
    time_str   = now.strftime("%I:%M %p")  # e.g. 10:28 AM

    event = NOTABLE_EVENTS.get((now.month, now.day), "")

    base = (
        f"📅 Today is **{day_name}, {month_name} {day}, {year}**.\n"
        f"🕐 Current time: {time_str} (IST)."
    )
    if event:
        return f"{base}\n\n✨ **Special today:** {event}"
    else:
        return f"{base}\n\nNo major global holiday today, but every day is what you make of it! 😊"

# -----------------------------
# Keyword pre-checks (run BEFORE ML model)
# These are patterns where we ALWAYS want a specific response
# regardless of what the ML model would predict.
# -----------------------------
CALENDAR_PATTERNS = [
    r"what.*(special|happening|occasion|event).*(today|now|this day)",
    r"what.*(today|this day).*(special|event|happening)",
    r"today.*(special|event|occasion|holiday)",
    r"what day is (it|today)",
    r"what.*date.*today",
    r"today.*date",
    r"current date",
    r"what is today",
    r"tell me.*date",
    r"what.*time is it",
    r"current time",
    r"tell me.*time",
    r"what time",
    r"calendar",
    r"any.*holiday.*today",
    r"is today.*holiday",
]

GREETING_KEYWORDS  = ["hi", "hello", "hey", "hiya", "howdy", "morning", "evening", "greetings"]
GOODBYE_KEYWORDS   = ["bye", "goodbye", "see you", "farewell", "cya", "take care", "night"]
THANKS_KEYWORDS    = ["thanks", "thank you", "thankyou", "cheers", "appreciated"]

def keyword_check(text_clean):
    """
    Fast keyword matching before hitting the ML model.
    Returns a response string if matched, None otherwise.
    """
    # Calendar / date / time
    for pattern in CALENDAR_PATTERNS:
        if re.search(pattern, text_clean):
            return get_calendar_info()

    # These short-word intents are easy to miss with TF-IDF
    words = set(text_clean.split())
    if words & {"hi", "hello", "hey", "hiya", "howdy"}:
        if len(text_clean.split()) <= 3:  # short greetings only
            return random.choice(intent_responses.get("greeting", [
                "Hello! How can I help you today?"
            ]))
    if words & {"bye", "goodbye", "farewell"}:
        if len(text_clean.split()) <= 3:
            return random.choice(intent_responses.get("goodbye", [
                "Goodbye! Have a great day!"
            ]))

    return None

# -----------------------------
# ML-based response with confidence gate
# -----------------------------
def get_response(text):
    if not model or not vectorizer:
        return "I'm not fully loaded yet. Please refresh and try again."

    text_clean = text.lower().strip().translate(str.maketrans('', '', string.punctuation))

    # 1. Try keyword / rule-based check first
    kw_response = keyword_check(text_clean)
    if kw_response:
        return kw_response

    try:
        vect_text = vectorizer.transform([text_clean])

        # 2. Check prediction confidence
        probabilities   = model.predict_proba(vect_text)[0]
        max_confidence  = max(probabilities)
        intent          = model.classes_[probabilities.argmax()]

        if max_confidence < CONFIDENCE_THRESHOLD:
            # Not confident — give a polite fallback
            return random.choice([
                "Hmm, I'm not quite sure about that. Could you rephrase it?",
                "I didn't quite catch that. Can you be more specific?",
                "That's a bit beyond me right now! Try asking something else.",
                "I'm still learning. Could you ask that differently?",
            ])

        # 3. Return response for the predicted intent
        if intent in intent_responses and intent_responses[intent]:
            return random.choice(intent_responses[intent])
        else:
            return "I'm not sure how to respond to that. Could you rephrase?"

    except Exception as e:
        print(f"Error in get_response: {e}")
        return "Sorry, I encountered an error. Please try again."

# -----------------------------
# View for index page
# -----------------------------
def index(request):
    return render(request, "chatbot/index.html")

# -----------------------------
# API view for AJAX requests
# -----------------------------
def chat_api(request):
    user_message = request.GET.get("message", "")
    if not user_message or user_message.strip() == "":
        return JsonResponse({"response": "Please type something!"})
    try:
        bot_response = get_response(user_message)
    except Exception as e:
        print(f"chat_api error: {e}")
        bot_response = "Sorry, something went wrong on my end!"
    return JsonResponse({"response": bot_response})