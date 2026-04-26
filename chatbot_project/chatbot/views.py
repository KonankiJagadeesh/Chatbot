import pickle
import random
import string
import os
import re
import math
import operator
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
CONFIDENCE_THRESHOLD = 0.60

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
    r"what is (is|today's) date",
    r"tell me the (date|time)",
    r"clock now",
    r"time check",
]

GREETING_KEYWORDS = ["hi", "hello", "hey", "hiya", "howdy", "morning", "evening", "greetings"]
GOODBYE_KEYWORDS  = ["bye", "goodbye", "see you", "farewell", "cya", "take care", "night"]
THANKS_KEYWORDS   = ["thanks", "thank you", "thankyou", "cheers", "appreciated"]

# Patterns to detect when user introduces their name
NAME_PATTERNS = [
    r"(?:my name is|i am|i'm|call me|this is|myself)\s+([a-zA-Z]+)",
    r"(?:hi|hello|hey)[,!]?\s+(?:i am|i'm|my name is)\s+([a-zA-Z]+)",
    r"(?:i am|i'm)\s+([a-zA-Z]+)[,!]?\s*(?:here|speaking|here to chat)?",
]

# Common non-name words to avoid false positives
NON_NAME_WORDS = {
    "fine", "good", "okay", "ok", "great", "well", "bad", "sad", "happy",
    "ready", "here", "back", "not", "sure", "yes", "no", "done", "just",
    "so", "very", "really", "trying", "going", "coming", "looking", "thinking",
    "learning", "working", "busy", "free", "new", "old", "lost", "confused"
}

def extract_name(text_original):
    """
    Try to extract a person's name from the message.
    Returns the capitalised name if found, None otherwise.
    """
    text_lower = text_original.lower().strip()

    # Explicit introduction patterns (case-insensitive)
    for pattern in NAME_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            name = match.group(1).capitalize()
            if name.lower() not in NON_NAME_WORDS and len(name) > 1:
                return name

    # Single-word message that looks like a proper name
    words = text_original.strip().split()
    if len(words) == 1:
        word = words[0]
        # Proper name: starts uppercase (or all lowercase but not a known keyword)
        if re.match(r'^[A-Za-z]{2,20}$', word):
            word_lower = word.lower()
            all_known_words = (
                NON_NAME_WORDS |
                {"hi", "hello", "hey", "bye", "okay", "thanks", "help",
                 "what", "when", "where", "who", "why", "how", "can",
                 "tell", "show", "give", "make", "do", "go", "run"}
            )
            if word_lower not in all_known_words:
                return word.capitalize()

    return None

def keyword_check(text_original):
    """
    Fast keyword/rule matching before hitting the ML model.
    Returns a response string if matched, None otherwise.
    text_original: the raw (uncleaned) user input
    """
    text_clean = text_original.lower().strip().translate(
        str.maketrans('', '', string.punctuation)
    )

    # 1. Name detection — HIGHEST PRIORITY
    detected_name = extract_name(text_original)
    if detected_name:
        greetings = [
            f"Hi {detected_name}! 👋 Nice to meet you, I'm Varahi. How can I help you today?",
            f"Hello {detected_name}! 😊 Great to meet you! I'm Varahi, your AI assistant.",
            f"Hey {detected_name}! 👋 Welcome! I'm Varahi. What can I do for you?",
            f"Nice to meet you, {detected_name}! I'm Varahi. Feel free to ask me anything! 😊",
        ]
        return random.choice(greetings)

    # 2. Calendar / date / time
    for pattern in CALENDAR_PATTERNS:
        if re.search(pattern, text_clean):
            return get_calendar_info()

    # 3. Short greeting keywords (avoid ML misclassifying "hi", "hey")
    words = set(text_clean.split())
    if words & {"hi", "hello", "hey", "hiya", "howdy"}:
        if len(text_clean.split()) <= 3:
            return random.choice(intent_responses.get("greeting", [
                "Hello! How can I help you today?"
            ]))

    # 4. Goodbye
    if words & {"bye", "goodbye", "farewell"}:
        if len(text_clean.split()) <= 3:
            return random.choice(intent_responses.get("goodbye", [
                "Goodbye! Have a great day!"
            ]))

    return None

# -----------------------------
# Math Evaluator (safe)
# -----------------------------
# Allowed names for safe eval
SAFE_MATH_NAMES = {
    k: v for k, v in math.__dict__.items() if not k.startswith("_")
}
SAFE_MATH_NAMES.update({"abs": abs, "round": round})

MATH_PATTERNS = [
    r"^[\d\s\.\+\-\*\/\(\)\^\%]+$",            # pure expression: 2+2, (3*4)/2
    r"(calculate|compute|solve|what is|whats)\s+([\d\s\.\+\-\*\/\(\)\^\%]+)",
    r"(sqrt|sin|cos|tan|log|pi|pow)\s*[\(\d]",  # math functions
    r"(\d+)\s*[\+\-\*\/]\s*(\d+)",              # simple: 5 * 10
    r"(\d+)\s*%\s*of\s*(\d+)",                  # percent: 15% of 200
    r"(\d+)\^(\d+)",                             # power: 2^8
    r"calculate (.*)",                           # generic calculation
    r"solve (.*)",                               # generic solving
]

def try_math(text):
    """Try to evaluate a math expression. Returns result string or None."""
    t = text.strip().lower()

    # Handle "X% of Y"
    pct = re.match(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", t)
    if pct:
        result = float(pct.group(1)) / 100 * float(pct.group(2))
        return f"🧮 {pct.group(1)}% of {pct.group(2)} = **{result:g}**"

    # Handle "X to the power of Y" / "X^Y"
    t = re.sub(r"(\d+)\s*\^\s*(\d+)", r"pow(\1, \2)", t)
    t = re.sub(r"to the power of", "**", t)
    t = re.sub(r"squared", "**2", t)
    t = re.sub(r"cubed", "**3", t)

    # Strip common prefix words
    for prefix in ["calculate", "compute", "solve", "what is", "whats"]:
        t = re.sub(r"^" + prefix + r"\s*", "", t).strip()

    # Only allow safe characters
    if not re.match(r"^[\d\s\.\+\-\*\/\(\)\,a-z_]+$", t):
        return None

    try:
        result = eval(t, {"__builtins__": {}}, SAFE_MATH_NAMES)  # noqa: S307
        if isinstance(result, (int, float)):
            formatted = f"{result:g}" if isinstance(result, float) else str(result)
            return f"🧮 {text.strip()} = **{formatted}**"
    except Exception:
        pass
    return None

# -----------------------------
# ML-based response with confidence gate
# -----------------------------
def get_response(text, session_name=None, session_context=None):
    if not model or not vectorizer:
        return "I'm not fully loaded yet. Please refresh and try again.", ""

    text_clean = text.lower().strip().translate(str.maketrans('', '', string.punctuation))

    # 0. Math evaluation (highest priority for numeric expressions)
    math_result = try_math(text)
    if math_result:
        return math_result, ""

    # 1. Try keyword / rule-based check first (pass raw text for name detection)
    kw_response = keyword_check(text)
    if kw_response:
        return kw_response, ""

    try:
        vect_text = vectorizer.transform([text_clean])

        # 3. Get prediction probabilities
        probabilities = model.predict_proba(vect_text)[0]
        
        # 4. Filter and weigh intents based on context
        best_intent = None
        max_score = -1
        
        for i, intent_tag in enumerate(model.classes_):
            score = probabilities[i]
            intent_info = intent_responses.get(intent_tag, {})
            
            # Boost score if intent matches current context
            expected_context = intent_info.get("context_filter", "")
            if expected_context:
                if session_context == expected_context:
                    # Very aggressive boost for matched context
                    score += 1.0  
                else:
                    # Heavy penalty for context-mismatched follow-up intents
                    score -= 1.0
            
            if score > max_score:
                max_score = score
                best_intent = intent_tag

        # Final confidence check
        # We allow a lower threshold (0.35) if the score was boosted significantly by context
        final_threshold = CONFIDENCE_THRESHOLD
        intent_info = intent_responses.get(best_intent, {})
        is_contextual = session_context and intent_info.get("context_filter") == session_context
        
        if is_contextual:
            final_threshold = 0.35 # Much lower threshold for context-boosted matches

        if best_intent is None or max_score < final_threshold:
            fallback = random.choice([
                "Hmm, I'm not quite sure about that. Could you rephrase it?",
                "I didn't quite catch that. Can you be more specific?",
                "That's a bit beyond me right now! Try asking something else.",
                "I'm still learning. Could you ask that differently?",
            ])
            msg = fallback + (f" By the way, I remember you, {session_name}! 😊" if session_name else "")
            return msg, (session_context if session_context else "")

        # 5. Return response for the predicted intent
        responses = intent_info.get("responses", [])
        new_context = intent_info.get("context_set", "")

        if responses:
            base = random.choice(responses)
            # Occasionally personalise with the user's name
            if session_name and random.random() < 0.25:  # 25% chance
                base = base.rstrip(".") + f", {session_name}."
            return base, new_context
        else:
            return "I'm not sure how to respond to that. Could you rephrase?", session_context

    except Exception as e:
        print(f"Error in get_response: {e}")
        return "Sorry, I encountered an error. Please try again.", ""

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

    # Retrieve stored name and context from session
    session_name = request.session.get("user_name", None)
    session_context = request.session.get("user_context", "")

    try:
        bot_response, new_context = get_response(
            user_message, 
            session_name=session_name, 
            session_context=session_context
        )
        # Update session with new context
        request.session["user_context"] = new_context
    except Exception as e:
        print(f"chat_api error: {e}")
        bot_response = "Sorry, something went wrong on my end!"

    # If a name was detected in this message, store it in session
    detected = extract_name(user_message)
    if detected:
        request.session["user_name"] = detected

    return JsonResponse({"response": bot_response})

def clear_session(request):
    """Clears the current session (called when user clears chat)."""
    request.session.flush()
    return JsonResponse({"status": "ok"})