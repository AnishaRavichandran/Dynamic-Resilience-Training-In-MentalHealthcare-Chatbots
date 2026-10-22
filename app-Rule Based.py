import streamlit as st
import json
import random
import uuid
import os
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# --- Setup ---
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# --- Load dataset ---
with open("MHResilience.json", "r", encoding="utf-8") as file:
    data = json.load(file)

pattern_to_responses = {}
for intent in data["intents"]:
    responses = intent.get("responses") or [intent.get("response")] or ["Sorry, I don't know how to respond."]
    for pattern in intent["patterns"]:
        pattern_to_responses[pattern.lower()] = responses

# --- Helper: Detect emotion ---
def detect_emotion(text):
    score = sia.polarity_scores(text)
    compound = score["compound"]
    if compound >= 0.3:
        emotion = "positive"
    elif compound <= -0.2:
        emotion = "negative"
    else:
        emotion = "neutral"
    return emotion, compound

# --- Helper: Get bot response ---
def get_response(user_input):
    user_input = user_input.strip().lower()
    if user_input in pattern_to_responses:
        return random.choice(pattern_to_responses[user_input])
    else:
        return "I'm here for you. Could you tell me more?"

# --- App UI ---
st.set_page_config(page_title="ResiliBot", page_icon="🧠")
st.title("🧠 ResiliBot (Rule-Based)")
st.warning(
    " **Disclaimer:** This chatbot is a simple rule-based assistant. It is *not* a substitute for professional mental health care. "
    "Please seek a licensed therapist for any serious concerns. ResiliBot is here to provide basic support when you need it."
)

# --- User session setup ---
if "user_info" not in st.session_state:
    with st.form("user_init_form"):
        nickname = st.text_input("What nickname should I use for you?")
        submitted = st.form_submit_button("Start Chat")
        if submitted and nickname:
            st.session_state.user_info = {
                "nickname": nickname,
                "ref": str(uuid.uuid4())[:8],
                "chat_history": [],
                "emotion_log": []
            }
            st.success(f"Hello, {nickname}! Your unique ID is: {st.session_state.user_info['ref']}")
            st.rerun()

# --- Chat logic ---
if "user_info" in st.session_state:
    user = st.session_state.user_info

    st.caption(f"Chatting as **{user['nickname']}** (ID: `{user['ref']}`)")

    user_input = st.text_input("You:", placeholder="Type something...")

    if user_input:
        emotion, score = detect_emotion(user_input)
        response = get_response(user_input)

        user["chat_history"].append({
            "user_msg": user_input,
            "bot_msg": response,
            "emotion": emotion,
            "score": score
        })
        user["emotion_log"].append({"emotion": emotion, "score": score})

    # --- Display chat history ---
    for msg in user["chat_history"]:
        st.markdown(f"**{user['nickname']}**: {msg['user_msg']}")
        st.markdown(f"*(Mood: {msg['emotion']})*")
        st.markdown(f"**ResiliBot**: {msg['bot_msg']}")
        st.markdown("---")

    # --- Mood chart button ---
    if st.button("Show My Mood Chart"):
        if len(user["emotion_log"]) < 2:
            st.warning("Not enough messages yet to plot a trend.")
        else:
            scores = [e["score"] for e in user["emotion_log"]]
            emotions = [e["emotion"] for e in user["emotion_log"]]
            x = list(range(1, len(scores) + 1))

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, scores, marker='o', color='purple', linestyle='-')
            ax.axhline(0, color='gray', linestyle='--')
            for i, (x_val, label) in enumerate(zip(x, emotions)):
                ax.text(x_val, scores[i] + 0.05, label, ha='center', fontsize=8)
            ax.set_ylim(-1, 1)
            ax.set_title(f"{user['nickname']}'s Emotional Trend")
            ax.set_xlabel("Message Count")
            ax.set_ylabel("Sentiment Score")
            ax.grid(True)
            st.pyplot(fig)
