import streamlit as st 
import json, uuid, os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# GPT2 model folder (your fine-tuned model files)
MODEL_PATH = "./"  # where config.json, pytorch_model.bin, vocab.json, merges.txt are located

# GPT2 model + tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# NLTK sentiment for emotion trend
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

USER_FOLDER = "user_data"
os.makedirs(USER_FOLDER, exist_ok=True)

# --- Crisis detection keywords ---
CRISIS_KEYWORDS = ["suicide", "kill myself", "end my life", "self harm", "self-harm"]

#  Harmful phrases we want to filter from GPT2 responses
BAD_PHRASES = [
    "waste of space",
    "kill yourself",
    "hopeless",
    "no one cares",
    "better off dead",
    "worthless",
    "meaningless"
]

#  Library of small resilience exercises
RESILIENCE_EXERCISES = {
    "grounding": (
        "Here’s a simple grounding exercise:\n\n"
        " Look around and name **5 things you can see**\n"
        "Touch **4 things you can feel**\n"
        " Listen for **3 sounds you can hear**\n"
        " Notice **2 things you can smell**\n"
        " Think of **1 thing you’re grateful for**\n\n"
        "This helps bring your mind back to the present. 💜"
    ),
    "breathing": (
        "Let’s try a short breathing exercise:\n\n"
        "🌿 **4-7-8 Breathing Technique**\n"
        "1. Inhale gently through your nose for **4 seconds**\n"
        "2. Hold your breath for **7 seconds**\n"
        "3. Exhale slowly through your mouth for **8 seconds**\n\n"
        "Repeat this 3 times and notice how your body feels calmer."
    )
}

def check_crisis(text):
    """Detect if user mentions crisis keywords."""
    for word in CRISIS_KEYWORDS:
        if word in text.lower():
            return True
    return False

def detect_emotion(text):
    """Quick sentiment-based emotion detection"""
    score = sia.polarity_scores(text)
    compound = score["compound"]
    if compound >= 0.3:
        emotion = "positive"
    elif compound <= -0.2:
        emotion = "negative"
    else:
        emotion = "neutral"
    return emotion, compound

def safe_reply(reply):
    """Filter harmful phrases & replace with supportive fallback"""
    for phrase in BAD_PHRASES:
        if phrase in reply.lower():
            return (
                "I'm really sorry you're feeling this way 💜. "
                "You are important, and you deserve care and kindness. "
                "Would you like to share what’s making you feel like this?"
            )
    return reply

def generate_gpt2_reply(user_input, history=None):
    """Generate a SAFE, concise GPT2 reply with empathetic prompt & filters"""

    #  Special case: user said YES / THANKS → give a real exercise
    if user_input.strip().lower() in ["yes", "okay", "sure", "thanks", "thank you"]:
        return RESILIENCE_EXERCISES["grounding"]

    #  Empathetic system prompt
    context = (
        "You are ResiliBot, a compassionate mental health support chatbot. "
        "Always respond with empathy, encouragement, and hope. "
        "Be concise (1-3 sentences). Never say anything harmful or negative.\n\n"
    )

    #  Auto-truncate long chat history (only last 3 exchanges)
    if history:
        for h in history[-3:]:
            context += f"User: {h['user_msg']}\nBot: {h['bot_msg']}\n"
    context += f"User: {user_input}\nBot:"

    #  Encode with GPT2
    inputs = tokenizer.encode(context, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)

    # Use max_new_tokens to avoid length error
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=60,   # generate only 60 new tokens beyond input
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.85,          # stricter sampling → less random
        temperature=0.6      # cooler → calmer tone
    )

    #  Decode & extract only the bot's last reply
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Bot:" in reply:
        reply = reply.split("Bot:")[-1].strip('", ')

    #  Take only the first 2 sentences → avoid long rambles
    sentences = reply.split(". ")
    reply = ". ".join(sentences[:2]).strip()
    if not reply.endswith("."):
        reply += "."

    #  Trigger-based supportive fallback for common feelings
    if "overwhelmed" in user_input.lower():
        reply = (
            "It’s okay to feel overwhelmed sometimes. "
            "Let’s take a deep breath together. "
            "Would you like me to share a simple grounding exercise?"
        )
    elif "alone" in user_input.lower():
        reply = (
            "Feeling alone can be really hard. 💜 "
            "Remember, you’re not truly alone—I’m here to listen. "
            "Would you like to talk about what’s making you feel this way?"
        )

    #  Safety filter for harmful output
    reply = safe_reply(reply)

    return reply

# --- User Session Setup ---
def get_user_session():
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}

    user_data = st.session_state.user_data

    if "nickname" not in user_data or "ref" not in user_data:
        st.markdown("### Hi there! Have you used ResiliBot before? (yes/no)")
        user_choice = st.text_input(">", key="user_check")

        if user_choice.lower().strip() == "no":
            nickname = st.text_input("What nickname should I use for you?", key="nickname_input")
            if nickname:
                ref = str(uuid.uuid4())[:8]
                st.session_state.user_data = {
                    "nickname": nickname,
                    "ref": ref,
                    "chat_history": [],
                    "emotion_log": []
                }
                with open(f"user_data/{ref}.json", "w") as f:
                    json.dump(st.session_state.user_data, f, indent=2)
                st.success(f"Thanks, {nickname}. Your reference number is: {ref} (save this!)")

        elif user_choice.lower().strip() == "yes":
            ref = st.text_input("Enter your reference number:", key="ref_input")
            if ref:
                user_file = f"user_data/{ref}.json"
                if os.path.exists(user_file):
                    with open(user_file) as f:
                        st.session_state.user_data = json.load(f)
                    st.success(f"Welcome back, {st.session_state.user_data['nickname']}! Your ref: {ref}")
                else:
                    st.warning("Sorry, that reference number wasn't found. Please try again.")
                    st.stop()

    return st.session_state.user_data

# --- Initialize Session ---
user = get_user_session()

# --- Streamlit UI ---
st.title("🧠 Mental Healthcare Chatbot")
if "nickname" in user:
    st.caption(f"Welcome, **{user['nickname']}** 🌿")

# --- Chat Form ---
with st.form("chat_form", clear_on_submit=True):
    if "nickname" in user:
        user_input = st.text_input(f"{user['nickname']}:", placeholder="Type your message here...")
    else:
        user_input = st.text_input("Say something:", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    #  Crisis keyword detection → immediate safe response
    if check_crisis(user_input):
        bot_response = (
            "I'm really sorry you're feeling this way. 💜 "
            "You are not alone, and help is available.\n\n"
            "👉 Please reach out to a mental health professional or call a crisis helpline.\n"
            "📞 In India: **+91-9152987821 (iCall)**\n"
            "📞 In the US: **988 Suicide & Crisis Lifeline**"
        )
    else:
        #  Normal GPT2 reply (with empathy & safety)
        bot_response = generate_gpt2_reply(user_input, user.get("chat_history"))

    #  Detect emotion for mood tracking
    emotion, score = detect_emotion(user_input)
    user["emotion_log"].append({"emotion": emotion, "score": score})

    #  Save chat in history
    user["chat_history"].append({
        "user_msg": str(user_input),
        "bot_msg": str(bot_response),
        "emotion": str(emotion)
    })

    #  Save session to disk
    if "ref" in user:
        with open(f"user_data/{user['ref']}.json", "w") as f:
            json.dump(user, f, indent=2)

# --- Display Chat Log ---
for chat in user.get("chat_history", []):
    st.markdown(f"**{user['nickname']}**: {chat['user_msg']}")
    st.markdown(f" *({chat['emotion']})*")
    st.markdown(f"**ResiliBot:** {chat['bot_msg']}")
    st.markdown("---")

# --- Quit & Mood Chart ---
if st.button("Quit and Show Mood Chart 📊"):
    if len(user.get("emotion_log", [])) < 2:
        st.warning("Not enough conversation yet to generate a chart.")
    else:
        scores = [e["score"] for e in user["emotion_log"]]
        emotions = [e["emotion"] for e in user["emotion_log"]]
        x = list(range(1, len(scores) + 1))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, scores, marker='o', linestyle='-', color='purple')
        ax.axhline(0, color='gray', linestyle='--')
        for i, (x_pt, label) in enumerate(zip(x, emotions)):
            ax.text(x_pt, scores[i] + 0.05, label, ha='center', fontsize=8)
        ax.set_ylim(-1, 1)
        ax.set_title(f"{user['nickname']}'s Emotional Trend")
        ax.set_xlabel("Message Count")
        ax.set_ylabel("Sentiment Score (-1 to 1)")
        ax.grid(True)
        st.pyplot(fig)
