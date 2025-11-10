

try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
import json
import random

st.set_page_config(
    page_title="Mental Health Companion",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)


with open('./data.json', 'r') as f:
    data = json.load(f)

def preprocess(sentence):
    words = word_tokenize(sentence.lower())
    words = [word for word in words if word not in stopwords.words('english') and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

intents_data = {}
for intent in data.get('intents', []):
    tag = intent.get('tag', '')
    patterns = intent.get('patterns', [])
    responses = intent.get('responses', [])
    if tag and patterns and responses:
        intents_data[tag] = {
            'patterns': [preprocess(p) for p in patterns],
            'responses': responses
        }

def find_best_match(user_input):
    if not user_input or not user_input.strip():
        return "no-response", None
    
    user_tokens = preprocess(user_input)
    if not user_tokens:
        return "no-response", None
    
    max_similarity = 0
    best_intent = "default"
    best_response = None
    
    for tag, intent_data in intents_data.items():
        for pattern_tokens in intent_data['patterns']:
            if not pattern_tokens:
                continue
            union = set(user_tokens).union(pattern_tokens)
            if len(union) > 0:
                similarity = len(set(user_tokens).intersection(pattern_tokens)) / float(len(union))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_intent = tag
                    best_response = random.choice(intent_data['responses'])
    
    if max_similarity > 0.1:  # Minimum threshold for intent matching
        return best_intent, best_response
    
    return "default", random.choice(intents_data.get("default", {}).get("responses", ["I'm here to support you. Could you tell me more about how you're feeling?"]))

def chatbot(question):
    """Generate response based on user question."""
    intent, response = find_best_match(question)
    return response if response else "I'm here to support you. Could you tell me more about how you're feeling?"

def main():
    st.title("🧠 Mental Health Companion")
    st.write("Welcome! I'm here to listen and support you. Share what's on your mind.")
    st.divider()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display conversation history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"**👤 You:** {message['content']}")
        else:
            st.write(f"**🤖 Companion:** {message['content']}")
    
    st.divider()
    
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("Your message:", key="user_input", placeholder="How are you feeling today?")
    with col2:
        submitted = st.button("Send", use_container_width=True)
    
    if submitted:
        if question and question.strip():
            response = chatbot(question)
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        else:
            st.warning("Please enter a message before sending.")

if __name__ == "__main__":
    main()
