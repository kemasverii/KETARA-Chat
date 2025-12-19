"""
KETARA Chatbot - Streamlit Application
Chatbot informasi kampus ITERA menggunakan BI-LSTM
"""

import streamlit as st
import json
import numpy as np
import pickle
import random
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="KETARA Chatbot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern chat UI
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 85%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin-top: 5px;
    }
    
    /* Quick chat buttons */
    .quick-btn {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        cursor: pointer;
        font-size: 0.85rem;
        transition: all 0.3s ease;
    }
    
    .quick-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 184, 148, 0.4);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #1a1a2e;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00b894;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #b2bec3;
    }
    
    /* Confidence badge */
    .confidence-high {
        background: #00b894;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.75rem;
        margin-left: 10px;
    }
    
    .confidence-medium {
        background: #fdcb6e;
        color: #2d3436;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.75rem;
        margin-left: 10px;
    }
    
    .confidence-low {
        background: #e17055;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.75rem;
        margin-left: 10px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollable chat area */
    .chat-history {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """Load model dan preprocessing objects"""
    model = load_model('ketara_chatbot.keras')
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open('responses.pkl', 'rb') as f:
        responses = pickle.load(f)
    
    with open('config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    return model, tokenizer, label_encoder, responses, config


def preprocess_text(text):
    """Preprocessing text input"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text


def predict_intent(user_input, model, tokenizer, label_encoder, config, threshold=0.3):
    """Predict intent dari user input"""
    cleaned = preprocess_text(user_input)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=config['max_sequence_length'], padding='post')
    
    prediction = model.predict(padded, verbose=0)
    class_idx = np.argmax(prediction[0])
    confidence = float(prediction[0][class_idx])
    
    if confidence < threshold:
        return None, confidence
    
    predicted_tag = label_encoder.inverse_transform([class_idx])[0]
    return predicted_tag, confidence


def get_response(user_input, model, tokenizer, label_encoder, responses, config):
    """Get chatbot response"""
    tag, confidence = predict_intent(user_input, model, tokenizer, label_encoder, config)
    
    if tag is None:
        return "Maaf, saya kurang memahami pertanyaan Anda. Bisakah Anda mengajukan pertanyaan tentang ITERA dengan lebih jelas? 🤔", 0.0, None
    
    response = random.choice(responses[tag])
    return response, confidence, tag


def main():
    # Load model
    try:
        model, tokenizer, label_encoder, responses, config = load_chatbot()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Pastikan file model dan preprocessing objects ada di direktori yang sama.")
        return
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🎓 KETARA Chatbot</h1>
        <p class="header-subtitle">Asisten Virtual Informasi Kampus ITERA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Pengaturan")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Minimum confidence untuk memberikan jawaban"
        )
        
        show_confidence = st.checkbox("Tampilkan Confidence Score", value=True)
        show_intent = st.checkbox("Tampilkan Intent Tag", value=False)
        
        st.markdown("---")
        st.markdown("### 📚 Tentang")
        st.markdown("""
        **KETARA** adalah chatbot berbasis **Bidirectional LSTM** 
        untuk memberikan informasi seputar kampus ITERA.
        """)
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Halo! 👋 Saya KETARA, asisten virtual kampus ITERA. Silakan tanyakan apapun tentang ITERA! Anda bisa menggunakan tombol quick chat di bawah atau ketik pertanyaan Anda langsung.",
            "confidence": 1.0,
            "intent": None
        })
    
    # Quick Chat Buttons
    st.markdown("### 💬 Quick Chat")
    
    quick_questions = [
        ("🏫 Apa itu ITERA?", "Apa itu ITERA?"),
        ("📍 Lokasi Kampus", "Dimana lokasi ITERA?"),
        ("🎓 Fakultas", "Fakultas apa saja yang ada di ITERA?"),
        ("🏠 Asrama", "ITERA punya asrama mahasiswa?"),
        ("💰 Beasiswa", "Ada beasiswa apa saja di ITERA?"),
        ("📝 Cara Daftar", "Bagaimana jalur masuk ITERA?"),
        ("🔬 Lab & Fasilitas", "Fasilitas apa saja di ITERA?"),
        ("🌿 Green Campus", "ITERA kampus hijau?"),
    ]
    
    # Create button grid
    cols = st.columns(4)
    for i, (label, question) in enumerate(quick_questions):
        with cols[i % 4]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                # Get response
                response, confidence, intent = get_response(
                    question, model, tokenizer, label_encoder, responses, config
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "confidence": confidence,
                    "intent": intent
                })
                st.rerun()
    
    st.markdown("---")
    
    # Display chat history
    st.markdown("### 💭 Percakapan")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            
            # Show confidence and intent for bot messages
            if message["role"] == "assistant" and message.get("confidence") is not None:
                confidence = message.get("confidence", 0)
                intent = message.get("intent")
                
                if show_confidence and confidence > 0:
                    if confidence >= 0.8:
                        badge_class = "confidence-high"
                        emoji = "🟢"
                    elif confidence >= 0.5:
                        badge_class = "confidence-medium"
                        emoji = "🟡"
                    else:
                        badge_class = "confidence-low"
                        emoji = "🔴"
                    
                    info_text = f"{emoji} Confidence: {confidence:.1%}"
                    if show_intent and intent:
                        info_text += f" | Intent: `{intent}`"
                    st.caption(info_text)
    
    # Chat input
    if prompt := st.chat_input("Ketik pertanyaan Anda tentang ITERA..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Sedang berpikir..."):
                response, confidence, intent = get_response(
                    prompt, model, tokenizer, label_encoder, responses, config
                )
            
            st.markdown(response)
            
            if show_confidence and confidence > 0:
                if confidence >= 0.8:
                    emoji = "🟢"
                elif confidence >= 0.5:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                
                info_text = f"{emoji} Confidence: {confidence:.1%}"
                if show_intent and intent:
                    info_text += f" | Intent: `{intent}`"
                st.caption(info_text)
        
        # Add bot message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "confidence": confidence,
            "intent": intent
        })
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #636e72; font-size: 0.8rem;">
        🎓 KETARA Chatbot | BI-LSTM Model | Institut Teknologi Sumatera<br>
        <span style="color: #00b894;">Deep Learning Project</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
