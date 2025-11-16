"""
Demo script untuk menggunakan KETARA Chatbot yang sudah ditraining
"""

from chatbot_lstm import KETARAChatbot

def main():
    # Load trained model
    chatbot = KETARAChatbot('data.json')
    
    try:
        chatbot.load_trained_model('ketara_chatbot.keras')
    except:
        print("Model belum ditraining. Silakan jalankan chatbot_lstm.py terlebih dahulu untuk training model.")
        print("Atau file model tidak ditemukan (ketara_chatbot.keras)")
        return
    
    # Start interactive chat
    chatbot.chat()

if __name__ == "__main__":
    main()
