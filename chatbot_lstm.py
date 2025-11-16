"""
KETARA Chatbot - LSTM-based Chatbot untuk Informasi ITERA
Menggunakan LSTM untuk intent classification dan response generation
"""

import json
import numpy as np
import pickle
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import re

class KETARAChatbot:
    def __init__(self, data_path='data.json'):
        """
        Initialize KETARA Chatbot
        
        Args:
            data_path (str): Path ke file dataset JSON
        """
        self.data_path = data_path
        self.tokenizer = Tokenizer(lower=True, oov_token='<OOV>')
        self.label_encoder = LabelEncoder()
        self.model = None
        self.max_sequence_length = 20
        self.vocab_size = None
        self.num_classes = None
        self.responses = {}
        
    def load_data(self):
        """Load dan parse dataset dari JSON"""
        print("Loading dataset...")
        with open(self.data_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        
        print(f"Dataset loaded: {len(self.data)} tags")
        return self.data
    
    def preprocess_text(self, text):
        """
        Preprocessing text: lowercase, remove special characters
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Lowercase
        text = text.lower()
        # Remove special characters, keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text
    
    def prepare_training_data(self):
        """
        Prepare data untuk training:
        - Extract patterns dan tags
        - Preprocessing
        - Tokenization
        - Encoding labels
        """
        print("\nPreparing training data...")
        
        patterns = []
        tags = []
        
        # Extract patterns dan tags dari dataset
        for intent in self.data:
            tag = intent['tag']
            # Simpan responses untuk setiap tag
            self.responses[tag] = intent['responses']
            
            for pattern in intent['patterns']:
                # Preprocess pattern
                cleaned_pattern = self.preprocess_text(pattern)
                patterns.append(cleaned_pattern)
                tags.append(tag)
        
        print(f"Total patterns: {len(patterns)}")
        print(f"Unique tags: {len(set(tags))}")
        
        # Tokenize patterns
        self.tokenizer.fit_on_texts(patterns)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Convert patterns to sequences
        sequences = self.tokenizer.texts_to_sequences(patterns)
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Encode labels
        self.label_encoder.fit(tags)
        y_encoded = self.label_encoder.transform(tags)
        self.num_classes = len(self.label_encoder.classes_)
        y = to_categorical(y_encoded, num_classes=self.num_classes)
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")
        
        return X, y
    
    def build_model(self, embedding_dim=64, lstm_units=64, dropout_rate=0.3):
        """
        Build LSTM model architecture (Optimized untuk dataset kecil)
        
        Args:
            embedding_dim (int): Dimensi embedding layer (default 64)
            lstm_units (int): Number of LSTM units (default 64)
            dropout_rate (float): Dropout rate untuk regularization (default 0.3)
            
        Returns:
            model: Compiled Keras model
        """
        print("\nBuilding LSTM model (Optimized)...")
        print(f"  Embedding Dim: {embedding_dim}")
        print(f"  LSTM Units: {lstm_units}")
        print(f"  Dropout Rate: {dropout_rate}")
        
        model = Sequential([
            # Embedding Layer
            Embedding(
                input_dim=self.vocab_size,
                output_dim=embedding_dim,
                input_length=self.max_sequence_length,
                name='embedding_layer'
            ),
            
            # Bidirectional LSTM Layer 1
            Bidirectional(LSTM(
                units=lstm_units,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=0.2
            ), name='bidirectional_lstm_1'),
            
            # Bidirectional LSTM Layer 2
            Bidirectional(LSTM(
                units=lstm_units // 2,
                return_sequences=False,
                dropout=dropout_rate,
                recurrent_dropout=0.2
            ), name='bidirectional_lstm_2'),
            
            # Dropout Layer
            Dropout(dropout_rate, name='dropout_layer'),
            
            # Dense Layer 1 (dikurangi dari 64 ke 32)
            Dense(32, activation='relu', name='dense_1'),
            Dropout(dropout_rate / 2, name='dropout_2'),
            
            # Output Layer
            Dense(self.num_classes, activation='softmax', name='output_layer')
        ])
        
        # Compile model dengan Adam optimizer
        from tensorflow.keras.optimizers import Adam
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def train(self, X, y, epochs=200, batch_size=4, validation_split=0.15):
        """
        Train LSTM model (Optimized untuk dataset kecil)
        
        Args:
            X: Input sequences
            y: Target labels (one-hot encoded)
            epochs (int): Number of training epochs (default 200)
            batch_size (int): Batch size (default 4)
            validation_split (float): Validation split ratio (default 0.15)
            
        Returns:
            history: Training history
        """
        print("\nTraining model (Optimized)...")
        print(f"  Max Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Validation Split: {validation_split}")
        
        # Callbacks
        from tensorflow.keras.callbacks import ReduceLROnPlateau
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        )
        
        model_checkpoint = ModelCheckpoint(
            'ketara_chatbot_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
        
        print("\nTraining completed!")
        return history
    
    def save_model(self, model_path='ketara_chatbot.keras'):
        """Save trained model dan preprocessing objects"""
        print(f"\nSaving model to {model_path}...")
        self.model.save(model_path)
        
        # Save tokenizer dan label encoder
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        with open('responses.pkl', 'wb') as f:
            pickle.dump(self.responses, f)
        
        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'num_classes': self.num_classes
        }
        with open('config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        print("Model and preprocessing objects saved!")
    
    def load_trained_model(self, model_path='ketara_chatbot.keras'):
        """Load trained model dan preprocessing objects"""
        print(f"Loading model from {model_path}...")
        
        self.model = load_model(model_path)
        
        with open('tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        with open('label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        with open('responses.pkl', 'rb') as f:
            self.responses = pickle.load(f)
        
        with open('config.pkl', 'rb') as f:
            config = pickle.load(f)
            self.vocab_size = config['vocab_size']
            self.max_sequence_length = config['max_sequence_length']
            self.num_classes = config['num_classes']
        
        print("Model loaded successfully!")
    
    def predict_intent(self, user_input, threshold=0.5):
        """
        Predict intent dari user input
        
        Args:
            user_input (str): Input dari user
            threshold (float): Confidence threshold
            
        Returns:
            tuple: (predicted_tag, confidence)
        """
        # Preprocess input
        cleaned_input = self.preprocess_text(user_input)
        
        # Tokenize dan pad
        sequence = self.tokenizer.texts_to_sequences([cleaned_input])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')
        
        # Predict
        prediction = self.model.predict(padded_sequence, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        
        if confidence < threshold:
            return None, confidence
        
        predicted_tag = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_tag, confidence
    
    def get_response(self, user_input):
        """
        Get chatbot response untuk user input
        
        Args:
            user_input (str): Input dari user
            
        Returns:
            str: Response dari chatbot
        """
        predicted_tag, confidence = self.predict_intent(user_input)
        
        if predicted_tag is None:
            return "Maaf, saya kurang memahami pertanyaan Anda. Bisakah Anda mengajukan pertanyaan tentang ITERA dengan lebih jelas?"
        
        # Get random response dari tag yang diprediksi
        response = random.choice(self.responses[predicted_tag])
        
        return response
    
    def chat(self):
        """Interactive chat mode"""
        print("\n" + "="*60)
        print("KETARA Chatbot - Informasi ITERA")
        print("="*60)
        print("Ketik 'quit' atau 'exit' untuk keluar\n")
        
        while True:
            user_input = input("Anda: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'keluar']:
                print("Chatbot: Terima kasih! Sampai jumpa!")
                break
            
            if not user_input:
                continue
            
            response = self.get_response(user_input)
            print(f"Chatbot: {response}\n")


def main():
    """Main function untuk training chatbot"""
    
    # Initialize chatbot
    chatbot = KETARAChatbot('data.json')
    
    # Load data
    chatbot.load_data()
    
    # Prepare training data
    X, y = chatbot.prepare_training_data()
    
    # Build model (Optimized hyperparameters)
    chatbot.build_model(
        embedding_dim=64,
        lstm_units=64,
        dropout_rate=0.3
    )
    
    # Train model (Optimized parameters)
    history = chatbot.train(
        X, y,
        epochs=200,
        batch_size=4,
        validation_split=0.15
    )
    
    # Save model
    chatbot.save_model('ketara_chatbot.keras')
    
    print("\n" + "="*60)
    print("Training selesai! Model telah disimpan.")
    print("="*60)
    
    # Test chatbot
    print("\nTesting chatbot...")
    test_questions = [
        "Apa itu ITERA?",
        "Dimana lokasi ITERA?",
        "Fakultas apa saja yang ada?",
        "Ceritakan tentang lambang ITERA"
    ]
    
    for question in test_questions:
        response = chatbot.get_response(question)
        print(f"\nQ: {question}")
        print(f"A: {response}")


if __name__ == "__main__":
    main()
