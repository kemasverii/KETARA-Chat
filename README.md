# KETARA Chatbot - LSTM Implementation

Chatbot berbasis LSTM untuk memberikan informasi seputar kampus ITERA.

## Struktur Project

```
KETARA-chat/
├── data.json                    # Dataset intents dan responses
├── chatbot_lstm.py              # Main implementation (Class-based)
├── demo_chatbot.py              # Demo script untuk testing
├── KETARA_Chatbot_LSTM.ipynb    # Jupyter notebook lengkap
├── requirements.txt             # Dependencies
└── README.md                    # Dokumentasi ini
```

## Arsitektur Model LSTM

Model chatbot menggunakan arsitektur Deep Learning dengan komponen:

1. **Embedding Layer (128 dimensi)**
   - Mengubah token kata menjadi dense vector representation
   - Vocabulary size dinamis berdasarkan dataset

2. **Bidirectional LSTM Layer 1 (128 units)**
   - Memproses sequence dari dua arah (forward & backward)
   - Return sequences untuk stacking
   - Dropout 0.5 dan recurrent dropout 0.2

3. **Bidirectional LSTM Layer 2 (64 units)**
   - Layer kedua untuk menangkap context lebih dalam
   - Dropout untuk regularization

4. **Dense Layers**
   - Dense layer (64 units) dengan ReLU activation
   - Dropout 0.25
   - Output layer dengan softmax untuk multi-class classification

## Fitur

- ✅ Intent classification menggunakan LSTM
- ✅ Preprocessing text otomatis
- ✅ Bidirectional LSTM untuk better context understanding
- ✅ Early stopping untuk mencegah overfitting
- ✅ Model checkpoint untuk save best model
- ✅ Confidence threshold untuk menangani pertanyaan di luar domain
- ✅ Interactive chat mode
- ✅ Comprehensive training visualization

## Instalasi

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verifikasi Dataset

Pastikan file `data.json` ada di direktori project dengan format:

```json
[
    {
        "tag": "nama_intent",
        "patterns": ["pertanyaan 1", "pertanyaan 2"],
        "responses": ["jawaban 1", "jawaban 2"]
    }
]
```

## Cara Menggunakan

### Opsi 1: Menggunakan Python Script

#### Training Model

```bash
python chatbot_lstm.py
```

Script ini akan:
- Load dataset dari `data.json`
- Preprocessing dan tokenization
- Build LSTM model
- Training dengan validation split
- Save model dan preprocessing objects

#### Testing Chatbot

```bash
python demo_chatbot.py
```

### Opsi 2: Menggunakan Jupyter Notebook

1. Buka notebook:
```bash
jupyter notebook KETARA_Chatbot_LSTM.ipynb
```

2. Jalankan cell secara berurutan untuk:
   - Explorasi dataset
   - Visualisasi distribusi data
   - Training model
   - Evaluasi dengan confusion matrix
   - Testing interaktif

## Model Architecture Details

```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
embedding_layer             (None, 20, 128)           Variable  
bidirectional_lstm_1        (None, 20, 256)           263,168   
bidirectional_lstm_2        (None, 128)               164,352   
dropout_layer               (None, 128)               0         
dense_1                     (None, 64)                8,256     
dropout_2                   (None, 64)                0         
output_layer                (None, num_classes)       Variable  
=================================================================
```

### Hyperparameters

- **Embedding Dimension**: 128
- **LSTM Units**: 128 (layer 1), 64 (layer 2)
- **Dropout Rate**: 0.5
- **Learning Rate**: Adam default (0.001)
- **Batch Size**: 8
- **Max Epochs**: 100 (dengan early stopping)
- **Validation Split**: 20%

## Output Files Setelah Training

Setelah training, akan dihasilkan file:

- `ketara_chatbot.h5` - Trained model
- `ketara_chatbot_best.h5` - Best model berdasarkan validation accuracy
- `tokenizer.pkl` - Tokenizer object
- `label_encoder.pkl` - Label encoder object
- `responses.pkl` - Dictionary responses
- `config.pkl` - Model configuration

## Contoh Penggunaan dalam Code

```python
from chatbot_lstm import KETARAChatbot

# Initialize chatbot
chatbot = KETARAChatbot('data.json')

# Load trained model
chatbot.load_trained_model('ketara_chatbot.h5')

# Get response
user_input = "Apa itu ITERA?"
response = chatbot.get_response(user_input)
print(response)

# Interactive chat
chatbot.chat()
```

## Dataset Structure

Dataset `data.json` berisi 7 intents:

1. `tentang_itera` - Informasi umum tentang ITERA
2. `lokasi_itera` - Lokasi kampus
3. `fakultas_itera` - Daftar fakultas
4. `fakultas_fs` - Fakultas Sains
5. `fakultas_ftik` - Fakultas Teknologi Infrastruktur dan Kewilayahan
6. `fakultas_fti` - Fakultas Teknologi Industri
7. `logo_itera` - Makna logo ITERA

Total: ~30 patterns untuk training

## Evaluasi Model

Model akan dievaluasi dengan:
- **Accuracy**: Training dan validation accuracy
- **Loss**: Categorical crossentropy loss
- **Confusion Matrix**: Visualisasi prediksi vs aktual
- **Classification Report**: Precision, recall, F1-score per intent
- **Confidence Distribution**: Analisis confidence scores

## Pengembangan Selanjutnya

Untuk meningkatkan performa, dapat ditambahkan:

1. **Data Augmentation**
   - Paraphrasing patterns
   - Synonym replacement
   - Back-translation

2. **Advanced Architecture** (Phase 2)
   - Attention Mechanism
   - Transformer-based models
   - BERT/GPT fine-tuning

3. **Feature Enhancement**
   - Context-aware responses
   - Multi-turn conversation
   - Entity recognition
   - Sentiment analysis

4. **Deployment**
   - REST API dengan Flask/FastAPI
   - Web interface
   - Integration dengan platform chat (Telegram, WhatsApp, dll)

## Troubleshooting

### Issue: Model overfitting
**Solution**: 
- Increase dropout rate
- Add more training data
- Use data augmentation

### Issue: Low accuracy
**Solution**:
- Increase LSTM units
- Add more training epochs
- Improve data quality
- Balance dataset distribution

### Issue: Out of memory
**Solution**:
- Reduce batch size
- Reduce LSTM units
- Use smaller embedding dimension

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- NumPy 1.23+
- scikit-learn 1.0+
- Matplotlib 3.5+
- Seaborn 0.12+

## Author

Deep Learning Project - ITERA Chatbot Implementation

## License

Untuk keperluan akademik dan pembelajaran.

---

**Note**: Ini adalah implementasi dasar LSTM untuk chatbot. Untuk production use, disarankan menggunakan arsitektur yang lebih advanced seperti Transformer atau fine-tuned pre-trained models.
