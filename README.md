# Next Word Prediction with LSTM

A neural network model that predicts the next word in a sequence using LSTM (Long Short-Term Memory) networks, trained on Franz Kafka's "The Metamorphosis".

## 📖 Overview

This project implements a deep learning approach to natural language processing by building a model that can predict the next word in a sentence based on the previous words. The LSTM neural network learns patterns and relationships in text to generate contextually appropriate word predictions.

**Key Features:**
- Text preprocessing and tokenization
- LSTM-based sequence modeling
- Word prediction functionality
- Training on classic literature text

## 🛠️ Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- Pickle (for saving tokenizer)

## 📦 Installation

```bash
pip install tensorflow numpy
```

## 🚀 Quick Start

### 1. Data Preparation

```python
# Load text data
file = open("metamorphosis_clean.txt", "r", encoding="utf8")
lines = []
for i in file:
    lines.append(i)

# Clean and preprocess text
data = ' '.join(lines)
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')
```

### 2. Tokenization

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
sequence_data = tokenizer.texts_to_sequences([data])[0]

vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")
```

### 3. Create Training Sequences

```python
sequence_length = 10
X_sequences = []
y_labels = []

for i in range(len(sequence_data) - sequence_length):
    X_sequences.append(sequence_data[i:i + sequence_length])
    y_labels.append(sequence_data[i + sequence_length])

X_sequences = np.array(X_sequences)
y_labels = to_categorical(y_labels, num_classes=vocab_size)
```

### 4. Build and Train Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(vocab_size, 10),
    LSTM(1000, return_sequences=True),
    LSTM(1000),
    Dense(1000, activation="relu"),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001)
)

# Train the model
model.fit(X_sequences, y_labels, epochs=20)
```

### 5. Make Predictions

```python
def predict_next_word(model, tokenizer, seed_text, sequence_length=10):
    tokens = tokenizer.texts_to_sequences([seed_text])[0]
    tokens = tokens[-sequence_length:]
    
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded = pad_sequences([tokens], maxlen=sequence_length)
    
    predicted_probs = model.predict(padded, verbose=0)
    predicted_id = predicted_probs.argmax()
    
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    return index_word.get(predicted_id, "<UNK>")

# Example usage
seed_text = "what a strenuous"
next_word = predict_next_word(model, tokenizer, seed_text)
print(f"Next word prediction: {next_word}")
```

## 🏗️ Model Architecture

The neural network consists of the following layers:

| Layer Type | Parameters | Description |
|------------|------------|-------------|
| Embedding | vocab_size, 10 | Converts word indices to dense vectors |
| LSTM | 1000 units, return_sequences=True | First LSTM layer for sequence processing |
| LSTM | 1000 units | Second LSTM layer for final sequence encoding |
| Dense | 1000 units, ReLU | Hidden layer for feature extraction |
| Dense | vocab_size, Softmax | Output layer for word probability distribution |

**Model Parameters:**
- Vocabulary Size: 2,617 unique words
- Sequence Length: 10 words
- Embedding Dimension: 10
- Total Parameters: ~28M parameters

## 📊 Dataset Information

- **Source**: Franz Kafka's "The Metamorphosis"
- **Preprocessing**: 
  - Removed newlines and special characters
  - Tokenized into word sequences
  - Created sliding window sequences of 10 words
- **Training Sequences**: 3,879 sequences
- **Unique Words**: 2,617 words

## 📈 Training Results

The model shows excellent learning progression:

```
Epoch 1/20 - Loss: 7.8758
Epoch 10/20 - Loss: 1.9993
Epoch 20/20 - Loss: 0.1033
```

The loss decreases significantly from ~7.88 to ~0.10, indicating successful learning of text patterns.

## 💡 Example Predictions

```python
# Example predictions
predict_next_word(model, tokenizer, "One morning when") → "Gregor"
predict_next_word(model, tokenizer, "what a strenuous") → "career"
predict_next_word(model, tokenizer, "the door") → "opened"
```

## 🔄 Future Improvements

- Implement beam search for better predictions
- Add temperature sampling for diverse outputs
- Experiment with different sequence lengths
- Try advanced architectures like Transformers
- Add text generation functionality
