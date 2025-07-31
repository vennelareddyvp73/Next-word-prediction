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
