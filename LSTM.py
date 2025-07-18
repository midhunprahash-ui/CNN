import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.text import Tokenizer # Keeping for text preprocessing convenience
from tensorflow.keras.preprocessing.sequence import pad_sequences # Keeping for text preprocessing convenience
from sklearn.model_selection import train_test_split
import numpy as np

# --- 1. Configuration and Hyperparameters ---
# Define vocabulary size based on your dataset.
VOCAB_SIZE = 10000
# Maximum length of input sequences.
MAX_SEQUENCE_LENGTH = 100
# Dimension of the word embeddings.
EMBEDDING_DIM = 128
# Number of LSTM units in each direction.
HIDDEN_DIM = 64 # Renamed for clarity in PyTorch context (often called hidden_size)
# Number of LSTM layers.
NUM_LAYERS = 1
# Dropout rate to prevent overfitting.
DROPOUT_RATE = 0.5
# Number of training epochs.
EPOCHS = 10
# Batch size for training.
BATCH_SIZE = 32
# Number of output classes (e.g., 'Anxious', 'Calm', 'Neutral').
# IMPORTANT: These labels represent general emotional or communication styles,
# NOT clinical diagnoses of psychopathy or any other mental health condition.
NUM_CLASSES = 3 # Example: 'Aggressive Tone', 'Neutral Tone', 'Empathetic Tone'

# Determine device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Sample Data (Replace with your actual dataset) ---
# For a real project, you would load a dataset containing user-generated
# text and corresponding labels for emotional tone or communication style.
# Example: 'text': "You always mess things up!", 'label': 'Aggressive Tone'
# 'text': "I understand your point of view.", 'label': 'Empathetic Tone'
# 'text': "The sky is blue today.", 'label': 'Neutral Tone'

sample_texts = [
    "You are always so careless, it's frustrating!", # Aggressive Tone
    "I appreciate your effort and understand your challenges.", # Empathetic Tone
    "The report needs to be submitted by Friday.", # Neutral Tone
    "Why do you never listen to what I say?", # Aggressive Tone
    "I can see why you feel that way, it's completely valid.", # Empathetic Tone
    "The meeting starts at 10 AM.", # Neutral Tone
    "This is unacceptable, fix it immediately!", # Aggressive Tone
    "I'm here to support you through this difficult time.", # Empathetic Tone
    "The weather forecast predicts rain tomorrow.", # Neutral Tone
    "You constantly make mistakes, it's infuriating.", # Aggressive Tone
    "It sounds like you're going through a lot, how can I help?", # Empathetic Tone
    "Please send the revised document.", # Neutral Tone
    "I'm fed up with your excuses!", # Aggressive Tone
    "I truly empathize with your situation.", # Empathetic Tone
    "The new software update is available.", # Neutral Tone
    "Your incompetence is costing us dearly.", # Aggressive Tone
    "I genuinely care about your well-being.", # Empathetic Tone
    "The cafeteria serves lunch from 12 to 1 PM.", # Neutral Tone
    "I demand an explanation for this failure.", # Aggressive Tone
    "I understand your perspective and respect your feelings." # Empathetic Tone
]

# Map labels to integers for classification
# 0: Aggressive Tone, 1: Empathetic Tone, 2: Neutral Tone
sample_labels = [
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1
]

# --- 3. Data Preprocessing ---

# Initialize Tokenizer (from Keras, as it's convenient for text processing)
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(sample_texts)

print(f"Vocabulary size: {len(tokenizer.word_index)}")

# Convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(sample_texts)

# Pad sequences to ensure uniform length
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,
                                 padding='post', truncating='post')

print(f"Shape of padded sequences: {padded_sequences.shape}")

# Convert numpy arrays to PyTorch tensors
# Labels need to be LongTensor for CrossEntropyLoss
X = torch.tensor(padded_sequences, dtype=torch.long)
y = torch.tensor(sample_labels, dtype=torch.long)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

# Create TensorDatasets and DataLoaders for batching
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. Build the BiLSTM Model in PyTorch ---

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout_rate):
        super(BiLSTMClassifier, self).__init__()
        # Embedding layer: Converts integer indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Bidirectional LSTM layer:
        # input_size: The number of expected features in the input (embedding_dim)
        # hidden_size: The number of features in the hidden state h
        # num_layers: Number of recurrent layers
        # bidirectional=True: Makes it a bidirectional LSTM
        # batch_first=True: Input and output tensors are provided as (batch, seq, feature)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer for classification
        # The output of a bidirectional LSTM with hidden_dim units is 2 * hidden_dim
        # (one for each direction).
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, text):
        # text shape: (batch_size, sequence_length)

        # Pass through embedding layer
        embedded = self.embedding(text)
        # embedded shape: (batch_size, sequence_length, embedding_dim)

        # Pass through LSTM layer
        # output: (batch_size, sequence_length, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)
        # cell: (num_layers * num_directions, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(embedded)

        # Concatenate the final forward and backward hidden states
        # hidden[-2, :, :] is the last forward hidden state
        # hidden[-1, :, :] is the last backward hidden state
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden shape: (batch_size, hidden_dim * 2)

        # Pass through the fully connected layer
        prediction = self.fc(hidden)
        # prediction shape: (batch_size, num_classes)

        return prediction

# Instantiate the model
model = BiLSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT_RATE).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification with integer labels
optimizer = optim.Adam(model.parameters())

print("\n--- Model Architecture ---")
print(model)

# --- 5. Train the Model ---
print("\n--- Training the model ---")
for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad() # Clear gradients

        predictions = model(texts) # Forward pass
        loss = criterion(predictions, labels) # Calculate loss

        loss.backward() # Backward pass (calculate gradients)
        optimizer.step() # Update weights

        total_loss += loss.item()
        _, predicted = torch.max(predictions.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    train_accuracy = correct_predictions / total_samples
    avg_train_loss = total_loss / len(train_loader)

    # --- Evaluate on test set after each epoch ---
    model.eval() # Set model to evaluation mode
    test_loss = 0
    correct_test_predictions = 0
    total_test_samples = 0

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            loss = criterion(predictions, labels)

            test_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            total_test_samples += labels.size(0)
            correct_test_predictions += (predicted == labels).sum().item()

    test_accuracy = correct_test_predictions / total_test_samples
    avg_test_loss = test_loss / len(test_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}, "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

# --- 6. Make Predictions (Example) ---
print("\n--- Making predictions on new text ---")
new_texts = [
    "I feel incredibly stressed and overwhelmed.", # Could be Aggressive or Neutral depending on context
    "This is a very peaceful and happy moment.", # Empathetic
    "The meeting was average.", # Neutral
    "I'm so worried about my future.", # Neutral or Aggressive (self-directed)
    "You are a genius, thank you so much!", # Empathetic
    "This is completely unacceptable and I'm very angry." # Aggressive
]

# Preprocess new texts using the same tokenizer
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, maxlen=MAX_SEQUENCE_LENGTH,
                                     padding='post', truncating='post')

# Convert to PyTorch tensor and move to device
new_input = torch.tensor(new_padded_sequences, dtype=torch.long).to(device)

model.eval() # Set model to evaluation mode for prediction
with torch.no_grad():
    predictions = model(new_input)
    # Apply softmax to get probabilities (CrossEntropyLoss already includes softmax implicitly)
    probabilities = torch.softmax(predictions, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)

# Map integer predictions back to labels
label_map = {0: 'Aggressive Tone', 1: 'Empathetic Tone', 2: 'Neutral Tone'}

print("\n--- Prediction Results ---")
for i, text in enumerate(new_texts):
    print(f"Text: '{text}'")
    print(f"Predicted Probability Distribution: {probabilities[i].cpu().numpy()}")
    print(f"Predicted Communication Style: {label_map[predicted_classes[i].item()]}\n")

