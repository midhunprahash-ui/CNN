import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
# Hyoerparameters
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
NUM_LAYERS = 10
DROPOUT_RATE = 0.5
EPOCHS = 25
BATCH_SIZE = 32
NUM_CLASSES = 3
#Saving the model
MODEL_SAVE_PATH = 'bilstm_psychology_model.pth'
TOKENIZER_SAVE_PATH = 'tokenizer.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Sample texts
sample_texts = [
    "You are always so careless, it's frustrating!",
    "I appreciate your effort and understand your challenges.",
    "The report needs to be submitted by Friday.",
    "Why do you never listen to what I say?",
    "I can see why you feel that way, it's completely valid.",
    "The meeting starts at 10 AM.",
    "This is unacceptable, fix it immediately!",
    "I'm here to support you through this difficult time.",
    "The weather forecast predicts rain tomorrow.",
    "You constantly make mistakes, it's infuriating.",
    "It sounds like you're going through a lot, how can I help?",
    "Please send the revised document.",
    "I'm fed up with your excuses!",
    "I truly empathize with your situation.",
    "The new software update is available.",
    "Your incompetence is costing us dearly.",
    "I genuinely care about your well-being.",
    "The cafeteria serves lunch from 12 to 1 PM.",
    "I demand an explanation for this failure.",
    "I understand your perspective and respect your feelings."
]

sample_labels = [
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1
]

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(sample_texts)

print(f"Vocabulary size: {len(tokenizer.word_index)}")

sequences = tokenizer.texts_to_sequences(sample_texts)

padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,
                                 padding='post', truncating='post')

print(f"Shape of padded sequences: {padded_sequences.shape}")

X = torch.tensor(padded_sequences, dtype=torch.long)
y = torch.tensor(sample_labels, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout_rate):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        prediction = self.fc(hidden)
        return prediction

model = BiLSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT_RATE).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

print("\n--- Model Architecture ---")
print(model)

print("\n--- Training the model ---")

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()

        predictions = model(texts)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(predictions.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    train_accuracy = correct_predictions / total_samples
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    test_loss = 0
    correct_test_predictions = 0
    total_test_samples = 0

    with torch.no_grad():
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
    test_losses.append(avg_test_loss)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch+1}/{EPOCHS}, "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

print("\n--- Plotting Learning Curve ---")

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, test_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, test_accuracies, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\n--- Saving the trained model to {MODEL_SAVE_PATH} ---")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved successfully.")

with open(TOKENIZER_SAVE_PATH, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Tokenizer saved successfully to {TOKENIZER_SAVE_PATH}.")

print(f"\n--- Loading the model from {MODEL_SAVE_PATH} and making predictions ---")

loaded_model = BiLSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT_RATE).to(device)

if os.path.exists(MODEL_SAVE_PATH):
    loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    loaded_model.eval()
    print("Model loaded successfully.")
else:
    print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Cannot load model.")
    exit()

loaded_tokenizer = None
if os.path.exists(TOKENIZER_SAVE_PATH):
    with open(TOKENIZER_SAVE_PATH, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    print("Tokenizer loaded successfully.")
else:
    print(f"Error: Tokenizer file not found at {TOKENIZER_SAVE_PATH}. Cannot load tokenizer.")
    exit()

new_texts = [
    "What he doesn’t understand is that Trump knows very few words in the dictionary.",
    "I'm italian and italy supports Brazil",
    "A consumer market of more than 210 million people, a historic ally of the United States and where North American companies profit! Unbelievable!"
]

new_sequences = loaded_tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, maxlen=MAX_SEQUENCE_LENGTH,
                                     padding='post', truncating='post')

new_input = torch.tensor(new_padded_sequences, dtype=torch.long).to(device)

with torch.no_grad():
    predictions = loaded_model(new_input)
    probabilities = torch.softmax(predictions, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)

label_map = {0: 'Aggressive Tone', 1: 'Empathetic Tone', 2: 'Neutral Tone'}

print("\n--- Prediction Results (from loaded model) ---")
for i, text in enumerate(new_texts):
    print(f"Text: '{text}'")
    print(f"Predicted Probability Distribution: {probabilities[i].cpu().numpy()}")
    print(f"Predicted Communication Style: {label_map[predicted_classes[i].item()]}\n")