import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


# Dummy dataset class
class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the text and truncate/pad to max_length
        tokens = self.tokenizer(self.texts[idx])
        tokens = tokens[:self.max_length]  # truncate
        if len(tokens) < self.max_length:
            tokens += [0] * (self.max_length - len(tokens))  # pad
        return torch.tensor(tokens)


# Simple tokenizer (for demonstration purposes)
def simple_tokenizer(text):
    return [ord(char) for char in text]  # Convert characters to ASCII values


# Sample texts
texts = [
            "Hello world!",
            "Transformer models are great.",
            "Let's train a simple model.",
            "This is an example sentence.",
            # Add more sentences to increase the dataset size
        ] * 100  # Duplicate to increase size for training

# Create dataset and dataloader
max_length = 512
dataset = SimpleTextDataset(texts, simple_tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
vocab_size = 50000  # Vocabulary size
d_model = 2048  # Hidden size
n_layers = 24  # Number of layers
n_heads = 16  # Number of attention heads
d_ff = 8192  # Feedforward size
dropout = 0.1  # Dropout rate

# Instantiate the model
model = TransformerModel(vocab_size, d_model, n_layers, n_heads, d_ff, max_length, dropout)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)

        # Create target tensor (shifted input for language modeling)
        targets = batch.clone()
        targets[:, :-1] = batch[:, 1:]  # Shift right for next token prediction
        targets[:, -1] = 0  # Use padding token for last position

        optimizer.zero_grad()
        outputs = model(batch)

        # Only use outputs for the real tokens (ignore padding)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

# Save the model
torch.save(model.state_dict(), 'transformer_model_1b.pth')