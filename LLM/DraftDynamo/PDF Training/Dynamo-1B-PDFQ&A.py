import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import fitz  # PyMuPDF
import os


# Dataset class for question answering
class QADataset(Dataset):
    def __init__(self, contexts, tokenizer, max_length):
        self.contexts = contexts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        # Tokenize the context
        context = self.tokenizer(self.contexts[idx])

        # Truncate/pad the tokens
        tokens = context[:self.max_length]  # Truncate
        if len(tokens) < self.max_length:
            tokens += [0] * (self.max_length - len(tokens))  # Pad

        return torch.tensor(tokens)


# Simple tokenizer (for demonstration purposes)
def simple_tokenizer(text):
    return [ord(char) for char in text]  # Convert characters to ASCII values


# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_folder):
    contexts = []

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            contexts.append(text)
            doc.close()

    return contexts


# Extract text from the PDF documents
pdf_folder = r"C:\Users\User\Desktop\PDF's"  # Update this path
contexts = extract_text_from_pdfs(pdf_folder)

# Create dataset and dataloader
max_length = 512
dataset = QADataset(contexts, simple_tokenizer, max_length)
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

        # For training, we will just shift the context as targets
        targets = batch.clone()  # You can adjust this for specific answer tokens

        optimizer.zero_grad()
        outputs = model(batch)

        # Calculate loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

# Save the model
torch.save(model.state_dict(), 'transformer_model_qa.pth')