import torch

# Simple tokenizer (for demonstration purposes)
def simple_tokenizer(text):
    return [ord(char) for char in text]  # Convert characters to ASCII values

# Function to generate text using the model
def generate_text(model, tokenizer, start_text, max_length, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Tokenize and prepare input
        input_ids = simple_tokenizer(start_text)
        input_ids = input_ids[:max_length]  # Truncate
        input_ids += [0] * (max_length - len(input_ids))  # Pad
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)  # Add batch dimension

        for _ in range(50):  # Generate 50 tokens
            outputs = model(input_tensor)
            next_token_logits = outputs[:, -1, :]  # Get the last token logits
            next_token = torch.argmax(next_token_logits, dim=-1)  # Predict the next token

            # Append predicted token to the input
            input_tensor = torch.cat((input_tensor, next_token.unsqueeze(0)), dim=1)
            input_tensor = input_tensor[:, -max_length:]  # Keep the last max_length tokens

        # Convert tensor back to text
        generated_ids = input_tensor.squeeze(0).cpu().numpy().tolist()
        generated_text = ''.join([chr(token) for token in generated_ids if token > 0])  # Filter out padding
        return generated_text

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = 50000    # Vocabulary size
d_model = 2048        # Hidden size
n_layers = 24         # Number of layers
n_heads = 16          # Number of attention heads
d_ff = 8192           # Feedforward size
dropout = 0.1         # Dropout rate

# Instantiate the model
model = TransformerModel(vocab_size, d_model, n_layers, n_heads, d_ff, max_length, dropout)
model.load_state_dict(torch.load('transformer_model_1b.pth'))
model.to(device)

# Test the model
start_text = "Once upon a time"
max_length = 512  # Max length to generate
generated_text = generate_text(model, simple_tokenizer, start_text, max_length, device)

print("Generated Text:")
print(generated_text)