import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        return self.layer_norm2(x + self.dropout2(ffn_output))

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_length, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

# Parameters
vocab_size = 50000    # Vocabulary size
d_model = 2048        # Hidden size (increased)
n_layers = 24         # Number of layers (remains)
n_heads = 16          # Number of attention heads
d_ff = 8192           # Feedforward size (increased)
max_length = 512      # Max sequence length
dropout = 0.1         # Dropout rate

# Model instantiation
model = TransformerModel(vocab_size, d_model, n_layers, n_heads, d_ff, max_length, dropout)

# Print model summary
print(model)

# Check total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')