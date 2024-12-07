import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, features, d_model, head, dropout, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(features, d_model)
        self.encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = self.embedding(X)  # Embed features
        X = self.encoding(X)   # Add positional encoding
        X = self.transformer_encoder(X)  # Pass through Transformer Encoder
        X = self.fc1(X[-1, :, :])  # Take the last sequence element
        X = self.dropout(X)
        X = self.fc2(X)
        return X
