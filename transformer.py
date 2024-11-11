import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, src_input_dim=4, tgt_input_dim=1, d_model=512, nhead=4, num_encoder_layers=5, num_decoder_layers=5, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer layers
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, batch_first=True)
        
        # Separate input layers for `src` and `tgt`
        self.fc_in_src = nn.Linear(src_input_dim, d_model)
        self.fc_in_tgt = nn.Linear(tgt_input_dim, d_model)
        
        # Output layer to transform model output to target feature dimension
        self.fc_out = nn.Linear(d_model, tgt_input_dim)

    def forward(self, src, tgt):
        # Apply separate input layers to `src` and `tgt`
        src = self.fc_in_src(src)
        tgt = self.fc_in_tgt(tgt)
        
        # Apply positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Pass through Transformer
        output = self.transformer(src, tgt)
        
        # Apply final output layer
        output = self.fc_out(output)
        
        return output
