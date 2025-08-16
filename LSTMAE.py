
#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn

# Encoder layer
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm_encode = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Attention mechanism 
        self.attention = nn.Linear(hidden_size * 2, 1) 
        
        # Project bidirectional output back to original hidden_size
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, (last_hidden_state, last_cell_state) = self.lstm_encode(x)
        # out shape: (batch_size, seq_len, hidden_size * 2) due to bidirectional
        
        # Apply attention to focus on important timesteps
        attention_scores = self.attention(out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        attended_out = out * attention_weights  # (batch_size, seq_len, hidden_size * 2)
        
        # Project back to original hidden size
        projected_out = self.projection(attended_out)  # (batch_size, seq_len, hidden_size)
        
        # Extract last hidden state as latent representation
        x_encode = projected_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Dropout on encoded representation 
        x_encode = self.dropout(x_encode)
        
        # Repeat for all timesteps for decoder input
        x_encode = x_encode.unsqueeze(1).repeat(1, x.shape[1], 1)  
        
        return x_encode, projected_out


# Decoder layer 
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM takes repeated hidden states as input
        self.lstm_decode = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # Project back to original feature space
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, z):
        # z shape
        decode_output, (hidden_state, cell_state) = self.lstm_decode(z)
        
        # Dropout before reconstruct
        decode_output = self.dropout(decode_output)
        
        # Reconstruct original features
        decode_output = self.fc(decode_output)  
        
        return decode_output, hidden_state


# LSTM-AE
class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, dropout):
        super(LSTMAE, self).__init__()
        self.input_size = input_size  
        self.hidden_size = hidden_size 
        self.seq_len = seq_len 
        self.dropout = dropout
        
        self.encoder = Encoder(input_size, hidden_size, dropout)
        self.decoder = Decoder(input_size, hidden_size, dropout)

    def forward(self, x):
        # Encode to latent representation
        x_encode, encode_out = self.encoder(x)
        
        # Decode back to original space
        x_decode, last_hidden = self.decoder(x_encode)
        
        return x_decode


