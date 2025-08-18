#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from LSTMAE import LSTMAE


# Class for pytorch dataloader
class IncubatorDataset(Dataset):
    def __init__(self, data, seq_len, step):
        
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.step = step
        
        # Calculates number of sequences
        self.n_seq = (len(data) - seq_len) // step + 1
        
    def __len__(self):
        return self.n_seq
    
    def __getitem__(self, ind):
        # Get a sequence starting at position index * step
        start = ind * self.step
        end = start + self.seq_len
        sequence = self.data[start:end]
        return sequence

# Loads and preprocesses dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Features 
    error = df.iloc[:, 1]  # temp pred error
    action = df.iloc[:, 2]  # action binary
    error_rate = np.diff(error, prepend=error[0]) # Change in error from last time step
    
    data = np.column_stack([error, action, error_rate])
    
    return data 


def create_sequences(data, seq_len, step, train_split):
    # Split data chronologically 
    split_ind = int(len(data) * train_split)
    train_data = data[:split_ind]
    val_data = data[split_ind:]
    
    # Create datasets
    train_dataset = IncubatorDataset(train_data, seq_len, step)
    val_dataset = IncubatorDataset(val_data, seq_len, step)
    
    return train_dataset, val_dataset



def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    # Optimiser and loss function 
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss()
    
    # logs loss
    train_losses = []
    val_losses = []
    
    # Training 
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch)
            
            # Backward pass
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            train_loss += loss.item()
        
        # Validation 
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item()
        
        # average loss per 10th epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses


def reconstruction_errors(model, data_loader):
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch)
            
            # calculates MSE for each sequence
            batch_errors = ((output - batch) ** 2).mean(dim=(1, 2))
            errors.extend(batch_errors.cpu().numpy())
    
    return np.array(errors)



def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_reconstruction_errors(errors):
    plt.figure(figsize=(12, 6))
    
    timesteps = np.arange(len(errors))
    
    plt.plot(timesteps, errors, label='Reconstruction Error', alpha=0.7)    
    plt.xlabel('Time')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    print(f"Mean reconstruction error: {errors.mean()}")
    
    


# ---- Main training script ----
if __name__ == "__main__":
    device = torch.device('cuda')
    
    csv = 'dataset/Trainingset'
    data = load_data(csv)
    print(f"Loaded data shape: {data.shape}")
    
    # input sequence length 
    seq_len = 60
    
    # Creates sequences
    train_dataset, val_dataset = create_sequences(data, seq_len=seq_len, step=1, train_split=0.8)
    print(f"Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")
    
    # Pytorch data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    
    # Initialise model
    model = LSTMAE(input_size=3, hidden_size=32, seq_len=seq_len, dropout=0.3)
    model = model.to(device)
    
    # Training model
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=100)
    
    # Plots training history
    plot_training_history(train_losses, val_losses)
    
    # Val reconstruction error plot
    val_errors = reconstruction_errors(model, val_loader)
    plot_reconstruction_errors(val_errors)
    
    # Saving model 
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses}, 
        'lstmae.pth')
    

# --- Additional val --- #

csv = 'dataset/Valset' 
data = load_data(csv)
print(f"Loaded data shape: {data.shape}")
val_data = IncubatorDataset(data, 60, 1)

device = torch.device('cuda')

model = LSTMAE(input_size=3, hidden_size=64, seq_len=60, dropout=0.3)
    
# Load trained parameters
trained_params = torch.load('lstmae.pth', map_location=device)
model.load_state_dict(trained_params['model_state_dict'])
model = model.to(device)
model.eval()


batch_size = 32
val_loader = DataLoader(val_data, batch_size, shuffle=False)

val_errors = reconstruction_errors(model, val_loader)
plot_reconstruction_errors(val_errors)


### --- Hyperparameter optimisation (Doesn't save model) --- ###
if __name__ == "__main__":
    device = torch.device('cuda')
    
    csv = 'dataset/Trainingset'
    data = load_data(csv)
    print(f"Loaded data shape: {data.shape}")
    
    # input sequence length 
    seq_len = 60
    
    # Creates sequences
    train_dataset, val_dataset = create_sequences(data, seq_len=seq_len, step=1, train_split=0.8)
    print(f"Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")
    
    # Pytorch data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    
    # Initialise model
    model = LSTMAE(input_size=3, hidden_size=64, seq_len=seq_len, dropout=0.3)
    model = model.to(device)
    
    # Training model
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=100)
    
    # Plots training history
    plot_training_history(train_losses, val_losses)
    
    # Val reconstruction error plot
    val_errors = reconstruction_errors(model, val_loader)
    plot_reconstruction_errors(val_errors)


