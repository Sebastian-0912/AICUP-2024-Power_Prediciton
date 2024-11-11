from torch.utils.data import DataLoader
from dataloader import SolarPowerDataset
import torch
import torch.nn as nn
import torch.optim as optim
from transformer import TransformerModel
from tqdm import tqdm
import os

# Directory to save model checkpoints
checkpoint_dir = "./model_pth/"

# Function to train the model
def train_model(model, dataloader, location_id, num_epochs=5, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Location {location_id} | Epoch {epoch}/{num_epochs}", leave=False)
        
        for X, Y in progress_bar:
            # Move data to the correct device
            src, tgt = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # Print average loss per epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Location {location_id} | Epoch {epoch} | Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 100 epochs
        if epoch % 200 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"location_{location_id}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved for Location {location_id} at epoch {epoch}.")

# Loop over each location to train a model for each
for location_id in range(1, 18):  # Assuming location IDs are 1 through 17
    print(f"\nTraining model for Location {location_id}")
    os.makedirs(os.path.join(checkpoint_dir, f"location_{location_id}") , exist_ok=True)
    
    # Load dataset specific to the current location
    dataset = SolarPowerDataset(
        data_path=f"/home/sebastian/Desktop/AICUP-2024-Power_Prediciton/dataset/36_TrainingData_process/L{location_id}_Train_resampled.csv"
    )
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # Initialize a new model for each location
    model = TransformerModel(
        src_input_dim=4,
        tgt_input_dim=1,
        d_model=256,
        nhead=4,
        num_encoder_layers=5,
        num_decoder_layers=5,
        dim_feedforward=512,
        dropout=0.1
    )
    
    # Train the model
    train_model(model, train_loader, location_id=location_id, num_epochs=2000, learning_rate=1e-4)
