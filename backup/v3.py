from torch.utils.data import Dataset
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

class SolarPowerDataset(Dataset):
    def __init__(self, data_path, lookback=72, predict_ahead=48):
        self.data = pd.read_csv(data_path)
        self.lookback = lookback  # 72 time steps for X
        self.predict_ahead = predict_ahead  # 48 time steps for Y
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Identify valid date ranges for constructing data points
        self.valid_data_indices = self._find_valid_pairs()
        
        self.data['Power(mW)'] = self.scaler.fit_transform(self.data[['Power(mW)']]) 
        
    def _find_valid_pairs(self):
        valid_indices = []
        for i in range(0, len(self.data)-60, 60):
            # Check for consecutive days
            start = pd.to_datetime(self.data.iloc[i]['DateTime'])
            end = pd.to_datetime(self.data.iloc[i + 60]['DateTime'])
            if end.normalize() - start.normalize() == pd.Timedelta(days=1):
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return len(self.valid_data_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_data_indices[idx]
        X = self.data.iloc[start_idx:start_idx + self.lookback][['WindSpeed(m/s)', 'Pressure(hpa)',
       'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)', 'Hour_sin', 'Hour_cos', 'Minute_sin', 'Minute_cos',
       'Month_sin', 'Month_cos']].values
        Y = self.data.iloc[start_idx + self.lookback - 12 : start_idx + self.lookback + self.predict_ahead][['Power(mW)']].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    

class SolarPowerDataset1min(Dataset):
    def __init__(self, data_path, lookback=720, predict_ahead=480):
        self.data = pd.read_csv(data_path)
        self.lookback = lookback  # 72 time steps for X
        self.predict_ahead = predict_ahead  # 48 time steps for Y
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Identify valid date ranges for constructing data points
        self.valid_data_indices = self._find_valid_pairs()
        
        self.data['Power(mW)'] = self.scaler.fit_transform(self.data[['Power(mW)']]) 
        
    def _find_valid_pairs(self):
        valid_indices = []
        for i in range(0, len(self.data)-600, 600):
            # Check for consecutive days
            start = pd.to_datetime(self.data.iloc[i]['DateTime'])
            end = pd.to_datetime(self.data.iloc[i + 600]['DateTime'])
            if end.normalize() - start.normalize() == pd.Timedelta(days=1):
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return len(self.valid_data_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_data_indices[idx]
        X = self.data.iloc[start_idx:start_idx + self.lookback][['WindSpeed(m/s)', 'Pressure(hpa)',
       'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)', 'Hour_sin', 'Hour_cos', 'Minute_sin', 'Minute_cos',
       'Month_sin', 'Month_cos']].values
        Y = self.data.iloc[start_idx + self.lookback - 120 : start_idx + self.lookback + self.predict_ahead][['Power(mW)']].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    

from torch.utils.data import DataLoader
from dataloader import SolarPowerDataset
import torch
import torch.nn as nn
import torch.optim as optim
from transformer import TransformerModel
from tqdm import tqdm
import os
from sklearn.preprocessing import MinMaxScaler
from utils.mask import generate_tgt_mask
from torch.optim.lr_scheduler import StepLR

# Directory to save model checkpoints
checkpoint_dir = "./model_pth/all_data/"

# Function to train the model
def train_model(model:TransformerModel, dataloader:DataLoader, scaler:MinMaxScaler, location_id, num_epochs=5, learning_rate=1e-4, model_pth=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_pth:
        model.load_state_dict(torch.load(model_pth))
    
    model.to(device)
    
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Add a learning rate scheduler
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)  # Reduce LR every 10 epochs by a factor of 0.5
    
    model.train()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"L{location_id} | Epoch {epoch}/{num_epochs}")
        
        for X, Y in progress_bar:
            # Move data to the correct device
            src = X.to(device)  # Input: first day + morning of second day (batch_size, 72, 18)
            tgt = Y[:, :-1, :].to(device)  # Target input: exclude the last timestep (batch_size, 60, 1)
            tgt_expected = Y[:, 1:, :].to(device)  # Target expected output: shifted by one (batch_size, 60, 1)
            
            # Generate target mask
            tgt_seq_len = tgt.size(1)  # Target sequence length (60)
            tgt_mask = generate_tgt_mask(tgt_seq_len).to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt, tgt_mask=tgt_mask)
            loss = criterion(output, tgt_expected)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # Print average loss per epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Location {location_id} | Epoch {epoch} | Loss: {avg_loss:.4f}")
    
        # Save checkpoint every 100 epochs
        if epoch % 200 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"location_{location_id}", f"L_{location_id}_ep_{epoch}_loss_{avg_loss:.6f}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved for Location {location_id} at epoch {epoch}.")
            
     # Step the scheduler after each epoch
    scheduler.step()
    
# Loop over each location to train a model for each
for location_id in range(1, 18):  # Assuming location IDs are 1 through 17
    print(f"\nTraining model for Location {location_id}")
    os.makedirs(os.path.join(checkpoint_dir, f"location_{location_id}") , exist_ok=True)
    
    # Load dataset specific to the current location
    dataset = SolarPowerDataset(
        # data_path=f"/home/sebastian/Desktop/AICUP-2024-Power_Prediciton/dataset/36_TrainingData_process/L{location_id}_Train_resampled.csv"
        data_path=f"/home/sebastian/Desktop/AICUP-2024-Power_Prediciton/dataset/36_TrainingData_interpolation_process/L{location_id}_Train_resampled.csv"
    )
    train_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    # Initialize a new model for each location
    model = TransformerModel(
        src_input_dim=12,
        tgt_input_dim=1,
        d_model=128,
        nhead=8,
        num_encoder_layers=5,
        num_decoder_layers=5,
        dim_feedforward=128,
        dropout=0.1
    )
    
    # Train the model
    train_model(model, train_loader, dataset.scaler, location_id=location_id, num_epochs=2000, learning_rate=1e-4)