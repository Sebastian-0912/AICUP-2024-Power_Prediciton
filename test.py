from torch.utils.data import DataLoader
from dataloader import SolarPowerDataset
import torch
import torch.nn as nn
import numpy as np
from transformer import TransformerModel
from tqdm import tqdm
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from utils.mask import generate_tgt_mask

def test_model(model: TransformerModel, dataloader: DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for X, Y in tqdm(dataloader, desc="Testing"):
            # Move data to device
            src = X.to(device)  # Shape: (batch_size, 72, 18)
            tgt_known = Y[:, :12, :].to(device)  # First 12 time steps of target (batch_size, 12, 1)
            tgt_expected = Y[:, 12:, :].to(device)  # Remaining 48 time steps for evaluation (batch_size, 48, 1)

            # Initialize target input for decoder
            tgt_input = tgt_known  # Start with the known first 12 steps

            # Autoregressive prediction for the next 48 time steps
            for _ in range(48):  # Predict 48 timesteps
                tgt_mask = generate_tgt_mask(tgt_input.size(1)).to(device)  # Shape: (current_seq_len, current_seq_len)

                # Predict next timestep
                output = model(src, tgt_input, tgt_mask=tgt_mask)  # Shape: (batch_size, current_seq_len, 1)
                next_step = output[:, -1:, :]  # Get the last timestep prediction (batch_size, 1, 1)
                # Append prediction to target input
                tgt_input = torch.cat([tgt_input, next_step], dim=1)  # Shape: (batch_size, current_seq_len + 1, 1)
                # print(tgt_input.shape)
            # Store predictions and actual values for evaluation
            predictions.append(tgt_input[:, 12:, :].cpu().numpy())  # Predicted last 48 timesteps
            actuals.append(tgt_expected.cpu().numpy())
            
    return np.concatenate(predictions, axis=0), np.concatenate(actuals, axis=0)

for location_id in range(1,18):
  model_path = f"./model_pth/v3/location_{location_id}/L_{location_id}_ep_2000_loss_0.019493.pth"
  data_path = f"./dataset/36_TrainingData_interpolation_process/L{location_id}_Train_resampled.csv"
  test_dataset = SolarPowerDataset(data_path)
  test_loader = DataLoader(test_dataset, batch_size=75, shuffle=False)

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

  model.load_state_dict(torch.load(model_path))

  predictions, actuals = test_model(model, test_loader)
  
  # Reshape predictions and actuals to match the scaler's input format
  predictions = predictions.reshape(-1, 1)  # (batch_size * time_steps, 1)
  actuals = actuals.reshape(-1, 1)

  # Inverse transform
  predictions = test_dataset.scaler.inverse_transform(predictions)
  actuals = test_dataset.scaler.inverse_transform(actuals)

  # Reshape back for evaluation
  predictions = predictions.reshape(-1, 48)  # (batch_size, 48)
  actuals = actuals.reshape(-1, 48)
  print(predictions[0])
  print(actuals[0])
  mae = mean_absolute_error(actuals.flatten(), predictions.flatten())

  print(f"Test MAE: {mae:.4f}")