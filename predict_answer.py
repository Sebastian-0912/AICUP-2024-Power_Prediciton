import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import torch
from sklearn.preprocessing import MinMaxScaler
from dataloader import SolarPowerDataset
from transformer import TransformerModel
from utils.mask import generate_tgt_mask


def predict_single_data(model: TransformerModel, data: pd.DataFrame, scaler: MinMaxScaler):
    """
    Predict solar power for a single input dataset.

    Args:
        model (TransformerModel): Trained model for prediction.
        data (pd.DataFrame): Input data for prediction (72 timesteps).
        scaler (MinMaxScaler): Scaler used for normalizing data during training.

    Returns:
        np.array: Predicted power for 48 timesteps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # print(data)
    # Preprocess the input data
    input_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 72, features)
    # print(input_data.shape)
    # Prepare input for the decoder (12 known timesteps from the second day)
    tgt_input = input_data[:, -12:, 5:6]  # (1, 12, features)
    # print(tgt_input.shape)
    # Perform autoregressive prediction
    with torch.no_grad():
        for _ in range(48):
            tgt_mask = generate_tgt_mask(tgt_input.size(1)).to(device)  # Update mask
            output = model(input_data, tgt_input, tgt_mask=tgt_mask)  # Model prediction
            next_step = output[:, -1:, :]  # Get the last timestep prediction
            tgt_input = torch.cat([tgt_input, next_step], dim=1)  # Append the prediction

    # Extract the predicted values and inverse transform
    predicted_power = tgt_input[:, 12:, :].squeeze(0).cpu().numpy()  # (48, features)
    predicted_power = scaler.inverse_transform(predicted_power)[:, 0]  # Extract power (first feature)
    return predicted_power


def process_upload_data(upload_path):
    """
    Process the upload.csv file into a structured DataFrame.

    Args:
        upload_path (str): Path to the upload.csv file.

    Returns:
        pd.DataFrame: Structured DataFrame with parsed DateTime and location.
    """
    data = pd.read_csv(upload_path)
    data['DateTime'] = data['序號'].apply(lambda x: str(x)[:12])
    data['DateTime'] = pd.to_datetime(data['DateTime'], format='%Y%m%d%H%M')
    data['location'] = data['序號'].apply(lambda x: str(x)[13:] if str(x)[12] == '0' else str(x)[12:])
    return data


def main():
    """
    Main function to process the public test dataset and generate predictions.
    """
    # Load the upload data
    upload_data = process_upload_data("dataset/upload.csv")

    # Initialize the prediction output
    predictions = []

    # Loop through each unique location and date pair
    for i in range(0, len(upload_data), 48):
        predicted_date, location_id = upload_data.iloc[i][['DateTime', 'location']]
        # print(predicted_date)
        # Load location-specific training data
        location_data = pd.read_csv(f"./dataset/36_TrainingData_for_test/L{location_id}_Train_resampled.csv")
        location_data['DateTime'] = pd.to_datetime(location_data['DateTime'])
        
        # Load the scaler used during training
        dataset = SolarPowerDataset(data_path=f"./dataset/36_TrainingData_interpolation_process/L{location_id}_Train_resampled.csv")
        scaler = dataset.scaler
        
        # Extract required input data for the model
        start_time = predicted_date - timedelta(days=1, hours=2, minutes=0)
        end_time = predicted_date
        input_data = location_data.set_index('DateTime').loc[start_time:end_time]
        
        input_data['Power(mW)'] = scaler.transform(input_data[['Power(mW)']])  # Normalize using the same scaler as training
        input_data.reset_index(inplace=True)
        input_data = input_data[['WindSpeed(m/s)', 'Pressure(hpa)',
       'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)', 'Hour_sin', 'Hour_cos', 'Minute_sin', 'Minute_cos',
       'Month_sin', 'Month_cos']]
        # print(input_data)
        
        # Initialize and load the trained model
        model_path = f"./model_pth/v3/location_{location_id}/L_{location_id}_ep_2000.pth"
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
        # print(input_data.columns)
        # Perform prediction
        predicted_power = predict_single_data(model, input_data.values, scaler)
        predictions.append(predicted_power)

    # Generate upload file
    flat_predictions = [val for sublist in predictions for val in sublist]
    upload_data['答案'] = flat_predictions
    upload_data.drop(columns=['DateTime', 'location'], inplace=True)
    upload_data.to_csv("upload_with_predictions.csv", index=False, encoding="utf-8-sig")
    print("Predictions saved to upload_with_predictions.csv")


if __name__ == "__main__":
    main()
