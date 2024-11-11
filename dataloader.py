from torch.utils.data import Dataset,DataLoader
import pandas as pd
import torch

class SolarPowerDataset(Dataset):
    def __init__(self, data_path, lookback=72, predict_ahead=48):
        self.data = pd.read_csv(data_path)
        self.lookback = lookback  # 72 time steps for X
        self.predict_ahead = predict_ahead  # 48 time steps for Y

        # Identify valid date ranges for constructing data points
        self.valid_data_indices = self._find_valid_pairs()

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
        X = self.data.iloc[start_idx:start_idx + self.lookback][['Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)','Power(mW)']].values
        Y = self.data.iloc[start_idx + self.lookback:start_idx + self.lookback + self.predict_ahead][['Power(mW)']].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)