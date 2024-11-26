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
    
    
class SolarPowerDatasetAllData(Dataset):
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
       'Month_sin', 'Month_cos','Location_1.0', 'Location_2.0', 'Location_3.0', 'Location_4.0',
       'Location_5.0', 'Location_6.0', 'Location_7.0', 'Location_8.0',
       'Location_9.0', 'Location_10.0', 'Location_11.0', 'Location_12.0',
       'Location_13.0', 'Location_14.0', 'Location_15.0', 'Location_16.0',
       'Location_17.0']].values
        Y = self.data.iloc[start_idx + self.lookback - 12 : start_idx + self.lookback + self.predict_ahead][['Power(mW)']].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    