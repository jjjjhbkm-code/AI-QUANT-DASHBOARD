import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_data(file_path, ticker, window_size=60):
    df = pd.read_csv(file_path, index_col=0)
    data = df[ticker].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x, y = [], []
    for i in range(window_size, len(scaled_data)):
        x.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])
        
    
    x = torch.tensor(np.array(x), dtype=torch.float32).unsqueeze(-1) # [Batch, Window, Feature]
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
    
    return x, y, scaler


class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        
        lstm_out, _ = self.lstm(x)
       
        last_time_step = lstm_out[:, -1, :]
        return self.linear(last_time_step)

if __name__ == "__main__":
    X, y, scaler = prepare_data("raw_prices.csv", ticker="AAPL")
    model = StockLSTM()
    print("PyTorch Model Initialized!")
    print(model)