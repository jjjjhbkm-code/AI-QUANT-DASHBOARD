import torch
import torch.nn as nn
from lstm_model import prepare_data, StockLSTM


def train_network(ticker = 'AAPL' , epochs = 20 , lr = 0.001):
  
    X , y , scaler = prepare_data("raw_prices.csv" , ticker)
    
   
    model = StockLSTM()
    criterian = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters() , lr = lr)
    
    
    
    
    print(f"Starting training for {ticker}......")
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        
        loss = criterian(outputs,y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch[{epoch+1} / {epoch}] , Loss : {loss.item():.6f}')
    return model , scaler

if __name__ == "__main__":
    trained_model , data_scaler = train_network()
    
    torch.save(trained_model.state_dict() , "stock_model.pth")
    
    print("Model saved to stock_model.pth")
    
    
                  
            