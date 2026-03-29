import pandas as pd
import numpy as np


def run_portfolio_optimization(file_path):
    #load data
    df = pd.read_csv(file_path,index_col = 0)
    returns = df.pct_change().dropna
    
    #calculate mean returns and covariance
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_portfolios = 5000
    results = np.zeros((3 , num_portfolios))
    weights_record = []
    
    #monte carlo simulation
    for i in range(num_portfolios):
        #generate random weights
        weights = np.random.random(len(mean_returns))
        
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        #calculate annualized portfolio return (252 trading days)
        portfolio_return = np.sum(mean_returns * weights)*252
        
        #calculate annualized portfolio volatility
        portfolio_std_dev = np.sqrt(np.dot(weights.T , np.dot(cov_matrix *252 , weights)))
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        results[2,i]= results[0,i] / results[1,i]
        
    #Identify the best portfolio
    max_sharp_idx = np.argmax(results[2])
    best_weights = weights_record[max_sharp_idx]
    
    return results , best_weights

if __name__ == "__main__":
    results , weights = run_portfolio_optimization("raw_prices.csv")
    print(f"Optimal Weights: {weights}")
    print(f"Max Sharp Ratio :{np.max(results[2]):.2f}")
    
    
     
        
    
    