import yfinance as yf
import pandas as pd

def get_stock_data(tickers,start_date = "2020-01-01" ,end_date = "2026-01-01"):
    
    """
    Fetches hidtorical dta and calculates daily returns
    """
    print(f"Downloading data for: {tickers}....")
    
    #Download data
    data = yf.download(tickers , start=start_date , end= end_date )['Adj Colse']
    
    #Calculation Daily Returns :(Price_t / Price_t-1) - 1
    
    returns = data.pct_change().dropna()
    
    return data , returns

if __name__ == "__main__":
    #Test with a popular portfolio
    
    portfolio = ['AAPL' , 'TSLA' , 'GOOGL' , 'MSFT']
    
    prices , daily_returns = get_stock_data(portfolio)
    
    print("\n---Price Data Head---")
    print(prices.head())
    
    #Save to CSV so we don't have to ping the API
    prices.to_csv("raw_prices.csv")
    print("\nSuccess! Data saved to raw_prices.csv")
    
    