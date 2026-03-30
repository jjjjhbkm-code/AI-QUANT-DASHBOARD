import streamlit as st


import pandas as pd
import plotly.graph_objects as go
import sys
import os







current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


try:
    from backend.lstm_model import StockLSTM, prepare_data
    from backend.optimizer import run_portfolio_optimization
except ModuleNotFoundError:
    st.error("Backend modules not found. Check your folder structure on GitHub!")

    

import torch
from backend.lstm_model import StockLSTM , prepare_data


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.optimizer import run_portfolio_optimization

st.set_page_config(page_title="AI Quant Dashboard (Founder-Neel Mondal)", layout="wide")

st.title(" AI-Driven Quant Trading Dashboard")
st.markdown("---")


st.sidebar.header("Configuration")
tickers = st.sidebar.multiselect("Select Tickers", ["AAPL", "TSLA", "MSFT", "GOOGL"], default=["AAPL", "MSFT"])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))


st.header("1. Portfolio Optimization (Risk vs Reward)")
if st.button("Optimize Portfolio"):
    with st.spinner("Calculating Efficient Frontier..."):
        
        results, weights = run_portfolio_optimization("backend/raw_prices.csv")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Optimal Weights")
            weight_df = pd.DataFrame({'Stock': tickers, 'Weight': weights})
            st.dataframe(weight_df.style.format({'Weight': '{:.2%}'}))
            
        with col2:
            st.subheader("Risk/Return Plot")
            fig = go.Figure(data=go.Scatter(
                x=results[1,:], y=results[0,:],
                mode='markers',
                marker=dict(color=results[2,:], colorscale='Viridis', showscale=True)
            ))
            fig.update_layout(xaxis_title="Volatility (Risk)", yaxis_title="Returns")
            st.plotly_chart(fig)

import torch
from backend.lstm_model import StockLSTM, prepare_data

def get_ai_prediction(ticker):
   
    X, y, scaler = prepare_data("backend/raw_prices.csv", ticker)
    
   
    model = StockLSTM()
    model.load_state_dict(torch.load("backend/stock_model.pth", weights_only=True))
    model.eval() 
    
    
    with torch.no_grad():
        
        last_window = X[-1].unsqueeze(0) 
        prediction_scaled = model(last_window)
        
        
        prediction_actual = scaler.inverse_transform(prediction_scaled.numpy())
        
    return prediction_actual[0][0]


st.header("2. AI Price Prediction")
selected_ticker = st.selectbox("Choose stock to predict:", tickers)

if st.button("Generate AI Forecast"):
    with st.spinner(f"Running LSTM Inference for {selected_ticker}..."):
        predicted_price = get_ai_prediction(selected_ticker)
        
       
        st.metric(label=f"Predicted Next Closing Price ({selected_ticker})", 
                  value=f"${predicted_price:.2f}")
        
        st.success("Disclaimer: AI predictions are based on historical patterns and are not financial advice.")