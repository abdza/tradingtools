import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import argparse

def find_doubled_stocks(csv_path):
   df = pd.read_csv(csv_path)
   tickers = df['Symbol'].tolist()
   
   end_date = datetime.now()
   start_date = end_date - timedelta(days=7)
   doubled_stocks = []
   
   for ticker in tickers:
       try:
           stock = yf.Ticker(ticker)
           hist = stock.history(start=start_date, end=end_date)
           
           if not hist.empty:
               min_price = hist['Low'].min()  # Find lowest price in the period
               last_price = hist['Close'].iloc[-1]
               min_date = hist[hist['Low'] == min_price].index[0].strftime('%Y-%m-%d')
               
               if last_price >= min_price * 2:
                   doubled_stocks.append({
                       'ticker': ticker,
                       'min_price': min_price,
                       'min_date': min_date,
                       'end_price': last_price,
                       'return': ((last_price/min_price) - 1) * 100
                   })
       except Exception as e:
           continue
           
   return doubled_stocks

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("csv_file", help="Path to CSV file containing stock symbols")
   args = parser.parse_args()
   
   doubled_stocks = find_doubled_stocks(args.csv_file)
   for stock in doubled_stocks:
       print(f"{stock['ticker']}: ${stock['min_price']:.2f} ({stock['min_date']}) -> ${stock['end_price']:.2f} ({stock['return']:.1f}% increase)")
