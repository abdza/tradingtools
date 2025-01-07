import argparse
import os
from datetime import datetime, timedelta
import concurrent.futures
import requests
from bs4 import BeautifulSoup

def load_symbols(symbols_file, min_volume=200_000):
    """
    Load symbols from a CSV file and filter by minimum volume

    Parameters:
    symbols_file (str): Path to CSV file containing symbols
    min_volume (int): Minimum daily volume filter

    Returns:
    list: List of filtered symbols to download
    """
    try:
        # Read the CSV file
        df = pd.read_csv(symbols_file)

        # Check if required columns exist
        required_columns = ["Symbol", "Volume 1 day"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"CSV file must contain these columns: {', '.join(missing_columns)}"
            )

        # Filter by minimum volume
        df_filtered = df[df["Volume 1 day"] >= min_volume]

        # Extract unique symbols
        symbols = df_filtered["Symbol"].unique().tolist()

        print(
            f"Loaded {len(symbols)} symbols from {symbols_file} (minimum volume: {min_volume:,})"
        )
        print(f"Filtered out {len(df) - len(df_filtered)} low-volume symbols")
        return symbols

    except Exception as e:
        print(f"Error loading symbols file: {str(e)}")
        return []

import pandas as pd
import pytz
import yfinance as yf
from sec_api import QueryApi
from newsapi.newsapi_client import NewsApiClient

class MarketDataCollector:
    def __init__(self, news_api_key=None, sec_api_key=None):
        self.news_api = NewsApiClient(api_key=news_api_key) if news_api_key else None
        self.sec_api = QueryApi(api_key=sec_api_key) if sec_api_key else None
    
    def get_sec_filings(self, symbol, start_date):
        """Get SEC filings for a symbol"""
        if not self.sec_api:
            return []
            
        query = {
            "query": {
                "query_string": {
                    "query": f"ticker:{symbol} AND filedAt:[{start_date} TO now]"
                }
            },
            "from": "0",
            "size": "10",
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        try:
            filings = self.sec_api.get_filings(query)
            return [{
                'type': filing['formType'],
                'timestamp': filing['filedAt'],
                'description': filing['description'],
                'url': filing['linkToFilingDetails']
            } for filing in filings['filings']]
        except Exception as e:
            print(f"Error getting SEC filings for {symbol}: {e}")
            return []

    def get_news_articles(self, symbol, start_date):
        """Get news articles for a symbol"""
        if not self.news_api:
            return []
            
        try:
            news = self.news_api.get_everything(
                q=symbol,
                from_param=start_date,
                language='en',
                sort_by='publishedAt'
            )
            
            return [{
                'title': article['title'],
                'timestamp': article['publishedAt'],
                'source': article['source']['name'],
                'url': article['url']
            } for article in news['articles']]
        except Exception as e:
            print(f"Error getting news for {symbol}: {e}")
            return []

    def get_social_media_mentions(self, symbol, start_date):
        """
        Get social media mentions (placeholder - you'd need to implement specific API calls
        for Twitter, Reddit, etc.)
        """
        return []

    def collect_catalyst_data(self, symbol, start_date):
        """Collect all catalyst data for a symbol"""
        catalysts = {
            'sec_filings': self.get_sec_filings(symbol, start_date),
            'news': self.get_news_articles(symbol, start_date),
            'social': self.get_social_media_mentions(symbol, start_date)
        }
        
        # Combine and sort all catalysts by timestamp
        all_catalysts = []
        for source, items in catalysts.items():
            for item in items:
                item['source'] = source
                all_catalysts.append(item)
        
        return sorted(all_catalysts, key=lambda x: x['timestamp'])

    def download_market_data(self, symbol, days=7):
        """Download market data including pre/post market"""
        try:
            end_date = datetime.now(pytz.timezone("US/Eastern"))
            start_date = end_date - timedelta(days=days+1)
            
            # Get price data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="5m", prepost=True)
            
            # Get catalyst data
            catalysts = self.collect_catalyst_data(symbol, start_date.strftime('%Y-%m-%d'))
            
            # Create catalyst markers in the dataframe
            df['has_catalyst'] = False
            df['catalyst_type'] = None
            df['catalyst_description'] = None
            
            for catalyst in catalysts:
                catalyst_time = pd.to_datetime(catalyst['timestamp'])
                # Find the closest time in our price data
                closest_idx = df.index[df.index.get_indexer([catalyst_time], method='nearest')[0]]
                df.at[closest_idx, 'has_catalyst'] = True
                df.at[closest_idx, 'catalyst_type'] = catalyst['source']
                df.at[closest_idx, 'catalyst_description'] = catalyst.get('description', 
                                                                       catalyst.get('title', ''))
            
            return df.reset_index()
            
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
            return None

def save_combined_data(data_frames, output_dir="data"):
    """Save market data and catalyst data"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save market data
    market_data = pd.concat([df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol',
                                'has_catalyst', 'catalyst_type']] 
                           for df in data_frames if df is not None],
                           ignore_index=True)
    market_filename = f"{output_dir}/market_data_{timestamp}.csv"
    market_data.to_csv(market_filename, index=False)
    
    # Save detailed catalyst data
    catalyst_data = pd.concat([df[['DateTime', 'Symbol', 'has_catalyst', 'catalyst_type',
                                 'catalyst_description']]
                             .loc[df['has_catalyst']]
                             for df in data_frames if df is not None],
                             ignore_index=True)
    catalyst_filename = f"{output_dir}/catalyst_data_{timestamp}.csv"
    catalyst_data.to_csv(catalyst_filename, index=False)
    
    return market_filename, catalyst_filename

def main(symbols_file=None, days=7, min_volume=200_000,
         news_api_key=None, sec_api_key=None):
    # Initialize collector
    collector = MarketDataCollector(news_api_key, sec_api_key)
    
    # Load symbols
    if symbols_file:
        symbols = load_symbols(symbols_file, min_volume)
    else:
        symbols = ["SPY"]
    
    # Download data
    all_data = []
    for symbol in symbols:
        df = collector.download_market_data(symbol, days)
        if df is not None:
            all_data.append(df)
    
    # Save data
    market_file, catalyst_file = save_combined_data(all_data)
    print(f"Market data saved to: {market_file}")
    print(f"Catalyst data saved to: {catalyst_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols-file", type=str)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--min-volume", type=int, default=200_000)
    parser.add_argument("--news-api-key", type=str)
    parser.add_argument("--sec-api-key", type=str)
    
    args = parser.parse_args()
    main(args.symbols_file, args.days, args.min_volume,
         args.news_api_key, args.sec_api_key)
