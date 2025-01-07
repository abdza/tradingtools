import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import yfinance as yf


def load_symbols(symbols_file, min_volume=1000000):
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


def download_5min_data(symbol, days=7):
    """
    Download 5-minute chart data for the specified symbol and number of days.

    Parameters:
    symbol (str): Stock/ETF symbol (e.g., 'SPY', 'AAPL')
    days (int): Number of days of historical data to download

    Returns:
    pandas.DataFrame: DataFrame with the chart data
    """
    try:
        # Calculate start and end dates
        end_date = datetime.now(pytz.timezone("US/Eastern"))
        # Start one day earlier to ensure we get pre-market data
        start_date = end_date - timedelta(days=days + 1)

        # Create ticker object
        ticker = yf.Ticker(symbol)

        # Download data
        print(f"Downloading 5-minute data for {symbol}...")
        df = ticker.history(start=start_date, end=end_date, interval="5m", prepost=True)

        # Reset index to make datetime a column
        df = df.reset_index()

        # Rename columns to match standard format
        df = df.rename(
            columns={
                "Datetime": "DateTime",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }
        )

        # Add symbol column
        df["Symbol"] = symbol

        return df

    except Exception as e:
        print(f"Error downloading data for {symbol}: {str(e)}")
        return None


def identify_market_session(df):
    """
    Identify market session (pre-market, regular, after-hours) for each data point
    
    Parameters:
    df (pandas.DataFrame): DataFrame with market data
    
    Returns:
    pandas.DataFrame: DataFrame with added session information
    """
    if df is None or len(df) == 0:
        return None

    df["DateTime"] = pd.to_datetime(df["DateTime"])

    # Convert to Eastern Time if not already
    if df["DateTime"].dt.tz is None:
        df["DateTime"] = df["DateTime"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    elif df["DateTime"].dt.tz.zone != "US/Eastern":
        df["DateTime"] = df["DateTime"].dt.tz_convert("US/Eastern")

    # Extract time component
    df["Time"] = df["DateTime"].dt.time

    # Define market sessions
    pre_market_start = pd.to_datetime("04:00").time()
    market_open = pd.to_datetime("09:30").time()
    market_close = pd.to_datetime("16:00").time()
    after_hours_end = pd.to_datetime("20:00").time()

    # Create session column
    conditions = [
        (df["Time"] >= pre_market_start) & (df["Time"] < market_open),
        (df["Time"] >= market_open) & (df["Time"] <= market_close),
        (df["Time"] > market_close) & (df["Time"] <= after_hours_end),
    ]
    choices = ["pre-market", "regular", "after-hours"]
    df["Session"] = np.select(conditions, choices, default="unknown")

    # Add date column if not present
    df["Date"] = df["DateTime"].dt.date

    # Initialize columns for session stats
    stats_columns = [
        "PreMarketStart", "PreMarketEnd", "PreMarketHigh", "PreMarketLow",
        "PreMarketVolume", "RegularOpen", "PreMarketGap", "OpenGap"
    ]
    for col in stats_columns:
        df[col] = np.nan

    # Calculate session stats for each date
    for date in df["Date"].unique():
        day_data = df[df["Date"] == date]

        # Get pre-market and regular session data
        pre_market = day_data[day_data["Session"] == "pre-market"]
        regular = day_data[day_data["Session"] == "regular"]

        if not pre_market.empty and not regular.empty:
            # Get indices for this date
            date_mask = df["Date"] == date

            # Calculate pre-market stats
            pre_market_start = pre_market.iloc[0]["Open"]
            pre_market_end = pre_market.iloc[-1]["Close"]
            pre_market_high = pre_market["High"].max()
            pre_market_low = pre_market["Low"].min()
            
            # Important: Use the actual volume from each pre-market row
            # Don't modify the original Volume column
            pre_market_volume = pre_market["Volume"].sum()

            regular_open = regular.iloc[0]["Open"]

            # Calculate gaps
            pre_market_gap = ((pre_market_end - pre_market_start) / pre_market_start) * 100
            open_gap = ((regular_open - pre_market_end) / pre_market_end) * 100

            # Assign values to the rows for this date
            df.loc[date_mask, "PreMarketStart"] = pre_market_start
            df.loc[date_mask, "PreMarketEnd"] = pre_market_end
            df.loc[date_mask, "PreMarketHigh"] = pre_market_high
            df.loc[date_mask, "PreMarketLow"] = pre_market_low
            df.loc[date_mask, "PreMarketVolume"] = pre_market_volume
            df.loc[date_mask, "RegularOpen"] = regular_open
            df.loc[date_mask, "PreMarketGap"] = pre_market_gap
            df.loc[date_mask, "OpenGap"] = open_gap

    return df
    

def filter_relevant_symbols(
    combined_df, min_dollar_volume=500_000, min_volatility=30.0
):
    """
    Filter symbols based on dollar volume and price volatility.

    Parameters:
    combined_df (pandas.DataFrame): Combined market data DataFrame
    min_dollar_volume (float): Minimum daily dollar volume
    min_volatility (float): Minimum price volatility percentage over the period

    Returns:
    pandas.DataFrame: Filtered DataFrame with relevant symbols only
    """
    # Group by symbol and date to calculate daily stats
    daily_stats = (
        combined_df.groupby(["Symbol", "Date"])
        .agg({"Close": "last", "High": "max", "Low": "min", "Volume": "sum"})
        .reset_index()
    )

    # Calculate dollar volume
    daily_stats["DollarVolume"] = daily_stats["Close"] * daily_stats["Volume"]

    # Calculate overall price range for each symbol
    symbol_stats = (
        daily_stats.groupby("Symbol")
        .agg(
            {
                "High": "max",
                "Low": "min",
                "DollarVolume": "mean",  # Use mean daily dollar volume
            }
        )
        .reset_index()
    )

    # Calculate total price movement percentage
    symbol_stats["TotalMovement"] = (
        (symbol_stats["High"] - symbol_stats["Low"]) / symbol_stats["Low"]
    ) * 100

    # Filter symbols based on both dollar volume and volatility
    valid_symbols = symbol_stats[
        (symbol_stats["DollarVolume"] >= min_dollar_volume)
        & (symbol_stats["TotalMovement"] >= min_volatility)
    ]["Symbol"].unique()

    # Filter the main DataFrame to keep only valid symbols
    filtered_df = combined_df[combined_df["Symbol"].isin(valid_symbols)].copy()

    print(f"\nFiltering results:")
    print(f"Total symbols: {len(daily_stats['Symbol'].unique())}")
    print(
        f"Symbols meeting dollar volume criteria: {len(symbol_stats[symbol_stats['DollarVolume'] >= min_dollar_volume])}"
    )
    print(
        f"Symbols meeting both dollar volume and {min_volatility}% volatility criteria: {len(valid_symbols)}"
    )
    print(f"Rows before filtering: {len(combined_df)}")
    print(f"Rows after filtering: {len(filtered_df)}")

    # Add volatility information to the filtered DataFrame
    volatility_info = symbol_stats[["Symbol", "TotalMovement"]]
    filtered_df = filtered_df.merge(volatility_info, on="Symbol", how="left")

    return filtered_df

def save_combined_data(all_data, output_dir="data"):
    """
    Save filtered data to multiple CSV files, organized by ticker and split by size
    """
    valid_data = [df for df in all_data if df is not None and len(df) > 0]

    if not valid_data:
        print("No valid data to save")
        return None

    # Combine all DataFrames
    combined_df = pd.concat(valid_data, ignore_index=True)

    # Apply pattern-based filtering
    filtered_df = filter_relevant_symbols(combined_df)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate base filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{output_dir}/filtered_market_data_{timestamp}"
    
    # Sort by Symbol first, then DateTime
    filtered_df = filtered_df.sort_values(['Symbol', 'DateTime'])
    
    # Group by symbol to maintain ticker integrity
    symbol_groups = filtered_df.groupby('Symbol')
    
    # Initialize variables for file splitting
    current_file_size = 0
    current_file_number = 1
    current_file_data = []
    saved_files = []
    
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB in bytes
    
    # Process each symbol group
    for symbol, group_df in symbol_groups:
        # Calculate approximate size of this group
        # Assuming UTF-8 encoding where each character is approximately 1 byte
        group_size = group_df.memory_usage(deep=True).sum()
        
        # If adding this group would exceed max file size and we already have data,
        # save current file and start a new one
        if current_file_data and (current_file_size + group_size > MAX_FILE_SIZE):
            # Save current file
            current_df = pd.concat(current_file_data, ignore_index=True)
            filename = f"{base_filename}_part{current_file_number}.csv"
            current_df.to_csv(filename, index=False)
            saved_files.append(filename)
            
            # Reset for next file
            current_file_data = []
            current_file_size = 0
            current_file_number += 1
        
        # Add current group to current file data
        current_file_data.append(group_df)
        current_file_size += group_size
    
    # Save any remaining data
    if current_file_data:
        current_df = pd.concat(current_file_data, ignore_index=True)
        filename = f"{base_filename}_part{current_file_number}.csv"
        current_df.to_csv(filename, index=False)
        saved_files.append(filename)
    
    # Generate and save session statistics
    stats_filename = f"{output_dir}/filtered_session_stats_{timestamp}.csv"
    session_stats = (
        filtered_df.groupby(["Symbol", "Date", "Session"])
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
            "PreMarketGap": "first",
            "OpenGap": "first",
        })
        .reset_index()
    )
    session_stats.to_csv(stats_filename, index=False)

    # Print summary
    print(f"\nData saved to {len(saved_files)} files:")
    for file in saved_files:
        file_size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"- {file} ({file_size_mb:.2f}MB)")
    print(f"Session statistics saved to {stats_filename}")
    print(f"Total rows: {len(filtered_df)}")
    print(f"Unique symbols: {filtered_df['Symbol'].nunique()}")
    print(f"Date range: {filtered_df['DateTime'].min()} to {filtered_df['DateTime'].max()}")

    return saved_files, stats_filename

def main(symbols_file=None, days=7, min_volume=200_000):
    """
    Main function to download and save data for multiple symbols

    Parameters:
    symbols_file (str): Path to CSV file containing symbols
    days (int): Number of days of historical data
    min_volume (int): Minimum daily volume filter
    """
    # Load symbols from file if provided, otherwise use default
    if symbols_file:
        symbols = load_symbols(symbols_file, min_volume)
        if not symbols:
            print("No symbols meeting volume criteria. Using default symbol.")
            symbols = ["SPY"]
    else:
        print("No symbols file provided. Using default symbol.")
        symbols = ["SPY"]

    # Download and process data for each symbol
    all_data = []
    for symbol in symbols:
        # Download data
        df = download_5min_data(symbol, days)

        # Add session information
        df = identify_market_session(df)

        if df is not None and len(df) > 0:
            all_data.append(df)

    # Save combined data
    save_combined_data(all_data)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Download and filter 5-minute chart data for pre-market and regular hours trading patterns"
    )
    parser.add_argument(
        "--symbols-file", type=str, help="Path to CSV file containing symbols"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of historical data to download",
    )
    parser.add_argument(
        "--min-volume", type=int, default=200_000, help="Minimum daily volume filter"
    )

    # Parse arguments
    args = parser.parse_args()

    main(args.symbols_file, args.days, args.min_volume)
