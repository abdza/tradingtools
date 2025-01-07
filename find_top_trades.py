import argparse
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import pytz
import yfinance as yf


def download_stock_data(symbol, start_date, end_date):
    """
    Download 5-minute interval stock data for the specified period
    Returns None if data is not available or invalid
    """
    try:
        # Clean up the symbol - remove any whitespace and convert to uppercase
        symbol = symbol.strip().upper()

        # Validate symbol format
        if not symbol or "/" in symbol or len(symbol) > 10:
            print(f"Skipping invalid symbol format: {symbol}")
            return None

        ticker = yf.Ticker(symbol)

        # Sleep for 2 seconds to avoid rate limiting
        time.sleep(2)

        # Attempt to get info to verify the ticker exists
        try:
            info = ticker.info
            if info is None:
                print(f"Warning: No ticker information available for {symbol}")
                return None
        except Exception as e:
            print(f"Warning: Could not verify ticker {symbol}: {str(e)}")
            return None

        # Sleep again before downloading historical data
        time.sleep(2)

        # Add buffer days to ensure we get full premarket data
        buffer_start = start_date - timedelta(days=1)
        buffer_end = end_date + timedelta(days=1)

        # Download with pre/post market data
        data = ticker.history(
            start=buffer_start,
            end=buffer_end,
            interval="5m",
            prepost=True,
            actions=False,
        )

        if data is None or data.empty:
            print(f"Warning: No data available for {symbol}")
            return None

        # Verify we have the required columns
        required_columns = {"Open", "High", "Low", "Close", "Volume"}
        if not all(col in data.columns for col in required_columns):
            print(f"Warning: Missing required columns for {symbol}")
            return None

        # Verify the index is a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            print(f"Warning: Invalid data format for {symbol}")
            return None

        # Ensure the index is timezone-aware
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")

        return data

    except Exception as e:
        print(f"Error downloading data for {symbol}: {str(e)}")
        return None


def find_trading_opportunities(data, period_type="premarket"):
    """
    Find periods where the stock price increased over 30-minute windows
    period_type: 'premarket' (4:00-9:30) or 'market' (9:30-16:00)
    """
    if data is None or data.empty:
        return []

    opportunities = []

    try:
        # Ensure index is timezone-aware and in US Eastern time
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        data.index = data.index.tz_convert("US/Eastern")

        # Define time ranges
        if period_type == "premarket":
            start_time = "04:00"
            end_time = "09:30"
        else:  # normal market hours
            start_time = "09:30"
            end_time = "16:00"

        # Filter data for the specified period
        period_data = data.between_time(start_time, end_time)

        if period_data.empty:
            return []

        # Look for 30-minute windows where price increased
        for i in range(len(period_data) - 6):  # 6 5-minute intervals = 30 minutes
            window = period_data.iloc[i : i + 6]
            start_price = window.iloc[0]["Low"]
            max_price = window["High"].max()

            if max_price > start_price:
                gain_percentage = ((max_price - start_price) / start_price) * 100
                opportunities.append(
                    {
                        "timestamp": period_data.index[i],
                        "start_price": start_price,
                        "max_price": max_price,
                        "gain_percentage": gain_percentage,
                    }
                )

        return opportunities

    except Exception as e:
        print(f"Error finding opportunities: {str(e)}")
        return []

    return opportunities


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Analyze stock data for trading opportunities."
    )
    parser.add_argument(
        "csv_file", type=str, help="Path to CSV file containing stock symbols"
    )
    parser.add_argument(
        "-d",
        "--days",
        type=int,
        default=7,
        help="Number of days to analyze (default: 7)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print detailed error messages"
    )
    parser.add_argument(
        "-t", "--top", type=int, default=5, help="Top number to analyze (default: 5)"
    )

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Handle --help or invalid arguments
        return

    # Validate file exists
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: File '{args.csv_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{args.csv_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{args.csv_file}': {str(e)}")
        sys.exit(1)

    # Validate CSV structure
    if "Symbol" not in df.columns:
        print("Error: CSV file must contain a 'Symbol' column")
        sys.exit(1)

    # Clean up symbols
    df["Symbol"] = df["Symbol"].str.strip().str.upper()
    # Remove any invalid symbols
    df = df[df["Symbol"].str.len() <= 10]
    df = df[~df["Symbol"].str.contains("/", na=False)]

    symbols = df["Symbol"].tolist()
    symbols = df["Symbol"].tolist()

    # Set date range for the past week
    end_date = datetime.now(pytz.timezone("US/Eastern"))
    start_date = end_date - timedelta(days=args.days)

    # Store results for each day
    daily_results = {}

    for symbol in symbols:
        try:
            # Download data
            data = download_stock_data(symbol, start_date, end_date)

            # Process each trading day
            for day in pd.date_range(start_date, end_date, freq="B"):
                day_str = day.strftime("%Y-%m-%d")
                if day_str not in daily_results:
                    daily_results[day_str] = {"premarket": [], "market": []}

                # Get data for this day
                day_data = data[data.index.date == day.date()]

                if len(day_data) == 0:
                    continue

                # Find opportunities
                premarket_ops = find_trading_opportunities(day_data, "premarket")
                market_ops = find_trading_opportunities(day_data, "market")

                # Add to results
                for op in premarket_ops:
                    daily_results[day_str]["premarket"].append({"symbol": symbol, **op})

                for op in market_ops:
                    daily_results[day_str]["market"].append({"symbol": symbol, **op})

        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    # Print results
    for day, periods in daily_results.items():
        print(f"\nResults for {day}")

        print("\nTop Premarket Opportunities:")
        premarket_sorted = sorted(
            periods["premarket"], key=lambda x: x["gain_percentage"], reverse=True
        )[: args.top]
        for op in premarket_sorted:
            print(
                f"{op['symbol']}: {op['gain_percentage']:.2f}% gain at {op['timestamp']}"
            )

        print("\nTop Market Hours Opportunities:")
        market_sorted = sorted(
            periods["market"], key=lambda x: x["gain_percentage"], reverse=True
        )[: args.top]
        for op in market_sorted:
            print(
                f"{op['symbol']}: {op['gain_percentage']:.2f}% gain at {op['timestamp']}"
            )


if __name__ == "__main__":
    main()
