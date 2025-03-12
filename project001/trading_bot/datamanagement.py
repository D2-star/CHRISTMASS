import pandas as pd
import os
import logging
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_historical_data(file_path: str) -> pd.DataFrame:
    """
    Load historical data from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing historical data.

    Returns:
        pd.DataFrame: The loaded historical data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return pd.DataFrame()

def create_placeholder_csv(file_path):
    """Create a placeholder CSV file with headers."""
    headers = ['date', 'open', 'high', 'low', 'close', 'volume']
    pd.DataFrame(columns=headers).to_csv(file_path, index=False)
    logging.info(f"Placeholder CSV file created at: {file_path}")

def save_historical_data(data, file_path):
    """Save historical market data to a CSV file."""
    try:
        data = data.drop_duplicates()  # Remove duplicate rows
        data = data.dropna()  # Drop rows with any missing values
        data.reset_index(drop=True, inplace=True)
        data.to_csv(file_path, index=False)  # Save without index
        logging.info(f"Historical data saved successfully to {file_path}.")
    except Exception as e:
        logging.error(f"Failed to save historical data: {e}")

def fetch_market_data(symbol, start_date, end_date):
    """Fetch historical market data for a given symbol."""
    try:
        logging.info(f"Fetching market data for {symbol} from {start_date} to {end_date}.")
        data = yf.download(symbol, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        logging.info("Market data fetched successfully.")
        return data
    except Exception as e:
        logging.error(f"Failed to fetch market data: {e}")
        return None

def load_symbols(file_path: str) -> list:
    """
    Load symbols from a configuration file.

    Args:
        file_path (str): The path to the configuration file containing symbols.

    Returns:
        list: A list of symbols.
    """
    try:
        with open(file_path, 'r') as file:
            symbols = file.read().splitlines()
        logging.info(f"Loaded symbols from {file_path}: {symbols}")
        return symbols
    except Exception as e:
        logging.error(f"Error loading symbols from {file_path}: {e}")
        return []

if __name__ == "__main__":
    symbols_file_path = '../symbols.txt'  # Adjust path as needed
    symbols = load_symbols(symbols_file_path)

    for symbol in symbols:
        historical_file_path = f'{symbol}_historical.csv'  # Adjust path as needed
        historical_data = load_historical_data(historical_file_path)

        if historical_data is None or historical_data.empty:
            start_date = '2025-01-01'
            end_date = '2025-12-31'
            
            new_data = fetch_market_data(symbol, start_date, end_date)
            
            if new_data is not None:
                save_historical_data(new_data, historical_file_path)
        else:
            logging.info(f"Historical data for {symbol} is already available.")
