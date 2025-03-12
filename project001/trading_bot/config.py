import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MetaTrader 5 configuration
MT5_CONFIG = {
    "username": "54399887",
    "password": "14513639@Davie",
    "server": "HFMarketsKE-Live2",
    "mt5Pathway": "C:\\Program Files\\HFM Metatrader 5\\terminal64.exe"
}



# File paths
HISTORICAL_DATA_FILE = os.path.join(BASE_DIR, 'data', 'historical.csv')  # Path to historical data CSV
TRADES_FILE = os.path.join(BASE_DIR, 'data', 'trades.csv')  # Path to trades CSV
LOGGING_FILE = os.path.join(BASE_DIR, 'logs', 'trading_bot.log')  # Log file path

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # Log level
    'format': '%(asctime)s - %(levelname)s - %(message)s',  # Log message format
}

# Trading settings
TRADING_SETTINGS = {
    'initial_capital': 10000,  # Initial capital for backtesting
    'risk_management': {
        'max_drawdown': 0.1,  # Maximum drawdown as a percentage
        'position_size': 0.05,  # Percentage of capital to risk per trade
        'lot_size': 0.04,  # Desired lot size updated to 0.04
    },
}

# Other constants
TRADING_SYMBOL = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", 
    "USDCHF", "NZDUSD", "USDCAD", "USDHKD"
]  # Default trading symbol (can be changed as needed)
TIMEFRAME = '1d'  # Default timeframe for historical data (e.g., daily)

# Debug print to confirm loading
print("Loading config.py")
print("Loading successful.....config.py")
