import logging
import requests
import time
import pandas as pd
import MetaTrader5 as mt5
from tabulate import tabulate
from datetime import datetime, time as dt_time
import pytz
import re
import os
# Tracker for last decisions
last_decision = {}
# Initialize global tracker for open trades
open_trades_by_symbol = {}

# Global list to store executed trade details
executed_trades = []

# List of major currency pairs
MAJOR_PAIRS = [
    "EURUSD.Z",
    "USDJPY.Z",
    "GBPUSD.Z",
    "AUDUSD.Z",
    "USDCHF.Z",
    "NZDUSD.Z",
    "USDCAD.Z",
    "XAUUSD",
    "GBPJPY.Z"
]
# List of timeframes
TIMEFRAMES = [
    mt5.TIMEFRAME_M1,
    mt5.TIMEFRAME_M3,
    mt5.TIMEFRAME_M5,
    mt5.TIMEFRAME_M15
]

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        filename="trading_bot.log",
        filemode="a",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging is set up.")

def log_trade(action, symbol, volume, price):
    """Log trade execution details."""
    logging.info(f"Executed {action} trade: {volume} of {symbol} at {price:.2f}")

def check_connection():
    """Check if MetaTrader 5 is connected."""
    if not mt5.initialize():
        logging.error("Failed to initialize MetaTrader 5. Please check the connection.")
        return False
    logging.info("MetaTrader 5 initialized successfully.")
    
    # Log the server being connected to
    server_info = mt5.version()
    logging.info(f"Connected to server: {server_info}")
    return True

def shutdown_mt5():
    """Shutdown the MetaTrader 5 connection."""
    mt5.shutdown()
    logging.info("MetaTrader 5 connection closed.")

def get_latest_data(symbol):
    """Fetch latest market data for the given symbol."""
    data_frames = []
    
    for timeframe in TIMEFRAMES:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
        if rates is None or len(rates) == 0:
            logging.warning(f"Failed to fetch market data for {symbol} at timeframe {timeframe}. No rates returned.")
            continue

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['timeframe'] = timeframe  # Add the timeframe column
        data_frames.append(df)

    if not data_frames:
        return pd.DataFrame()  # Return empty DataFrame if no data was fetched

    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data

def get_market_status():
    """Determine if the market is open and which session it is in."""
    utc_now = datetime.now(pytz.utc)

    sessions = {
        "Sydney": (dt_time(22, 0), dt_time(6, 59)),
        "Tokyo": (dt_time(0, 0), dt_time(8, 59)),
        "London": (dt_time(8, 0), dt_time(16, 59)),
        "New York": (dt_time(13, 0), dt_time(21, 59))
    }

    for session, (start, end) in sessions.items():
        if start <= utc_now.time() < end:
            return f"Market is open. Current session: {session}"

    return "Market is closed."

def calculate_signals(data, symbol):
    """Calculate trading signals based on the latest data."""
    from indicators import indicator1, indicator2, indicator3, indicator4, indicator5, indicator6, indicator_30
    from newindicators import indicator8_macd, indicator9_adx, indicator10_bollinger_bands, indicator11_stochastic, indicator13_donchian
    from newindicators3 import indicator17_cci, indicator18_rsi

    signals = {
        "Indicator 1": indicator1(data).iloc[-1],
        "Indicator 2": indicator2(data).iloc[-1],
        "Indicator 3": indicator3(data).iloc[-1],
        "Indicator 4": indicator4(data).iloc[-1],
        "Indicator 5": indicator5(data).iloc[-1],
        "Indicator 6": indicator6(data, ma_lengths=[(5, 7), (5, 34), (7, 11), (11, 15)], ma_type="EMA"),
        "Indicator 8": indicator8_macd(data),
        "Indicator 10 Bollinger Bands": indicator10_bollinger_bands(data),
        "Indicator 11 Stochastic": indicator11_stochastic(data),
        "Indicator 30 ZeroLagAlgoAlpha": indicator_30(data).iloc[-1]  # Ensure it returns a single value
    }

    table_data = []
    total_buys = 0
    total_sells = 0

    # Collecting the latest signals and counts
    for key, signal in signals.items():
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(signal, str):
            if "buys" in signal and "sells" in signal:
                table_data.append((timestamp, key, signal))
                buys, sells = map(int, re.findall(r'(\d+)', signal))
                total_buys += buys
                total_sells += sells
            elif signal == "buy":
                total_buys += 1
                table_data.append((timestamp, key, "buy"))
            elif signal == "sell":
                total_sells += 1
                table_data.append((timestamp, key, "sell"))
            else:  # "hold" or any other status
                table_data.append((timestamp, key, signal))

    return table_data, total_buys, total_sells

def get_current_balance():
    """Fetch current balance from the trading account."""
    account_info = mt5.account_info()
    if account_info is not None:
        return account_info.balance
    return 0.0

def get_current_profit_loss():
    """Calculate current profit or loss from open trades."""
    positions = mt5.positions_get()
    total_profit = sum(pos.profit for pos in positions) if positions else 0.0
    return total_profit

def fetch_news():
    """Fetch relevant financial news using an external API.""" 
    url = "https://www.forexfactory.com/"
    params = {
        "apiKey": os.getenv("NEWS_API_KEY")  # Use environment variable for API key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        articles = response.json().get('articles', [])
        return {article['title']: article['description'] for article in articles}
    except requests.RequestException as e:
        logging.warning(f"Failed to fetch news: {e}")
        return {}

# Opening trade
def open_trade(symbol, decision, volume, current_tick):
    """Open a trade for the given symbol and decision."""
    try:
        price = current_tick.ask if decision == "buy" else current_tick.bid
        order_type = mt5.ORDER_TYPE_BUY if decision == "buy" else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": f"Opening {decision} trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Successfully opened {decision} trade for {symbol}.")
            # Set Stop Loss and Take Profit
            sl, tp = set_sl_tp(symbol, price, decision)
            set_sl_tp_order(symbol, result.order, sl, tp)
        else:
            print(f"Failed to open {decision} trade for {symbol}, Retcode={result.retcode}")
    except Exception as e:
        print(f"Error opening trade for {symbol}: {e}")


def set_sl_tp(symbol, price, decision):
    """Set Stop Loss and Take Profit based on the symbol and decision."""
    try:
        # Define stop loss distance based on the symbol
        sl_distances = {
            "XAUUSD": 6, # 60 pips for XAUUSD 5.99
            "GBPJPY.Z": 0.500, # 43 pips for GBPJPY 3.33
            "USDJPY.Z": 0.430,
            "AUDUSD.Z": 0.00140,
            "NZDUSD.Z": 0.00130, # 13 pips for NZDUSD 1.30
            "USDCAD.Z": 0.00284,# 28 pips for USDCAD 1..86
            "USDCHF.Z": 0.00180, # 18 pips for USDCHF  2.01
            "GBPUSD.Z": 0.00271,# 25 pips for GBPUSD 2.71
            "EURUSD.Z": 0.00250
        }
        
        # Default stop loss distance for other symbols
        sl_distance = sl_distances.get(symbol, 0.0010)
        
        # Calculate Stop Loss
        sl = price - sl_distance if decision == "buy" else price + sl_distance
        
        # Set Take Profit as three times the Stop Loss distance
        tp = price + (3 * abs(price - sl)) if decision == "buy" else price - (3 * abs(sl - price))

        return sl, tp

    except Exception as e:
        logging.error(f"Error setting SL/TP for {symbol}: {e}")
        return None, None

def set_sl_tp_order(symbol, order_ticket, sl, tp):
    """Set Stop Loss and Take Profit for an existing order."""
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": order_ticket,
        "sl": sl,
        "tp": tp,
        "magic": 234000,
        "comment": "Setting SL and TP",
    }
    
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Successfully set SL={sl} and TP={tp} for order {order_ticket}.")
    else:
        print(f"Failed to set SL and TP for order {order_ticket}, Retcode={result.retcode}")

def count_open_trades(symbol):
    """Count the number of currently opened trades for a given symbol."""
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            print(f"No positions found for {symbol}.")
            return 0, 0
        
        total_buys = sum(1 for pos in positions if pos.type == mt5.ORDER_TYPE_BUY)
        total_sells = sum(1 for pos in positions if pos.type == mt5.ORDER_TYPE_SELL)
        
        print(f"Open positions for {symbol}: Buy={total_buys}, Sell={total_sells}")
        return total_buys, total_sells
    except Exception as e:
        print(f"Error counting open trades for {symbol}: {e}")
        return 0, 0

def show_open_positions(symbol):
    """Display all open positions for a given symbol."""
    try:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            print(f"No open positions for {symbol}.")
            return
        
        print(f"Open Positions for {symbol}:")
        for pos in positions:
            pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            print(f"  Symbol: {pos.symbol}, Type: {pos_type}, Ticket: {pos.ticket}, Volume: {pos.volume:.2f}, Profit: {pos.profit:.2f}, SL: {pos.sl:.5f}, TP: {pos.tp:.5f}")
    except Exception as e:
        print(f"Error displaying positions for {symbol}: {e}")

def execute_trade(symbol, decision):
    """Execute a trade based on the decision for a specific symbol."""
    global last_decision

    max_trades = 5
    volume = 0.3
    current_tick = mt5.symbol_info_tick(symbol)
    if not current_tick:
        print(f"Failed to retrieve tick data for {symbol}.")
        return

    # Get current open positions
    open_buys, open_sells = count_open_trades(symbol)

    # Check for decision change and handle conflicting positions
    if decision == "buy":
        if open_sells > 0:
            print(f"Closing all SELL positions for {symbol} before opening a new trade.")
            close_all_open_positions(symbol, "sell")  # Close all SELL positions
        if open_buys < max_trades:
            open_trade(symbol, "buy", volume, current_tick)
            last_decision[symbol] = "buy"
    elif decision == "sell":
        if open_buys > 0:
            print(f"Closing all BUY positions for {symbol} before opening a new trade.")
            close_all_open_positions(symbol, "buy")  # Close all BUY positions
        if open_sells < max_trades:
            open_trade(symbol, "sell", volume, current_tick)
            last_decision[symbol] = "sell"

    # Recheck for any residual conflicts
    open_buys, open_sells = count_open_trades(symbol)
    if open_buys > 0 and open_sells > 0:
        print(f"Residual conflict detected for {symbol}: Buy={open_buys}, Sell={open_sells}. Manual intervention may be required.")

# Closing Trades
def close_position(position, retries=3):
    """Close a specific position with retry logic, returning closing price and profit/loss."""
    tick = mt5.symbol_info_tick(position.symbol)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
        "price": tick.ask if position.type == 1 else tick.bid,
        "deviation": 50,
        "magic": 100,
        "comment": "python script close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    for attempt in range(retries):
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            closing_price = tick.ask if position.type == 1 else tick.bid
            profit_loss = position.profit
            print(f"Closed position for {position.symbol}: {position.ticket} at {closing_price} with P/L: {profit_loss}")
            return closing_price, profit_loss
        
        print(f"Failed to close position {position.ticket} for {position.symbol}. Attempt {attempt + 1}/{retries}. Error: {result.retcode}")
        time.sleep(1)

    return None, None

def close_all_open_positions(symbol, position_type):
    """Close all open positions for a specific symbol and position type."""
    positions = mt5.positions_get(symbol=symbol)

    if not positions:
        print(f"No open positions for {symbol}.")
        return

    print(f"Closing all {position_type.upper()} positions for {symbol} before opening a new trade.")

    for position in positions:
        if (position_type == 'buy' and position.type == mt5.ORDER_TYPE_BUY) or (position_type == 'sell' and position.type == mt5.ORDER_TYPE_SELL):
            print(f"Closing position for {position.symbol}: {position.ticket}...")
            closing_price, profit_loss = close_position(position)
            if closing_price is not None:
                print(f"Closed position {position.ticket} at {closing_price} with P/L: {profit_loss:.2f}")

#Market open
def is_market_open(symbol):
    """Check if the market for a given symbol is open."""
    market_info = mt5.symbol_info(symbol)
    if market_info is None:
        logging.error(f"Failed to retrieve market info for {symbol}.")
        return False

    if not market_info.visible:
        logging.error(f"{symbol} is not visible for trading.")
        return False

    return True

def escape_markdown(text):
    """Escape special characters for Markdown V2."""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

def color_text(text, color):
    """Return colored text for terminal output."""
    colors = {
        'green': '\033[92m',       # Green
        'red': '\033[91m',         # Red
        'yellow': '\033[93m',      # Yellow
        'blue': '\033[94m',        # Blue
        'maroon': '\033[38;5;88m', # Maroon (approximation)
        'cyan': '\033[96m',        # Cyan
        'magenta': '\033[95m',     # Magenta
        'white': '\033[97m',       # White
        'black': '\033[90m',       # Black (dark gray)
        'light_green': '\033[92m', # Light Green
        'light_yellow': '\033[93m',# Light Yellow
        'light_blue': '\033[94m',  # Light Blue
        'light_cyan': '\033[96m',  # Light Cyan
        'light_magenta': '\033[95m', # Light Magenta
        'reset': '\033[0m'         # Reset to default
    }
    return f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"

def send_message_to_telegram(message):
    """Send a message to the Telegram bot."""
    TOKEN = "7578688531:AAG1Z-SdqYU76b-4c_IEmsiQrqqsSG_iPRA"  # Keep this secure
    CHAT_ID = "6985786951"  # Updated with the numeric chat ID
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": escape_markdown(message),
        "parse_mode": "MarkdownV2"
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        logging.info("Message sent successfully.")
    else:
        logging.error(f"Failed to send message to Telegram: {response.status_code} - {response.text}")
        print(response.json())  # Log the full response for debugging

def analyze_and_trade(symbols):
    """Analyze indicators for multiple symbols and execute trades based on their signals."""
    log_data = []
    
    for symbol in symbols:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.error(f"Failed to get tick data for {symbol}.")
            continue

        logging.info(f"Analyzing indicators for {symbol}...")
        
        data = get_latest_data(symbol)
        if data.empty:
            logging.warning(f"No valid data retrieved for {symbol}. Skipping...")
            continue

        # Get the table data and counts
        table_data, total_buys, total_sells = calculate_signals(data, symbol)

        # Print the signals in the specified table format
        if table_data:
            print(color_text(f"\nSignal Calculation Table for {symbol}", 'light_magenta'))
            print(tabulate(table_data, headers=["Timestamp", "Indicator", "Latest Signal"], tablefmt="grid"))

        print(color_text(f"\nTotal Buy Signals: {total_buys}", 'blue') + f", " + color_text(f"Total Sell Signals: {total_sells}", 'red'))

        zero_lag_signal = table_data[-1][2]  # Assuming it's the last row in the table_data
        decision = "hold"  # Default decision
        if total_buys > total_sells and zero_lag_signal == "buy":
            decision = "buy"
        elif total_sells > total_buys and zero_lag_signal == "sell":
            decision = "sell"
        
        decision_color = 'green' if decision == "buy" else 'red' if decision == "sell" else 'yellow'
        print(color_text(f"Final decision based on signals for {color_text(symbol, 'cyan')}: {decision}", decision_color))
        
        # Count opened positions
        open_buys, open_sells = count_open_trades(symbol)
        print(f"Opened Positions for {color_text(symbol, 'cyan')}: {open_buys} buys, {open_sells} sells")

        # Execute trade if there is a valid decision
        if decision != "hold":
            logging.info(f"Signal generated for {color_text(symbol, 'cyan')}: {decision}. Executing trade...")
            execute_trade(symbol, decision)
            log_data.append([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                symbol,
                decision,
                f"{tick.ask:.5f}" if decision == "buy" else f"{tick.bid:.5f}",
                f"{(tick.ask - tick.bid):.5f}"
            ])

    if log_data:
        # Prepare a formatted summary message for Telegram
        summary_message = "=== Latest Symbols Summary ===\n"
        header = f"{'Timestamp':<20} {'Symbol':<10} {'Action':<10} {'Current Price':<15} {'Spread':<10}\n"
        separator = "-" * len(header) + "\n"
        summary_message += header + separator

        for entry in log_data:
            summary_message += f"{entry[0]:<20} {entry[1]:<10} {entry[2]:<10} {entry[3]:<15} {entry[4]:<10}\n"

        print("\n=== Latest Symbols Summary ===")
        print(tabulate(log_data, headers=["Timestamp", "Symbol", "Action", "Current Price", "Spread"], tablefmt="grid"))

        # Send the formatted summary to Telegram
        send_message_to_telegram(summary_message)

def main():
    """Main function to run the trading bot."""
    setup_logging()
    logging.info("Starting trading bot...")
    print(color_text("Starting trading bot...", 'blue'))

    if not check_connection():
        logging.error("Initialization failed. Exiting...")
        print(color_text("Initialization failed. Exiting...", 'red'))
        return

    logging.info("MetaTrader 5 connected successfully.")
    print(color_text("MetaTrader 5 connected successfully.", 'green'))
    logging.info("Starting main trading loop...")
    print(color_text("Starting main trading loop...", 'green'))

    try:
        while True:
            market_status = get_market_status()
            logging.info(f"Market Status: {market_status}")
            print(color_text(f"Market Status: {market_status}", 'yellow'))

            if "closed" in market_status.lower():
                time.sleep(60)
                continue

            current_balance = get_current_balance()
            current_profit_loss = get_current_profit_loss()
            news_data = fetch_news()

            # Ensure current_balance and current_profit_loss are floats
            current_balance = float(current_balance)
            current_profit_loss = float(current_profit_loss)
            
            # Analyze and trade based on signals
            analyze_and_trade(MAJOR_PAIRS)

            # Calculate total opened trades
            total_opened_buys = sum(count_open_trades(symbol)[0] for symbol in MAJOR_PAIRS)  # Total buys
            total_opened_sells = sum(count_open_trades(symbol)[1] for symbol in MAJOR_PAIRS)  # Total sells

            # Print Total Summary Table
            print(color_text("\nTotal Summary Table", 'light_magenta'))
            print(tabulate([
                [color_text("Current Balance", 'cyan'), f"${current_balance:.2f}"],
                [color_text("Current Profit/Loss", 'cyan'), f"${current_profit_loss:.2f}"],
                [color_text("Total Opened Buys", 'green'), total_opened_buys],
                [color_text("Total Opened Sells", 'red'), total_opened_sells],
                [color_text("Current Market Session", 'cyan'), market_status.split(":")[-1].strip()],
            ] + [[color_text(f"{symbol}", 'cyan'), news] for symbol, news in news_data.items()], 
            headers=["Description", "Value"], tablefmt="grid"))

            time.sleep(10)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(color_text(f"An error occurred: {e}", 'red'))
    finally:
        logging.info("Shutting down the trading bot...")
        print(color_text("Shutting down the trading bot...", 'blue'))

if __name__ == "__main__":
    main()
