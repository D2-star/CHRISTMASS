from flask import Flask, render_template, jsonify, request
from threading import Thread
import main as trading_bot  # Ensure main.py is in the same directory
import time
import webbrowser
import threading

app = Flask(__name__)

# Global variables to track bot status and logs
bot_thread = None
bot_running = False
console_logs = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_bot():
    global bot_thread, bot_running
    console_logs.clear()  # Clear previous logs when starting the bot

    if not bot_running:
        bot_running = True
        bot_thread = Thread(target=run_bot)
        bot_thread.start()
        return jsonify({'status': 'Bot started', 'logs': console_logs})

    return jsonify({'status': 'Bot is already running'})

def run_bot():
    try:
        trading_bot.main()  # Call main() without arguments
    except Exception as e:
        console_logs.append(f"Error in bot: {e}")

@app.route('/stop')
def stop_bot():
    global bot_thread, bot_running
    if bot_thread is not None:
        bot_thread.join()  # Wait for the bot thread to finish

    if bot_running:
        bot_running = False
        console_logs.append("Bot stopping...")
        return jsonify({'status': 'Bot stopping...', 'logs': console_logs})

    return jsonify({'status': 'Bot is not running'})

@app.route('/console_output')
def console_output():
    return jsonify(console_logs)

@app.route('/latest_summary')
def latest_summary():
    try:
        symbols = ["AAPL", "GOOGL", "MSFT"]  # Replace with your actual symbols
        summary_data = trading_bot.analyze_and_trade(symbols)  # Execute the function
        return jsonify({'status': 'success', 'data': summary_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        # Save settings
        settings_data = request.json
        # Implement logic to save settings
        return jsonify({'status': 'Settings saved'})
    else:
        # Load settings
        # Implement logic to load settings
        settings_data = {}
        return jsonify(settings_data)

@app.route('/performance_metrics')
def performance_metrics():
    try:
        # Implement logic to fetch performance metrics
        metrics_data = {
            'current_balance': trading_bot.get_current_balance(),
            'current_profit_loss': trading_bot.get_current_profit_loss(),
            'total_opened_trades': len(trading_bot.executed_trades)
        }
        return jsonify({'status': 'success', 'data': metrics_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()  # Delay to ensure the server is up
    app.run(host='0.0.0.0', port=5000, debug=True)  # Use host='0.0.0.0' for PythonAnywhere
