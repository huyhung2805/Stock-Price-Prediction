import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from flask_caching import Cache
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Cấu hình cache
cache = Cache(app, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'Models')
ALLOWED_MODELS = ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer', 'XGBoost', 'RF']
SEQUENCE_LENGTH = 20
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

# Danh sách các thị trường chứng khoán
STOCK_MARKETS = {
    'US': ['^GSPC', '^DJI', '^IXIC'],  # S&P 500, Dow Jones, NASDAQ
    'VN': ['^VNINDEX', 'VN30.VN'],      # Vietnam Index
}

print(f"Current working directory: {os.getcwd()}")
print(f"MODEL_DIR path: {MODEL_DIR}")
print(f"MODEL_DIR exists: {os.path.exists(MODEL_DIR)}")

class ModelPredictor:
    def __init__(self):
        self.scalers = {}
        self.models = {}
        
    def load_model(self, symbol, model_type):
        """Load model và scaler cho một symbol và model type cụ thể"""
        model_key = f"{symbol}_{model_type}"
        
        if model_key not in self.models:
            try:
                if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
                    model_path = os.path.join(MODEL_DIR, f'{symbol}_{model_type}_model.keras')
                    self.models[model_key] = load_model(model_path)
                else:
                    model_path = os.path.join(MODEL_DIR, f'{symbol}_{model_type}_model.pkl')
                    self.models[model_key] = joblib.load(model_path)
                
                # Load scaler nếu tồn tại
                scaler_path = os.path.join(MODEL_DIR, f'{symbol}_scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scalers[symbol] = joblib.load(scaler_path)
                else:
                    logger.warning(f"No scaler found for {symbol}, creating new one")
                    self.scalers[symbol] = MinMaxScaler()
                    
            except Exception as e:
                logger.error(f"Error loading model for {symbol} {model_type}: {e}")
                return None
                
        return self.models[model_key]

    def prepare_data(self, data, symbol, model_type):
        """Chuẩn bị dữ liệu cho prediction"""
        try:
            # Đảm bảo data có đủ các cột cần thiết
            required_columns = FEATURE_COLUMNS
            if not all(col in data.columns for col in required_columns):
                raise ValueError("Missing required columns in data")

            # Tạo features
            features = data[required_columns].values
            
            # Scale features
            if symbol not in self.scalers:
                logger.info(f"Creating new scaler for {symbol}")
                self.scalers[symbol] = MinMaxScaler()
                self.scalers[symbol].fit(features)
            elif not hasattr(self.scalers[symbol], 'n_features_in_'):
                logger.info(f"Fitting scaler for {symbol}")
                self.scalers[symbol].fit(features)
                
            features_scaled = self.scalers[symbol].transform(features)

            # Tạo sequences cho deep learning models
            if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
                # Tạo sequences
                sequences = []
                for i in range(len(features_scaled) - SEQUENCE_LENGTH + 1):
                    sequences.append(features_scaled[i:(i + SEQUENCE_LENGTH)])
                
                # Thêm các features phụ trợ nếu cần
                sequences = np.array(sequences)
                if sequences.shape[2] < 58:  # Nếu cần 58 features
                    padding = np.zeros((sequences.shape[0], sequences.shape[1], 58 - sequences.shape[2]))
                    sequences = np.concatenate([sequences, padding], axis=2)
                return sequences
            else:
                # Cho các mô hình ML truyền thống (Random Forest, XGBoost)
                # Flatten sequence thành vector
                sequence = features_scaled[-SEQUENCE_LENGTH:].flatten()
                # Pad nếu cần
                if len(sequence) < 1160:  # Nếu cần 1160 features
                    padding = np.zeros(1160 - len(sequence))
                    sequence = np.concatenate([sequence, padding])
                return sequence.reshape(1, -1)

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

predictor = ModelPredictor()

def get_company_info(symbol):
    """Lấy thông tin công ty từ yfinance và kiểm tra trạng thái training"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Kiểm tra trạng thái training
        is_trained = False
        trained_models = []
        
        # Kiểm tra thư mục Models có tồn tại không
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        # Kiểm tra các model đã train
        for model_type in ALLOWED_MODELS:
            model_path = ''
            if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
                model_path = os.path.join(MODEL_DIR, f'{symbol}_{model_type}_model.keras')
            else:
                model_path = os.path.join(MODEL_DIR, f'{symbol}_{model_type}_model.pkl')
            
            # Log để debug
            logger.info(f"Checking model path: {model_path}")
            logger.info(f"Model exists: {os.path.exists(model_path)}")
            
            if os.path.exists(model_path):
                is_trained = True
                trained_models.append(model_type)

        company_info = {
            "symbol": symbol,
            "name": info.get('longName', symbol),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "country": info.get('country', 'N/A'),
            "is_trained": is_trained,
            "trained_models": trained_models,
            "market_cap": info.get('marketCap', 'N/A'),
            "currency": info.get('currency', 'USD'),
            "current_price": info.get('regularMarketPrice', 'N/A'),
            "price_change": info.get('regularMarketChangePercent', 0)
        }
        
        # Log thông tin công ty để debug
        logger.info(f"Company info for {symbol}: {company_info}")
        
        return company_info
        
    except Exception as e:
        logger.error(f"Error getting info for {symbol}: {e}")
        return None

@app.route('/')
def index():
    """Render trang chủ"""
    return render_template('index.html')

@app.route('/api/search_companies')
@cache.cached(timeout=3600, query_string=True)
def search_companies():
    """Tìm kiếm công ty với yfinance và kiểm tra trạng thái training"""
    try:
        query = request.args.get('query', '').strip().upper()
        if not query or len(query) < 2:
            return jsonify([])

        # Tìm kiếm công ty với yfinance
        search_results = []
        
        def process_symbol(symbol):
            try:
                company_info = get_company_info(symbol)
                if company_info:
                    search_results.append(company_info)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Tìm kiếm song song
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Tìm kiếm trực tiếp với query
            executor.submit(process_symbol, query)
            
            # Tìm kiếm các công ty liên quan
            try:
                ticker = yf.Ticker(query)
                info = ticker.info
                if info:
                    search_results.append({
                        "symbol": query,
                        "name": info.get('longName', query),
                        "sector": info.get('sector', 'N/A'),
                        "industry": info.get('industry', 'N/A'),
                        "country": info.get('country', 'N/A'),
                        "is_trained": False,  # Mặc định là chưa train
                        "trained_models": [],
                        "market_cap": info.get('marketCap', 'N/A'),
                        "currency": info.get('currency', 'USD'),
                        "current_price": info.get('regularMarketPrice', 'N/A'),
                        "price_change": info.get('regularMarketChangePercent', 0)
                    })
            except Exception as e:
                logger.error(f"Error searching for {query}: {e}")

        # Lọc kết quả trùng lặp và sắp xếp
        unique_results = []
        seen_symbols = set()
        for company in search_results:
            if company and company['symbol'] not in seen_symbols:
                seen_symbols.add(company['symbol'])
                unique_results.append(company)

        # Sắp xếp kết quả
        unique_results.sort(key=lambda x: (
            not x['is_trained'],  # Đã train lên đầu
            x['symbol'] != query,  # Kết quả khớp chính xác lên đầu
            x['symbol']  # Sắp xếp theo alphabet
        ))

        return jsonify(unique_results[:10])  # Giới hạn 10 kết quả

    except Exception as e:
        logger.error(f"Error in search_companies: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/market_overview')
@cache.cached(timeout=300)
def market_overview():
    """Lấy tổng quan về thị trường và danh sách công ty đã train"""
    try:
        # Lấy danh sách công ty đã train
        trained_companies = []
        for file in os.listdir(MODEL_DIR):
            if file.endswith('_model.keras') or file.endswith('_model.pkl'):
                symbol = file.split('_')[0]
                if symbol not in trained_companies:
                    trained_companies.append(symbol)

        # Lấy thông tin thị trường
        market_data = {}
        for market, indices in STOCK_MARKETS.items():
            market_data[market] = []
            for index in indices:
                try:
                    ticker = yf.Ticker(index)
                    hist = ticker.history(period='1d')
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        prev_close = ticker.info.get('previousClose', current)
                        change = ((current - prev_close) / prev_close) * 100
                        
                        market_data[market].append({
                            'symbol': index,
                            'name': ticker.info.get('shortName', index),
                            'price': current,
                            'change': change
                        })
                except Exception as e:
                    logger.error(f"Error getting market data for {index}: {e}")

        # Lấy thông tin chi tiết về các công ty đã train
        trained_companies_info = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(get_company_info, symbol): symbol 
                for symbol in trained_companies
            }
            
            for future in as_completed(future_to_symbol):
                try:
                    result = future.result()
                    if result:
                        trained_companies_info.append(result)
                except Exception as e:
                    logger.error(f"Error getting trained company info: {e}")

        return jsonify({
            'market_indices': market_data,
            'trained_companies': trained_companies_info
        })

    except Exception as e:
        logger.error(f"Error in market_overview: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/check_company_status/<symbol>')
def check_company_status(symbol):
    """Kiểm tra trạng thái training của công ty"""
    try:
        company_info = get_company_info(symbol)
        if company_info is None:
            return jsonify({"error": "Company not found"}), 404
        return jsonify(company_info)

    except Exception as e:
        logger.error(f"Error checking company status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/historical/<symbol>')
def historical_data(symbol):
    """Lấy dữ liệu lịch sử"""
    try:
        days = int(request.args.get('days', 0))
        months = int(request.args.get('months', 1))
        years = int(request.args.get('years', 0))
        
        total_days = days + (months * 30) + (years * 365)
        period = f"{total_days}d"

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            return jsonify({"error": "No historical data available"}), 404

        return jsonify({
            'dates': hist.index.strftime('%Y-%m-%d').tolist(),
            'prices': hist['Close'].tolist(),
            'volumes': hist['Volume'].tolist(),
            'high': hist['High'].tolist(),
            'low': hist['Low'].tolist()
        })

    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Tạo dự đoán giá"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        symbol = data.get('company')
        model_type = data.get('model_type')
        days = int(data.get('days', 0))
        months = int(data.get('months', 0))
        years = int(data.get('years', 0))

        if not symbol or not model_type:
            return jsonify({"error": "Missing company or model type"}), 400

        # Load model
        model = predictor.load_model(symbol, model_type)
        if model is None:
            return jsonify({"error": f"No trained {model_type} model found for {symbol}"}), 404

        # Lấy dữ liệu lịch sử
        try:
            ticker = yf.Ticker(symbol)
            # Lấy dữ liệu từ 60 ngày trước đến hiện tại
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            hist_data = ticker.history(start=start_date, end=end_date)
            
            if hist_data.empty:
                return jsonify({"error": "No historical data available"}), 400

            # Lưu lại giá thực tế gần nhất, bao gồm cả ngày hiện tại nếu có
            historical_prices = {
                'dates': hist_data.index.strftime('%Y-%m-%d').tolist(),
                'values': hist_data['Close'].tolist()
            }

            logger.info(f"Historical data from {start_date} to {end_date}")
            logger.info(f"Last historical date: {hist_data.index[-1]}")

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return jsonify({"error": "Failed to fetch historical data"}), 500

        # Chuẩn bị input data
        try:
            input_data = predictor.prepare_data(hist_data, symbol, model_type)
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return jsonify({"error": "Failed to prepare prediction data"}), 500
        
        # Tính số ngày cần dự đoán
        total_days = days + (months * 30) + (years * 365)
        
        # Tạo dự đoán
        predictions = []
        scaled_predictions = []  # Lưu giá trị scaled để cập nhật sequence
        dates = []
        current_date = datetime.now()
        
        if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
            last_sequence = input_data[-1:]
        else:
            last_sequence = input_data

        # Tạo mock data cho việc inverse transform
        mock_data = np.zeros((1, len(FEATURE_COLUMNS)))
        
        for i in range(total_days):
            if current_date.weekday() < 5:  # Chỉ dự đoán cho ngày trong tuần
                try:
                    if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
                        pred = model.predict(last_sequence, verbose=0)
                    else:
                        pred = model.predict(last_sequence)
                    
                    # Lưu giá trị scaled
                    scaled_pred = float(pred[0][0]) if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer'] else float(pred[0])
                    scaled_predictions.append(scaled_pred)
                    
                    # Inverse transform để lấy giá trị thực
                    mock_data[0, 3] = scaled_pred  # Giả sử cột 3 là Close price
                    real_pred = predictor.scalers[symbol].inverse_transform(mock_data)[0, 3]
                    
                    predictions.append(float(real_pred))
                    dates.append(current_date.strftime('%Y-%m-%d'))
                    
                    # Cập nhật sequence cho lần dự đoán tiếp theo
                    if model_type in ['LSTM', 'GRU', 'BiLSTM', 'TCN', 'Transformer']:
                        last_sequence = np.roll(last_sequence, -1, axis=1)
                        last_sequence[0, -1, 3] = scaled_pred  # Cập nhật Close price
                        # Thêm nhiễu ngẫu nhiên cho các features khác để tạo dự đoán động hơn
                        noise = np.random.normal(0, 0.01, last_sequence.shape[2])
                        last_sequence[0, -1, :] += noise
                    else:
                        # Cập nhật vector đầu vào cho mô hình ML truyền thống
                        last_sequence = np.roll(last_sequence, -5)  # Roll 5 features
                        last_sequence[0, -5:] = scaled_pred  # Cập nhật Close price
                        # Thêm nhiễu ngẫu nhiên
                        noise = np.random.normal(0, 0.01, 5)
                        last_sequence[0, -5:] += noise
                    
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    return jsonify({"error": "Prediction failed"}), 500
                
            current_date += timedelta(days=1)

        if not predictions:
            return jsonify({"error": "No predictions generated"}), 500

        # Tính toán thống kê
        current_price = hist_data['Close'].iloc[-1]
        avg_pred = np.mean(predictions)
        price_change = ((avg_pred - current_price) / current_price) * 100

        return jsonify({
            'historical': historical_prices,
            'predictions': {
                'dates': dates,
                'values': predictions
            },
            'statistics': {
                'current_price': float(current_price),
                'average_prediction': float(avg_pred),
                'price_change_percent': float(price_change),
                'min_prediction': float(min(predictions)),
                'max_prediction': float(max(predictions))
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)