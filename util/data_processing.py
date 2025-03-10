import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

MODEL_DIR = 'models'

def get_historical_data(company):
    # Giả sử bạn có dữ liệu lịch sử cho mỗi công ty
    data_path = f'data/{company}_historical_data.csv'
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        return data
    else:
        return None

def prepare_input_data(company, model_type):
    data = get_historical_data(company)
    if data is None:
        return None
    # Chọn các tính năng cần thiết
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    # Tải scaler nếu cần thiết
    if model_type in ['lstm', 'gru']:
        scaler_path = os.path.join(MODEL_DIR, f'{company}_{model_type}_scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            features = scaler.transform(features)
        else:
            return None
        # Tạo dữ liệu với time_steps
        time_steps = 30  # Ví dụ
        input_data = []
        input_data.append(features[-time_steps:])
        input_data = np.array(input_data)
        return input_data
    else:
        # Với mô hình học máy truyền thống
        input_data = features[-1]
        return input_data

def get_future_dates(num_dates):
    last_date = datetime.now()
    dates = [last_date + timedelta(days=i) for i in range(1, num_dates+1)]
    dates = [date.strftime('%Y-%m-%d') for date in dates]
    return dates
