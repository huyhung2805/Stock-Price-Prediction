import yfinance as yf
import pandas as pd

# Mã cổ phiếu bạn quan tâm
ticker = 'NVDA'

# Khoảng thời gian bạn muốn lấy dữ liệu
start_date = '2015-01-01'
end_date = '2025-03-03'

# Tải dữ liệu từ Yahoo! Finance
data = yf.download(ticker, start=start_date, end=end_date)

# Chọn và sắp xếp lại các cột
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Đặt lại chỉ số để đưa cột 'Date' vào dữ liệu
data.reset_index(inplace=True)
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# Lưu dữ liệu vào file CSV mà không định dạng số và ngày tháng
file_name = f'{ticker}_historical_data.csv'
data.to_csv(file_name, index=False)
