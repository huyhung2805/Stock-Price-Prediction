import yfinance as yf
import pandas as pd

# Mã cổ phiếu bạn quan tâm
ticker = 'SONY'

# Khoảng thời gian bạn muốn lấy dữ liệu
start_date = '2015-01-01'
end_date = '2025-03-03'

# Tải dữ liệu từ Yahoo! Finance
data = yf.download(ticker, start=start_date, end=end_date)

# Chọn và sắp xếp lại các cột
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Đặt lại chỉ số để đưa cột 'Date' vào dữ liệu
data.reset_index(inplace=True)
data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Định dạng cột 'Date' theo 'Mar 3, 2025'
# Lưu ý: Sử dụng '%-d' trên hệ thống Unix/Mac và '%#d' trên Windows để loại bỏ số 0 ở đầu ngày
import sys
if sys.platform.startswith('win'):
    data['Date'] = data['Date'].dt.strftime('%b %#d, %Y')
else:
    data['Date'] = data['Date'].dt.strftime('%b %-d, %Y')

# Định dạng các cột số với dấu phân cách hàng nghìn và 2 chữ số thập phân
for col in ['Open', 'High', 'Low', 'Close']:
    data[col] = data[col].apply(lambda x: f"{x:,.2f}")

# Định dạng cột 'Volume' với dấu phân cách hàng nghìn
data['Volume'] = data['Volume'].apply(lambda x: f"{x:,}")

# Lưu dữ liệu vào file CSV với tùy chỉnh
file_name = f"{ticker}_historical_data.csv"
data.to_csv(file_name, sep=';', index=False, encoding='utf-8-sig')
