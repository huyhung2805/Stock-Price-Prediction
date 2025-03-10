# Stock Price Prediction

A web-based application for predicting stock prices using various machine learning models including LSTM, GRU, BiLSTM, TCN, Transformer, XGBoost, and Random Forest.

## Features

- Real-time stock market overview
- Company search with detailed information
- Historical price data visualization
- Multiple prediction models support
- Interactive price charts
- Prediction statistics and recommendations
- Responsive design for all devices

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Libraries**:
  - TensorFlow/Keras for deep learning models
  - XGBoost and Scikit-learn for traditional ML models
  - yfinance for stock data
  - Chart.js for data visualization
  - Select2 for enhanced dropdowns
  - Bootstrap for UI components

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd stock-price-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

1. Make sure you have all required model files in the `Models` directory
2. Configure your API keys if needed (for additional data sources)

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
stock-price-prediction/
├── app.py              # Main Flask application
├── static/
│   ├── css/           # CSS styles
│   ├── js/            # JavaScript files
│   └── img/           # Images and icons
├── templates/         # HTML templates
├── Models/           # Trained ML models
└── requirements.txt  # Python dependencies
```

## Usage

1. Search for a company using the search bar
2. View company information and historical data
3. Select a prediction model and timeframe
4. Generate and view price predictions
5. Analyze prediction statistics and recommendations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- yfinance for providing stock market data
- TensorFlow and scikit-learn communities
- Chart.js for visualization capabilities 