import unittest
import pandas as pd
import numpy as np
from data_processing import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.data_dir = 'Data'
        self.symbol = 'AAPL'
        self.processor = DataProcessor(self.data_dir, self.symbol)

    def test_load_data(self):
        data = self.processor.load_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertFalse(data.isnull().any().any())

    def test_preprocess_data(self):
        data = self.processor.load_data()
        scaled_data, scaler = self.processor.preprocess_data(data)
        self.assertIsNotNone(scaled_data)
        self.assertIsNotNone(scaler)
        self.assertEqual(scaled_data.shape[1], len(self.processor.feature_columns))

    def test_create_sequences(self):
        data = self.processor.load_data()
        scaled_data, _ = self.processor.preprocess_data(data)
        X, y = self.processor.create_sequences(scaled_data, 20, 0)
        self.assertEqual(X.shape[1], 20)  # time_step
        self.assertEqual(X.shape[2], scaled_data.shape[1])  # features