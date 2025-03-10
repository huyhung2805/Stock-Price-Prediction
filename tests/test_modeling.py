import unittest
import numpy as np
from modeling import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.model_dir = 'Models'
        self.time_step = 20
        self.num_features = 58
        self.trainer = ModelTrainer(self.model_dir, self.time_step, self.num_features)

    def test_build_lstm_model(self):
        input_shape = (self.time_step, self.num_features)
        model = self.trainer.build_advanced_lstm_model(
            input_shape, 
            units=64, 
            dropout_rate=0.2,
            optimizer='adam'
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape[1:], input_shape)
        self.assertEqual(model.output_shape[1], 1)