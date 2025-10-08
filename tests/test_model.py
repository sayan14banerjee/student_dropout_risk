# tests/test_model.py
import unittest
import numpy as np
import joblib
import pandas as pd

# Import or define the predict function
def predict(model, X):
    return model.predict(X)

class TestStudentDropoutModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the trained model
        cls.model = joblib.load("models/best_model.pkl")
        # Load a sample of test data for shape and feature reference
        cls.X_test = pd.read_csv("data/processed/X_test.csv")
        cls.y_test = pd.read_csv("data/processed/y_test.csv")

    def test_prediction_shape(self):
        """Ensure model returns output with correct shape for test set."""
        output = predict(self.model, self.X_test)
        self.assertEqual(output.shape, (self.X_test.shape[0],))

    def test_known_input(self):
        """Check that model predicts expected value for a known input (first row)."""
        test_input = self.X_test.iloc[[0]]
        prediction = predict(self.model, test_input)
        # We can't know the true label, but we can check type and shape
        self.assertEqual(prediction.shape, (1,))
        self.assertIn(prediction[0], np.unique(self.y_test))

    def test_multiple_inputs(self):
        """Check predictions for multiple inputs (first 5 rows)."""
        test_inputs = self.X_test.iloc[:5]
        predictions = predict(self.model, test_inputs)
        self.assertEqual(predictions.shape[0], 5)

    def test_invalid_input(self):
        """Ensure model raises error for invalid input shape."""
        # Remove a column to make it invalid
        test_input = self.X_test.iloc[[0], :-1]
        with self.assertRaises(Exception):
            predict(self.model, test_input)


if __name__ == "__main__":
    unittest.main()
