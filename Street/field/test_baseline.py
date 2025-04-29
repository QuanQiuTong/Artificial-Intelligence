import unittest
import pandas as pd
import os
import numpy as np
from collections import Counter
from baseline import write2csv, parse2class

import matplotlib.pyplot as plt

# Import functions from baseline.py


CSV = './result1.csv'

class TestCSVResults(unittest.TestCase):
    """Test suite for evaluating results.csv without retraining the model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.csv_path = CSV
        self.df = pd.read_csv(self.csv_path)
    
    def test_csv_format(self):
        """Test if the CSV file has the expected format"""
        # Check column names
        self.assertIn('file_name', self.df.columns)
        self.assertIn('file_code', self.df.columns)
        
        # Check if there are any empty values
        self.assertFalse(self.df['file_name'].isnull().any())
        
        # Check if file_name has the expected format
        self.assertTrue(all(name.endswith('.png') for name in self.df['file_name']))
        
        # Check if all file names contain a directory prefix
        self.assertTrue(all('mchar_test_a' in name for name in self.df['file_name']))
    
    def test_predictions_distribution(self):
        """Test the distribution of predicted codes"""
        # Check if all predictions are numbers
        self.assertTrue(all(str(code).isdigit() for code in self.df['file_code']))
        
        # Analyze prediction length distribution
        prediction_lengths = self.df['file_code'].astype(str).apply(len)
        length_counts = Counter(prediction_lengths)
        
        # Print distribution information
        print(f"Prediction length distribution: {length_counts}")
        
        # The majority of predictions should have 1-4 digits
        self.assertTrue(sum(length_counts[i] for i in range(1, 5)) > 0.5 * len(self.df))
    
    def test_prediction_statistics(self):
        """Test statistics of predictions"""
        # Convert to numeric for statistical analysis
        self.df['file_code'] = pd.to_numeric(self.df['file_code'])
        
        # Calculate statistics
        mean = self.df['file_code'].mean()
        median = self.df['file_code'].median()
        std = self.df['file_code'].std()
        min_val = self.df['file_code'].min()
        max_val = self.df['file_code'].max()
        
        # Print statistics
        print(f"Statistics of predictions:")
        print(f"Mean: {mean}")
        print(f"Median: {median}")
        print(f"Standard deviation: {std}")
        print(f"Min: {min_val}")
        print(f"Max: {max_val}")
        
        # Ensure the predictions are within reasonable ranges
        self.assertTrue(0 <= min_val <= 9999)
        self.assertTrue(1 <= max_val <= 10000)
    
    def test_frequency_analysis(self):
        """Test frequency analysis of predictions"""
        # Analyze digit frequency
        digits = ''.join(self.df['file_code'].astype(str))
        digit_counter = Counter(digits)
        
        # Print digit distribution
        print(f"Digit distribution: {digit_counter}")
        
        # All digits should be used at least once
        for digit in '0123456789':
            self.assertIn(digit, digit_counter)
    
    def test_plot_predictions(self):
        """Generate a histogram of predictions"""
        # Skip this test if running in a non-interactive environment
        try:
            plt.figure(figsize=(12, 6))
            plt.hist(self.df['file_code'], bins=50, alpha=0.7)
            plt.title('Distribution of Predictions')
            plt.xlabel('Predicted Code')
            plt.ylabel('Frequency')
            plt.savefig('prediction_distribution.png')
            plt.close()
            print("Generated prediction distribution plot: prediction_distribution.png")
        except Exception as e:
            print(f"Could not generate plot: {e}")


def analyze_accuracy_against_ground_truth(results_path, ground_truth_path=None):
    """
    Analyze the accuracy of predictions against ground truth if available
    
    Args:
        results_path: Path to results CSV file
        ground_truth_path: Path to ground truth file (if available)
    
    Returns:
        Accuracy metrics or None if ground truth is not available
    """
    results_df = pd.read_csv(results_path)
    
    # If no ground truth available, just return distribution analysis
    if ground_truth_path is None or not os.path.exists(ground_truth_path):
        print("No ground truth available for accuracy calculation")
        
        # Calculate distribution statistics
        prediction_counts = Counter(results_df['file_code'])
        top_predictions = prediction_counts.most_common(10)
        
        print("Top 10 most common predictions:")
        for prediction, count in top_predictions:
            print(f"  {prediction}: {count} occurrences ({count/len(results_df)*100:.2f}%)")
        
        return None
    
    # If ground truth is available, calculate accuracy
    # This would need implementation based on the ground truth format
    pass


if __name__ == "__main__":
    # Run the tests
    unittest.main()
    
    # Additional analysis
    print("\n--- Additional Analysis ---")
    analyze_accuracy_against_ground_truth(CSV, ground_truth_path=None)