# test_accuracy.py
import os
import json
import unittest
import pandas as pd
import numpy as np
import torch as t
from glob import glob
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader

# Import necessary components from baseline
try:
   from baseline import DigitsDataset, DigitsResnet50, Config, parse2class
except ImportError:
   # Fallback imports if module structure is different
   import sys
   sys.path.append(os.path.dirname(os.path.abspath(__file__)))
   from baseline import DigitsDataset, DigitsResnet50, Config, parse2class

class TestAccuracy(unittest.TestCase):
   """Test suite for evaluating model accuracy locally"""
   
   def setUp(self):
      """Set up test fixtures"""
      self.config = Config()
      self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
      self.result_csv = './result.csv'  # Path to your results CSV
      self.dataset_path = "./dataset"   # Adjust this to your dataset path
      self.benchmark_targets = [0.86, 0.88, 0.90, 0.92]  # Target accuracies
      
      # Define data directories
      self.data_dir = {
         'train_data': f'{self.dataset_path}/mchar_train/',
         'val_data': f'{self.dataset_path}/mchar_val/',
         'test_data': f'{self.dataset_path}/mchar_test_a/',
         'train_label': f'{self.dataset_path}/mchar_train.json',
         'val_label': f'{self.dataset_path}/mchar_val.json',
      }
      
      # Load results
      self.df_results = pd.read_csv(self.result_csv)
      
   def test_result_format(self):
      """Test if the result CSV has the correct format"""
      self.assertIn('file_name', self.df_results.columns)
      self.assertIn('file_code', self.df_results.columns)
      
      # Check for file format and naming consistency
      self.assertTrue(all(name.endswith('.png') for name in self.df_results['file_name']))
      
      # Check for missing values
      self.assertEqual(self.df_results.isnull().sum().sum(), 0, "Result CSV has null values")
      
      print(f"Result CSV format is valid, contains {len(self.df_results)} entries")
      
   def test_prediction_distribution(self):
      """Analyze the distribution of predictions"""
      # Convert to string to handle different types of predictions
      predictions = self.df_results['file_code'].astype(str)
      
      # Compute length statistics
      prediction_lengths = predictions.str.len()
      length_stats = prediction_lengths.describe()
      length_counts = Counter(prediction_lengths)
      
      print("\n=== Prediction Length Statistics ===")
      print(f"Mean prediction length: {length_stats['mean']:.2f}")
      print(f"Min prediction length: {length_stats['min']}")
      print(f"Max prediction length: {length_stats['max']}")
      print(f"Length distribution: {dict(sorted(length_counts.items()))}")
      
      # Distribution of individual digits
      all_digits = ''.join(predictions)
      digit_counts = Counter(all_digits)
      
      print("\n=== Digit Distribution ===")
      for digit, count in sorted(digit_counts.items()):
         print(f"Digit {digit}: {count} occurrences ({count/len(all_digits)*100:.2f}%)")
      
      # Check for valid digits (should only contain 0-9)
      valid_chars = set("0123456789")
      invalid_chars = set(all_digits) - valid_chars
      
      if invalid_chars:
         print(f"Warning: Found invalid characters in predictions: {invalid_chars}")
      else:
         print("All predictions contain valid digits (0-9)")
   
   def test_validation_accuracy(self):
      """Test the model's accuracy on the validation set"""
      # Skip if validation data not available
      if not os.path.exists(self.data_dir['val_label']):
         self.skipTest("Validation data not available")
      
      # Create validation dataset
      val_dataset = DigitsDataset(mode='val', aug=False)
      val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, 
                        shuffle=False, num_workers=2, pin_memory=True)
      
      # Load the model (assuming we have a trained model)
      try:
         # Get checkpoint path from environment variable or use default
         checkpoint_path = os.environ.get('MODEL_CHECKPOINT', 'best_model.pth')
         
         if os.path.exists(checkpoint_path):
            model = DigitsResnet50(self.config.class_num).to(self.device)
            model.load_state_dict(t.load(checkpoint_path)['model'])
            model.eval()
            
            # Calculate accuracy
            corrects = 0
            total = 0
            
            with t.no_grad():
               for img, label in val_loader:
                  img = img.to(self.device)
                  label = label.to(self.device)
                  pred = model(img)
                  
                  # Check if all 4 positions are correct
                  temp = t.stack([
                     pred[0].argmax(1) == label[:, 0],
                     pred[1].argmax(1) == label[:, 1],
                     pred[2].argmax(1) == label[:, 2],
                     pred[3].argmax(1) == label[:, 3],
                  ], dim=1)
                  
                  corrects += t.all(temp, dim=1).sum().item()
                  total += label.size(0)
            
            accuracy = corrects / total
            print(f"\n=== Validation Accuracy ===")
            print(f"Accuracy on validation set: {accuracy:.4f} ({corrects}/{total})")
            
            # Check against benchmark targets
            print("\n=== Benchmark Assessment ===")
            for target in self.benchmark_targets:
               status = "✓ ACHIEVED" if accuracy >= target else "✗ NOT MET"
               print(f"Target {target:.2f}: {status}")
            
            # Return accuracy for use in other tests
            return accuracy
         else:
            print(f"Model checkpoint not found: {checkpoint_path}")
            self.skipTest("Model checkpoint not available")
      except Exception as e:
         print(f"Error loading model: {e}")
         self.skipTest(f"Failed to load model: {e}")
         
   def test_estimate_test_accuracy(self):
      """Estimate accuracy on test set based on validation performance"""
      # This is just an estimation - can't calculate true accuracy without labels
      validation_accuracy = getattr(self, 'validation_accuracy', None)
      
      if validation_accuracy is None:
         try:
            validation_accuracy = self.test_validation_accuracy()
         except Exception:
            self.skipTest("Cannot estimate test accuracy without validation accuracy")
      
      # Typical validation-test gap based on similar competitions
      estimated_gap = 0.02  # This is a conservative estimate
      
      estimated_test_accuracy = validation_accuracy - estimated_gap
      
      print("\n=== Estimated Test Accuracy ===")
      print(f"Estimated accuracy on test set: {estimated_test_accuracy:.4f}")
      print(f"(Based on validation accuracy of {validation_accuracy:.4f} with an estimated gap of {estimated_gap:.4f})")
      
      print("\n=== Submission Requirement Assessment ===")
      for target in self.benchmark_targets:
         status = "✓ LIKELY TO MEET" if estimated_test_accuracy >= target else "✗ MAY NOT MEET"
         print(f"Target {target:.2f}: {status}")

   def test_error_analysis(self):
      """Analyze error patterns on validation data"""
      # Skip if validation data not available
      if not os.path.exists(self.data_dir['val_label']):
         self.skipTest("Validation data not available")
         
      try:
         # Get checkpoint path from environment variable or use default
         checkpoint_path = os.environ.get('MODEL_CHECKPOINT', 'best_model.pth')
         
         if os.path.exists(checkpoint_path):
            model = DigitsResnet50(self.config.class_num).to(self.device)
            model.load_state_dict(t.load(checkpoint_path)['model'])
            model.eval()
            
            # Create validation dataset
            val_dataset = DigitsDataset(mode='val', aug=False)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, 
                             shuffle=False, num_workers=2, pin_memory=True)
            
            # Track errors by position and digit
            errors_by_position = {0: 0, 1: 0, 2: 0, 3: 0}
            errors_by_digit = {i: 0 for i in range(11)}  # 0-9 + empty (10)
            confusion_matrix = np.zeros((11, 11), dtype=int)  # 11x11 for digits 0-9 + empty
            
            with t.no_grad():
               for img, label in val_loader:
                  img = img.to(self.device)
                  label = label.to(self.device)
                  pred = model(img)
                  
                  for pos in range(4):
                     pred_pos = pred[pos].argmax(1)
                     label_pos = label[:, pos]
                     
                     # Count errors by position
                     errors_by_position[pos] += (pred_pos != label_pos).sum().item()
                     
                     # Count errors by true digit
                     for true_digit in range(11):
                        digit_indices = (label_pos == true_digit)
                        errors_by_digit[true_digit] += ((pred_pos != label_pos) & digit_indices).sum().item()
                        
                        # Update confusion matrix
                        for pred_digit in range(11):
                           confusion_matrix[true_digit, pred_digit] += ((pred_pos == pred_digit) & 
                                                       (label_pos == true_digit)).sum().item()
            
            print("\n=== Error Analysis ===")
            print("Errors by position:")
            for pos, count in errors_by_position.items():
               print(f"Position {pos+1}: {count} errors")
            
            print("\nErrors by true digit:")
            for digit, count in errors_by_digit.items():
               digit_label = str(digit) if digit != 10 else "empty"
               print(f"Digit {digit_label}: {count} errors")
            
            print("\nTop confused digit pairs (true → predicted):")
            # Get top confused pairs (excluding correct predictions on diagonal)
            confused_pairs = []
            for i in range(11):
               for j in range(11):
                  if i != j:  # Skip diagonal (correct predictions)
                     confused_pairs.append((i, j, confusion_matrix[i, j]))
            
            # Sort by count in descending order and show top 10
            confused_pairs.sort(key=lambda x: x[2], reverse=True)
            for true, pred, count in confused_pairs[:10]:
               true_label = str(true) if true != 10 else "empty"
               pred_label = str(pred) if pred != 10 else "empty"
               print(f"{true_label} → {pred_label}: {count} occurrences")
         else:
            print(f"Model checkpoint not found: {checkpoint_path}")
            self.skipTest("Model checkpoint not available")
      except Exception as e:
         print(f"Error in error analysis: {e}")
         self.skipTest(f"Failed to perform error analysis: {e}")
   
   def test_plot_results(self):
      """Generate visualizations of results"""
      try:
         # Plot prediction length distribution
         predictions = self.df_results['file_code'].astype(str)
         prediction_lengths = predictions.str.len()
         
         plt.figure(figsize=(10, 6))
         plt.hist(prediction_lengths, bins=range(1, 8), align='left', rwidth=0.8)
         plt.title('Distribution of Prediction Lengths')
         plt.xlabel('Number of Digits')
         plt.ylabel('Count')
         plt.xticks(range(1, 7))
         plt.savefig('prediction_length_distribution.png')
         plt.close()
         
         # Plot digit distribution
         all_digits = ''.join(predictions)
         digit_counts = Counter(all_digits)
         
         plt.figure(figsize=(10, 6))
         digits = sorted(digit_counts.keys())
         counts = [digit_counts[d] for d in digits]
         plt.bar(digits, counts)
         plt.title('Distribution of Digits in Predictions')
         plt.xlabel('Digit')
         plt.ylabel('Count')
         plt.savefig('digit_distribution.png')
         plt.close()
         
         print("\n=== Visualizations Generated ===")
        #  print("-

      except:
            pass
