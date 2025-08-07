#!/usr/bin/env python3
"""
Data perturbation tester for ML drift detection
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import sys

# Import monitor module
from monitor import EvidentiallyMonitor

def perturb_data(data, modifications=None):
    if modifications is None:
        modifications = {
            'alcohol': -1.2,
            'volatile acidity': 0.1
        }
    
    print(f"Starting data perturbation:")
    print(f"Modification rules: {modifications}")
    
    perturbed_data = data.copy()
    applied_changes = {}
    
    for column, change in modifications.items():
        if column in perturbed_data.columns:
            original_mean = perturbed_data[column].mean()
            perturbed_data[column] = perturbed_data[column] + change
            new_mean = perturbed_data[column].mean()
            
            applied_changes[column] = {
                'change_value': change,
                'original_mean': float(original_mean),
                'new_mean': float(new_mean)
            }
            
            print(f"{column}: {original_mean:.3f} → {new_mean:.3f} (change: {change:+.1f})")
        else:
            print(f"Column '{column}' does not exist, skipping modification")
    
    return perturbed_data, applied_changes

def save_results(original_data, perturbed_data, comparison_results, applied_changes, output_dir):
    perturbed_data_path = f"{output_dir}/perturbed_data.csv"
    perturbed_data.to_csv(perturbed_data_path, index=False)
    print(f"Perturbed data saved: {perturbed_data_path}")
    
    complete_results = {
        'timestamp': datetime.now().isoformat(),
        'data_info': {
            'original_shape': list(original_data.shape),
            'perturbed_shape': list(perturbed_data.shape),
            'features': list(original_data.columns)
        },
        'perturbations_applied': applied_changes,
        'performance_comparison': comparison_results
    }
    
    results_path = f"{output_dir}/perturb_test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, indent=2, ensure_ascii=False)
    
    print(f"Complete results saved: {results_path}")
    
    return results_path

def main():
    """Main function - Execute perturbation test"""
    # Configuration parameters
    DATA_PATH = "/Users/ashley/Downloads/data/validation_data_complete.csv"
    MODEL_PATH = "/Users/ashley/Downloads/Data/StackedEnsemble_BestOfFamily_2_AutoML_1_20250806_231050"
    OUTPUT_DIR = "/Users/ashley/Desktop/evidently_reports"
    
    # Perturbation configuration
    PERTURBATIONS = {
        'alcohol': -1.2,
        'volatile acidity': 0.1
    }
    
    print("Starting data perturbation test")
    print("=" * 50)
    
    # Load original data
    print(f"Loading original data: {DATA_PATH}")
    original_data = pd.read_csv(DATA_PATH)
    original_data = original_data.dropna()
    print(f"Original data loaded successfully, shape: {original_data.shape}")
    
    perturbed_data, applied_changes = perturb_data(original_data, PERTURBATIONS)
    
    monitor = EvidentiallyMonitor(output_dir=OUTPUT_DIR)
    
    if not monitor.load_h2o_model(MODEL_PATH):
        print("Model loading failed, exiting")
        return None
    
    print(f"\nGenerating drift detection report...")
    drift_html, comparison_results = monitor.generate_drift_report(
        reference_data=original_data,
        current_data=perturbed_data,
        report_name="drift-after"
    )
    
    # Save results
    if comparison_results:
        results_path = save_results(
            original_data, 
            perturbed_data, 
            comparison_results, 
            applied_changes, 
            OUTPUT_DIR
        )
        
        # Output final summary
        print(f"\nPerturbation test summary:")
        print("=" * 50)
        print(f"Drift report: {drift_html}")
        print(f"Results file: {results_path}")
        
        # Output key metrics
        baseline_acc = comparison_results['baseline_metrics']['accuracy']
        drift_acc = comparison_results['drift_metrics']['accuracy']
        acc_change = comparison_results['changes']['accuracy_change_percent']
        f1_change = comparison_results['changes']['f1_change_percent']
        
        print(f"\nKey performance metrics:")
        print(f"Pre-perturbation accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
        print(f"Post-perturbation accuracy: {drift_acc:.4f} ({drift_acc*100:.2f}%)")
        print(f"Accuracy change: {acc_change:+.2f}%")
        print(f"F1 score change: {f1_change:+.2f}%")
        
        if acc_change < -10:
            print(f"Severe performance degradation detected!")
        elif acc_change < -5:
            print(f"Significant performance degradation detected!")
        else:
            print(f"Performance impact within acceptable range")
    
    return drift_html, comparison_results

if __name__ == "__main__":
    main()
