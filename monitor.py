#!/usr/bin/env python3
"""
Evidently Monitor - monitor.py
Features:
1. Generate baseline monitoring reports
2. Integrate H2O AutoML model for predictions
3. Calculate classification metrics (accuracy, F1, etc.)
4. Support drift detection and comparative analysis
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
from datetime import datetime
import os
import h2o

# Evidently imports - Fix import issues
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset, ClassificationPreset
    from evidently import ColumnMapping
    print("✅ Evidently imports successful")
except ImportError as e:
    print(f"❌ Evidently import failed: {e}")
    exit(1)

class EvidentiallyMonitor:
    def __init__(self, output_dir="./reports"):
        """
        Initialize Evidently Monitor
        """
        self.model = None
        self.output_dir = output_dir
        self.h2o_initialized = False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize H2O
        self._init_h2o()
    
    def _init_h2o(self):
        """Initialize H2O cluster"""
        try:
            h2o.init()
            self.h2o_initialized = True
            print("✅ H2O cluster initialized successfully")
        except Exception as e:
            print(f"❌ H2O initialization failed: {e}")
            self.h2o_initialized = False
    
    def load_h2o_model(self, model_path):
        """Load H2O model"""
        if not self.h2o_initialized:
            print("❌ H2O not initialized, cannot load model")
            return False
        
        try:
            self.model = h2o.load_model(model_path)
            print(f"✅ H2O model loaded successfully: {model_path}")
            print(f"✅ Model ID: {self.model.model_id}")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.model = None
            return False
    
    def setup_column_mapping(self, data):
        """Setup Evidently column mapping - Explicitly define numerical and categorical columns"""
        # Explicitly define numerical features (all physicochemical indicators)
        numerical_features = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
        
        # Explicitly define categorical features
        categorical_features = ['type']  # Red/white wine type
        
        # Filter out columns that don't exist in the data
        numerical_features = [col for col in numerical_features if col in data.columns]
        categorical_features = [col for col in categorical_features if col in data.columns]
        
        column_mapping = ColumnMapping(
            target='quality',
            prediction='prediction' if 'prediction' in data.columns else None,
            numerical_features=numerical_features,
            categorical_features=categorical_features
        )
        
        print(f"📊 Column Mapping setup:")
        print(f"   Target column: quality")
        print(f"   Prediction column: {'prediction' if 'prediction' in data.columns else 'None'}")
        print(f"   Numerical columns ({len(numerical_features)}): {numerical_features}")
        print(f"   Categorical columns ({len(categorical_features)}): {categorical_features}")
        
        return column_mapping
    
    def predict_with_h2o(self, data):
        """Make predictions using H2O model"""
        if not self.model or not self.h2o_initialized:
            return None
        
        try:
            # Prepare feature data
            feature_cols = [col for col in data.columns if col != 'quality']
            
            # Convert to H2O Frame
            h2o_frame = h2o.H2OFrame(data[feature_cols])
            
            # Set categorical features
            if 'type' in h2o_frame.columns:
                h2o_frame['type'] = h2o_frame['type'].asfactor()
            
            # Make predictions
            predictions = self.model.predict(h2o_frame)
            pred_df = predictions.as_data_frame()
            
            return pred_df['predict'].astype(int)
            
        except Exception as e:
            print(f"❌ H2O prediction failed: {e}")
            return None
    
    def calculate_metrics(self, data, data_name=""):
        """Calculate classification metrics"""
        if not self.model:
            print("⚠️  Model not loaded, skipping metrics calculation")
            return None
        
        print(f"\n📊 Calculating {data_name} classification metrics:")
        
        try:
            y_true = data['quality']
            y_pred = self.predict_with_h2o(data)
            
            if y_pred is None:
                return None
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            results = {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'sample_count': len(y_true),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"   📈 Accuracy: {accuracy:.4f}")
            print(f"   📈 F1 Score: {f1:.4f}")
            print(f"   📈 Sample Count: {len(y_true)}")
            
            return results
            
        except Exception as e:
            print(f"❌ Metrics calculation failed: {e}")
            return None
    
    def generate_drift_report(self, reference_data, current_data, report_name="drift"):
        """Generate Evidently drift detection report and calculate performance comparison"""
        print(f"\n🔄 Generating Evidently drift detection report...")
        
        try:
            # Add prediction columns to both datasets
            reference_with_pred = reference_data.copy()
            current_with_pred = current_data.copy()
            
            baseline_metrics = None
            drift_metrics = None
            
            if self.model:
                # Add predictions to reference data
                reference_predictions = self.predict_with_h2o(reference_data)
                if reference_predictions is not None:
                    reference_with_pred['prediction'] = reference_predictions
                    baseline_metrics = self.calculate_metrics(reference_data, "baseline")
                
                # Add predictions to current data
                current_predictions = self.predict_with_h2o(current_data)
                if current_predictions is not None:
                    current_with_pred['prediction'] = current_predictions
                    drift_metrics = self.calculate_metrics(current_data, "after drift")
                
                print("✅ Added prediction columns to both datasets")
            
            # Setup column mapping
            column_mapping = self.setup_column_mapping(reference_with_pred)
            
            # Create report
            metrics = [
                DataDriftPreset(),
                DataQualityPreset(),
                TargetDriftPreset()
            ]
            
            if 'prediction' in reference_with_pred.columns and 'prediction' in current_with_pred.columns:
                try:
                    metrics.append(ClassificationPreset())
                    print("✅ Added classification metrics")
                except:
                    print("⚠️  Failed to add classification metrics")
            
            # Generate drift detection report
            report = Report(metrics=metrics)
            report.run(
                reference_data=reference_with_pred,
                current_data=current_with_pred,
                column_mapping=column_mapping
            )
            
            # Save HTML report
            report_html = f"{self.output_dir}/{report_name}.html"
            report.save_html(report_html)
            print(f"✅ Drift report saved: {report_html}")
            
            # Save JSON format report
            report_json = f"{self.output_dir}/{report_name}.json"
            report.save_json(report_json)
            print(f"✅ Drift report JSON saved: {report_json}")
            
            # Calculate performance changes
            comparison_results = None
            if baseline_metrics and drift_metrics:
                accuracy_change = drift_metrics['accuracy'] - baseline_metrics['accuracy']
                f1_change = drift_metrics['f1_score'] - baseline_metrics['f1_score']
                
                comparison_results = {
                    'baseline_metrics': baseline_metrics,
                    'drift_metrics': drift_metrics,
                    'changes': {
                        'accuracy_change': float(accuracy_change),
                        'f1_change': float(f1_change),
                        'accuracy_change_percent': float(accuracy_change * 100),
                        'f1_change_percent': float(f1_change * 100)
                    }
                }
                
                print(f"\n🎯 Performance comparison analysis:")
                print(f"📊 Accuracy change: {baseline_metrics['accuracy']:.4f} → {drift_metrics['accuracy']:.4f} ({accuracy_change:+.4f})")
                print(f"📊 F1 score change: {baseline_metrics['f1_score']:.4f} → {drift_metrics['f1_score']:.4f} ({f1_change:+.4f})")
            
            return report_html, comparison_results
            
        except Exception as e:
            print(f"❌ Drift report generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main function - Generate baseline report"""
    # Configuration parameters
    DATA_PATH = "/Users/ashley/Downloads/data/validation_data_complete.csv"
    MODEL_PATH = "/Users/ashley/Downloads/Data/StackedEnsemble_BestOfFamily_2_AutoML_1_20250806_231050"
    OUTPUT_DIR = "/Users/ashley/Desktop/evidently_reports"
    
    print("🚀 Starting Evidently baseline monitoring")
    print("=" * 50)
    
    # Load data
    print(f"📂 Loading data: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    data = data.dropna()
    print(f"✅ Data loaded successfully, shape: {data.shape}")
    
    # Create monitor
    monitor = EvidentiallyMonitor(output_dir=OUTPUT_DIR)
    
    # Load model
    if not monitor.load_h2o_model(MODEL_PATH):
        print("❌ Model loading failed, exiting")
        return
    
    # Generate baseline report (reference and current data are the same)
    baseline_html, _ = monitor.generate_drift_report(
        reference_data=data,
        current_data=data,
        report_name="baseline"
    )
    
    # Calculate and save baseline metrics
    baseline_metrics = monitor.calculate_metrics(data, "baseline")
    if baseline_metrics:
        metrics_path = f"{OUTPUT_DIR}/baseline_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(baseline_metrics, f, indent=2, ensure_ascii=False)
        print(f"💾 Baseline metrics saved: {metrics_path}")
    
    print(f"\n🎯 Baseline monitoring completed:")
    print(f"✅ Baseline report: {baseline_html}")
    print(f"📁 All files saved in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
