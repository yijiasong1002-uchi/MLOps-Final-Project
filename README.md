# MLOps-Final-Project

## ML Monitoring with Evidently

### Files

- `monitor.py` - Baseline monitoring script
- `perturb_test.py` - Data drift simulation and testing
- `requirements.txt` - Required dependencies

### Setup

```bash
pip install h2o evidently pandas scikit-learn numpy
```

### Usage

#### 1. Generate Baseline Report

```bash
python monitor.py
```

This creates a baseline monitoring report using the original dataset as both reference and current data.

**Output:**
- `baseline.html` - Evidently monitoring report
- `baseline_metrics.json` - Performance metrics

#### 2. Test Data Drift

```bash
python perturb_test.py
```

This simulates data drift by modifying alcohol (-1.2) and volatile acidity (+0.1) features, then generates a drift detection report.

**Output:**
- `drift-after.html` - Drift detection report
- `perturb_test_results.json` - Complete comparison results

### Results

The monitoring system successfully detects significant performance degradation:

- **Baseline Accuracy:** 68.40%
- **After Drift Accuracy:** 49.92%
- **Performance Drop:** -18.48%

### Technical Details

#### Column Mapping
- **Numerical features:** 11 physicochemical properties
- **Categorical features:** Wine type (red/white)
- **Target:** Quality score
- **Predictions:** H2O AutoML model outputs

#### Monitoring Metrics
- Data drift detection
- Data quality analysis
- Target drift monitoring
- Classification performance tracking
