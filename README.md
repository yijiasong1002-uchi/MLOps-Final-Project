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

**Note:** Due to file size limitations, HTML reports need to be downloaded and opened locally in your browser.

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

## How to View HTML Reports

### Method 1: Download and Open Locally
1. Download the HTML files from the repository
2. Open them in your web browser locally

### Method 2: GitHub Pages (Recommended)
Visit the live reports at: [GitHub Pages Site](https://yijiasong1002-uchi.github.io/MLOps-Final-Project/)

### Method 3: Alternative Hosting
- Upload to any static hosting service (Netlify, Vercel, etc.)
- Use online HTML viewers like htmlpreview.github.io
