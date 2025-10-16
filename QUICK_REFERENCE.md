# Quick Reference Guide
## ARP Spoofing Detection System - Common Commands

This guide provides quick access to the most commonly used commands in the project.

## 📋 Table of Contents
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Real-Time Detection](#real-time-detection)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Setup

### Initial Setup
```bash
# Navigate to project directory
cd /home/nimisha/Files/Courses/ARP_SPOOFING/arp_spoofing_detection_project

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sklearn, pandas, numpy; print('✓ All dependencies installed!')"
```

### Activate Environment (Each Session)
```bash
# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

---

## Training

### Train All Models
```bash
# Train with default configuration
python scripts/train_model.py

# Expected output: models/saved_models/arp_spoofing_detector.pkl
# Training time: ~60-90 seconds
# Expected accuracy: ~96%
```

### Training Output
- **Model file:** `models/saved_models/arp_spoofing_detector.pkl`
- **Logs:** `outputs/logs/training_YYYYMMDD_HHMMSS.log`
- **Metrics:** Printed to console

---

## Evaluation

### Evaluate Saved Model
```bash
# Evaluate the trained model
python scripts/evaluate_model.py --model models/saved_models/arp_spoofing_detector.pkl

# Output: Comprehensive metrics to console
```

### Expected Metrics
- Accuracy: ~96.00%
- Precision: ~96.51%
- Recall: ~95.46%
- F1-Score: ~95.98%

---

## Visualization

### Generate All Plots
```bash
# Generate all 14 visualizations
python scripts/generate_plots.py

# Output: 14 PNG files in outputs/plots/
# Generation time: ~20-30 seconds
```

### Generated Plots
1. `class_distribution.png` - Bar chart
2. `class_distribution_pie.png` - Pie chart
3. `correlation_matrix.png` - Feature correlations
4. `feature_distributions.png` - Feature analysis
5. `confusion_matrix_random_forest.png` - Best model
6. `confusion_matrix_gradient_boosting.png` - GB model
7. `confusion_matrix_neural_network.png` - NN model
8. `confusion_matrix_logistic_regression.png` - LR model
9. `confusion_matrix_isolation_forest.png` - Unsupervised
10. `roc_curves.png` - ROC comparison
11. `feature_importance.png` - Feature rankings
12. `detection_timeline.png` - Detection over time
13. `alert_level_distribution.png` - Alert severity
14. `realtime_detection_results.png` - 8-panel comprehensive

### View Plots
```bash
# List all generated plots
ls -lh outputs/plots/

# Open specific plot (Linux)
xdg-open outputs/plots/realtime_detection_results.png

# Open specific plot (Mac)
open outputs/plots/realtime_detection_results.png
```

---

## Real-Time Detection

### Run Detection Simulation
```bash
# Default: 100 packets
python scripts/detect_realtime.py \
  --model models/saved_models/arp_spoofing_detector.pkl \
  --packets 100

# Custom packet count
python scripts/detect_realtime.py \
  --model models/saved_models/arp_spoofing_detector.pkl \
  --packets 500

# Output: Console display + realtime_detection_results.png
```

### Detection Output
- **Console:** Packet-by-packet analysis with colors
  - 🟢 SAFE (0.0-0.3 confidence)
  - 🟡 MEDIUM (0.3-0.7 confidence)
  - 🟠 HIGH (0.7-0.9 confidence)
  - 🔴 CRITICAL (0.9-1.0 confidence)
- **Plot:** `outputs/plots/realtime_detection_results.png` (8 panels)

### Expected Performance
- Accuracy: ~99%
- Processing time: <2ms per packet
- Missed attacks: 0
- False positive rate: ~2-4%

---

## Testing

### Run All Tests
```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
```

### Run Specific Tests
```bash
# Test data loader
pytest tests/test_data_loader.py -v

# Test feature engineering
pytest tests/test_feature_engineering.py -v

# Test models
pytest tests/test_models.py -v
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Model File Not Found
**Error:** `FileNotFoundError: models/saved_models/arp_spoofing_detector.pkl`

**Solution:**
```bash
# Train the model first
python scripts/train_model.py
```

#### 2. Dataset Not Found
**Error:** `FileNotFoundError: dataset/CIC_MITM_ArpSpoofing_All_Labelled.csv`

**Solution:**
```bash
# Ensure datasets are in the correct location
ls dataset/
# Should show:
# - CIC_MITM_ArpSpoofing_All_Labelled.csv
# - All_Labelled.csv
# - GIT_arpspoofLabelledData.csv
```

#### 3. Old Model Incompatible
**Error:** `TypeError: __init__() got an unexpected keyword argument`

**Solution:**
```bash
# Retrain model with updated code
python scripts/train_model.py
```

**Note:** Models saved before October 17, 2025 are incompatible due to scaler fix.

#### 4. Import Errors
**Error:** `ModuleNotFoundError: No module named 'sklearn'`

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 5. Memory Errors
**Error:** `MemoryError` during training

**Solution:**
```bash
# Close other applications
# Or use a machine with more RAM (4GB+ recommended)
```

#### 6. Permission Errors
**Error:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
```bash
# Create output directories with proper permissions
mkdir -p outputs/plots outputs/logs outputs/reports
mkdir -p models/saved_models
```

---

## Quick Workflow

### Complete Pipeline (Start to Finish)
```bash
# 1. Setup (first time only)
source venv/bin/activate
pip install -r requirements.txt

# 2. Train model
python scripts/train_model.py

# 3. Generate visualizations
python scripts/generate_plots.py

# 4. Evaluate model
python scripts/evaluate_model.py --model models/saved_models/arp_spoofing_detector.pkl

# 5. Test real-time detection
python scripts/detect_realtime.py --model models/saved_models/arp_spoofing_detector.pkl --packets 100

# 6. View results
ls -lh outputs/plots/
```

**Expected Time:** ~2-3 minutes total

---

## File Locations

### Input Files
- **Datasets:** `dataset/*.csv`
- **Configuration:** `config/*.yaml`

### Output Files
- **Models:** `models/saved_models/*.pkl`
- **Plots:** `outputs/plots/*.png`
- **Logs:** `outputs/logs/*.log`
- **Reports:** `outputs/reports/`

### Documentation
- **README:** `README.md`
- **Changelog:** `CHANGELOG.md`
- **Quick Reference:** `QUICK_REFERENCE.md` (this file)
- **Project Deliverables:** `docs/PROJECT_DELIVERABLES.md`
- **Hybrid Approach:** `docs/HYBRID_APPROACH_DOCUMENTATION.md`
- **API Docs:** `docs/API_DOCUMENTATION.md`
- **Deployment:** `docs/DEPLOYMENT_GUIDE.md`

---

## Key Metrics Quick Reference

| Metric | Value | Meaning |
|--------|-------|---------|
| **Accuracy** | 96.00% | Overall correctness |
| **Precision** | 96.51% | Attack predictions accuracy |
| **Recall** | 95.46% | Attack detection rate |
| **F1-Score** | 95.98% | Balance of precision/recall |
| **FPR** | 4.09% | False alarm rate |
| **FNR** | 3.15% | Missed attack rate |
| **Inference** | <2ms | Prediction latency |
| **Throughput** | ~500 pkt/s | Processing rate |

---

## Alert Level Reference

| Level | Confidence Range | Color | Action |
|-------|-----------------|-------|--------|
| **SAFE** | 0.0 - 0.3 | 🟢 Green | Normal - No action |
| **MEDIUM** | 0.3 - 0.7 | 🟡 Yellow | Monitor activity |
| **HIGH** | 0.7 - 0.9 | 🟠 Orange | Investigate packet |
| **CRITICAL** | 0.9 - 1.0 | 🔴 Red | Block immediately |

---

## Python API Quick Reference

### Load Model and Detect
```python
from src.detector import ARPSpoofingDetector

# Load trained detector
detector = ARPSpoofingDetector.load(
    "models/saved_models/arp_spoofing_detector.pkl"
)

# Prepare packet features (25 features required)
packet = {
    'bidirectional_packets': 15,
    'duration_ms': 1200,
    'bidirectional_bytes': 8400,
    # ... (22 more features)
}

# Detect attack
result = detector.detect(packet)

# Check result
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Alert Level: {result['alert_level']}")

# Take action based on alert level
if result['alert_level'] in ['HIGH', 'CRITICAL']:
    print("⚠️ ATTACK DETECTED - Take action!")
```

---

## Configuration Quick Reference

### Model Hyperparameters (config/model_config.yaml)

**Random Forest (Best Model):**
- n_estimators: 200
- max_depth: 15
- min_samples_split: 5
- min_samples_leaf: 2
- class_weight: balanced

**Isolation Forest (Anomaly Detection):**
- contamination: 0.1
- n_estimators: 100

### System Settings (config/config.yaml)

**Training:**
- test_size: 0.2 (20% test split)
- random_state: 42 (reproducibility)

**Features:**
- n_features: 25
- selection_method: "hybrid"
- scaler: "standard"

---

## Support & Resources

### Documentation
- Full documentation in `docs/` directory
- API reference: `docs/API_DOCUMENTATION.md`
- Deployment guide: `docs/DEPLOYMENT_GUIDE.md`

### Logs
- Training logs: `outputs/logs/`
- Check logs for detailed error messages

### Contact
- Check README.md for contact information
- Review CHANGELOG.md for recent updates

---

**Last Updated:** October 17, 2025

**Version:** 1.0.0

**Status:** ✅ Production Ready
