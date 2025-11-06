# Changelog

All notable changes to the ARP Spoofing Detection System.

## [2.1.0] - 2025-11-07

### Major Changes

#### Real-Time Detection Overhaul
- **Batch Processing Mode**: Changed from incremental packet-by-packet HTTP requests to batch processing (all packets at once)
  - **Before**: `/api/realtime/next` endpoint called repeatedly for each packet
  - **After**: `/api/realtime/start` processes all packets in single request
  - **Benefit**: Matches CLI script behavior (`detect_realtime.py`), better performance, comprehensive statistics
  
- **6-Panel Visualization Dashboard**: Added matplotlib-based comprehensive visualization
  1. Detection timeline (attack/normal patterns over time)
  2. Confusion matrix heatmap (TP/TN/FP/FN)
  3. Alert level distribution pie chart
  4. Confidence score histogram
  5. Performance metrics comparison table
  6. Real-time accuracy/precision/recall statistics
  
- **Client-Side Animation**: Results received in single batch, animated display for better UX
  - Maintains visual packet-by-packet effect without network overhead
  - Configurable speed (50ms - 500ms per packet)

#### Hybrid Model Implementation Fix
- **Root Cause**: Hybrid models were never saved as `.pkl` files - they're computed on-the-fly
- **Solution**: Removed non-existent pickle file references, implemented dynamic loading
  
**Changes Made:**
1. **Removed `HybridModel` class** from `app.py` (was causing unpickling errors)
2. **Updated `AVAILABLE_MODELS` configuration**:
   ```python
   'hybrid_rf_if': {
       'supervised': 'random_forest',
       'unsupervised': 'isolation_forest',
       'weights': (0.7, 0.3),
       # No 'path' key - computed on-the-fly
   }
   ```
3. **Modified `load_specific_model()`**: Detects hybrid models, loads both components separately
4. **Added helper functions**:
   - `predict_with_model()`: Handles predictions for regular and hybrid models
   - `predict_proba_with_model()`: Handles probability predictions for both types
5. **Updated all prediction calls**: Use helper functions throughout codebase

**Hybrid Model Behavior:**
- Load supervised model (e.g., Random Forest)
- Load unsupervised model (e.g., Isolation Forest)
- Combine predictions: `(0.7 * supervised) + (0.3 * unsupervised)`
- No separate pickle file needed
- Automatic fallback if unsupervised component unavailable

#### Validation on Unseen Data (UQ Dataset)
- **Added comprehensive validation** on completely independent dataset
- **Dataset**: University of Queensland MITM ARP dataset (10,055 flows, 2% attacks)
- **Test Script**: `scripts/test_uq_dataset.py`

**Results - Imbalanced (Real-World Scenario):**
- Random Forest: **96.95% accuracy**, **90.55% recall**, ROC-AUC: 0.9908
- Hybrid (LOF + RF): 82.92% accuracy, **93.03% recall**
- Validates excellent generalization to unseen data
- Results: `outputs/reports/uq_dataset_results_imbalanced.json`

**Results - Balanced (Research Scenario):**
- Random Forest: **93.78% accuracy**, **96.81% precision**, 90.55% recall
- Hybrid (LOF + RF): 86.57% accuracy, 82.38% precision, **93.03% recall**
- Results: `outputs/reports/uq_dataset_results_balanced.json`

### Technical Improvements

#### Code Quality
- Fixed unpickling errors for hybrid models
- Improved error handling with traceback logging
- Removed orphaned code from failed edits
- Better separation of concerns (prediction logic in helper functions)

#### Performance
- Batch processing: Process 500 packets in <2 seconds
- Single HTTP request vs. hundreds for real-time detection
- Reduced server load and network overhead
- Matplotlib plot generation: ~1-2 seconds for comprehensive dashboard

#### Documentation
- Updated README.md with UQ dataset validation results
- Added real-world implications section
- Documented hybrid model on-the-fly approach
- Added deployment recommendations based on validation results
- Created CHANGELOG.md (this file)

### Files Modified

#### Core Application
- `app.py`:
  - Removed `HybridModel` class (lines 11-59)
  - Updated `AVAILABLE_MODELS` hybrid model definitions
  - Modified `load_specific_model()` to handle hybrid models
  - Added `predict_with_model()` and `predict_proba_with_model()` helpers
  - Updated `/api/upload` to use helper functions
  - Updated `/api/realtime/start` to batch process and generate plot
  - Added `generate_realtime_plot()` for 6-panel visualization
  - Fixed startup model checking for hybrid models

#### Frontend
- `templates/realtime.html`:
  - Removed incremental `/api/realtime/next` polling
  - Updated to receive all results in single response
  - Added client-side animation logic
  - Display comprehensive visualization plot
  - Maintained terminal-style feed with color coding

#### Documentation
- `README.md`:
  - Added UQ dataset validation section
  - Updated real-time detection description
  - Added hybrid model on-the-fly explanation
  - Updated performance metrics tables
  - Added deployment recommendations
  - Updated project status to v2.1

### Files Created

- `CHANGELOG.md`: This file - comprehensive change tracking
- `outputs/reports/uq_dataset_results_imbalanced.json`: UQ dataset validation results
- `outputs/reports/uq_dataset_results_balanced.json`: UQ balanced dataset results

### Breaking Changes

⚠️ **Hybrid Model Pickle Files**: If you have existing hybrid model `.pkl` files, they will no longer work. Hybrid models are now computed on-the-fly by loading component models.

**Migration:**
- Remove `models/saved_models/hybrid_*.pkl` files (if they exist)
- Ensure component models exist (e.g., `random_forest.pkl`, `isolation_forest.pkl`)
- Hybrid models will be automatically created when selected

### Bug Fixes

1. **Fixed KeyError in app startup**: Hybrid models don't have 'path' key
2. **Fixed unpickling errors**: Removed non-existent `HybridModel` class references
3. **Fixed AttributeError**: Removed reliance on non-existent model attributes
4. **Fixed prediction errors**: Use helper functions that handle both model types

### Known Issues

- Hybrid RF+LOF shows as unavailable (LOF model missing or incorrect naming)
- Hard Voting and Soft Voting ensemble models not available (not implemented)

### Deployment Notes

**Production Recommendations:**
1. **Standard Networks**: Use Random Forest (96.00% test accuracy, 96.95% UQ accuracy)
2. **High-Security Environments**: Use Hybrid (LOF + RF) for maximum recall (93.03%)
3. **Low False Alarm Priority**: Use Random Forest (96.81% precision on balanced UQ)
4. **Real-Time Processing**: Flask batch mode processes 500 packets in <2 seconds

### Performance Benchmarks

**Test Set (27,726 packets - Balanced):**
- Random Forest: 96.00% accuracy, 95.46% recall, 3.45% FPR

**UQ Dataset (10,055 flows - 2% attacks, Imbalanced):**
- Random Forest: 96.95% accuracy, 90.55% recall, 67.66% precision
- Catches 182/201 attacks (only 19 missed)
- 0.67% false alarm rate on normal traffic

**UQ Dataset (402 flows - Balanced):**
- Random Forest: 93.78% accuracy, 90.55% recall, 96.81% precision
- Hybrid (LOF + RF): 86.57% accuracy, 93.03% recall, 82.38% precision

### Testing

All changes tested with:
- Flask web application (manual testing)
- CLI real-time detection script
- UQ dataset validation script
- Model loading and prediction tests

### Contributors

- Thallapally Nimisha (CS22B1082) - All changes

---

## [2.0.0] - 2025-11-06

### Initial Production Release

- 13 ML models (5 supervised + 4 unsupervised + 4 hybrid)
- Flask web application
- Batch analysis and real-time detection
- Comprehensive visualizations
- Documentation and deliverables

---

## Version History

- **2.1.0** (2025-11-07): Real-time batch processing, hybrid on-the-fly, UQ validation
- **2.0.0** (2025-11-06): Initial production release with web interface
- **1.0.0**: CLI-only version with ML models
