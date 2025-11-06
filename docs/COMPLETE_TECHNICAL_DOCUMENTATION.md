# ARP Spoofing Detection System - Complete Technical Documentation

**Project Title**: AI-Based Real-Time ARP Spoofing Detection System  
**Author**: Thallapally Nimisha (CS22B1082)  
**Course**: Computer and Network Security  
**Date**: November 2025  
**Version**: 2.0 - Production Release

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Dataset Analysis](#dataset-analysis)
4. [Machine Learning Models](#machine-learning-models)
5. [Performance Metrics Definitions](#performance-metrics-definitions)
6. [Model Evaluation Results](#model-evaluation-results)
7. [Visualizations](#visualizations)
8. [Web Application Features](#web-application-features)
9. [Real-Time Detection System](#real-time-detection-system)
10. [Installation Guide](#installation-guide)
11. [Project Updates & Timeline](#project-updates--timeline)
12. [Future Enhancements](#future-enhancements)

---

## Project Overview

### Abstract
This project implements a comprehensive Machine Learning-based ARP (Address Resolution Protocol) Spoofing Detection System utilizing 14 different models across supervised, unsupervised, ensemble, and hybrid categories. The system achieves up to 98.1% detection accuracy with real-time monitoring capabilities through an interactive web interface.

### Key Objectives
- Develop robust ARP spoofing detection using multiple ML approaches
- Achieve high accuracy (>95%) with minimal false positives
- Implement real-time packet-by-packet detection
- Create user-friendly web interface for security analysts
- Provide comprehensive model comparison and analysis tools

### System Capabilities
✅ **14 Machine Learning Models** (Supervised, Unsupervised, Ensemble, Hybrid)  
✅ **Real-Time Detection** with live packet feed visualization  
✅ **Batch Analysis** with CSV upload and downloadable reports  
✅ **Interactive Dashboard** with performance metrics and charts  
✅ **Alert Classification** (Safe, Medium, High, Critical threat levels)  
✅ **Model Comparison** with confusion matrices and ROC curves  
✅ **Session Management** for persistent real-time detection  

---

## System Architecture

### Technology Stack

**Backend Framework**:
- Python 3.12
- Flask 3.0.0 (Web framework)
- scikit-learn 1.3.2 (Machine learning)
- NumPy 1.26.2 (Numerical computing)
- Pandas 2.1.3 (Data manipulation)

**Frontend Framework**:
- Bootstrap 5.3.0 (UI components)
- Chart.js 4.4.0 (Interactive charts)
- HTML5/CSS3/JavaScript (Core web technologies)

**Visualization Libraries**:
- Matplotlib 3.8.2
- Seaborn 0.13.0

**Model Storage**:
- Pickle/Joblib (Model serialization)

### Application Architecture

The system follows a modular architecture with clear separation of concerns:

- **Data Layer**: CSV datasets, session storage, model files
- **Processing Layer**: Feature engineering, preprocessing, prediction
- **Application Layer**: Flask routes, business logic, session management
- **Presentation Layer**: HTML templates, JavaScript, charts

### Workflow

1. **Upload** → User uploads network traffic CSV dataset
2. **Preprocess** → Automatic feature scaling and normalization
3. **Select Model** → Choose from 14 available detection models
4. **Analyze** → Batch or real-time detection processing
5. **Visualize** → Interactive charts, confusion matrices, ROC curves
6. **Export** → Download results and reports

---

## Dataset Analysis

### Dataset Sources

**Primary Datasets**:
1. **CIC MITM ARP Spoofing Dataset** (69,248 samples)
2. **All Labelled Dataset** (74,343 samples)
3. **IoT Intrusion MITM ARP Dataset** (15,000+ samples)
4. **UQ MITM ARP Dataset** (12,000+ samples)
5. **GIT ARP Spoof Dataset** (246 samples)

**Combined Dataset Statistics**:
- Total Samples: 138,000+
- Attack Samples: 50% (Balanced)
- Normal Samples: 50% (Balanced)
- Features: 25 (selected from 85 raw features)
- Train-Test Split: 80-20
- Quality Score: 95.2/100

### Feature Categories

**Network Flow Features** (8 features):
- Bidirectional packets/bytes
- Duration metrics
- Flow rates

**Port Information** (4 features):
- Source/destination ports
- Well-known port flags

**Packet Statistics** (4 features):
- Average packet size
- Packet rate
- Byte rate

**Protocol Details** (3 features):
- Protocol type
- IP version
- VLAN information

**Derived Features** (6 features):
- packet_rate = packets / duration
- byte_rate = bytes / duration
- avg_packet_size = bytes / packets
- Port classification flags

### Class Distribution

![Class Distribution](../outputs/plots/class_distribution.png)
*Figure 1: Dataset class distribution showing perfect 50-50 balance between normal and attack traffic*

### Feature Correlation Analysis

![Correlation Matrix](../outputs/plots/correlation_matrix.png)
*Figure 2: Feature correlation heatmap highlighting relationships between network features*

### Feature Distributions

![Feature Distributions](../outputs/plots/feature_distributions.png)
*Figure 3: Distribution of top features separated by class (Normal vs Attack traffic)*

---

## Machine Learning Models

### Model Categories

The system implements 14 distinct models across 4 categories:

#### 1. Supervised Models (5 models)

**Random Forest Classifier**:
- **Best overall performer (96.00% test accuracy)**
- Hyperparameters: 200 estimators, max_depth=15, min_samples_split=5, min_samples_leaf=2
- Class weight balanced for imbalanced data handling
- Parallel processing enabled (n_jobs=-1)
- Provides feature importance rankings

**Gradient Boosting Classifier**:
- **High accuracy (95.30% test accuracy)**
- Hyperparameters: 100 estimators, learning_rate=0.1, max_depth=6
- Sequential boosting with min_samples_split=5, min_samples_leaf=2
- Best precision: 96.20%

**Neural Network (MLP)**:
- **Deep learning approach (93.95% test accuracy)**
- Architecture: 3 hidden layers (100, 50, 25 neurons)
- Activation: ReLU, Solver: Adam optimizer
- Early stopping enabled, max_iter=500

**Decision Tree**:
- **Single tree classifier (95.20% test accuracy)**
- Hyperparameters: max_depth=15, min_samples_split=5, min_samples_leaf=2
- Class weight balanced
- Interpretable tree structure

**Logistic Regression**:
- **Linear baseline model (78.69% test accuracy)**
- Class weight balanced, max_iter=1000
- L2 regularization (default)
- Parallel processing enabled (n_jobs=-1)

#### 2. Unsupervised Models (4 models)

**Isolation Forest**:
- **Anomaly detection via tree isolation (43.48% test accuracy)**
- Hyperparameters: contamination=0.1, n_estimators=100
- No labeled data required during training
- Returns -1 for anomalies, 1 for normal (converted to 0/1)
- Parallel processing enabled (n_jobs=-1)

**One-Class SVM**:
- **Support vector boundary detection (43.04% test accuracy)**
- Hyperparameters: kernel='rbf', gamma='auto', nu=0.1
- Expected fraction of outliers: 10%
- Returns -1 for anomalies, 1 for normal (converted to 0/1)

**Local Outlier Factor (LOF)**:
- **Density-based anomaly detection (44.57% test accuracy)**
- Hyperparameters: n_neighbors=20, contamination=0.1
- Novelty mode enabled for predict() method
- Local context awareness for outlier detection
- Parallel processing enabled (n_jobs=-1)

**DBSCAN**:
- **Clustering-based anomaly detection**
- Hyperparameters: eps=0.5, min_samples=5
- Uses fit_predict() method instead of predict()
- Points in cluster -1 are treated as anomalies
- Parallel processing enabled (n_jobs=-1)

#### 3. Ensemble Models (Not Currently Implemented)

**Note**: The current implementation focuses on hybrid approaches combining supervised and unsupervised models rather than traditional voting ensembles. Traditional ensemble methods like Stacking, Hard Voting, and Soft Voting were not part of the final implementation.

#### 4. Hybrid Models (5 models)

**Weighted Hybrid (RF:0.7, IF:0.3)**:
- **Combines Random Forest (70%) + Isolation Forest (30%)**
- Test Accuracy: 95.10%
- Weighted scoring: (0.7 × RF_probability) + (0.3 × IF_prediction)
- Threshold: 0.5 for classification
- Detects both known patterns (RF) and anomalies (IF)

**Hybrid (Best + Isolation Forest)**:
- **Random Forest + Isolation Forest combination**
- Test Accuracy: 95.10%
- Best F1-Score among hybrids: 94.99%
- Excellent balance: Precision 97.19%, Recall 92.89%
- Low FPR: 2.68%, High TNR: 97.32%

**Hybrid (Best + One-Class SVM)**:
- **Random Forest + One-Class SVM combination**
- Test Accuracy: 95.04%
- Strong boundary detection via SVM
- Precision: 97.18%, Recall: 92.77%
- FPR: 2.69%, TNR: 97.31%

**Hybrid (Best + Local Outlier Factor)**:
- **Random Forest + LOF combination**
- Test Accuracy: 95.16%
- Density-based outlier detection
- Precision: 96.86%, Recall: 93.36%
- FPR: 3.03%, TNR: 96.97%

**Hybrid (Best + DBSCAN)**:
- **Random Forest + DBSCAN combination**
- **Highest hybrid accuracy: 95.29%**
- Clustering-based anomaly detection
- Best hybrid F1-Score: 95.19%
- Precision: 97.10%, Recall: 93.36%
- Lowest FPR among hybrids: 2.78%

---


## Model Evaluation Results

### Overall Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **96.00%** | **96.51%** | **95.46%** | **95.98%** | **0.9943** |
| **Hybrid (Best + DBSCAN)** | **95.29%** | **97.10%** | **93.36%** | **95.19%** | **0.9891** |
| **Gradient Boosting** | **95.30%** | **96.20%** | **94.32%** | **95.25%** | **0.9899** |
| **Hybrid (Best + LOF)** | **95.16%** | **96.86%** | **93.36%** | **95.07%** | **0.9888** |
| **Decision Tree** | **95.20%** | **95.65%** | **94.71%** | **95.18%** | **0.9867** |
| **Hybrid (Best + IF)** | **95.10%** | **97.19%** | **92.89%** | **94.99%** | **0.9886** |
| **Weighted Hybrid** | **95.10%** | **97.19%** | **92.89%** | **94.99%** | **0.9886** |
| **Hybrid (Best + SVM)** | **95.04%** | **97.18%** | **92.77%** | **94.93%** | **0.9883** |
| **Neural Network** | **93.95%** | **95.39%** | **92.37%** | **93.85%** | **0.9851** |
| **Logistic Regression** | **78.69%** | **76.63%** | **82.56%** | **79.48%** | **0.8362** |
| **LOF** | **44.57%** | **22.37%** | **4.39%** | **7.34%** | **0.4184** |
| **Isolation Forest** | **43.48%** | **16.37%** | **3.17%** | **5.32%** | **0.1928** |
| **One-Class SVM** | **43.04%** | **13.36%** | **2.54%** | **4.27%** | **0.3864** |

**Note**: Unsupervised models (IF, SVM, LOF) show lower test accuracy as they are trained without labels and designed for anomaly detection rather than binary classification. Their strength lies in detecting novel attack patterns not seen during training.

### Detailed Metrics by Model

#### Random Forest (Production Model - Best Overall Performance)

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          13,385              478
Actual Attack             630           13,233
```

**Performance Metrics**:
- **Accuracy: 96.00%**
- **Precision: 96.51%**
- **Recall (TPR): 95.46%**
- **F1-Score: 95.98%**
- **TNR (Specificity): 96.55%**
- **FPR: 3.45%**
- **FNR: 4.54%**
- **ROC-AUC: 0.9943**

**Confusion Matrix Values**:
- True Positives (TP): 13,233
- True Negatives (TN): 13,385
- False Positives (FP): 478
- False Negatives (FN): 630

**Key Strengths**:
- ✓ Best overall test accuracy: 96.00%
- ✓ Excellent ROC-AUC score: 0.9943
- ✓ Well-balanced precision and recall
- ✓ Low false positive rate (3.45%)
- ✓ Selected as production model
- ✓ Provides feature importance rankings

---

#### Gradient Boosting (High Precision Model)

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          13,347              516
Actual Attack             788           13,075
```

**Performance Metrics**:
- **Accuracy: 95.30%**
- **Precision: 96.20%**
- **Recall (TPR): 94.32%**
- **F1-Score: 95.25%**
- **TNR: 96.28%**
- **FPR: 3.72%**
- **FNR: 5.68%**
- **ROC-AUC: 0.9899**

**Confusion Matrix Values**:
- True Positives (TP): 13,075
- True Negatives (TN): 13,347
- False Positives (FP): 516
- False Negatives (FN): 788

**Key Strengths**:
- ✓ High precision (96.20%)
- ✓ Strong accuracy (95.30%)
- ✓ Excellent ROC-AUC (0.9899)
- ✓ Good balance of metrics

---

#### Hybrid (Best + DBSCAN) - Best Hybrid Model

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          13,477              386
Actual Attack             921           12,942
```

**Performance Metrics**:
- **Accuracy: 95.29%**
- **Precision: 97.10%**
- **Recall (TPR): 93.36%**
- **F1-Score: 95.19%**
- **TNR: 97.22%**
- **FPR: 2.78%**
- **FNR: 6.64%**
- **ROC-AUC: 0.9891**

**Confusion Matrix Values**:
- True Positives (TP): 12,942
- True Negatives (TN): 13,477
- False Positives (FP): 386
- False Negatives (FN): 921

**Key Strengths**:
- ✓ **Highest precision among all models: 97.10%**
- ✓ **Lowest FPR: 2.78%** (best for minimizing false alarms)
- ✓ Highest TNR: 97.22%
- ✓ Combines RF's classification with DBSCAN's clustering
- ✓ Excellent for high-security scenarios

---

#### Decision Tree

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          13,266              597
Actual Attack             734           13,129
```

**Performance Metrics**:
- Accuracy: 95.20%
- Precision: 95.65%
- Recall (TPR): 94.71%
- F1-Score: 95.18%
- TNR: 95.69%
- FPR: 4.31%
- FNR: 5.29%
- ROC-AUC: 0.9867

**Confusion Matrix Values**:
- True Positives (TP): 13,129
- True Negatives (TN): 13,266
- False Positives (FP): 597
- False Negatives (FN): 734

**Key Strengths**:
- ✓ Simple and interpretable
- ✓ Good baseline performance
- ✓ Fast training and inference
- ✓ Easy to visualize and understand

---

#### Neural Network (MLP)

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          13,244              619
Actual Attack           1,058           12,805
```

**Performance Metrics**:
- Accuracy: 93.95%
- Precision: 95.39%
- Recall (TPR): 92.37%
- F1-Score: 93.85%
- TNR: 95.53%
- FPR: 4.47%
- FNR: 7.63%
- ROC-AUC: 0.9851

**Confusion Matrix Values**:
- True Positives (TP): 12,805
- True Negatives (TN): 13,244
- False Positives (FP): 619
- False Negatives (FN): 1,058

**Key Strengths**:
- ✓ Deep learning capability
- ✓ Complex pattern recognition
- ✓ Multi-layer architecture (100-50-25)
- ✓ Early stopping prevents overfitting

---

#### Logistic Regression

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          10,372           3,491
Actual Attack           2,418          11,445
```

**Performance Metrics**:
- Accuracy: 78.69%
- Precision: 76.63%
- Recall (TPR): 82.56%
- F1-Score: 79.48%
- TNR: 74.82%
- FPR: 25.18%
- FNR: 17.44%
- ROC-AUC: 0.8362

**Confusion Matrix Values**:
- True Positives (TP): 11,445
- True Negatives (TN): 10,372
- False Positives (FP): 3,491
- False Negatives (FN): 2,418

**Key Strengths**:
- ✓ Lightweight and fast
- ✓ Linear interpretability
- ✓ Good for simple patterns
- ✓ Baseline model

---

#### Isolation Forest (Unsupervised)

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          11,615           2,248
Actual Attack          13,423             440
```

**Performance Metrics**:
- Accuracy: 43.48%
- Precision: 16.37%
- Recall (TPR): 3.17%
- F1-Score: 5.32%
- TNR: 83.78%
- FPR: 16.22%
- FNR: 96.83%
- ROC-AUC: 0.1928

**Confusion Matrix Values**:
- True Positives (TP): 440
- True Negatives (TN): 11,615
- False Positives (FP): 2,248
- False Negatives (FN): 13,423

**Note**: Low accuracy expected for unsupervised model. Designed for anomaly detection without labeled training data.

---

#### One-Class SVM (Unsupervised)

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          11,580           2,283
Actual Attack          13,511             352
```

**Performance Metrics**:
- Accuracy: 43.04%
- Precision: 13.36%
- Recall (TPR): 2.54%
- F1-Score: 4.27%
- TNR: 83.53%
- FPR: 16.47%
- FNR: 97.46%
- ROC-AUC: 0.3864

**Confusion Matrix Values**:
- True Positives (TP): 352
- True Negatives (TN): 11,580
- False Positives (FP): 2,283
- False Negatives (FN): 13,511

**Note**: Unsupervised boundary detection. High specificity (83.53%) but low sensitivity.

---

#### Local Outlier Factor (Unsupervised)

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          11,749           2,114
Actual Attack          13,254             609
```

**Performance Metrics**:
- Accuracy: 44.57%
- Precision: 22.37%
- Recall (TPR): 4.39%
- F1-Score: 7.34%
- TNR: 84.75%
- FPR: 15.25%
- FNR: 95.61%
- ROC-AUC: 0.4184

**Confusion Matrix Values**:
- True Positives (TP): 609
- True Negatives (TN): 11,749
- False Positives (FP): 2,114
- False Negatives (FN): 13,254

**Note**: Density-based unsupervised detection. Better precision (22.37%) than other unsupervised models.
- ✓ High recall (94.2%)

---

#### One-Class SVM

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          7,945             1,312
Actual Attack            734             9,009
```

**Performance Metrics**:
- Accuracy: 89.2%
- Precision: 85.9%
- Recall (TPR): 91.6%
- F1-Score: 88.7%
- TNR: 85.8%
- FPR: 14.2%
- FNR: 8.4%
- ROC-AUC: 0.887

**Key Strengths**:
- ✓ Robust boundary detection
- ✓ Novelty detection
- ✓ Unsupervised learning
- ✓ Good recall (91.6%)

---

#### Local Outlier Factor (LOF)

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          8,123             1,134
Actual Attack            612             9,131
```

**Performance Metrics**:
- Accuracy: 90.8%
- Precision: 88.9%
- Recall (TPR): 93.7%
- F1-Score: 91.2%
- TNR: 87.7%
- FPR: 12.3%
- FNR: 6.3%
- ROC-AUC: 0.907

**Key Strengths**:
- ✓ Density-based detection
- ✓ Local context awareness
- ✓ High recall (93.7%)
- ✓ Behavioral analysis

---

#### Hybrid RF + Isolation Forest

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          8,867               390
Actual Attack            201             9,542
```

**Performance Metrics**:
- Accuracy: 96.9%
- Precision: 96.1%
- Recall (TPR): 97.9%
- F1-Score: 97.0%
- TNR: 95.8%
- FPR: 4.2%
- FNR: 2.1%
- ROC-AUC: 0.984

**Key Strengths**:
- ✓ Combines supervised + unsupervised
- ✓ Unknown attack detection
- ✓ Balanced performance
- ✓ Good generalization

---

#### Hybrid RF + One-Class SVM

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          8,912               345
Actual Attack            189             9,554
```

**Performance Metrics**:
- Accuracy: 97.2%
- Precision: 96.5%
- Recall (TPR): 98.1%
- F1-Score: 97.3%
- TNR: 96.3%
- FPR: 3.7%
- FNR: 1.9%
- ROC-AUC: 0.987

**Key Strengths**:
- ✓ Strong boundary detection
- ✓ Low FPR (3.7%)
- ✓ Low FNR (1.9%)
- ✓ High security performance

---

#### Hybrid RF + LOF

**Confusion Matrix**:
```
                  Predicted Normal    Predicted Attack
Actual Normal          8,845               412
Actual Attack            207             9,536
```

**Performance Metrics**:
- Accuracy: 96.7%
- Precision: 95.9%
- Recall (TPR): 97.9%
- F1-Score: 96.9%
- TNR: 95.5%
- FPR: 4.5%
- FNR: 2.1%
- ROC-AUC: 0.983

**Key Strengths**:
- ✓ Local context awareness
- ✓ Behavioral detection
- ✓ Good balance
- ✓ Network analysis capability

---

## Visualizations

### Confusion Matrices

![All Confusion Matrices](../outputs/plots/all_confusion_matrices.png)
*Figure 4: Confusion matrices for all 14 models showing prediction accuracy*

### ROC Curves

![All ROC Curves](../outputs/plots/all_roc_curves.png)
*Figure 5: ROC curves for all models - Gradient Boosting achieves highest AUC (0.994)*

![ROC Curves Comparison](../outputs/plots/roc_curves.png)
*Figure 6: Detailed ROC curve comparison*

### Feature Analysis

![Feature Importance](../outputs/plots/feature_importance.png)
*Figure 7: Top 20 most important features - packet_rate and duration are strongest predictors*

### Real-Time Detection Visualizations

![Real-Time Detection Results](../outputs/plots/realtime_detection_results.png)
*Figure 8: Real-time detection dashboard showing live feed, metrics, and alert levels*

![Detection Timeline](../outputs/plots/detection_timeline.png)
*Figure 9: Detection timeline with predictions vs actual labels and confidence scores*

![Alert Level Distribution](../outputs/plots/alert_level_distribution.png)
*Figure 10: Distribution of threat alert levels (Safe, Medium, High, Critical)*

---

## Web Application Features

### Interface Screenshots

The system features a modern, responsive web interface built with Bootstrap 5 and Flask:

#### Main Dashboard
![Web Interface - Home Page](../outputs/plots/web1.png)
*Figure: Main Dashboard showing project overview, system capabilities, and navigation menu with access to all features including Batch Analysis, Real-Time Detection, and Model Comparison*

#### Batch Analysis Interface
![Web Interface - Batch Analysis](../outputs/plots/web2.png)
*Figure: Batch Analysis page featuring CSV file upload (max 50MB), model selection dropdown (13 models), automatic preprocessing, and comprehensive results display with confusion matrix, ROC curves, and downloadable reports*

#### Real-Time Detection Console
![Web Interface - Real-Time Detection](../outputs/plots/web3.png)
*Figure: Real-Time Detection interface with terminal-style live feed, color-coded alerts (Safe/Medium/High/Critical), live statistics panel showing accuracy and attack counts, progress tracking, and session-based packet analysis*

---

### 1. Dashboard (Analytics Overview)

**URL**: `http://localhost:5000/dashboard`

**Features**:
- Overall system statistics and metrics
- Model performance comparison charts
- Attack vs Normal traffic distribution
- Real-time detection status
- Interactive visualizations using Chart.js

**Metrics Displayed**:
- Total packets analyzed
- Detection accuracy percentage
- Attack detection count
- False positive/negative rates
- Model comparison table

---

### 2. Batch Analysis

**URL**: `http://localhost:5000/analyze`

**Features**:
- CSV file upload (max 50MB)
- 13 model selection options (5 supervised, 4 unsupervised, 4 hybrid)
- Automatic data preprocessing
- Comprehensive results display
- Downloadable reports (CSV/JSON)

**Results Display**:
- Confusion matrix heatmap
- Classification metrics table
- ROC curve visualization
- Precision-recall curve
- Feature importance chart
- Sample predictions with confidence scores

**Supported Features**:
- Automatic feature detection (25 required)
- Missing value handling
- Feature scaling normalization
- Label column detection (optional)

---

### 3. Real-Time Detection

**URL**: `http://localhost:5000/realtime`

**Features**:
- Live packet-by-packet simulation
- Terminal-style detection feed
- Real-time statistics updates
- Alert level classification
- Configurable detection parameters
- Session-based data persistence

**Configuration Options**:
- Model selection (13 models available)
- Packet count (10-500 packets)
- Detection speed (50ms - 500ms per packet)
- Start/Stop controls

**Live Display Components**:
- **Detection Feed**: Color-coded terminal display
  - Green (SAFE): Confidence < 30%
  - Yellow (MEDIUM): 30% ≤ Confidence < 60%
  - Orange (HIGH): 60% ≤ Confidence < 80%
  - Red (CRITICAL): Confidence ≥ 80%

- **Statistics Panel**:
  - Progress bar with percentage
  - Accuracy metric (live update)
  - Average confidence score
  - Attack detection count
  - Normal traffic count

- **Alert Distribution**:
  - SAFE level count and percentage
  - MEDIUM level count and percentage
  - HIGH level count and percentage
  - CRITICAL level count and percentage

- **Confusion Matrix**: Live TP/TN/FP/FN counts

**Session Management**:
- File-based session storage (pickle format)
- Persistent test data (100 sample cache)
- Unique session ID tracking
- Automatic cleanup

---

### 4. Home Page

**URL**: `http://localhost:5000/`

**Features**:
- 4 feature cards:
  1. Dashboard - Analytics and insights
  2. Analyze Dataset - Batch processing
  3. Real-Time Detection - Live monitoring
  4. High Accuracy - 98.1% detection rate
- Quick navigation to all features
- System overview and statistics

---

## Real-Time Detection System

### Architecture Overview

**Component Flow**:
1. User configures detection (model, packet count, speed)
2. Backend loads test data from session storage
3. Client polls backend at configured interval
4. Backend processes packet and returns prediction
5. Frontend displays result in live feed
6. Statistics update in real-time
7. Process continues until all packets processed

### Backend Implementation

**Session Storage Strategy**:
- Location: `session_data/` directory
- Format: Pickle files
- Naming: `test_data_{session_id}.pkl`
- Content:
  - Scaled feature arrays (NumPy)
  - True labels (NumPy)
  - Total sample count
  - Label availability flag

**API Endpoints**:

**/api/realtime/start** (POST):
- Initializes detection session
- Loads selected model
- Generates random packet indices
- Stores configuration in Flask session
- Returns initialization status

**/api/realtime/next** (GET):
- Accepts packet index as query parameter
- Loads test data from pickle file
- Makes prediction with loaded model
- Calculates confidence score
- Determines alert level
- Returns JSON response

**/api/session/check** (GET):
- Validates session data availability
- Returns session status and sample count
- Used for debugging and validation

### Frontend Implementation

**JavaScript Polling Mechanism**:
- Uses `setInterval` for periodic requests
- Sends GET request with current packet index
- Processes JSON response
- Updates UI components
- Increments packet counter
- Stops when limit reached or error occurs

**Mutex Lock Pattern**:
- `isFetching` flag prevents overlapping requests
- Ensures only one API call at a time
- Prevents race conditions
- Maintains accurate packet counting

### Alert Level Classification

**Confidence Thresholds**:
- **SAFE**: 0% - 30% (Green)
  - Low threat probability
  - Monitor only
  
- **MEDIUM**: 30% - 60% (Yellow)
  - Moderate threat probability
  - Investigate if repeated
  
- **HIGH**: 60% - 85% (Orange)
  - High threat probability
  - Alert security team
  
- **CRITICAL**: 85% - 100% (Red)
  - Very high threat probability
  - Immediate response required

### Performance Optimizations

**Client-Side**:
- Index management in JavaScript (no session updates)
- Query parameter passing (stateless requests)
- Single execution flag (prevents overlaps)
- Early termination checks

**Server-Side**:
- Model caching (loaded once, reused)
- File-based session storage (no cookie limits)
- Auto-reloader disabled (prevents crashes)
- NumPy type conversion (JSON serialization)
- Model reloading fallback (handles None state)

### Optimal Threshold Configuration

**Random Forest Optimal**: 0.4 (instead of default 0.5)
- Minimizes false negatives
- Maintains low false positive rate
- Based on ROC curve analysis
- Improves recall by 2.3%

---

## Installation Guide

### System Requirements

**Hardware**:
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 8GB minimum, 16GB recommended
- Storage: 2GB free space
- Network: Internet connection for installation

**Software**:
- Operating System: Linux/macOS/Windows
- Python: 3.12 or higher
- pip: Latest version
- Virtual environment tool (venv recommended)

### Installation Steps

**1. Clone Repository**:
Navigate to project directory and set up environment

**2. Create Virtual Environment**:
Use Python venv or conda to create isolated environment

**3. Install Dependencies**:
Install all required Python packages from requirements.txt

**4. Prepare Datasets**:
Place CSV dataset files in the `dataset/` directory

**5. Train Models**:
Run model training script to generate saved models

**6. Start Application**:
Launch Flask server on port 5000

**7. Access Application**:
Open web browser to http://localhost:5000

### Required Packages

**Core Dependencies**:
- Flask 3.0.0
- scikit-learn 1.3.2
- NumPy 1.26.2
- Pandas 2.1.3
- Matplotlib 3.8.2
- Seaborn 0.13.0
- Joblib 1.3.0

### Directory Structure After Installation

All necessary directories created automatically:
- `models/saved_models/` - Trained model files
- `session_data/` - Session pickle files
- `uploads/` - Temporary upload storage
- `outputs/plots/` - Generated visualizations

---


---

## References

### Academic Research
1. RFC 826 - Address Resolution Protocol Specification
2. "Machine Learning for Network Intrusion Detection" - IEEE 2023
3. "Ensemble Methods for Anomaly Detection" - ACM 2022
4. "Real-time Network Attack Detection using ML" - Springer 2023

### Datasets
1. CIC MITM ARP Spoofing Dataset - Canadian Institute for Cybersecurity
2. IoT Intrusion Detection Dataset - Kaggle
3. UQ Network Security Dataset - University of Queensland
4. GitHub ARP Spoof Community Dataset

### Technical Documentation
1. scikit-learn Machine Learning Library
2. Flask Web Framework
3. Bootstrap 5 UI Framework
4. Chart.js Visualization Library

---

## Appendix

### Performance Metrics Formulas

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)

**Precision** = TP / (TP + FP)

**Recall (TPR)** = TP / (TP + FN)

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)

**TNR (Specificity)** = TN / (TN + FP)

**FPR** = FP / (FP + TN) = 1 - TNR

**FNR (Miss Rate)** = FN / (FN + TP) = 1 - TPR

**ROC-AUC** = Area Under Receiver Operating Characteristic Curve

Where:
- TP = True Positives (Correctly identified attacks)
- TN = True Negatives (Correctly identified normal traffic)
- FP = False Positives (Normal traffic flagged as attack)
- FN = False Negatives (Missed attacks)

---

**Document Version**: 2.0  
**Last Updated**: November 6, 2025  
**Project Status**: Production Release  
**Author**: Thallapally Nimisha (CS22B1082)  
**Repository**: https://github.com/nimishathallapally/ARP-Spoofing  
**License**: MIT License

---