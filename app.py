import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from datetime import datetime
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['SESSION_FOLDER'] = 'session_data'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variable to store loaded model
current_model = None
current_scaler = None
current_feature_names = None

# Define available models with metadata
AVAILABLE_MODELS = {
    # Supervised Models
    'random_forest': {
        'name': 'Random Forest',
        'path': 'models/saved_models/random_forest.pkl',
        'description': 'Bagging ensemble of decision trees',
        'icon': 'tree-fill',
        'category': 'Supervised'
    },
    'decision_tree': {
        'name': 'Decision Tree',
        'path': 'models/saved_models/decision_tree.pkl',
        'description': 'Single tree classifier',
        'icon': 'diagram-3-fill',
        'category': 'Supervised'
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'path': 'models/saved_models/gradient_boosting.pkl',
        'description': 'Boosting ensemble - Sequential error correction',
        'icon': 'graph-up-arrow',
        'category': 'Supervised'
    },
    'neural_network': {
        'name': 'Neural Network (MLP)',
        'path': 'models/saved_models/neural_network.pkl',
        'description': 'Multi-layer perceptron',
        'icon': 'cpu-fill',
        'category': 'Supervised'
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'path': 'models/saved_models/logistic_regression.pkl',
        'description': 'Linear classifier - Lightweight',
        'icon': 'calculator-fill',
        'category': 'Supervised'
    },
    
    # Unsupervised Models
    'isolation_forest': {
        'name': 'Isolation Forest',
        'path': 'models/saved_models/isolation_forest.pkl',
        'description': 'Unsupervised anomaly detector',
        'icon': 'shield-fill-exclamation',
        'category': 'Unsupervised'
    },
    'one_class_svm': {
        'name': 'One-Class SVM',
        'path': 'models/saved_models/one-class_svm.pkl',
        'description': 'Support Vector Machine for outliers',
        'icon': 'bullseye',
        'category': 'Unsupervised'
    },
    'local_outlier_factor': {
        'name': 'Local Outlier Factor',
        'path': 'models/saved_models/local_outlier_factor.pkl',
        'description': 'LOF density-based detector',
        'icon': 'geo-alt-fill',
        'category': 'Unsupervised'
    },
    
    # Ensemble Models
    'hard_voting': {
        'name': 'Hard Voting',
        'path': 'models/saved_models/hard_voting.pkl',
        'description': 'Majority voting from multiple classifiers',
        'icon': 'people-fill',
        'category': 'Ensemble'
    },
    'soft_voting': {
        'name': 'Soft Voting',
        'path': 'models/saved_models/soft_voting.pkl',
        'description': 'Probability-weighted voting',
        'icon': 'bar-chart-fill',
        'category': 'Ensemble'
    },
    'stacking': {
        'name': 'Stacking',
        'path': 'models/saved_models/stacking.pkl',
        'description': 'Meta-learner stacking multiple models',
        'icon': 'layers-fill',
        'category': 'Ensemble'
    },
    
    # Hybrid Models (0.7 RF + 0.3 Unsupervised)
    # These are computed on-the-fly, not loaded from pickle
    'hybrid_rf_if': {
        'name': 'Hybrid RF+IF',
        'supervised': 'random_forest',
        'unsupervised': 'isolation_forest',
        'weights': (0.7, 0.3),
        'description': '70% Random Forest + 30% Isolation Forest',
        'icon': 'plus-circle-fill',
        'category': 'Hybrid'
    },
    'hybrid_rf_svm': {
        'name': 'Hybrid RF+SVM',
        'supervised': 'random_forest',
        'unsupervised': 'one_class_svm',
        'weights': (0.7, 0.3),
        'description': '70% Random Forest + 30% One-Class SVM',
        'icon': 'intersect',
        'category': 'Hybrid'
    },
    'hybrid_rf_lof': {
        'name': 'Hybrid RF+LOF',
        'supervised': 'random_forest',
        'unsupervised': 'lof',
        'weights': (0.7, 0.3),
        'description': '70% Random Forest + 30% Local Outlier Factor',
        'icon': 'distribute-vertical',
        'category': 'Hybrid'
    }
}

def load_specific_model(model_key):
    """Load a specific model from the saved models"""
    global current_model, current_scaler, current_feature_names
    
    try:
        model_info = AVAILABLE_MODELS.get(model_key)
        if not model_info:
            return False, f"Model '{model_key}' not found"
        
        # Check if it's a hybrid model
        if 'supervised' in model_info:
            # Hybrid model - load both components
            sup_key = model_info['supervised']
            unsup_key = model_info['unsupervised']
            weights = model_info['weights']
            
            # Load supervised model
            sup_info = AVAILABLE_MODELS.get(sup_key)
            if not sup_info or not os.path.exists(sup_info['path']):
                return False, f"Supervised model '{sup_key}' not found"
            
            with open(sup_info['path'], 'rb') as f:
                sup_package = pickle.load(f)
            
            # Load unsupervised model
            unsup_info = AVAILABLE_MODELS.get(unsup_key)
            if not unsup_info or not os.path.exists(unsup_info['path']):
                return False, f"Unsupervised model '{unsup_key}' not found"
            
            with open(unsup_info['path'], 'rb') as f:
                unsup_package = pickle.load(f)
            
            # Create hybrid model wrapper
            current_model = {
                'type': 'hybrid',
                'supervised': sup_package['model'],
                'unsupervised': unsup_package['model'],
                'weights': weights
            }
            current_scaler = sup_package['scaler']
            current_feature_names = sup_package['feature_names']
            
            print(f"✓ Loaded hybrid model: {model_info['name']}")
            print(f"  Supervised: {sup_info['name']}")
            print(f"  Unsupervised: {unsup_info['name']}")
            print(f"  Weights: {weights}")
            
        else:
            # Regular model - load from pickle
            model_path = model_info['path']
            
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
            
            # Load the packaged model
            with open(model_path, 'rb') as f:
                package = pickle.load(f)
            
            current_model = package['model']
            current_scaler = package['scaler']
            current_feature_names = package['feature_names']
            
            print(f"✓ Loaded model: {model_info['name']}")
        
        print(f"  Features: {len(current_feature_names)}")
        
        return True, model_info['name']
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error loading model: {str(e)}"

def predict_with_model(model, X):
    """Make predictions - handles both regular and hybrid models"""
    if isinstance(model, dict) and model.get('type') == 'hybrid':
        # Hybrid model prediction
        sup_model = model['supervised']
        unsup_model = model['unsupervised']
        weights = model['weights']
        
        # Get supervised predictions
        if hasattr(sup_model, 'predict_proba'):
            sup_pred = sup_model.predict_proba(X)[:, 1]
        else:
            sup_pred = sup_model.predict(X).astype(float)
        
        # Get unsupervised predictions
        unsup_pred_raw = unsup_model.predict(X)
        # Convert -1 (outlier/attack) to 1, 1 (normal) to 0
        unsup_pred = np.where(unsup_pred_raw == -1, 1, 0).astype(float)
        
        # Weighted combination
        weighted_score = (weights[0] * sup_pred) + (weights[1] * unsup_pred)
        return (weighted_score >= 0.5).astype(int)
    else:
        # Regular model prediction
        return model.predict(X)

def predict_proba_with_model(model, X):
    """Get prediction probabilities - handles both regular and hybrid models"""
    if isinstance(model, dict) and model.get('type') == 'hybrid':
        # Hybrid model probabilities
        sup_model = model['supervised']
        unsup_model = model['unsupervised']
        weights = model['weights']
        
        # Get supervised probabilities
        if hasattr(sup_model, 'predict_proba'):
            sup_proba = sup_model.predict_proba(X)[:, 1]
        else:
            sup_pred = sup_model.predict(X)
            sup_proba = np.where(sup_pred == 1, 0.8, 0.2)
        
        # Get unsupervised predictions
        unsup_pred_raw = unsup_model.predict(X)
        unsup_pred = np.where(unsup_pred_raw == -1, 1, 0).astype(float)
        
        # Weighted combination for probabilities
        weighted_proba = (weights[0] * sup_proba) + (weights[1] * unsup_pred)
        
        # Return as probability matrix
        return np.column_stack([1 - weighted_proba, weighted_proba])
    else:
        # Regular model probabilities
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            # No probabilities available, use binary predictions
            pred = model.predict(X)
            return np.column_stack([1 - pred, pred])

@app.route('/')
def index():
    """Home page"""
    # Get stats from session if available
    stats = session.get('analysis_stats', None)
    return render_template('index.html', models=AVAILABLE_MODELS, stats=stats)

@app.route('/analyze')
def analyze():
    """Analysis page with file upload"""
    return render_template('analyze.html', models=AVAILABLE_MODELS)

@app.route('/dashboard')
def dashboard():
    """Dashboard showing results"""
    stats = session.get('analysis_stats', {
        'total_flows': 0,
        'attacks_detected': 0,
        'normal_flows': 0,
        'accuracy': 0,
        'recall': 0,
        'precision': 0,
        'detections': []
    })
    
    return render_template('dashboard.html', stats=stats)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    global current_model, current_scaler, current_feature_names
    
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        model_key = request.form.get('model', 'random_forest')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Only CSV files are allowed'}), 400
        
        # Load the selected model
        success, message = load_specific_model(model_key)
        if not success:
            return jsonify({'success': False, 'error': message}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and process the CSV
        df = pd.read_csv(filepath)
        print(f"✓ Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
        
        # Prepare features
        # Check if 'Label' column exists for evaluation
        has_labels = 'Label' in df.columns or 'label' in df.columns
        
        if has_labels:
            label_col = 'Label' if 'Label' in df.columns else 'label'
            y_true_raw = df[label_col].values
            
            # Convert string labels to binary (0 or 1)
            # Common label formats: 'normal'/0 and 'arp_spoofing'/'attack'/1
            y_true = np.where(
                (y_true_raw == 'normal') | (y_true_raw == 0) | (y_true_raw == '0'),
                0,
                1
            )
            X = df.drop([label_col], axis=1)
        else:
            y_true = None
            X = df.copy()
        
        # Ensure we have the right features
        if current_feature_names:
            # Reorder columns to match training
            missing_cols = set(current_feature_names) - set(X.columns)
            extra_cols = set(X.columns) - set(current_feature_names)
            
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                # Add missing columns with zeros
                for col in missing_cols:
                    X[col] = 0
            
            if extra_cols:
                print(f"Info: Dropping extra columns: {extra_cols}")
            
            # Select and reorder columns
            X = X[current_feature_names]
        
        # Scale features
        if current_scaler:
            X_scaled = current_scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make predictions with optimal threshold
        OPTIMAL_THRESHOLD = 0.4  # Based on test results from test_uq_dataset.py
        
        # Get prediction probabilities using helper function
        try:
            y_proba_matrix = predict_proba_with_model(current_model, X_scaled)
            y_proba = y_proba_matrix[:, 1]
            # Use optimal threshold instead of default 0.5
            y_pred_binary = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
        except:
            # Fallback to regular predictions
            y_pred_binary = predict_with_model(current_model, X_scaled)
            # For unsupervised models, convert -1 to 1 (attack) and 1 to 0 (normal)
            if -1 in y_pred_binary:
                y_pred_binary = np.where(y_pred_binary == -1, 1, 0)
            y_proba = np.abs(y_pred_binary).astype(float)
        
        # Calculate metrics
        total_flows = len(y_pred_binary)
        attacks_detected = int(np.sum(y_pred_binary))
        normal_flows = total_flows - attacks_detected
        
        # Calculate accuracy metrics if ground truth is available
        if has_labels:
            accuracy = accuracy_score(y_true, y_pred_binary) * 100
            recall = recall_score(y_true, y_pred_binary) * 100
            precision = precision_score(y_true, y_pred_binary, zero_division=0) * 100
            cm = confusion_matrix(y_true, y_pred_binary)
            
            confusion_data = {
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]),
                'tp': int(cm[1, 1])
            }
        else:
            accuracy = 0
            recall = 0
            precision = 0
            confusion_data = None
        
        # Prepare detection samples (top 50 most suspicious)
        detections = []
        suspicious_indices = np.argsort(y_proba)[-50:][::-1]  # Top 50 instead of 10
        
        for idx in suspicious_indices:
            detection = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'src_ip': f"192.168.1.{np.random.randint(1, 254)}",
                'dst_ip': f"192.168.1.{np.random.randint(1, 254)}",
                'probability': float(y_proba[idx])
            }
            detections.append(detection)
        
        # Store results in session
        stats = {
            'total_flows': total_flows,
            'attacks_detected': attacks_detected,
            'normal_flows': normal_flows,
            'accuracy': round(accuracy, 2),
            'recall': round(recall, 2),
            'precision': round(precision, 2),
            'detections': detections,
            'model_name': AVAILABLE_MODELS[model_key]['name']
        }
        
        session['analysis_stats'] = stats
        session.permanent = True  # Make session permanent
        
        # Store test data in a file instead of session (to avoid cookie size limits)
        max_samples = min(100, len(X_scaled))
        sample_indices = np.random.choice(len(X_scaled), size=max_samples, replace=False)
        
        print(f"Storing {max_samples} samples for real-time detection")
        
        # Create a unique session ID if not exists
        if 'session_id' not in session:
            session['session_id'] = datetime.now().strftime('%Y%m%d%H%M%S%f')
        
        session_id = session['session_id']
        
        # Save test data to file
        test_data_file = os.path.join(app.config['SESSION_FOLDER'], f'test_data_{session_id}.pkl')
        test_data = {
            'X': X_scaled[sample_indices],
            'y': y_true[sample_indices] if has_labels else y_pred_binary[sample_indices],
            'total_samples': max_samples,
            'has_labels': has_labels
        }
        
        with open(test_data_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        print(f"Test data saved to: {test_data_file}")
        print(f"Session ID: {session_id}")
        
        # Store only the session ID in session (small!)
        session['has_test_data'] = True
        
        # Prepare response
        response = {
            'success': True,
            'stats': stats,
            'confusion_matrix': confusion_data,
            'message': f'Analysis complete using {AVAILABLE_MODELS[model_key]["name"]}'
        }
        
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/realtime')
def realtime():
    """Real-time detection demo page"""
    return render_template('realtime.html', models=AVAILABLE_MODELS)

@app.route('/api/session/check')
def check_session():
    """Debug endpoint to check session contents"""
    has_test_data = session.get('has_test_data', False)
    session_id = session.get('session_id', None)
    
    test_data_info = {}
    test_data_samples = 0
    
    if has_test_data and session_id:
        test_data_file = os.path.join(app.config['SESSION_FOLDER'], f'test_data_{session_id}.pkl')
        if os.path.exists(test_data_file):
            try:
                with open(test_data_file, 'rb') as f:
                    test_data = pickle.load(f)
                test_data_info = {
                    'total_samples': test_data['total_samples'],
                    'has_labels': test_data['has_labels']
                }
                test_data_samples = test_data['total_samples']
            except Exception as e:
                test_data_info = {'error': str(e)}
    
    return jsonify({
        'session_keys': list(session.keys()),
        'has_test_data': has_test_data,
        'session_id': session_id,
        'test_data_info': test_data_info,
        'test_data_samples': test_data_samples
    })

@app.route('/api/realtime/start', methods=['POST'])
def start_realtime_detection():
    """Start real-time detection simulation - Process all packets like detect_realtime.py"""
    global current_model, current_scaler, current_feature_names
    
    print("=== Real-time detection start request received ===")
    
    try:
        data = request.get_json()
        print(f"Request data: {data}")
        
        model_key = data.get('model', 'random_forest')
        n_packets = int(data.get('packets', 100))
        
        print(f"Model: {model_key}, Packets: {n_packets}")
        
        # Load model if needed
        success, message = load_specific_model(model_key)
        if not success:
            print(f"Model loading failed: {message}")
            return jsonify({'success': False, 'error': message}), 400
        
        print("Model loaded successfully")
        
        # Check if we have test data
        print(f"Current session keys: {list(session.keys())}")
        print(f"Session ID: {session.get('session_id', 'None')}")
        print(f"Has test data flag: {session.get('has_test_data', False)}")
        
        if not session.get('has_test_data', False) or 'session_id' not in session:
            print("No test_data flag or session_id in session")
            return jsonify({
                'success': False,
                'error': 'No test data available. Please run batch analysis first.'
            }), 400
        
        # Load test data from file
        session_id = session['session_id']
        test_data_file = os.path.join(app.config['SESSION_FOLDER'], f'test_data_{session_id}.pkl')
        
        print(f"Looking for test data file: {test_data_file}")
        
        if not os.path.exists(test_data_file):
            print(f"Test data file not found: {test_data_file}")
            return jsonify({
                'success': False,
                'error': 'Test data file not found. Please run batch analysis again.'
            }), 400
        
        # Load the test data
        try:
            with open(test_data_file, 'rb') as f:
                test_data = pickle.load(f)
            
            print(f"Test data loaded successfully")
            print(f"Total samples: {test_data['total_samples']}")
            print(f"Has labels: {test_data['has_labels']}")
        except Exception as e:
            print(f"Error loading test data: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error loading test data: {str(e)}'
            }), 500
        
        # Get test data from loaded file
        total_available = test_data['total_samples']
        
        if n_packets > total_available:
            n_packets = total_available
        
        # Select random packets
        indices = np.random.choice(total_available, size=n_packets, replace=False)
        
        X_test = test_data['X'][indices]
        y_test = test_data['y'][indices]
        
        # Process ALL packets at once (like detect_realtime.py)
        print(f"\n{'='*60}")
        print(f"Processing {n_packets} packets for real-time detection...")
        print(f"{'='*60}\n")
        
        # Make predictions on all packets
        OPTIMAL_THRESHOLD = 0.4
        
        all_results = []
        predictions = []
        confidences = []
        
        for i in range(n_packets):
            packet_features = X_test[i].reshape(1, -1)
            true_label = int(y_test[i])
            
            # Make prediction using helper function
            try:
                y_proba_matrix = predict_proba_with_model(current_model, packet_features)
                confidence = float(y_proba_matrix[0][1])  # Probability of attack
                prediction = 1 if confidence >= OPTIMAL_THRESHOLD else 0
            except:
                # Fallback to binary prediction
                y_pred = predict_with_model(current_model, packet_features)
                prediction = 1 if y_pred[0] == -1 or y_pred[0] == 1 else 0
                # For unsupervised: -1 = attack (1), 1 = normal (0)
                if y_pred[0] == -1:
                    prediction = 1
                    confidence = 0.8
                elif y_pred[0] == 1:
                    prediction = 0
                    confidence = 0.2
                else:
                    prediction = int(y_pred[0])
                    confidence = 0.8 if prediction == 1 else 0.2
            
            # Determine alert level
            if confidence < 0.3:
                alert_level = 'SAFE'
            elif confidence < 0.6:
                alert_level = 'MEDIUM'
            elif confidence < 0.8:
                alert_level = 'HIGH'
            else:
                alert_level = 'CRITICAL'
            
            result = {
                'packet_id': i + 1,
                'prediction': 'ATTACK' if prediction == 1 else 'NORMAL',
                'true_label': 'ATTACK' if true_label == 1 else 'NORMAL',
                'correct': prediction == true_label,
                'confidence': confidence,
                'alert_level': alert_level,
                'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3]
            }
            
            all_results.append(result)
            predictions.append(prediction)
            confidences.append(confidence)
        
        # Calculate comprehensive statistics
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        total = len(predictions)
        correct = sum(1 for r in all_results if r['correct'])
        accuracy = correct / total if total > 0 else 0
        
        tp = sum(1 for r in all_results if r['prediction'] == 'ATTACK' and r['true_label'] == 'ATTACK')
        tn = sum(1 for r in all_results if r['prediction'] == 'NORMAL' and r['true_label'] == 'NORMAL')
        fp = sum(1 for r in all_results if r['prediction'] == 'ATTACK' and r['true_label'] == 'NORMAL')
        fn = sum(1 for r in all_results if r['prediction'] == 'NORMAL' and r['true_label'] == 'ATTACK')
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Alert level counts
        alert_counts = {}
        for r in all_results:
            level = r['alert_level']
            alert_counts[level] = alert_counts.get(level, 0) + 1
        
        # Generate visualization (like detect_realtime.py)
        plot_filename = generate_realtime_plot(
            all_results, 
            y_test, 
            predictions, 
            confidences,
            AVAILABLE_MODELS[model_key]['name'],
            session_id
        )
        
        # Prepare comprehensive statistics
        stats = {
            'total_packets': total,
            'correct_predictions': correct,
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'fpr': round(fpr * 100, 2),
            'confusion_matrix': {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            },
            'alert_counts': alert_counts,
            'avg_confidence': round(np.mean(confidences) * 100, 1),
            'plot_url': f'/static/plots/{plot_filename}' if plot_filename else None
        }
        
        # Store results in session
        session['realtime_results'] = all_results
        session['realtime_stats'] = stats
        session['realtime_model'] = AVAILABLE_MODELS[model_key]['name']
        session.modified = True
        
        print(f"\n{'='*60}")
        print(f"Real-time Detection Complete!")
        print(f"Accuracy: {stats['accuracy']:.2f}%")
        print(f"Precision: {stats['precision']:.2f}%")
        print(f"Recall: {stats['recall']:.2f}%")
        print(f"F1-Score: {stats['f1_score']:.2f}%")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'model_name': AVAILABLE_MODELS[model_key]['name'],
            'total_packets': n_packets,
            'stats': stats,
            'results': all_results,  # Send all results at once
            'message': f'Processed all {n_packets} packets successfully'
        })
        
    except Exception as e:
        print(f"Error starting real-time detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

def generate_realtime_plot(results, y_true, y_pred, confidences, model_name, session_id):
    """Generate comprehensive real-time detection plot (like detect_realtime.py)"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Create figure with subplots (similar to detect_realtime.py)
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Prediction Timeline
        ax1 = fig.add_subplot(gs[0, :])
        packet_ids = [r['packet_id'] for r in results]
        
        # Color code by correctness
        colors = ['green' if r['correct'] else 'red' for r in results]
        ax1.scatter(packet_ids, confidences, c=colors, alpha=0.6, s=50)
        ax1.axhline(y=0.4, color='orange', linestyle='--', label='Threshold (0.4)')
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Default (0.5)')
        ax1.set_xlabel('Packet ID', fontsize=12)
        ax1.set_ylabel('Attack Confidence', fontsize=12)
        ax1.set_title(f'Real-Time Detection Timeline - {model_name}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confusion Matrix Heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        tp = sum(1 for r in results if r['prediction'] == 'ATTACK' and r['true_label'] == 'ATTACK')
        tn = sum(1 for r in results if r['prediction'] == 'NORMAL' and r['true_label'] == 'NORMAL')
        fp = sum(1 for r in results if r['prediction'] == 'ATTACK' and r['true_label'] == 'NORMAL')
        fn = sum(1 for r in results if r['prediction'] == 'NORMAL' and r['true_label'] == 'ATTACK')
        
        cm = np.array([[tn, fp], [fn, tp]])
        im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['Normal', 'Attack'])
        ax2.set_yticklabels(['Normal', 'Attack'])
        ax2.set_xlabel('Predicted', fontsize=12)
        ax2.set_ylabel('Actual', fontsize=12)
        ax2.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax2.text(j, i, cm[i, j], ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max()/2 else "black",
                              fontsize=16, fontweight='bold')
        
        # 3. Alert Level Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        alert_counts = {}
        for r in results:
            level = r['alert_level']
            alert_counts[level] = alert_counts.get(level, 0) + 1
        
        levels = ['SAFE', 'MEDIUM', 'HIGH', 'CRITICAL']
        counts = [alert_counts.get(level, 0) for level in levels]
        colors_bar = ['green', 'yellow', 'orange', 'red']
        
        bars = ax3.bar(levels, counts, color=colors_bar, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('Alert Level Distribution', fontsize=13, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Confidence Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(confidences, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.axvline(x=0.4, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax4.set_xlabel('Confidence Score', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Confidence Score Distribution', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Metrics Table
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Accuracy', f'{accuracy*100:.2f}%'],
            ['Precision', f'{precision*100:.2f}%'],
            ['Recall', f'{recall*100:.2f}%'],
            ['F1-Score', f'{f1*100:.2f}%'],
            ['False Positive Rate', f'{fpr*100:.2f}%'],
            ['Total Packets', f'{len(results)}'],
            ['Correct', f'{sum(1 for r in results if r["correct"])}'],
            ['Incorrect', f'{sum(1 for r in results if not r["correct"])}']
        ]
        
        table = ax5.table(cellText=metrics_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(metrics_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        ax5.set_title('Performance Metrics', fontsize=13, fontweight='bold', pad=20)
        
        # Save plot
        plots_dir = os.path.join('static', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_filename = f'realtime_{session_id}_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
        plot_path = os.path.join(plots_dir, plot_filename)
        
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved: {plot_path}")
        return plot_filename
        
    except Exception as e:
        print(f"Error generating plot: {str(e)}")
        import traceback
        traceback.print_exc()
        return None



@app.route('/api/realtime/stats', methods=['GET'])
def get_realtime_stats():
    """Get current real-time detection statistics"""
    try:
        if 'realtime_stats' not in session:
            return jsonify({'success': False, 'error': 'No results available. Please run detection first.'}), 400
        
        stats = session['realtime_stats']
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("=" * 60)
    print("ARP Spoofing Detection System")
    print("=" * 60)
    print(f"Available models: {len(AVAILABLE_MODELS)}")
    for key, info in AVAILABLE_MODELS.items():
        # Check if it's a hybrid model
        if 'supervised' in info:
            # Hybrid model - check if component models exist
            sup_info = AVAILABLE_MODELS.get(info['supervised'], {})
            unsup_info = AVAILABLE_MODELS.get(info['unsupervised'], {})
            sup_exists = os.path.exists(sup_info.get('path', ''))
            unsup_exists = os.path.exists(unsup_info.get('path', ''))
            status = "✓" if (sup_exists and unsup_exists) else "✗"
        else:
            # Regular model - check path
            status = "✓" if os.path.exists(info.get('path', '')) else "✗"
        print(f"  {status} {info['name']} ({info['category']})")
    print("=" * 60)
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    
    # Disable auto-reloader to prevent crashes during real-time detection
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
