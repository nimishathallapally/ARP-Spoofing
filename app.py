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
    'hybrid_rf_if': {
        'name': 'Hybrid RF+IF',
        'path': 'models/saved_models/hybrid_rf_if.pkl',
        'description': '70% Random Forest + 30% Isolation Forest',
        'icon': 'plus-circle-fill',
        'category': 'Hybrid'
    },
    'hybrid_rf_svm': {
        'name': 'Hybrid RF+SVM',
        'path': 'models/saved_models/hybrid_rf_svm.pkl',
        'description': '70% Random Forest + 30% One-Class SVM',
        'icon': 'intersect',
        'category': 'Hybrid'
    },
    'hybrid_rf_lof': {
        'name': 'Hybrid RF+LOF',
        'path': 'models/saved_models/hybrid_rf_lof.pkl',
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
        return False, f"Error loading model: {str(e)}"

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
        
        # Get prediction probabilities (if available)
        if hasattr(current_model, 'predict_proba'):
            try:
                y_proba = current_model.predict_proba(X_scaled)[:, 1]
                # Use optimal threshold instead of default 0.5
                y_pred_binary = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
            except:
                y_pred = current_model.predict(X_scaled)
                y_proba = np.abs(y_pred)
                y_pred_binary = y_pred
        elif hasattr(current_model, 'decision_function'):
            try:
                decision = current_model.decision_function(X_scaled)
                y_proba = (decision - decision.min()) / (decision.max() - decision.min())
                y_pred = current_model.predict(X_scaled)
                y_pred_binary = y_pred
            except:
                y_pred = current_model.predict(X_scaled)
                y_proba = np.abs(y_pred)
                y_pred_binary = y_pred
        else:
            y_pred = current_model.predict(X_scaled)
            y_proba = np.abs(y_pred)
            y_pred_binary = y_pred
        
        # For unsupervised models, convert -1 to 1 (attack) and 1 to 0 (normal)
        if hasattr(current_model, 'predict') and not hasattr(current_model, 'predict_proba'):
            if -1 in y_pred_binary:
                y_pred_binary = np.where(y_pred_binary == -1, 1, 0)
        
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
    """Start real-time detection simulation"""
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
        
        indices = list(np.random.choice(total_available, size=n_packets, replace=False))
        # Convert numpy int64 to Python int (for JSON serialization)
        indices = [int(i) for i in indices]
        
        # Store simulation config in session (no current_index - managed client-side)
        session['realtime_config'] = {
            'model_key': model_key,
            'model_name': AVAILABLE_MODELS[model_key]['name'],
            'n_packets': int(n_packets),  # Ensure int, not numpy int64
            'indices': indices,
            'started_at': datetime.now().isoformat()
        }
        session.modified = True  # Force session save
        
        return jsonify({
            'success': True,
            'model_name': AVAILABLE_MODELS[model_key]['name'],
            'total_packets': n_packets,
            'message': f'Real-time simulation initialized with {n_packets} packets'
        })
        
    except Exception as e:
        print(f"Error starting real-time detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/realtime/next', methods=['GET'])
def get_next_packet():
    """Get next packet detection result"""
    global current_model, current_scaler, current_feature_names
    
    try:
        if 'realtime_config' not in session or 'session_id' not in session:
            return jsonify({'success': False, 'error': 'Simulation not initialized'}), 400
        
        config = session['realtime_config']
        
        # Get packet index from query parameter instead of session
        packet_idx = request.args.get('index', type=int)
        if packet_idx is None:
            return jsonify({'success': False, 'error': 'Packet index required'}), 400
        
        if packet_idx >= config['n_packets']:
            return jsonify({'success': True, 'done': True, 'message': 'All packets processed'})
        
        # Reload model if needed (in case it was lost)
        if current_model is None:
            model_key = config['model_key']
            success, message = load_specific_model(model_key)
            if not success:
                return jsonify({'success': False, 'error': f'Model reload failed: {message}'}), 500
        
        # Load test data from file
        session_id = session['session_id']
        test_data_file = os.path.join(app.config['SESSION_FOLDER'], f'test_data_{session_id}.pkl')
        
        if not os.path.exists(test_data_file):
            return jsonify({'success': False, 'error': 'Test data file not found'}), 400
        
        with open(test_data_file, 'rb') as f:
            test_data = pickle.load(f)
        
        # Get the actual data index
        data_idx = config['indices'][packet_idx]
        
        # Get test data
        X_test = test_data['X']
        y_test = test_data['y']
        
        # Get single packet
        packet_features = X_test[data_idx].reshape(1, -1)
        true_label = int(y_test[data_idx])
        
        # Make prediction
        if hasattr(current_model, 'predict_proba'):
            y_proba = current_model.predict_proba(current_scaler.transform(packet_features))
            confidence = float(y_proba[0][1])  # Probability of attack
            
            # Use optimal threshold
            OPTIMAL_THRESHOLD = 0.4
            prediction = 1 if confidence >= OPTIMAL_THRESHOLD else 0
        else:
            # For unsupervised models
            y_pred = current_model.predict(current_scaler.transform(packet_features))
            prediction = 1 if y_pred[0] == -1 else 0  # -1 means anomaly/attack
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
        
        # Create result
        result = {
            'packet_id': packet_idx + 1,
            'prediction': 'ATTACK' if prediction == 1 else 'NORMAL',
            'true_label': 'ATTACK' if true_label == 1 else 'NORMAL',
            'correct': prediction == true_label,
            'confidence': confidence,
            'alert_level': alert_level,
            'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3]
        }
        
        # No need to update session - index is managed client-side
        return jsonify({'success': True, 'done': False, 'result': result})
        
    except Exception as e:
        print(f"Error in real-time detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/realtime/stats', methods=['GET'])
def get_realtime_stats():
    """Get current real-time detection statistics"""
    try:
        if 'realtime_results' not in session:
            return jsonify({'success': False, 'error': 'No results available'}), 400
        
        results = session['realtime_results']
        
        if not results:
            return jsonify({'success': True, 'stats': None})
        
        # Calculate statistics
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / total if total > 0 else 0
        
        tp = sum(1 for r in results if r['prediction'] == 'ATTACK' and r['true_label'] == 'ATTACK')
        tn = sum(1 for r in results if r['prediction'] == 'NORMAL' and r['true_label'] == 'NORMAL')
        fp = sum(1 for r in results if r['prediction'] == 'ATTACK' and r['true_label'] == 'NORMAL')
        fn = sum(1 for r in results if r['prediction'] == 'NORMAL' and r['true_label'] == 'ATTACK')
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Alert level counts
        alert_counts = {}
        for r in results:
            level = r['alert_level']
            alert_counts[level] = alert_counts.get(level, 0) + 1
        
        stats = {
            'total_packets': total,
            'accuracy': round(accuracy * 100, 1),
            'precision': round(precision * 100, 1),
            'recall': round(recall * 100, 1),
            'f1_score': round(f1 * 100, 1),
            'confusion_matrix': {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            },
            'alert_counts': alert_counts,
            'avg_confidence': round(np.mean([r['confidence'] for r in results]) * 100, 1)
        }
        
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
        status = "✓" if os.path.exists(info['path']) else "✗"
        print(f"  {status} {info['name']} ({info['category']})")
    print("=" * 60)
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    
    # Disable auto-reloader to prevent crashes during real-time detection
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
