#!/usr/bin/env python3
"""
Real-Time ARP Spoofing Detection Demo

This script demonstrates real-time detection capabilities by:
1. Loading a trained model
2. Simulating packet-by-packet detection
3. Displaying results with color-coded alerts
4. Generating visualizations
"""

import sys
from pathlib import Path
import argparse
import logging
import time
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from detector import ARPSpoofingDetector
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from utils import setup_logging, load_config, print_header, print_colored, get_color_code

logger = logging.getLogger(__name__)


def generate_realtime_plot(results: list, model_name: str, output_dir: str) -> str:
    """
    Generate a comprehensive visualization of real-time detection results.
    
    Args:
        results: List of detection result dictionaries
        model_name: Name of the model used
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path
    
    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = {
        'NORMAL': '#2ecc71',
        'ATTACK': '#e74c3c',
        'SAFE': '#27ae60',
        'MEDIUM': '#f39c12',
        'HIGH': '#e67e22',
        'CRITICAL': '#c0392b'
    }
    
    # Extract data
    packet_ids = [r['packet_id'] for r in results]
    predictions = [r['prediction'] for r in results]
    true_labels = [r['true_label'] for r in results]
    confidences = [r['confidence'] for r in results]
    alert_levels = [r['alert_level'] for r in results]
    correct = [r['correct'] for r in results]
    
    # 1. Detection Timeline (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Plot predictions and true labels
    pred_colors = [colors['ATTACK'] if p == 'ATTACK' else colors['NORMAL'] for p in predictions]
    true_colors = [colors['ATTACK'] if t == 'ATTACK' else colors['NORMAL'] for t in true_labels]
    
    ax1.scatter(packet_ids, [1] * len(packet_ids), c=pred_colors, s=100, alpha=0.7, 
                label='Predicted', marker='o', edgecolors='black', linewidths=1.5)
    ax1.scatter(packet_ids, [0] * len(packet_ids), c=true_colors, s=100, alpha=0.7,
                label='Actual', marker='s', edgecolors='black', linewidths=1.5)
    
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Actual', 'Predicted'], fontsize=11, fontweight='bold')
    ax1.set_xlabel('Packet ID', fontsize=12, fontweight='bold')
    ax1.set_title('Real-Time Detection Timeline', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Add background for incorrect predictions
    for i, corr in enumerate(correct):
        if not corr:
            ax1.axvspan(packet_ids[i]-0.5, packet_ids[i]+0.5, alpha=0.1, color='red')
    
    # 2. Confidence Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(confidences, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.2%}')
    ax2.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Confidence Distribution', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[1, 0])
    
    tp = sum(1 for r in results if r['prediction'] == 'ATTACK' and r['true_label'] == 'ATTACK')
    tn = sum(1 for r in results if r['prediction'] == 'NORMAL' and r['true_label'] == 'NORMAL')
    fp = sum(1 for r in results if r['prediction'] == 'ATTACK' and r['true_label'] == 'NORMAL')
    fn = sum(1 for r in results if r['prediction'] == 'NORMAL' and r['true_label'] == 'ATTACK')
    
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax3.imshow(cm, cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax3.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                          color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=16, fontweight='bold')
    
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Normal', 'Attack'], fontsize=11, fontweight='bold')
    ax3.set_yticklabels(['Normal', 'Attack'], fontsize=11, fontweight='bold')
    ax3.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax3.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. Alert Level Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    
    alert_counts = {}
    for level in ['SAFE', 'MEDIUM', 'HIGH', 'CRITICAL']:
        alert_counts[level] = sum(1 for a in alert_levels if a == level)
    
    levels = list(alert_counts.keys())
    counts = list(alert_counts.values())
    bar_colors = [colors[level] for level in levels]
    
    bars = ax4.bar(levels, counts, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Number of Packets', fontsize=11, fontweight='bold')
    ax4.set_title('Alert Level Distribution', fontsize=12, fontweight='bold', pad=10)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. Performance Metrics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    accuracy = sum(correct) / len(correct) if correct else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f"""
    Performance Metrics
    ========================
    
    Accuracy:     {accuracy:.1%}
    Precision:    {precision:.1%}
    Recall:       {recall:.1%}
    F1-Score:     {f1:.1%}
    
    ========================
    
    Total Packets:    {len(results)}
    Correct:          {sum(correct)}
    Incorrect:        {len(correct) - sum(correct)}
    
    True Positives:   {tp}
    True Negatives:   {tn}
    False Positives:  {fp}
    False Negatives:  {fn}
    """
    
    ax5.text(0.1, 0.95, metrics_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 6. Prediction Distribution (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    
    pred_counts = {'NORMAL': predictions.count('NORMAL'), 'ATTACK': predictions.count('ATTACK')}
    true_counts = {'NORMAL': true_labels.count('NORMAL'), 'ATTACK': true_labels.count('ATTACK')}
    
    x = np.arange(len(pred_counts))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, list(pred_counts.values()), width, 
                    label='Predicted', color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax6.bar(x + width/2, list(true_counts.values()), width,
                    label='Actual', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax6.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax6.set_title('Prediction vs Actual Distribution', fontsize=12, fontweight='bold', pad=10)
    ax6.set_xticks(x)
    ax6.set_xticklabels(list(pred_counts.keys()), fontsize=11, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 7. Confidence vs Correctness (bottom middle)
    ax7 = fig.add_subplot(gs[2, 1])
    
    correct_conf = [confidences[i] for i in range(len(confidences)) if correct[i]]
    incorrect_conf = [confidences[i] for i in range(len(confidences)) if not correct[i]]
    
    ax7.scatter(range(len(correct_conf)), correct_conf, c='green', alpha=0.6,
               label='Correct', s=60, edgecolors='black', linewidths=0.5)
    ax7.scatter(range(len(correct_conf), len(correct_conf) + len(incorrect_conf)), 
               incorrect_conf, c='red', alpha=0.6,
               label='Incorrect', s=60, edgecolors='black', linewidths=0.5)
    
    ax7.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax7.set_xlabel('Prediction Index', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Confidence Score', fontsize=11, fontweight='bold')
    ax7.set_title('Confidence vs Correctness', fontsize=12, fontweight='bold', pad=10)
    ax7.legend(fontsize=10)
    ax7.grid(alpha=0.3)
    ax7.set_ylim(0, 1)
    
    # 8. Summary Statistics (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary_text = f"""
    Detection Summary
    ========================
    
    Model: {model_name}
    
    Detection Rate
    Attacks Detected:  {tp}/{tp+fn}
    Attack Rate:       {(tp+fn)/len(results):.1%}
    
    Correct Predictions
    Normal: {tn}/{tn+fp}
    Attack: {tp}/{tp+fn}
    
    Errors
    False Positives: {fp}
    False Negatives: {fn}
    
    Alert Levels
    """
    
    for level in ['SAFE', 'MEDIUM', 'HIGH', 'CRITICAL']:
        count = alert_counts[level]
        if count > 0:
            summary_text += f"    {level}: {count}\n"
    
    ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Overall title
    fig.suptitle(f'Real-Time ARP Spoofing Detection Results - {model_name}',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    plot_filename = 'realtime_detection_results.png'
    plot_path = output_path / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(plot_path)


def display_detection_result(result: dict, packet_num: int):
    """Display a single detection result with colors."""
    alert_level = result['alert_level']
    prediction = result['prediction']
    confidence = result['confidence']
    
    # Color codes
    color = get_color_code(alert_level)
    reset = get_color_code('RESET')
    
    # Status symbol
    symbol = "✓" if prediction == "NORMAL" else "⚠"
    
    # Format output
    print(f"{color}Packet #{packet_num:03d}: {symbol} {prediction:<8} "
          f"[Confidence: {confidence:>5.1%}] [{alert_level:>8}]{reset}")


def main(model_path: str = None, config_path: str = None, n_packets: int = 100):
    """
    Main real-time detection demo.
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        n_packets: Number of packets to simulate
    """
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    
    config = load_config(str(config_path))
    
    # Setup logging
    setup_logging(level='WARNING')  # Less verbose for demo
    
    print_header("ARP SPOOFING DETECTION - REAL-TIME DEMO", width=70)
    
    # ===== LOAD MODEL =====
    print("\n[1/4] Loading trained model...")
    
    if model_path is None:
        model_path = Path(config['output']['models_path']) / 'arp_spoofing_detector.pkl'
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print("  Please run 'python scripts/train_model.py' first")
        return 1
    
    try:
        detector = ARPSpoofingDetector.load(str(model_path))
        print(f"✓ Loaded model: {detector.model_name}")
        print(f"  Features: {len(detector.feature_names)}")
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)}")
        return 1
    
    # ===== LOAD TEST DATA =====
    print("\n[2/4] Loading test data...")
    
    try:
        # Load data
        loader = DataLoader(data_dir=config['data']['raw_data_path'])
        combined_df = loader.load_all_datasets(
            filenames=config['data']['dataset_files'],
            select_best=False,  # Use all data
            balance_classes=True
        )
        
        # Prepare data - we need to do the same steps but keep unscaled version
        engineer = FeatureEngineer()
        
        # Step 1: Clean data
        df_clean = engineer.clean_data(combined_df)
        
        # Step 2: Create derived features
        df_enhanced = engineer.create_derived_features(df_clean)
        
        # Step 3: Separate target
        y = df_enhanced['Label']
        X = df_enhanced.drop(columns=['Label'])
        
        # Keep only numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Step 4: Encode labels
        y_encoded = engineer.encode_labels(y)
        
        # Step 5: Feature selection (using same method as training)
        if config['features']['selection_method'] == 'hybrid':
            selected_features = engineer.select_features_hybrid(
                X, y_encoded, k=config['features']['n_features']
            )
        else:
            selected_features = engineer.select_features_hybrid(X, y_encoded, k=25)
        
        X_selected = X[selected_features]
        
        # Step 6: Train-test split (same random state to get same split)
        from sklearn.model_selection import train_test_split
        X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(
            X_selected, y_encoded,
            test_size=config['training']['test_size'],
            random_state=config['training']['random_state'],
            stratify=y_encoded
        )
        
        # NOW we have unscaled test data that matches the training features
        
        # Ensure features match the model
        if set(selected_features) != set(detector.feature_names):
            print(f"⚠ Warning: Feature mismatch detected")
            print(f"  Model expects: {len(detector.feature_names)} features")
            print(f"  Data provides: {len(selected_features)} features")
            
            # Try to reorder to match model
            if set(selected_features).issuperset(set(detector.feature_names)):
                X_test_unscaled = X_test_unscaled[detector.feature_names]
                print(f"  ✓ Reordered features to match model")
            else:
                print(f"✗ Cannot proceed: Missing required features")
                return 1
        
        print(f"✓ Loaded {len(X_test_unscaled):,} test samples (unscaled)")
    except Exception as e:
        print(f"✗ Failed to load test data: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== SIMULATE REAL-TIME DETECTION =====
    print(f"\n[3/4] Simulating real-time detection ({n_packets} packets)...")
    print("-" * 70)
    
    try:
        # Select random packets from UNSCALED data
        np.random.seed(42)
        indices = np.random.choice(len(X_test_unscaled), size=n_packets, replace=False)
        
        # Get subset - X_test_unscaled is a DataFrame, so use iloc
        sim_X_df = X_test_unscaled.iloc[indices]
        
        # Use .iloc for pandas Series or .values to convert to numpy
        if hasattr(y_test, 'iloc'):
            sim_y = y_test.iloc[indices].values
        else:
            sim_y = y_test[indices]
        
        print(f"  Selected data shape: {sim_X_df.shape}")
        print(f"  Model expects: {len(detector.feature_names)} features")
        print()
        
        # Process packets one by one
        results = []
        start_time = time.time()
        
        for i in range(n_packets):
            # Simulate processing time
            if i > 0 and i % 10 == 0:
                time.sleep(0.1)  # Small delay every 10 packets
            
            try:
                # Get packet as dictionary (like in notebook)
                packet_features = sim_X_df.iloc[i].to_dict()
                
                # Detect using the detector's detect method
                detection_result = detector.detect(packet_features)
                
                # Extract results
                prediction = 1 if detection_result['label'] == 'arp_spoofing' else 0
                confidence = detection_result['probability']
                alert_level = detection_result['alert_level']
                
                result = {
                    'packet_id': i + 1,
                    'prediction': 'ATTACK' if prediction == 1 else 'NORMAL',
                    'true_label': 'ATTACK' if sim_y[i] == 1 else 'NORMAL',
                    'correct': prediction == sim_y[i],
                    'confidence': float(confidence),
                    'alert_level': alert_level
                }
                
                results.append(result)
                display_detection_result(result, i + 1)
                
            except Exception as packet_error:
                print(f"\n✗ Error at packet {i+1}: {str(packet_error)}")
                import traceback
                traceback.print_exc()
                return 1
        
        elapsed_time = time.time() - start_time
        
    except Exception as e:
        print(f"\n✗ Detection setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== DISPLAY SUMMARY =====
    print("\n" + "-" * 70)
    print(f"\n[4/4] Detection Summary")
    print("=" * 70)
    
    # Calculate statistics
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / n_packets
    
    # Count by prediction
    attack_detected = sum(1 for r in results if r['prediction'] == 'ATTACK')
    normal_detected = sum(1 for r in results if r['prediction'] == 'NORMAL')
    
    # Count by true label
    true_attacks = sum(1 for r in results if r['true_label'] == 'ATTACK')
    true_normal = sum(1 for r in results if r['true_label'] == 'NORMAL')
    
    # Count by alert level
    alert_counts = {}
    for r in results:
        level = r['alert_level']
        alert_counts[level] = alert_counts.get(level, 0) + 1
    
    # Calculate confusion matrix
    tp = sum(1 for r in results if r['prediction'] == 'ATTACK' and r['true_label'] == 'ATTACK')
    tn = sum(1 for r in results if r['prediction'] == 'NORMAL' and r['true_label'] == 'NORMAL')
    fp = sum(1 for r in results if r['prediction'] == 'ATTACK' and r['true_label'] == 'NORMAL')
    fn = sum(1 for r in results if r['prediction'] == 'NORMAL' and r['true_label'] == 'ATTACK')
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display results
    print(f"\nPerformance Metrics:")
    print(f"  Total Packets Analyzed:    {n_packets}")
    print(f"  Processing Time:           {elapsed_time:.2f} seconds")
    print(f"  Throughput:                {n_packets/elapsed_time:.1f} packets/sec")
    print(f"\n  Accuracy:                  {accuracy:.2%}")
    print(f"  Precision:                 {precision:.2%}")
    print(f"  Recall:                    {recall:.2%}")
    print(f"  F1-Score:                  {f1_score:.2%}")
    
    print(f"\nDetection Results:")
    print(f"  Attacks Detected:          {attack_detected} / {true_attacks} actual")
    print(f"  Normal Traffic:            {normal_detected} / {true_normal} actual")
    print(f"  Correct Predictions:       {correct} ({accuracy:.1%})")
    print(f"  Incorrect Predictions:     {n_packets - correct} ({(1-accuracy):.1%})")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Normal    Predicted Attack")
    print(f"  Actual Normal        {tn:4d}               {fp:4d}")
    print(f"  Actual Attack        {fn:4d}               {tp:4d}")
    
    print(f"\nAlert Level Distribution:")
    for level in ['SAFE', 'MEDIUM', 'HIGH', 'CRITICAL']:
        count = alert_counts.get(level, 0)
        percentage = count / n_packets * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length
        color = get_color_code(level)
        reset = get_color_code('RESET')
        print(f"  {color}{level:>8}{reset}: {count:3d} packets ({percentage:5.1f}%) {bar}")
    
    print("\n" + "=" * 70)
    
    # Display key findings
    if fn == 0:
        print_colored(f"\n✓ EXCELLENT: No attacks were missed (100% Recall)!", 'SAFE')
    elif fn <= 2:
        print_colored(f"\n✓ GOOD: Only {fn} attack(s) missed", 'MEDIUM')
    else:
        print_colored(f"\n⚠ WARNING: {fn} attacks were missed!", 'HIGH')
    
    if fp == 0:
        print_colored(f"✓ PERFECT: No false positives!", 'SAFE')
    elif fp <= 5:
        print_colored(f"✓ ACCEPTABLE: {fp} false positive(s)", 'MEDIUM')
    else:
        print_colored(f"⚠ CAUTION: {fp} false positives detected", 'HIGH')
    
    print(f"\n{'='*70}\n")
    
    # ===== GENERATE VISUALIZATION =====
    try:
        plot_path = generate_realtime_plot(results, detector.model_name, config['output']['plots_path'])
        print(f"✓ Visualization saved: {plot_path}")
    except Exception as e:
        print(f"⚠ Could not generate visualization: {str(e)}")
    
    print("\nDetection demo complete!")
    print(f"Model: {detector.model_name}")
    print(f"Saved model: {model_path}\n")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-Time ARP Spoofing Detection Demo')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (default: models/saved_models/arp_spoofing_detector.pkl)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--packets',
        type=int,
        default=100,
        help='Number of packets to simulate (default: 100)'
    )
    
    args = parser.parse_args()
    
    try:
        exit_code = main(
            model_path=args.model,
            config_path=args.config,
            n_packets=args.packets
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)
