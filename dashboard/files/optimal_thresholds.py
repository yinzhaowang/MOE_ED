# Optimal Threshold Calculator
# This script determines the best probability threshold for each disease outcome
# by maximizing the AUC score using calibrated probabilities.

import os
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torch.utils.data import TensorDataset, DataLoader
from mmoe_iso_calibration import MMoE, IsotonicCalibrator, outcome_list, col_list
import seaborn as sns
from tqdm import tqdm

# Create directory for saving threshold plots
os.makedirs("threshold_plots", exist_ok=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data_and_model():
    """Load test data, model, and calibrator"""
    print("Loading data, model, and calibrator...")
    
    # Load test data (same as in app.py)
    df_test = pd.read_csv("../../data/test_with_outcomes.csv")
    raw_feats = df_test.loc[:, col_list]
    feat_df = raw_feats.select_dtypes(include=[np.number, bool])
    feat_df = feat_df.fillna(feat_df.mean())
    X_np = feat_df.to_numpy(dtype=float)
    y_np = df_test.loc[:, outcome_list].to_numpy(dtype=float)
    
    # Convert to torch tensors
    X_t = torch.from_numpy(X_np).float()
    y_t = torch.from_numpy(y_np).float()
    
    # Create dataloader
    batch_size = 256
    dataset = TensorDataset(X_t, y_t)
    test_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    input_size = X_np.shape[1]
    num_experts = 8
    expert_hidden = [128, 64]
    expert_output_dim = 32
    tower_hidden_dim = 16
    task_output_dims = [1] * len(outcome_list)
    
    model = MMoE(input_size, num_experts, expert_hidden, expert_output_dim, 
                tower_hidden_dim, task_output_dims).to(device)
    model.load_state_dict(torch.load("best_mmoe_iso.pt", map_location=device))
    model.eval()
    
    # Load calibrator
    calibrator = joblib.load("calibrator.pkl")
    
    return df_test, X_np, y_np, model, calibrator, test_dl

def get_calibrated_predictions(model, test_dl, calibrator):
    """Get calibrated predictions for all test data"""
    print("Generating calibrated predictions...")
    
    all_logits = []
    all_targets = []
    
    # Get model predictions
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_dl):
            X_batch = X_batch.to(device)
            logits_list = model(X_batch)
            logits = torch.cat(logits_list, dim=1).cpu().numpy()
            all_logits.append(logits)
            all_targets.append(y_batch.numpy())
    
    # Concatenate batches
    all_logits = np.vstack(all_logits)
    all_targets = np.vstack(all_targets)
    
    # Apply calibration
    calibrated_probs = calibrator.predict(all_logits)
    
    return calibrated_probs, all_targets

def find_optimal_thresholds(calibrated_probs, y_true):
    """Find optimal threshold for each outcome by maximizing Youden's J statistic (sensitivity + specificity - 1)"""
    print("Finding optimal thresholds...")
    
    optimal_thresholds = {}
    threshold_metrics = {}
    
    for i, outcome in enumerate(outcome_list):
        # Get actual outcomes and predicted probabilities for this disease
        y_true_outcome = y_true[:, i]
        y_prob_outcome = calibrated_probs[:, i]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_outcome, y_prob_outcome)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Youden's J statistic (sensitivity + specificity - 1)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true_outcome, y_prob_outcome)
        pr_auc = average_precision_score(y_true_outcome, y_prob_outcome)
        
        # Store results
        disease_name = outcome.replace("outcome_", "")
        optimal_thresholds[disease_name] = optimal_threshold
        
        # Store additional metrics
        threshold_metrics[disease_name] = {
            'threshold': optimal_threshold,
            'sensitivity': tpr[optimal_idx],
            'specificity': 1 - fpr[optimal_idx],
            'youden_j': j_scores[optimal_idx],
            'prevalence': np.mean(y_true_outcome),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
        # Create and save plots
        plot_threshold_curves(disease_name, fpr, tpr, roc_auc, precision, recall, 
                             pr_auc, optimal_threshold, y_true_outcome, y_prob_outcome)
    
    return optimal_thresholds, threshold_metrics

def plot_threshold_curves(disease_name, fpr, tpr, roc_auc, precision, recall, 
                         pr_auc, optimal_threshold, y_true, y_prob):
    """Create and save ROC and PR curves with optimal threshold marked"""
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot ROC curve
    ax1.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Mark the optimal threshold on ROC curve
    optimal_idx = np.argmin(np.abs(np.array([x for x in fpr]) - (1 - np.array([x for x in tpr]))))
    ax1.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
            label=f'Optimal threshold = {optimal_threshold:.3f}')
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.set_title(f'ROC Curve for {disease_name}')
    ax1.legend(loc="lower right")
    
    # Plot PR curve
    ax2.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.axhline(y=np.mean(y_true), color='k', linestyle='--', 
               label=f'Baseline (prevalence = {np.mean(y_true):.3f})')
    
    # Mark the optimal threshold on PR curve
    threshold_indices = np.digitize(optimal_threshold, [0.5])
    if len(precision) > threshold_indices:
        pr_idx = min(threshold_indices, len(precision)-1)
        ax2.plot(recall[pr_idx], precision[pr_idx], 'ro', markersize=8,
                label=f'Optimal threshold = {optimal_threshold:.3f}')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve for {disease_name}')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(f"threshold_plots/{disease_name}_threshold_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_threshold_heatmap(threshold_metrics):
    """Create a heatmap of threshold metrics across all diseases"""
    
    # Create a DataFrame for the heatmap
    metrics_df = pd.DataFrame({
        'Disease': [],
        'Threshold': [],
        'Prevalence': [],
        'Sensitivity': [],
        'Specificity': [],
        'Youden J': [],
        'ROC AUC': [],
        'PR AUC': []
    })
    
    # Fill the DataFrame with metrics
    for disease, metrics in threshold_metrics.items():
        metrics_df = metrics_df.append({
            'Disease': disease,
            'Threshold': round(metrics['threshold'], 3),
            'Prevalence': round(metrics['prevalence'], 3),
            'Sensitivity': round(metrics['sensitivity'], 3),
            'Specificity': round(metrics['specificity'], 3),
            'Youden J': round(metrics['youden_j'], 3),
            'ROC AUC': round(metrics['roc_auc'], 3),
            'PR AUC': round(metrics['pr_auc'], 3)
        }, ignore_index=True)
    
    # Sort by AUC
    metrics_df = metrics_df.sort_values('ROC AUC', ascending=False)
    
    # Save the metrics to CSV
    metrics_df.to_csv("threshold_metrics.csv", index=False)
    print(f"Saved threshold metrics to threshold_metrics.csv")
    
    # Create heatmap visualization
    plt.figure(figsize=(15, len(threshold_metrics)*0.6))
    
    # Create a heatmap for numeric columns only
    heatmap_data = metrics_df.set_index('Disease')
    heatmap_data = heatmap_data.astype(float)
    
    # Create custom colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Create heatmap
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap=cmap, 
                    linewidths=0.5, cbar=True)
    plt.title('Optimal Thresholds and Performance Metrics by Disease')
    plt.tight_layout()
    plt.savefig("threshold_plots/threshold_metrics_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to find optimal thresholds"""
    print("=" * 80)
    print("OPTIMAL THRESHOLD CALCULATOR")
    print("=" * 80)
    
    # Load data and model
    df_test, X_np, y_np, model, calibrator, test_dl = load_data_and_model()
    
    # Get calibrated predictions
    calibrated_probs, y_true = get_calibrated_predictions(model, test_dl, calibrator)
    
    # Find optimal thresholds
    optimal_thresholds, threshold_metrics = find_optimal_thresholds(calibrated_probs, y_true)
    
    # Display results
    print("\nOptimal Thresholds:")
    print("-" * 80)
    for disease, threshold in sorted(optimal_thresholds.items(), 
                                   key=lambda x: threshold_metrics[x[0]]['roc_auc'], 
                                   reverse=True):
        auc = threshold_metrics[disease]['roc_auc']
        sensitivity = threshold_metrics[disease]['sensitivity']
        specificity = threshold_metrics[disease]['specificity']
        prevalence = threshold_metrics[disease]['prevalence']
        
        print(f"{disease.ljust(20)}: {threshold:.3f} (AUC: {auc:.3f}, "
              f"Sens: {sensitivity:.3f}, Spec: {specificity:.3f}, Prev: {prevalence:.3f})")
    
    # Create heatmap and save metrics
    create_threshold_heatmap(threshold_metrics)
    
    print("\nAnalysis complete! Threshold plots saved to the 'threshold_plots' directory.")
    print("Threshold metrics saved to 'threshold_metrics.csv'.")

if __name__ == "__main__":
    main()
