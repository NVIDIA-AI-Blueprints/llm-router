"""
nn_router.py

Uses pre-computed embeddings from HuggingFace evaluations to train a neural network router.

Expects:
- hf_evaluations_train.json / hf_evaluations_test.json with model_scores
- hf_train_embeddings.npy / hf_test_embeddings.npy with pre-computed embeddings

Features:
- Multi-layer neural network with batch normalization and dropout
- Hyperparameter tuning with random search
- Class weighting to handle data imbalance (e.g., few positive examples for Llama model)
- Early stopping to prevent overfitting
- GPU support

Installs:
pip install torch scikit-learn joblib
"""

from pathlib import Path
import numpy as np
import json
import joblib
import random
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ---------- Configuration ----------
TRAIN_JSON = "hf_evaluations_train.json"
TEST_JSON = "hf_evaluations_test.json"
TRAIN_EMBEDDINGS = "hf_train_embeddings.npy"
TEST_EMBEDDINGS = "hf_test_embeddings.npy"
RANDOM_STATE = 42
TUNE_HYPERPARAMETERS = True  # Set to False to skip tuning and use default params
USE_CLASS_WEIGHTS = True  # Set to False to disable class weighting for imbalanced data

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Check for GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Default hyperparameters
default_config = {
    'hidden_dims': [512, 256, 128],  # Hidden layer dimensions
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 50,
    'weight_decay': 1e-5,
    'patience': 10,  # Early stopping patience
}

# Hyperparameter search space for tuning
param_grid = {
    'hidden_dims': [
        [256, 128],
        [512, 256],
        [512, 256, 128],
        [1024, 512, 256],
        [256, 128, 64],
    ],
    'dropout': [0.2, 0.3, 0.4, 0.5],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.002],
    'batch_size': [32, 64, 128],
    'weight_decay': [0, 1e-6, 1e-5, 1e-4],
}

# ---------- Neural Network Architecture ----------
class RouterNetwork(nn.Module):
    """
    Multi-output neural network for routing.
    Predicts probability that each model will be correct for a given input.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super(RouterNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (one sigmoid output per model)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        logits = self.network(x)
        # Apply sigmoid to get probabilities for each model independently
        return torch.sigmoid(logits)

# ---------- Helpers ----------
def load_data(json_path: str, embeddings_path: str, selected_models=None):
    """
    Load data from JSON file and corresponding embeddings.
    
    Args:
        json_path: Path to JSON file with model_scores
        embeddings_path: Path to embeddings .npy file
        selected_models: Optional list of model names to keep. If None, keeps all models.
    
    Returns:
        embeddings (X), labels (Y), model_names, and prompts.
    """
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    
    # Verify alignment
    if len(data) != len(embeddings):
        raise ValueError(f"Mismatch: {len(data)} JSON records vs {len(embeddings)} embeddings")
    
    # Extract labels from model_scores
    if not data:
        raise ValueError("Empty JSON file")
    
    all_model_names = list(data[0]['model_scores'].keys())
    print(f"Found models in data: {all_model_names}")
    
    # Filter to selected models if specified
    if selected_models is not None:
        model_names = [m for m in all_model_names if m in selected_models]
        if len(model_names) != len(selected_models):
            missing = set(selected_models) - set(model_names)
            print(f"Warning: The following selected models were not found in data: {missing}")
        print(f"Using selected models: {model_names}")
    else:
        model_names = all_model_names
    
    # Build label matrix
    labels = []
    prompts = []
    for record in data:
        prompts.append(record['prompt'])
        label_row = [record['model_scores'].get(model, 0) for model in model_names]
        labels.append(label_row)
    
    Y = np.array(labels, dtype=np.float32)
    X = embeddings.astype(np.float32)
    
    return X, Y, model_names, prompts

def compute_class_weights(y_train, model_names, method='inverse'):
    """
    Compute class weights to handle imbalanced data.
    
    Args:
        y_train: Training labels (n_samples, n_models)
        model_names: List of model names
        method: 'inverse' or 'balanced'
    
    Returns:
        torch.Tensor of shape (n_models,) with weights for each model
    """
    weights = []
    pos_rates = []
    
    print("\n" + "="*80)
    print("Computing class weights to handle data imbalance:")
    print("="*80)
    
    for i, model_name in enumerate(model_names):
        pos_rate = y_train[:, i].mean()
        pos_rates.append(pos_rate)
        
        if method == 'inverse':
            # Inverse frequency: weight = 1 / pos_rate
            # Higher weight for rarer classes
            if pos_rate > 0:
                weight = 1.0 / pos_rate
            else:
                weight = 1.0
        elif method == 'balanced':
            # Balanced: weight = n_samples / (n_classes * n_samples_for_class)
            n_pos = y_train[:, i].sum()
            if n_pos > 0:
                weight = len(y_train) / (2 * n_pos)
            else:
                weight = 1.0
        else:
            weight = 1.0
        
        weights.append(weight)
        print(f"  {model_name:40s}: pos_rate={pos_rate:.4f}, weight={weight:.4f}")
    
    # Normalize weights so they sum to number of models (keeps loss scale similar)
    weights = np.array(weights)
    weights = weights * len(weights) / weights.sum()
    
    print(f"\nNormalized weights (sum={weights.sum():.2f}):")
    for i, model_name in enumerate(model_names):
        print(f"  {model_name:40s}: {weights[i]:.4f}")
    print("="*80)
    
    return torch.FloatTensor(weights)

def train_model(model, train_loader, val_loader, config, model_names, class_weights=None):
    """
    Train the neural network with early stopping.
    Supports class weights to handle data imbalance.
    """
    # Use weighted BCE loss if class weights are provided
    if class_weights is not None:
        # BCELoss doesn't support weights directly, so we'll use a weighted version
        def weighted_bce_loss(outputs, targets):
            # Compute BCE loss element-wise
            bce = -(targets * torch.log(outputs + 1e-8) + (1 - targets) * torch.log(1 - outputs + 1e-8))
            # Apply class weights
            weighted = bce * class_weights.to(outputs.device)
            return weighted.mean()
        criterion = weighted_bce_loss
    else:
        criterion = nn.BCELoss()
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{config['epochs']}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_loss

def evaluate_router(model, X_test, y_test, model_names, model_thresholds=None):
    """
    Evaluate the router on test set.
    
    Args:
        model: trained RouterNetwork
        X_test: test embeddings
        y_test: test labels
        model_names: list of model names
        model_thresholds: optional dict of model name -> threshold for selection
    """
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        proba_test = model(X_test_tensor).cpu().numpy()
    
    # Evaluate per-model metrics
    metrics = {}
    for i, model_name in enumerate(model_names):
        y_true = y_test[:, i]
        y_score = proba_test[:, i]
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float("nan")
        preds = (y_score >= 0.5).astype(int)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        metrics[model_name] = {
            "auc": auc, 
            "accuracy": acc, 
            "f1": f1, 
            "positive_rate": y_true.mean()
        }
    
    # Router evaluation (system accuracy)
    chosen_idx = np.argmax(proba_test, axis=1)
    
    # Apply model thresholds if specified
    if model_thresholds is not None and len(model_thresholds) > 0:
        # Get indices for models with thresholds
        threshold_map = {}
        for model_name, threshold in model_thresholds.items():
            if model_name in model_names:
                model_idx = model_names.index(model_name)
                threshold_map[model_idx] = threshold
        
        # Override choices that don't meet thresholds
        for i in range(len(chosen_idx)):
            chosen_model_idx = chosen_idx[i]
            
            # Check if chosen model has a threshold and doesn't meet it
            if chosen_model_idx in threshold_map:
                threshold = threshold_map[chosen_model_idx]
                if proba_test[i, chosen_model_idx] < threshold:
                    # Find next best model that either has no threshold or meets its threshold
                    proba_copy = proba_test[i].copy()
                    
                    # Try models in order of probability until we find one that meets threshold
                    sorted_indices = np.argsort(proba_copy)[::-1]  # Descending order
                    
                    for candidate_idx in sorted_indices:
                        # Skip the originally chosen model (already failed threshold)
                        if candidate_idx == chosen_model_idx:
                            continue
                        
                        # Check if candidate has a threshold
                        if candidate_idx in threshold_map:
                            candidate_threshold = threshold_map[candidate_idx]
                            if proba_copy[candidate_idx] >= candidate_threshold:
                                chosen_idx[i] = candidate_idx
                                break
                        else:
                            # No threshold for this model, so it's acceptable
                            chosen_idx[i] = candidate_idx
                            break
    
    chosen_label = np.array([y_test[i, chosen_idx[i]] for i in range(len(chosen_idx))])
    system_accuracy = chosen_label.mean()
    
    # Model selection counts
    model_selection_counts = {}
    for i, model_name in enumerate(model_names):
        count = np.sum(chosen_idx == i)
        model_selection_counts[model_name] = {
            'count': int(count),
            'percentage': count / len(chosen_idx) * 100,
            'accuracy_when_chosen': chosen_label[chosen_idx == i].mean() if count > 0 else 0.0
        }
    
    # Oracle accuracy
    any_correct = (y_test.sum(axis=1) >= 1).mean()
    
    # Baseline accuracies
    always_acc = {}
    for i, model_name in enumerate(model_names):
        always_acc[model_name] = y_test[:, i].mean()
    
    return {
        'metrics': metrics,
        'system_accuracy': system_accuracy,
        'model_selection_counts': model_selection_counts,
        'any_correct': any_correct,
        'always_acc': always_acc,
        'proba_test': proba_test
    }

def test_model_thresholds(model, X_test, y_test, model_names, 
                          threshold_configs=None):
    """
    Test different threshold configurations and show their impact on model selection and accuracy.
    
    Args:
        model: trained RouterNetwork
        X_test: test embeddings
        y_test: test labels
        model_names: list of model names
        threshold_configs: list of dicts with threshold configurations to test
                          Example: [
                              {},  # No threshold
                              {'gpt-5-chat': 0.65},
                              {'gpt-5-chat': 0.65, 'nemotron-nano-12b-v2-vl': 0.55},
                          ]
                          If None, uses default configurations
    """
    if threshold_configs is None:
        # Default configurations: test GPT-5 thresholds alone, then with Qwen-VL
        threshold_configs = [
            {},  # No threshold
            {'gpt-5-chat': 0.60},
            {'gpt-5-chat': 0.65},
            {'gpt-5-chat': 0.70},
            {'gpt-5-chat': 0.65, 'Qwen/Qwen3-VL-8B-Instruct': 0.50},
            {'gpt-5-chat': 0.65, 'Qwen/Qwen3-VL-8B-Instruct': 0.55},
            {'gpt-5-chat': 0.70, 'Qwen/Qwen3-VL-8B-Instruct': 0.55},
        ]
    
    print("\n" + "="*120)
    print("TESTING MODEL THRESHOLD CONFIGURATIONS")
    print("="*120)
    print("Thresholds control which models are used based on confidence levels")
    print("Higher thresholds = more cost savings (redirect to cheaper models)")
    print("="*120)
    
    results = []
    
    for config in threshold_configs:
        # Evaluate with this threshold configuration
        eval_results = evaluate_router(model, X_test, y_test, model_names, model_thresholds=config if config else None)
        
        # Extract key metrics
        gpt5_selection = eval_results['model_selection_counts'].get('gpt-5-chat', {})
        qwen_selection = eval_results['model_selection_counts'].get('Qwen/Qwen3-VL-8B-Instruct', {})
        nano_text_selection = eval_results['model_selection_counts'].get('nvidia/nvidia-nemotron-nano-9b-v2', {})
        
        # Format threshold description
        if not config:
            threshold_desc = "No thresholds"
        else:
            parts = []
            if 'gpt-5-chat' in config:
                parts.append(f"GPT5={config['gpt-5-chat']:.2f}")
            if 'Qwen/Qwen3-VL-8B-Instruct' in config:
                parts.append(f"Qwen={config['Qwen/Qwen3-VL-8B-Instruct']:.2f}")
            threshold_desc = ", ".join(parts)
        
        results.append({
            'config': config,
            'desc': threshold_desc,
            'accuracy': eval_results['system_accuracy'],
            'gpt5_pct': gpt5_selection.get('percentage', 0),
            'qwen_pct': qwen_selection.get('percentage', 0),
            'nano_text_pct': nano_text_selection.get('percentage', 0),
        })
    
    # Print results table
    print(f"\n{'Configuration':>30} {'Accuracy':>10} {'GPT-5 %':>10} {'Qwen-VL %':>12} {'Nano-Text %':>12} {'Cost Savings':>15}")
    print("-" * 125)
    
    baseline_gpt5_pct = results[0]['gpt5_pct']
    baseline_qwen_pct = results[0]['qwen_pct']
    
    for r in results:
        # Calculate cost savings (assuming GPT-5 is most expensive, then Qwen-VL, then Nano-Text)
        # Use relative costs: GPT-5 = 1.0, Qwen-VL = 0.4, Nano-Text = 0.1 (example ratios)
        baseline_cost = baseline_gpt5_pct * 1.0 + baseline_qwen_pct * 0.4 + (100 - baseline_gpt5_pct - baseline_qwen_pct) * 0.1
        current_cost = r['gpt5_pct'] * 1.0 + r['qwen_pct'] * 0.4 + r['nano_text_pct'] * 0.1
        cost_savings = ((baseline_cost - current_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        
        savings_str = f"-{cost_savings:.1f}%" if cost_savings > 0 else "baseline"
        
        print(f"{r['desc']:>30} {r['accuracy']:>10.3f} {r['gpt5_pct']:>9.1f}% "
              f"{r['qwen_pct']:>11.1f}% {r['nano_text_pct']:>11.1f}% {savings_str:>15}")
    
    print("="*125)
    print("\nRecommended Configurations:")
    print("  1. GPT-5 only (0.65): Moderate savings, redirects expensive model when unsure")
    print("  2. GPT-5 (0.65) + Qwen-VL (0.55): Balanced savings, redirects both premium models ← RECOMMENDED")
    print("  3. GPT-5 (0.70) + Qwen-VL (0.55): Aggressive savings, maximum cost reduction")
    print("\nNote: Cost savings assume relative costs of GPT-5:Qwen-VL:Nano-Text = 10:4:1")
    print("="*125)
    
    return results

def tune_hyperparameters(X_train, y_train, X_val, y_val, model_names, class_weights=None, n_trials=10):
    """
    Perform random search over hyperparameters.
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING (Neural Network)")
    print("="*80)
    print(f"Running {n_trials} random search trials...")
    
    if class_weights is not None:
        print("Using class weights to handle data imbalance")
    
    best_val_loss = float('inf')
    best_config = None
    best_model = None
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    for trial in range(n_trials):
        # Sample random hyperparameters
        config = {
            'hidden_dims': random.choice(param_grid['hidden_dims']),  # Use random.choice for lists
            'dropout': float(np.random.choice(param_grid['dropout'])),
            'learning_rate': float(np.random.choice(param_grid['learning_rate'])),
            'batch_size': int(np.random.choice(param_grid['batch_size'])),
            'weight_decay': float(np.random.choice(param_grid['weight_decay'])),
            'epochs': default_config['epochs'],
            'patience': default_config['patience'],
        }
        
        print(f"\nTrial {trial+1}/{n_trials}")
        print(f"  Config: hidden={config['hidden_dims']}, dropout={config['dropout']:.2f}, "
              f"lr={config['learning_rate']:.4f}, bs={config['batch_size']}, wd={config['weight_decay']:.6f}")
        
        # Create model
        model = RouterNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout']
        ).to(DEVICE)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # Train model
        model, val_loss = train_model(model, train_loader, val_loader, config, model_names, class_weights)
        
        print(f"  Final validation loss: {val_loss:.4f}")
        
        # Update best config
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = config.copy()
            best_model = model
            print(f"  *** New best model! ***")
    
    print("\n" + "="*80)
    print("Best hyperparameters found:")
    print("="*80)
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    
    return best_model, best_config

# ---------- Main training pipeline ----------
def train_router():
    """Train router using pre-computed embeddings and labels."""
    # Define models to train on
    selected_models = [
        'gpt-5-chat',
        'nvidia/nvidia-nemotron-nano-9b-v2',
        'Qwen/Qwen3-VL-8B-Instruct'
    ]
    
    # Load training data (filtered to selected models only)
    X_train_full, y_train_full, model_names, train_prompts = load_data(
        TRAIN_JSON, TRAIN_EMBEDDINGS, selected_models=selected_models
    )
    
    # Load test data (filtered to selected models only)
    X_test, y_test, _, test_prompts = load_data(
        TEST_JSON, TEST_EMBEDDINGS, selected_models=selected_models
    )
    
    print(f"Training set: {X_train_full.shape[0]} samples, {X_train_full.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Models: {model_names}")
    print(f"Label distribution (train):")
    for i, model in enumerate(model_names):
        pos_rate = y_train_full[:, i].mean()
        print(f"  {model}: {pos_rate:.3f} positive rate")
    
    # Split training data into train/val for hyperparameter tuning
    val_size = int(0.15 * len(X_train_full))
    indices = np.random.permutation(len(X_train_full))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_train = X_train_full[train_indices]
    y_train = y_train_full[train_indices]
    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]
    
    print(f"\nSplit: {len(X_train)} train, {len(X_val)} validation")
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    # Compute class weights to handle data imbalance (if enabled)
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(y_train, model_names, method='balanced')
    else:
        print("\nClass weighting disabled (USE_CLASS_WEIGHTS=False)")
    
    # Train model with or without hyperparameter tuning
    if TUNE_HYPERPARAMETERS:
        model, best_config = tune_hyperparameters(
            X_train, y_train, X_val, y_val, model_names, class_weights, n_trials=10
        )
    else:
        print("\nTraining neural network with default parameters...")
        print("(Set TUNE_HYPERPARAMETERS=True to optimize hyperparameters)")
        
        config = default_config.copy()
        
        model = RouterNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout']
        ).to(DEVICE)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        print("\nTraining...")
        model, val_loss = train_model(model, train_loader, val_loader, config, model_names, class_weights)
        best_config = config
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATION ON TEST SET")
    print("="*80)
    
    results = evaluate_router(model, X_test, y_test, model_names)
    
    # Print results
    print("\nPer-model metrics (on test set):")
    print("="*80)
    for model_name, m in results['metrics'].items():
        print(f"  {model_name:40s}: AUC={m['auc']:.3f}, acc={m['accuracy']:.3f}, "
              f"f1={m['f1']:.3f}, pos_rate={m['positive_rate']:.3f}")
    
    print("\n" + "="*80)
    print("Baseline (always choose single model) accuracies (fraction correct):")
    print("="*80)
    for model_name, acc in results['always_acc'].items():
        print(f"  {model_name:40s}: {acc:.3f}")
    
    print("\n" + "="*80)
    print("Router model selection distribution on test set:")
    print("="*80)
    for model_name, stats in results['model_selection_counts'].items():
        print(f"  {model_name:40s}: selected {stats['count']:4d} times "
              f"({stats['percentage']:5.1f}%) - accuracy when chosen: {stats['accuracy_when_chosen']:.3f}")
    
    print("\n" + "="*80)
    print(f"Router system accuracy (choose model with highest predicted prob): "
          f"{results['system_accuracy']:.3f}")
    print(f"Oracle (if you always picked any correct model when available): "
          f"{results['any_correct']:.3f}")
    print("="*80)
    print("\nNote: Oracle is an upper bound; a perfect router achieves that when it "
          "picks a correct model whenever one exists.")
    
    # Save model and config
    out_dir = Path("router_artifacts")
    out_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': best_config,
        'model_names': model_names,
        'input_dim': input_dim,
        'output_dim': output_dim,
    }, out_dir / "nn_router.pth")
    
    print(f"\nSaved model to {out_dir.resolve()}/nn_router.pth")
    
    return model, model_names, best_config, (X_test, y_test, results['proba_test'], test_prompts)

# ---------- Inference utility ----------
def load_router(model_path="router_artifacts/nn_router.pth"):
    """Load the trained router model."""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    model = RouterNetwork(
        input_dim=checkpoint['input_dim'],
        output_dim=checkpoint['output_dim'],
        hidden_dims=checkpoint['config']['hidden_dims'],
        dropout=checkpoint['config']['dropout']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['model_names'], checkpoint['config']

def route_embeddings(embeddings, model, model_names, return_probs=False, model_thresholds=None):
    """
    Given embeddings, returns chosen model names and optionally probabilities.
    
    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        model: trained RouterNetwork
        model_names: list of model names
        return_probs: whether to return probability scores
        model_thresholds: optional dict mapping model names to minimum confidence thresholds.
                         If a model is chosen but prob < threshold, route to next best model.
                         Example: {'gpt-5-chat': 0.65, 'Qwen/Qwen3-VL-8B-Instruct': 0.55}
                         Suggested values: 0.60 (moderate), 0.65 (significant), 0.70+ (aggressive)
    
    Returns:
        chosen_models: list of chosen model names
        proba (optional): probability scores for each model
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(embeddings).to(DEVICE)
        proba = model(X_tensor).cpu().numpy()
    
    # Choose model with highest probability
    chosen_idx = np.argmax(proba, axis=1)
    
    # Apply model thresholds if specified
    if model_thresholds is not None and len(model_thresholds) > 0:
        # Get indices for models with thresholds
        threshold_map = {}
        for model_name, threshold in model_thresholds.items():
            if model_name in model_names:
                model_idx = model_names.index(model_name)
                threshold_map[model_idx] = threshold
        
        # Override choices that don't meet thresholds
        for i in range(len(chosen_idx)):
            chosen_model_idx = chosen_idx[i]
            
            # Check if chosen model has a threshold and doesn't meet it
            if chosen_model_idx in threshold_map:
                threshold = threshold_map[chosen_model_idx]
                if proba[i, chosen_model_idx] < threshold:
                    # Find next best model that either has no threshold or meets its threshold
                    proba_copy = proba[i].copy()
                    
                    # Try models in order of probability until we find one that meets threshold
                    sorted_indices = np.argsort(proba_copy)[::-1]  # Descending order
                    
                    for candidate_idx in sorted_indices:
                        # Skip the originally chosen model (already failed threshold)
                        if candidate_idx == chosen_model_idx:
                            continue
                        
                        # Check if candidate has a threshold
                        if candidate_idx in threshold_map:
                            candidate_threshold = threshold_map[candidate_idx]
                            if proba_copy[candidate_idx] >= candidate_threshold:
                                chosen_idx[i] = candidate_idx
                                break
                        else:
                            # No threshold for this model, so it's acceptable
                            chosen_idx[i] = candidate_idx
                            break
    
    chosen_models = [model_names[i] for i in chosen_idx]
    if return_probs:
        return chosen_models, proba
    return chosen_models

# ---------- Example usage ----------
if __name__ == "__main__":
    # Train
    model, model_names, config, test_info = train_router()
    X_test, y_test, proba_test, test_prompts = test_info
    
    # Test different threshold configurations (GPT-5 and Nemotron)
    test_model_thresholds(model, X_test, y_test, model_names)
    
    # Show some example routing decisions on test set (with different threshold configs)
    print("\n" + "="*100)
    print("Sample routing decisions on test set (NO THRESHOLD):")
    print("="*100)
    
    chosen, probs = route_embeddings(X_test[:10], model, model_names, return_probs=True)
    for i, (prompt, choice, prob) in enumerate(zip(test_prompts[:10], chosen, probs)):
        print(f"\n{i+1}. PROMPT: {prompt[:80]}...")
        print(f"   ROUTED TO: {choice}")
        prob_dict = {model_names[j]: f"{float(prob[j]):.3f}" for j in range(len(model_names))}
        print(f"   PROBS: {prob_dict}")
        # Show which models were actually correct
        correct_models = [model_names[j] for j in range(len(model_names)) if y_test[i, j] == 1]
        print(f"   CORRECT: {correct_models}")
    
    # Show routing with GPT-5 threshold only
    print("\n" + "="*105)
    print("Sample routing decisions (WITH GPT-5 THRESHOLD=0.65):")
    print("="*105)
    
    chosen_gpt5, probs_gpt5 = route_embeddings(
        X_test[:10], model, model_names, return_probs=True, 
        model_thresholds={'gpt-5-chat': 0.65}
    )
    for i, (prompt, choice, prob) in enumerate(zip(test_prompts[:10], chosen_gpt5, probs_gpt5)):
        print(f"\n{i+1}. PROMPT: {prompt[:80]}...")
        print(f"   ROUTED TO: {choice}")
        prob_dict = {model_names[j]: f"{float(prob[j]):.3f}" for j in range(len(model_names))}
        print(f"   PROBS: {prob_dict}")
        correct_models = [model_names[j] for j in range(len(model_names)) if y_test[i, j] == 1]
        print(f"   CORRECT: {correct_models}")
        if chosen[i] != choice:
            print(f"   ⚠️  ROUTING CHANGED: {chosen[i]} → {choice} (due to GPT-5 threshold)")
    
    # Show routing with both GPT-5 and Qwen-VL thresholds
    print("\n" + "="*105)
    print("Sample routing decisions (WITH GPT-5=0.65 & QWEN-VL=0.55 THRESHOLDS):")
    print("="*105)
    
    chosen_both, probs_both = route_embeddings(
        X_test[:10], model, model_names, return_probs=True,
        model_thresholds={'gpt-5-chat': 0.65, 'Qwen/Qwen3-VL-8B-Instruct': 0.55}
    )
    for i, (prompt, choice, prob) in enumerate(zip(test_prompts[:10], chosen_both, probs_both)):
        print(f"\n{i+1}. PROMPT: {prompt[:80]}...")
        print(f"   ROUTED TO: {choice}")
        prob_dict = {model_names[j]: f"{float(prob[j]):.3f}" for j in range(len(model_names))}
        print(f"   PROBS: {prob_dict}")
        correct_models = [model_names[j] for j in range(len(model_names)) if y_test[i, j] == 1]
        print(f"   CORRECT: {correct_models}")
        if chosen[i] != choice:
            print(f"   ⚠️  ROUTING CHANGED: {chosen[i]} → {choice} (due to thresholds)")

