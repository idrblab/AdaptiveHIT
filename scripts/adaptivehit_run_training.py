# adaptivehit_run_training.py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import time
import warnings
warnings.filterwarnings('ignore')
import json

from adaptivehit_config import AdaptiveHITConfig
from adaptivehit_data_loader import create_data_loaders
from adaptivehit_learner import AdaptiveHITModel, CombinedLoss


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, optimizer, criterion, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_weights = []

    for batch_idx, batch_data in enumerate(dataloader):
        if batch_data is None:
            continue

        if len(batch_data) == 5:
            indices, prot_feats, drug_feats, base_preds, labels = batch_data
        elif len(batch_data) == 4:
            indices, prot_feats, drug_feats, base_preds = batch_data
            labels = None
        else:
            continue

        if prot_feats is not None:
            prot_feats = prot_feats.to(device)
        if drug_feats is not None:
            drug_feats = drug_feats.to(device)
        base_preds = base_preds.to(device)
        if labels is not None:
            labels = labels.to(device)

        optimizer.zero_grad()
        final_pred, weights = model(prot_feats, drug_feats, base_preds)

        if labels is not None:
            loss, loss_dict = criterion(final_pred, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(final_pred.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_weights.extend(weights.detach().cpu().numpy())

        if batch_idx % 100 == 0 and config.debug_mode:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")

    if len(all_preds) > 0:
        avg_loss = total_loss / len(dataloader)
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        return avg_loss, auc, auprc, all_preds, all_labels, all_weights
    else:
        return 0, 0, 0, [], [], []


def validate(model, dataloader, criterion, device, config):
    """Validation function"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_weights = []

    with torch.no_grad():
        for batch_data in dataloader:
            if batch_data is None:
                continue

            if len(batch_data) == 5:
                indices, prot_feats, drug_feats, base_preds, labels = batch_data
            elif len(batch_data) == 4:
                indices, prot_feats, drug_feats, base_preds = batch_data
                labels = None
            else:
                continue

            if prot_feats is not None:
                prot_feats = prot_feats.to(device)
            if drug_feats is not None:
                drug_feats = drug_feats.to(device)
            base_preds = base_preds.to(device)
            if labels is not None:
                labels = labels.to(device)

            final_pred, weights = model(prot_feats, drug_feats, base_preds)

            if labels is not None:
                loss, loss_dict = criterion(final_pred, labels, weights)
                total_loss += loss.item()
                all_preds.extend(final_pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_weights.extend(weights.cpu().numpy())

    if len(all_preds) > 0:
        avg_loss = total_loss / len(dataloader)
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)

        best_threshold = 0.5
        pred_labels = [1 if p >= best_threshold else 0 for p in all_preds]
        accuracy = accuracy_score(all_labels, pred_labels)
        f1 = f1_score(all_labels, pred_labels)

        return avg_loss, auc, auprc, accuracy, f1, all_preds, all_labels, all_weights
    else:
        return 0, 0, 0, 0, 0, [], [], []


def test(model, dataloader, device, config):
    """Test function"""
    model.eval()
    all_preds = []
    all_labels = []
    all_weights = []

    with torch.no_grad():
        for batch_data in dataloader:
            if batch_data is None:
                continue

            if len(batch_data) == 5:
                indices, prot_feats, drug_feats, base_preds, labels = batch_data
            elif len(batch_data) == 4:
                indices, prot_feats, drug_feats, base_preds = batch_data
                labels = None
            else:
                continue

            if prot_feats is not None:
                prot_feats = prot_feats.to(device)
            if drug_feats is not None:
                drug_feats = drug_feats.to(device)
            base_preds = base_preds.to(device)
            if labels is not None:
                labels = labels.to(device)

            final_pred, weights = model(prot_feats, drug_feats, base_preds)

            if labels is not None:
                all_preds.extend(final_pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_weights.extend(weights.cpu().numpy())

    if len(all_preds) > 0:
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)

        best_threshold = 0.5
        pred_labels = [1 if p >= best_threshold else 0 for p in all_preds]
        accuracy = accuracy_score(all_labels, pred_labels)
        f1 = f1_score(all_labels, pred_labels)

        return auc, auprc, accuracy, f1, all_preds, all_labels, all_weights
    else:
        return 0, 0, 0, 0, [], [], []


def main():
    parser = argparse.ArgumentParser(description='AdaptiveHIT for DTI prediction')

    # Data parameters
    parser.add_argument('--dataset', type=str, default='alldataset',
                        help='Dataset name')
    parser.add_argument('--data_subdir', type=str, default='alldataset',
                        help='Data subdirectory')
    parser.add_argument('--input_dir', type=str, default='data/',
                        help='Input data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/',
                        help='Output directory for results')

    # Model parameters
    parser.add_argument('--strategy', type=str, default='probability_only',
                        choices=['probability_only', 'full_representations'],
                        help='AdaptiveHIT strategy')
    parser.add_argument('--fusion_method', type=str, default='concat',
                        choices=['concat', 'gate', 'prob_attention'],
                        help='Fusion method (for full_representations strategy)')
    parser.add_argument('--hidden_dims', type=str, default='256,128',
                        help='Hidden dimensions for meta network')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience')

    # Weight balance parameters
    parser.add_argument('--use_weight_balance', action='store_true',
                        help='Enable weight balance loss')
    parser.add_argument('--balance_loss_weight', type=float, default=0.5,
                        help='Weight for balance loss')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--model_names', type=str, default='TransformerCPI,ConPLex,DeepConv-DTI,DrugBAN',
                    help='Comma-separated list of base model names used in the AdaptiveHIT')
    args = parser.parse_args()

    # Create config
    config = AdaptiveHITConfig()

    # Update config with command line arguments
    config.strategy = args.strategy
    config.fusion_method = args.fusion_method
    config.hidden_dims = tuple(int(x) for x in args.hidden_dims.split(','))
    config.dropout = args.dropout
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.early_stopping = args.early_stopping
    config.use_weight_balance = args.use_weight_balance
    config.balance_loss_weight = args.balance_loss_weight
    config.seed = args.seed
    config.num_workers = args.num_workers
    config.debug_mode = args.debug
    config.data_subdir = args.data_subdir
    config.data_dir = args.input_dir
    config.dataset = args.dataset
    config.model_names = [name.strip() for name in args.model_names.split(',') if name.strip()]
    
    if args.strategy == "full_representations":
        config.use_protein_representation = True
        config.use_drug_representation = True

    # Set device
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed
    set_seed(config.seed)

    print(f"\n{'='*80}")
    print(f"AdaptiveHIT Training")
    print(f"{'='*80}")
    print(f"Strategy: {config.strategy}")
    print(f"Device: {config.device}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("[Data] Loading data...")
    train_df = pd.read_csv(os.path.join(args.input_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(args.input_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(args.input_dir, 'test.csv'))

    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, config
    )

    # Initialize model
    print("\n[Model] Initializing AdaptiveHIT...")
    model = AdaptiveHITModel(config).to(config.device)

    # Initialize loss function
    criterion = CombinedLoss(config)

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Initialize scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # Training loop
    print("\n[Training] Starting training...")
    best_val_auprc = 0
    best_epoch = 0
    patience_counter = 0

    train_results = []
    val_results = []

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_auc, train_auprc, train_preds, train_labels, train_weights = train_epoch(
            model, train_loader, optimizer, criterion, config.device, config
        )

        # Validate
        val_loss, val_auc, val_auprc, val_acc, val_f1, val_preds, val_labels, val_weights = validate(
            model, val_loader, criterion, config.device, config
        )

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print results
        print(f"Epoch {epoch+1:3d}/{config.epochs} | Time: {epoch_time:.1f}s")
        print(f"  Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, AUPRC: {train_auprc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, AUPRC: {val_auprc:.4f}, ACC: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Save results
        train_results.append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'auc': train_auc,
            'auprc': train_auprc
        })

        val_results.append({
            'epoch': epoch + 1,
            'loss': val_loss,
            'auc': val_auc,
            'auprc': val_auprc,
            'accuracy': val_acc,
            'f1': val_f1
        })

        # Check for improvement
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auprc': val_auprc,
                'config': config
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  -> New best model saved! (AUPRC: {val_auprc:.4f})")
            # ========== save config.json ==========

            config_path = os.path.join(args.output_dir, 'config.json')
            with open(config_path, 'w') as f:
                # Convert config to serializable dict
                config_dict = {
                    k: v for k, v in config.__dict__.items()
                    if not callable(v) and not isinstance(v, torch.device)
                }
                # Handle device specially
                config_dict['device'] = str(config.device)
                json.dump(config_dict, f, indent=4, default=str)
            print(f"  -> Config saved to {config_path}")

        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.early_stopping:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    print(f"\n[Training] Completed! Best validation AUPRC: {best_val_auprc:.4f} at epoch {best_epoch}")

    # Load best model and test
    print("\n[Testing] Loading best model and evaluating on test set...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_auc, test_auprc, test_acc, test_f1, test_preds, test_labels, test_weights = test(
        model, test_loader, config.device, config
    )

    print(f"\n{'='*80}")
    print(f"Test Results")
    print(f"{'='*80}")
    print(f"AUC:   {test_auc:.4f}")
    print(f"AUPRC: {test_auprc:.4f}")
    print(f"ACC:   {test_acc:.4f}")
    print(f"F1:    {test_f1:.4f}")
    print(f"{'='*80}\n")

    # Save results
    results_df = pd.DataFrame({
        'epoch': [r['epoch'] for r in val_results],
        'val_auc': [r['auc'] for r in val_results],
        'val_auprc': [r['auprc'] for r in val_results],
        'val_accuracy': [r['accuracy'] for r in val_results],
        'val_f1': [r['f1'] for r in val_results]
    })
    results_df.to_csv(os.path.join(args.output_dir, 'validation_results.csv'), index=False)

    # Save test results
    test_summary = pd.DataFrame([{
        'test_auc': test_auc,
        'test_auprc': test_auprc,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'best_epoch': best_epoch
    }])
    test_summary.to_csv(os.path.join(args.output_dir, 'test_results.csv'), index=False)

    if args.save_predictions:
        predictions_df = pd.DataFrame({
            'true_label': test_labels,
            'pred_score': test_preds
        })
        predictions_df.to_csv(os.path.join(args.output_dir, 'test_predictions.csv'), index=False)
        print(f"Predictions saved to {os.path.join(args.output_dir, 'test_predictions.csv')}")

    # Print average weights if available
    if len(test_weights) > 0:
        avg_weights = np.mean(test_weights, axis=0)
        print(f"\nAverage model weights on test set:")
        for i, model_name in enumerate(config.model_names):
            print(f"  {model_name}: {avg_weights[i]:.4f}")


if __name__ == "__main__":
    main()