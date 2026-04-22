"""
src/training/ontology_finetuning.py

The proposed mitigation: Ontology-Guided Fine-Tuning (OGFT).

Key idea: standard fine-tuning treats all misclassifications equally.
OGFT re-weights the loss so that:
  - Misclassifying pneumonia as 'Lung Opacity' (adjacent) = small penalty
  - Misclassifying pneumonia as 'Normal' or 'Atelectasis' (distant) = large penalty

The weights come directly from the BioMedBERT semantic similarity scores,
making the training objective aware of clinical ontology.

This is the methodological contribution of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json


# ── ONTOLOGY-GUIDED LOSS ──────────────────────────────────────────────────────

class OntologyGuidedLoss(nn.Module):
    """
    Weighted Binary Cross Entropy where weights are derived from
    semantic similarity between CheXpert training labels and
    the Kaggle target label (Pneumonia).

    For a model trained on CheXpert labels and fine-tuned for Kaggle:
    - Labels semantically close to Kaggle/Pneumonia get LOWER loss weight
      (the model's 'dialect' confusion is penalized less aggressively)
    - Labels semantically distant get HIGHER loss weight

    Formally:
        w_i = 1 + alpha * (1 - sim(label_i, target_label))

    where sim is cosine similarity from BioMedBERT embeddings.
    """

    def __init__(self, label_weights, alpha=1.0, reduction='mean'):
        """
        Args:
            label_weights: dict {label_name: weight} or tensor of shape (num_labels,)
            alpha: scaling factor for ontological weighting
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

        if isinstance(label_weights, dict):
            # Convert to tensor — order must match model output order
            self.label_names = list(label_weights.keys())
            weights = torch.tensor(
                list(label_weights.values()), dtype=torch.float32
            )
        else:
            weights = torch.tensor(label_weights, dtype=torch.float32)

        self.register_buffer('weights', weights)

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, num_labels) raw model output
            targets: (batch, num_labels) binary targets in [0,1]
        Returns:
            scalar loss
        """
        # Per-label BCE loss (unreduced)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )  # (batch, num_labels)

        # Apply ontological weights — broadcast over batch dimension
        weighted = bce * self.weights.unsqueeze(0)  # (batch, num_labels)

        if self.reduction == 'mean':
            return weighted.mean()
        elif self.reduction == 'sum':
            return weighted.sum()
        return weighted


def compute_ontology_weights(similarity_matrix, chexpert_labels,
                              target_label='Kaggle/Pneumonia', alpha=1.0):
    """
    Compute per-label ontological weights from semantic similarity matrix.

    w_i = 1 + alpha * (1 - sim(chexpert_label_i, target_label))

    Labels close to target get weight ~1.0 (no extra penalty)
    Labels far from target get weight ~1 + alpha (strong penalty)

    Args:
        similarity_matrix: SemanticDistanceMatrix instance
        chexpert_labels: list of CheXpert label names (in model output order)
        target_label: the Kaggle label we're adapting toward
        alpha: ontological weighting strength

    Returns:
        dict {label: weight}, also prints weight table
    """
    weights = {}
    print('\n=== Ontological Loss Weights ===')
    print(f'{"Label":<25} {"Similarity":>12} {"Weight":>10}')
    print('-' * 50)

    for label in chexpert_labels:
        full_label = f'CheXpert/{label}' if not label.startswith('CheXpert/') else label
        sim = 1 - similarity_matrix.get_distance(full_label, target_label)
        weight = 1 + alpha * (1 - sim)
        weights[label] = weight
        print(f'  {label:<23} {sim:>12.4f} {weight:>10.4f}')

    print('-' * 50)
    print(f'  Alpha = {alpha}')
    print(f'  Range: [{min(weights.values()):.4f}, {max(weights.values()):.4f}]')
    return weights


# ── FINE-TUNING TRAINER ───────────────────────────────────────────────────────

class OntologyFineTuner:
    """
    Fine-tunes a CheXpert-pretrained model on Kaggle data using
    both standard BCE and ontology-guided loss, for comparison.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.history = {'standard': [], 'ontology': []}

    def finetune(self, train_loader, val_loader, loss_fn, optimizer,
                 scheduler=None, epochs=10, label='standard'):
        """
        Single fine-tuning run. Call twice — once with BCE, once with OGFT.
        """
        history = []
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_losses = []
            for batch in tqdm(train_loader, desc=f'[{label}] Epoch {epoch+1}/{epochs}'):
                if batch is None:
                    continue
                imgs, targets, _ = batch
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits = self.model(imgs)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            if scheduler:
                scheduler.step()

            # Validate
            val_metrics = self.evaluate(val_loader)
            val_metrics['train_loss'] = np.mean(train_losses)
            val_metrics['epoch'] = epoch + 1
            history.append(val_metrics)

            print(f'  Epoch {epoch+1}: loss={val_metrics["train_loss"]:.4f} | '
                  f'val_auc={val_metrics["auc"]:.4f} | '
                  f'val_f1={val_metrics["f1"]:.4f}')

        self.history[label] = history
        return history

    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate on Kaggle binary pneumonia task."""
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
        from torchmetrics.classification import BinaryCalibrationError

        self.model.eval()
        all_probs = []
        all_labels = []

        for batch in loader:
            if batch is None:
                continue
            imgs, labels, _ = batch
            imgs = imgs.to(self.device)
            logits = self.model(imgs)

            # Get pneumonia probability (assumes first output or binary head)
            probs = torch.sigmoid(logits)
            if probs.dim() > 1:
                # Multi-label model: take Pneumonia column (index 0)
                probs = probs[:, 0]

            all_probs.extend(probs.cpu().numpy())
            if labels.dim() > 1:
                labels = labels[:, 0]
            all_labels.extend(labels.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs >= 0.5).astype(int)

        return {
            'auc': roc_auc_score(all_labels, all_probs),
            'f1': f1_score(all_labels, preds, zero_division=0),
            'accuracy': accuracy_score(all_labels, preds),
            'ece': self._compute_ece(all_probs, all_labels),
        }

    def _compute_ece(self, probs, labels, n_bins=10):
        """Expected Calibration Error."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            if mask.sum() == 0:
                continue
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
        return ece / len(probs)

    def plot_comparison(self, save_path=None):
        """
        Figure 5 — Compare standard vs ontology-guided fine-tuning.
        Shows AUC, F1, and ECE curves over epochs for both methods.
        """
        if not self.history['standard'] or not self.history['ontology']:
            print('Run both fine-tuning variants before plotting.')
            return

        metrics = ['auc', 'f1', 'ece']
        metric_labels = ['AUC-ROC', 'F1 Score', 'ECE (↓)']
        colors = {'standard': '#2196F3', 'ontology': '#FF5722'}

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, metric, mlabel in zip(axes, metrics, metric_labels):
            for method, history in self.history.items():
                if not history:
                    continue
                epochs = [h['epoch'] for h in history]
                values = [h[metric] for h in history]
                ax.plot(epochs, values,
                        label=f'{"Standard BCE" if method=="standard" else "Ontology-Guided (Ours)"}',
                        color=colors[method],
                        linewidth=2.5,
                        marker='o', markersize=5)

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(mlabel, fontsize=12)
            ax.set_title(f'{mlabel} During Fine-Tuning', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

        plt.suptitle(
            'Standard vs Ontology-Guided Fine-Tuning\n'
            'CheXpert → Kaggle Transfer',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


# ── RECOVERY RATE METRIC ──────────────────────────────────────────────────────

def compute_recovery_rate(baseline_metrics, finetuned_standard, finetuned_ontology,
                           metric='auc'):
    """
    Custom metric from the paper: what fraction of degradation does each
    fine-tuning strategy recover?

    Recovery Rate = (finetuned - baseline) / (oracle - baseline)

    where oracle = performance on source domain (upper bound)

    Args:
        baseline_metrics: dict — CheXpert model zero-shot on Kaggle
        finetuned_standard: dict — after standard fine-tuning
        finetuned_ontology: dict — after ontology-guided fine-tuning

    Returns: dict with recovery rates
    """
    # We use the source domain performance as oracle approximation
    # In practice: run model on CheXpert validation set
    # For now: assume oracle is max of (standard, ontology) as lower bound

    oracle = max(finetuned_standard[metric], finetuned_ontology[metric])
    baseline = baseline_metrics[metric]

    if oracle == baseline:
        print('Warning: oracle equals baseline, recovery rate undefined')
        return {}

    def rr(method_val):
        return (method_val - baseline) / (oracle - baseline)

    results = {
        'baseline': baseline,
        'oracle_approx': oracle,
        'standard_recovery_rate': rr(finetuned_standard[metric]),
        'ontology_recovery_rate': rr(finetuned_ontology[metric]),
        'metric': metric,
    }

    print(f'\n=== Recovery Rate Analysis ({metric.upper()}) ===')
    print(f'  Baseline (zero-shot):       {baseline:.4f}')
    print(f'  Oracle (approx):            {oracle:.4f}')
    print(f'  Standard fine-tuning:       {finetuned_standard[metric]:.4f}  '
          f'(RR = {results["standard_recovery_rate"]:.2%})')
    print(f'  Ontology-guided (ours):     {finetuned_ontology[metric]:.4f}  '
          f'(RR = {results["ontology_recovery_rate"]:.2%})')
    delta = results["ontology_recovery_rate"] - results["standard_recovery_rate"]
    print(f'  OGFT improvement over standard: {delta:.2%}')

    return results
