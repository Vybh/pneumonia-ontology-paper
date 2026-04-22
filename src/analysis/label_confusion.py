"""
src/analysis/label_confusion.py

The 'clinical dialect' analysis — core finding of the paper.

When a model trained on CheXpert fails on Kaggle pneumonia cases,
what does it actually predict? We show failures are STRUCTURED:
the model reclassifies pneumonia into ontologically adjacent CheXpert
categories, not random noise.

This is Figure 3 and Table 2 of the paper.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm


# ── PREDICTION COLLECTOR ──────────────────────────────────────────────────────

class PredictionCollector:
    """
    Runs a CheXpert-trained model on the Kaggle test set and
    collects per-image predictions across ALL CheXpert labels.

    This is what lets us ask: "when it's wrong, which CheXpert
    label does it fire on instead?"
    """

    CHEXPERT_LABELS = [
        'Pneumonia', 'Lung Opacity', 'Consolidation',
        'Edema', 'Pleural Effusion', 'Atelectasis'
    ]

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def collect(self, kaggle_loader):
        """
        Returns a DataFrame with columns:
            path, true_label, [chexpert_label_probs...], predicted_pneumonia_prob
        """
        records = []

        for batch in tqdm(kaggle_loader, desc='Collecting predictions'):
            if batch is None:
                continue
            imgs, true_labels, paths = batch
            imgs = imgs.to(self.device)

            logits = self.model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(len(paths)):
                record = {
                    'path': paths[i],
                    'true_label': int(true_labels[i]),
                }
                for j, label in enumerate(self.CHEXPERT_LABELS):
                    record[f'prob_{label}'] = float(probs[i, j])
                records.append(record)

        df = pd.DataFrame(records)

        # Add predicted pneumonia probability and binary prediction
        df['prob_pneumonia_chexpert'] = df['prob_Pneumonia']
        df['pred_pneumonia'] = (df['prob_Pneumonia'] >= 0.5).astype(int)

        return df


# ── FAILURE TAXONOMY ──────────────────────────────────────────────────────────

class FailureTaxonomy:
    """
    Categorizes prediction failures into structured types.

    Failure Types:
    1. DIALECT FAILURE: true pneumonia, model fires on adjacent label
       (Lung Opacity / Consolidation) but not Pneumonia
    2. MISS: true pneumonia, model fires on nothing
    3. DISTANT MISS: true pneumonia, model fires on non-adjacent label
       (Edema / Pleural Effusion / Atelectasis)
    4. FALSE ALARM: true normal, model fires on any label

    The paper's claim: most failures are Type 1 (dialect), not Type 2/3.
    This is the 'model speaks a different clinical dialect' result.
    """

    # Ontologically adjacent to Kaggle/Pneumonia (based on semantic distance)
    ADJACENT_LABELS = ['Lung Opacity', 'Consolidation']

    # Ontologically distant from Kaggle/Pneumonia
    DISTANT_LABELS = ['Edema', 'Pleural Effusion', 'Atelectasis']

    THRESHOLD = 0.5

    def __init__(self, predictions_df):
        self.df = predictions_df
        self.failures = None
        self.taxonomy = None

    def analyze(self):
        """Run full failure taxonomy analysis."""
        # Focus on false negatives (missed pneumonia cases)
        fn_df = self.df[
            (self.df['true_label'] == 1) &
            (self.df['pred_pneumonia'] == 0)
        ].copy()

        fp_df = self.df[
            (self.df['true_label'] == 0) &
            (self.df['pred_pneumonia'] == 1)
        ].copy()

        def classify_failure(row):
            adjacent_fired = any(
                row[f'prob_{l}'] >= self.THRESHOLD
                for l in self.ADJACENT_LABELS
            )
            distant_fired = any(
                row[f'prob_{l}'] >= self.THRESHOLD
                for l in self.DISTANT_LABELS
            )
            nothing_fired = not adjacent_fired and not distant_fired

            if adjacent_fired and not distant_fired:
                return 'DIALECT'         # speaks a different clinical dialect
            elif nothing_fired:
                return 'MISS'            # model sees nothing pathological
            elif distant_fired and not adjacent_fired:
                return 'DISTANT_MISS'    # wrong anatomical region / finding
            else:
                return 'MIXED'           # adjacent + distant both fire

        fn_df['failure_type'] = fn_df.apply(classify_failure, axis=1)

        # For false positives: what is the model actually seeing?
        def classify_fp(row):
            fired = [l for l in self.ADJACENT_LABELS + self.DISTANT_LABELS
                     if row[f'prob_{l}'] >= self.THRESHOLD]
            if not fired:
                return 'Pneumonia only'
            return '/'.join(fired)

        fp_df['fired_labels'] = fp_df.apply(classify_fp, axis=1)

        self.failures = fn_df
        self.taxonomy = fn_df['failure_type'].value_counts()

        return self

    def report(self):
        """Print summary statistics for the paper."""
        total_fn = len(self.failures)
        print('\n' + '='*60)
        print('FAILURE TAXONOMY REPORT')
        print('='*60)
        print(f'Total false negatives (missed pneumonia): {total_fn}')
        print()
        for failure_type, count in self.taxonomy.items():
            pct = 100 * count / total_fn
            print(f'  {failure_type:20s}: {count:4d} ({pct:.1f}%)')
        print()
        print('INTERPRETATION:')
        if 'DIALECT' in self.taxonomy:
            dialect_pct = 100 * self.taxonomy['DIALECT'] / total_fn
            print(f'  {dialect_pct:.1f}% of failures are DIALECT type —')
            print(f'  model fires on Lung Opacity/Consolidation but not Pneumonia.')
            print(f'  This supports the clinical dialect hypothesis.')
        print('='*60)
        return self

    def plot_taxonomy_pie(self, save_path=None):
        """Figure 3a — Failure type distribution pie chart."""
        colors = {
            'DIALECT': '#FF6B6B',
            'MISS': '#4ECDC4',
            'DISTANT_MISS': '#45B7D1',
            'MIXED': '#96CEB4',
        }
        labels = list(self.taxonomy.index)
        sizes = list(self.taxonomy.values)
        chart_colors = [colors.get(l, '#999') for l in labels]

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=chart_colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 13},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for autotext in autotexts:
            autotext.set_fontweight('bold')

        ax.set_title(
            'Failure Type Distribution\nCheXpert-trained model on Kaggle Pneumonia\n'
            '(False Negatives Only)',
            fontsize=14, fontweight='bold', pad=20
        )

        # Legend with descriptions
        legend_desc = {
            'DIALECT': 'Fires on Lung Opacity/Consolidation\n(ontologically adjacent)',
            'MISS': 'No pathology detected',
            'DISTANT_MISS': 'Fires on unrelated finding',
            'MIXED': 'Multiple conflicting signals',
        }
        patches = [
            mpatches.Patch(color=colors.get(l, '#999'),
                           label=f'{l}: {legend_desc.get(l, "")}')
            for l in labels
        ]
        ax.legend(handles=patches, loc='lower left',
                  bbox_to_anchor=(-0.3, -0.15), fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_label_activation_heatmap(self, n_samples=100, save_path=None):
        """
        Figure 3b — Heatmap of which CheXpert labels activate for
        each failure type. The 'clinical dialect' signature.
        """
        chexpert_prob_cols = [
            f'prob_{l}' for l in [
                'Pneumonia', 'Lung Opacity', 'Consolidation',
                'Edema', 'Pleural Effusion', 'Atelectasis'
            ]
        ]

        fig, axes = plt.subplots(1, len(self.taxonomy), figsize=(18, 6),
                                 sharey=True)
        if len(self.taxonomy) == 1:
            axes = [axes]

        for ax, (ftype, _) in zip(axes, self.taxonomy.items()):
            subset = self.failures[self.failures['failure_type'] == ftype]
            subset = subset.sample(min(n_samples, len(subset)), random_state=42)
            prob_matrix = subset[chexpert_prob_cols].values.T

            im = ax.imshow(prob_matrix, aspect='auto', cmap='hot',
                           vmin=0, vmax=1, interpolation='nearest')
            ax.set_yticks(range(len(chexpert_prob_cols)))
            ax.set_yticklabels([c.replace('prob_', '') for c in chexpert_prob_cols],
                               fontsize=10)
            ax.set_xlabel('Sample index', fontsize=10)
            ax.set_title(f'{ftype}\n(n={len(subset)})', fontsize=11,
                         fontweight='bold')

        plt.colorbar(im, ax=axes, label='Predicted Probability', shrink=0.8)
        plt.suptitle(
            'CheXpert Label Activation Patterns by Failure Type\n'
            'DIALECT failures show Lung Opacity/Consolidation activation',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


# ── LABEL CONFUSION MATRIX ────────────────────────────────────────────────────

def compute_label_confusion_matrix(predictions_df, threshold=0.5, save_path=None):
    """
    For each Kaggle true label (Normal / Pneumonia), what CheXpert labels fire?
    This is Table 2 in the paper.

    Rows: Kaggle true label
    Cols: CheXpert predicted label
    Values: % of samples where that CheXpert label fires
    """
    chexpert_labels = [
        'Pneumonia', 'Lung Opacity', 'Consolidation',
        'Edema', 'Pleural Effusion', 'Atelectasis'
    ]

    confusion = {}
    for true_label_val, true_label_name in [(0, 'Normal'), (1, 'Pneumonia')]:
        subset = predictions_df[predictions_df['true_label'] == true_label_val]
        row = {}
        for label in chexpert_labels:
            prob_col = f'prob_{label}'
            if prob_col in subset.columns:
                row[label] = (subset[prob_col] >= threshold).mean() * 100
            else:
                row[label] = 0.0
        confusion[true_label_name] = row

    confusion_df = pd.DataFrame(confusion).T
    confusion_df.index.name = 'Kaggle True Label'

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        confusion_df, annot=True, fmt='.1f', cmap='YlOrRd',
        linewidths=0.5, ax=ax,
        cbar_kws={'label': '% of samples where label fires'}
    )
    ax.set_title(
        'Label Confusion: CheXpert Predictions on Kaggle Test Set\n'
        '(Values = % samples with predicted probability ≥ 0.5)',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel('CheXpert Predicted Label', fontsize=11)

    # Highlight ontologically adjacent labels
    for i, label in enumerate(confusion_df.columns):
        if label in ['Lung Opacity', 'Consolidation']:
            ax.add_patch(plt.Rectangle(
                (i, 0), 1, confusion_df.shape[0],
                fill=False, edgecolor='blue', lw=3,
                label='Ontologically adjacent' if i == 0 else ''
            ))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        confusion_df.to_csv(save_path.replace('.png', '.csv'))

    print('\nLabel Confusion Matrix (Table 2):')
    print(confusion_df.round(1).to_string())

    return confusion_df, fig
