"""
src/analysis/semantic_alignment.py

Core novelty of the paper:
Quantifies semantic distance between CheXpert and Kaggle label spaces
using BioMedBERT embeddings. Tests whether transfer performance
correlates with semantic label distance rather than just domain shift.
"""

import numpy as np
import torch
import json
import os
from pathlib import Path
from itertools import combinations
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity


# ── LABEL DESCRIPTIONS ────────────────────────────────────────────────────────
# Rich clinical descriptions fed to BioMedBERT — more context = better embeddings

LABEL_DESCRIPTIONS = {
    # CheXpert labels
    'CheXpert/Pneumonia': (
        'Pneumonia on chest X-ray: infectious or inflammatory process causing '
        'airspace consolidation, ground-glass opacity, and lobar or segmental '
        'infiltrates. May present as bronchopneumonia or lobar pneumonia.'
    ),
    'CheXpert/Lung Opacity': (
        'Lung opacity on chest radiograph: increased attenuation of lung parenchyma '
        'that obscures underlying vessels and bronchi. Non-specific finding that '
        'encompasses consolidation, ground-glass opacity, and atelectasis.'
    ),
    'CheXpert/Consolidation': (
        'Pulmonary consolidation: complete replacement of alveolar air by fluid, '
        'pus, blood, or cells. Appears as homogeneous opacification on chest X-ray '
        'with air bronchograms. Common in bacterial pneumonia and organizing pneumonia.'
    ),
    'CheXpert/Edema': (
        'Pulmonary edema: accumulation of fluid in lung interstitium and alveoli. '
        'Manifests as perihilar haziness, Kerley B lines, and bilateral airspace '
        'opacities. Often due to cardiac failure or capillary leak.'
    ),
    'CheXpert/Pleural Effusion': (
        'Pleural effusion: fluid collection in the pleural space between lung and '
        'chest wall. Appears as blunting of costophrenic angles and dependent opacity '
        'on upright chest radiograph.'
    ),
    'CheXpert/Atelectasis': (
        'Atelectasis: collapse or incomplete expansion of lung tissue. Produces '
        'increased opacity with volume loss, shift of fissures, and elevation of '
        'hemidiaphragm. Can be subsegmental, segmental, lobar, or complete.'
    ),

    # Kaggle label
    'Kaggle/Pneumonia': (
        'Pneumonia diagnosis on pediatric chest X-ray: presence of lung infection '
        'with radiographic evidence including focal or diffuse opacities, '
        'consolidation, or interstitial infiltrates indicating bacterial or viral '
        'pneumonia in a child.'
    ),
    'Kaggle/Normal': (
        'Normal chest X-ray with clear lung fields, no infiltrates, no consolidation, '
        'no pleural effusion, and no cardiomegaly. Trachea midline, costophrenic '
        'angles sharp, diaphragm well-defined.'
    ),
}


# ── BIOMEDBERT ENCODER ────────────────────────────────────────────────────────

class BioMedBERTEncoder:
    """
    Encodes clinical text descriptions using Microsoft BiomedBERT.
    Uses mean pooling over token embeddings as the sentence representation.
    """

    MODEL_NAME = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Loading BioMedBERT on {self.device}...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()
        print('BioMedBERT loaded.')

    def encode(self, texts, batch_size=8):
        """Encode a list of strings, return (N, 768) numpy array."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**encoded)
            # Mean pool over non-padding tokens
            attention_mask = encoded['attention_mask']
            token_embeddings = output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            embeddings = embeddings / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def encode_labels(self, label_descriptions=None):
        """
        Encode all labels. Returns dict of {label_name: embedding}.
        """
        if label_descriptions is None:
            label_descriptions = LABEL_DESCRIPTIONS
        labels = list(label_descriptions.keys())
        texts = list(label_descriptions.values())
        embeddings = self.encode(texts)
        return {label: emb for label, emb in zip(labels, embeddings)}


# ── SEMANTIC DISTANCE MATRIX ──────────────────────────────────────────────────

class SemanticDistanceMatrix:
    """
    Computes and stores pairwise semantic distances between all labels.
    Core of the paper's methodology.
    """

    def __init__(self, encoder=None):
        self.encoder = encoder
        self.embeddings = None
        self.labels = None
        self.similarity_matrix = None
        self.distance_matrix = None

    def compute(self, label_descriptions=None, save_path=None):
        """Compute full pairwise similarity and distance matrices."""
        if label_descriptions is None:
            label_descriptions = LABEL_DESCRIPTIONS

        print('Computing BioMedBERT embeddings for all labels...')
        embedding_dict = self.encoder.encode_labels(label_descriptions)

        self.labels = list(embedding_dict.keys())
        embedding_array = np.array([embedding_dict[l] for l in self.labels])

        # Cosine similarity → distance
        self.similarity_matrix = cosine_similarity(embedding_array)
        self.distance_matrix = 1 - self.similarity_matrix

        if save_path:
            np.save(f'{save_path}/label_embeddings.npy', embedding_array)
            np.save(f'{save_path}/similarity_matrix.npy', self.similarity_matrix)
            np.save(f'{save_path}/distance_matrix.npy', self.distance_matrix)
            with open(f'{save_path}/label_names.json', 'w') as f:
                json.dump(self.labels, f)
            print(f'Matrices saved to {save_path}')

        return self

    def load(self, load_path):
        """Load precomputed matrices."""
        self.similarity_matrix = np.load(f'{load_path}/similarity_matrix.npy')
        self.distance_matrix = np.load(f'{load_path}/distance_matrix.npy')
        with open(f'{load_path}/label_names.json') as f:
            self.labels = json.load(f)
        return self

    def get_distance(self, label_a, label_b):
        """Get semantic distance between two specific labels."""
        i = self.labels.index(label_a)
        j = self.labels.index(label_b)
        return self.distance_matrix[i, j]

    def get_chexpert_to_kaggle_distances(self):
        """
        Key result: distance from each CheXpert label to Kaggle/Pneumonia.
        This is what we correlate with transfer performance.
        """
        target = 'Kaggle/Pneumonia'
        chexpert_labels = [l for l in self.labels if l.startswith('CheXpert/')]
        distances = {
            label: self.get_distance(label, target)
            for label in chexpert_labels
        }
        return dict(sorted(distances.items(), key=lambda x: x[1]))

    def plot_heatmap(self, save_path=None, figsize=(12, 10)):
        """Plot similarity heatmap — Figure 1 of the paper."""
        fig, ax = plt.subplots(figsize=figsize)

        # Clean label names for display
        display_labels = [l.replace('CheXpert/', 'CXP: ').replace('Kaggle/', 'KGL: ')
                          for l in self.labels]

        # Highlight Kaggle labels differently
        colors = ['#2196F3' if l.startswith('KGL') else '#FF9800'
                  for l in display_labels]

        mask = np.zeros_like(self.similarity_matrix, dtype=bool)

        sns.heatmap(
            self.similarity_matrix,
            xticklabels=display_labels,
            yticklabels=display_labels,
            cmap='RdYlGn',
            vmin=0, vmax=1,
            annot=True,
            fmt='.2f',
            linewidths=0.5,
            ax=ax,
            cbar_kws={'label': 'Cosine Similarity'}
        )

        ax.set_title(
            'Semantic Similarity Matrix: CheXpert vs Kaggle Labels\n'
            '(BioMedBERT embeddings, cosine similarity)',
            fontsize=14, fontweight='bold', pad=20
        )
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Heatmap saved to {save_path}')
        return fig

    def plot_distance_bar(self, save_path=None):
        """
        Bar chart of semantic distances from CheXpert labels to Kaggle/Pneumonia.
        Figure 2 of the paper — shows which CheXpert labels are 'closest' clinically.
        """
        distances = self.get_chexpert_to_kaggle_distances()
        labels = [l.replace('CheXpert/', '') for l in distances.keys()]
        values = list(distances.values())

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(labels, values,
                       color=['#4CAF50' if v < 0.3 else '#FF9800' if v < 0.5 else '#F44336'
                              for v in values])

        ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.5,
                   label='High similarity (d < 0.3)')
        ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5,
                   label='Moderate similarity (d < 0.5)')

        ax.set_xlabel('Semantic Distance from Kaggle/Pneumonia', fontsize=12)
        ax.set_title(
            'CheXpert Label Distance to Kaggle "Pneumonia"\n'
            'Predicts transfer performance degradation',
            fontsize=13, fontweight='bold'
        )
        ax.legend()
        for bar, val in zip(bars, values):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


# ── CORRELATION ANALYSIS ──────────────────────────────────────────────────────

def correlate_distance_with_performance(distance_matrix, performance_results,
                                         save_path=None):
    """
    KEY EXPERIMENT: Does semantic distance predict transfer failure?

    performance_results: dict like
        {'CheXpert/Pneumonia': {'auc': 0.89, 'f1': 0.82},
         'CheXpert/Lung Opacity': {'auc': 0.71, 'f1': 0.65}, ...}

    Returns: correlation coefficients and p-values
    """
    distances = []
    aucs = []
    f1s = []

    for label, metrics in performance_results.items():
        d = distance_matrix.get_distance(label, 'Kaggle/Pneumonia')
        distances.append(d)
        aucs.append(metrics['auc'])
        f1s.append(metrics['f1'])

    # Pearson and Spearman correlations
    r_auc, p_auc = pearsonr(distances, aucs)
    rho_auc, p_rho_auc = spearmanr(distances, aucs)
    r_f1, p_f1 = pearsonr(distances, f1s)

    results = {
        'pearson_r_auc': r_auc, 'pearson_p_auc': p_auc,
        'spearman_rho_auc': rho_auc, 'spearman_p_auc': p_rho_auc,
        'pearson_r_f1': r_f1, 'pearson_p_f1': p_f1,
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, values, r, p in [
        (axes[0], 'AUC-ROC', aucs, r_auc, p_auc),
        (axes[1], 'F1 Score', f1s, r_f1, p_f1),
    ]:
        labels_clean = [l.replace('CheXpert/', '') for l in performance_results.keys()]
        ax.scatter(distances, values, s=100, zorder=5, color='#2196F3')
        for i, label in enumerate(labels_clean):
            ax.annotate(label, (distances[i], values[i]),
                        textcoords='offset points', xytext=(5, 5), fontsize=9)

        # Regression line
        m, b = np.polyfit(distances, values, 1)
        x_line = np.linspace(min(distances), max(distances), 100)
        ax.plot(x_line, m * x_line + b, 'r--', alpha=0.7, linewidth=2)

        ax.set_xlabel('Semantic Distance from Kaggle/Pneumonia', fontsize=12)
        ax.set_ylabel(f'Transfer {metric}', fontsize=12)
        ax.set_title(f'Semantic Distance vs Transfer {metric}\n'
                     f'r = {r:.3f}, p = {p:.4f}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

    plt.suptitle('Semantic Label Distance Predicts Cross-Dataset Transfer Failure',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print('\n=== Correlation Results ===')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')

    return results, fig
