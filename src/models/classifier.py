"""
src/models/classifier.py

Model architectures for the paper.
We use ResNet50 as the backbone (standard in medical imaging literature).
Two heads:
  - MultiLabelHead: for CheXpert pretraining (6 labels)
  - BinaryHead: for Kaggle fine-tuning (1 label)

The backbone weights transfer between the two.
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path


CHEXPERT_LABELS = [
    'Pneumonia', 'Lung Opacity', 'Consolidation',
    'Edema', 'Pleural Effusion', 'Atelectasis'
]


class CheXpertClassifier(nn.Module):
    """
    ResNet50 trained on CheXpert 6-label multi-label classification.
    This is the pretrained model that we then analyze and adapt.
    """

    def __init__(self, num_labels=6, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            'resnet50', pretrained=pretrained, num_classes=0
        )
        feature_dim = self.backbone.num_features  # 2048 for ResNet50

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_labels)
        )
        self.label_names = CHEXPERT_LABELS[:num_labels]

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        """Return penultimate layer features — used for t-SNE / MMD analysis."""
        return self.backbone(x)

    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'label_names': self.label_names,
        }, path)
        print(f'Model saved to {path}')

    @classmethod
    def load(cls, path, **kwargs):
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(**kwargs)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'Model loaded from {path}')
        return model


class KaggleAdaptedClassifier(nn.Module):
    """
    Same ResNet50 backbone, binary head for Kaggle fine-tuning.
    Backbone weights are initialized from a trained CheXpertClassifier.
    """

    def __init__(self, chexpert_model=None, freeze_backbone=False, dropout=0.3):
        super().__init__()

        if chexpert_model is not None:
            # Transfer backbone weights
            self.backbone = chexpert_model.backbone
        else:
            self.backbone = timm.create_model(
                'resnet50', pretrained=True, num_classes=0
            )

        feature_dim = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Binary head — predicts P(Pneumonia)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze(-1)

    def get_features(self, x):
        return self.backbone(x)


class MultiOutputWrapper(nn.Module):
    """
    Wraps CheXpertClassifier to also output binary pneumonia prediction.
    Used during label confusion analysis — we want both the multi-label
    CheXpert outputs AND a binary pneumonia prediction simultaneously.
    """

    def __init__(self, chexpert_model):
        super().__init__()
        self.chexpert_model = chexpert_model

    def forward(self, x):
        # Returns (batch, 6) — all CheXpert labels
        return self.chexpert_model(x)

    def predict_pneumonia_prob(self, x):
        """Returns P(Pneumonia) from the CheXpert head (index 0)."""
        logits = self.forward(x)
        return torch.sigmoid(logits[:, 0])
