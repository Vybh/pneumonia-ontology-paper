"""
src/data/datasets.py

Dataset classes for Kaggle Chest X-Ray and CheXpert.
Handles label normalization, uncertain label strategies,
and unified DataLoader creation for both datasets.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ── LABEL DEFINITIONS ─────────────────────────────────────────────────────────

# CheXpert labels and their clinical descriptions (used for BioMedBERT embedding)
CHEXPERT_LABEL_DESCRIPTIONS = {
    'Pneumonia':         'Pneumonia is an infection causing inflammation and fluid in the lungs',
    'Lung Opacity':      'Lung opacity refers to increased density in the lung parenchyma visible on X-ray',
    'Consolidation':     'Consolidation is alveolar airspace filling with fluid, pus, or cells',
    'Edema':             'Pulmonary edema is fluid accumulation in the lung tissue and air spaces',
    'Pleural Effusion':  'Pleural effusion is fluid accumulation in the pleural space surrounding the lung',
    'Atelectasis':       'Atelectasis is partial or complete collapse of a lung or lobe',
}

# Kaggle binary label description
KAGGLE_LABEL_DESCRIPTIONS = {
    'Pneumonia': 'Pneumonia is a lung infection causing inflammation, consolidation, and opacity on chest X-ray',
    'Normal':    'Normal chest X-ray with no pathological findings',
}

# CheXpert uncertain label handling strategies
UNCERTAIN_STRATEGIES = {
    'ignore':   -1,   # exclude from training
    'negative':  0,   # treat as no finding
    'positive':  1,   # treat as finding present
    'smooth':    0.5, # label smoothing — encode uncertainty
}


# ── KAGGLE DATASET ─────────────────────────────────────────────────────────────

class KagglePneumoniaDataset(Dataset):
    """
    Kaggle Chest X-Ray dataset (Kermany).
    Binary classification: Normal (0) vs Pneumonia (1).

    Directory structure expected:
        kaggle_root/
            chest_xray/
                train/NORMAL/*.jpeg
                train/PNEUMONIA/*.jpeg
                val/NORMAL/*.jpeg
                val/PNEUMONIA/*.jpeg
                test/NORMAL/*.jpeg
                test/PNEUMONIA/*.jpeg
    """

    def __init__(self, root, split='train', transform=None):
        assert split in ('train', 'val', 'test')
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []  # list of (path, label)
        self._load_samples()

    def _load_samples(self):
        # Handle nested chest_xray subfolder if present
        split_dir = self.root / 'chest_xray' / self.split
        if not split_dir.exists():
            split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f'Could not find split directory at {split_dir}')

        for label_name, label_idx in [('NORMAL', 0), ('PNEUMONIA', 1)]:
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue
            for img_path in label_dir.glob('*'):
                if img_path.suffix.lower() in ('.jpeg', '.jpg', '.png'):
                    self.samples.append((str(img_path), label_idx))

        print(f'[Kaggle/{self.split}] Loaded {len(self.samples)} images '
              f'({sum(1 for _,l in self.samples if l==1)} pneumonia, '
              f'{sum(1 for _,l in self.samples if l==0)} normal)')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, path


# ── CHEXPERT DATASET ───────────────────────────────────────────────────────────

class CheXpertDataset(Dataset):
    """
    CheXpert dataset (Stanford).
    Multi-label classification with uncertain label handling.

    The dataset CSV contains columns:
        Path, Sex, Age, Frontal/Lateral, AP/PA,
        Atelectasis, Cardiomegaly, Consolidation, Edema,
        Lung Opacity, Lung Lesion, Pleural Effusion, Pleural Other,
        Pneumonia, Pneumothorax, Fracture, Support Devices

    Labels: 1=positive, 0=negative, -1=uncertain, NaN=unmentioned
    """

    TARGET_LABELS = [
        'Pneumonia', 'Lung Opacity', 'Consolidation',
        'Edema', 'Pleural Effusion', 'Atelectasis'
    ]

    def __init__(self, root, split='train', transform=None,
                 uncertain_strategy='smooth', frontal_only=True):
        assert split in ('train', 'valid')
        assert uncertain_strategy in UNCERTAIN_STRATEGIES
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.uncertain_strategy = uncertain_strategy
        self.frontal_only = frontal_only
        self.df = self._load_csv()

    def _load_csv(self):
        csv_name = 'train.csv' if self.split == 'train' else 'valid.csv'

        # Try multiple possible locations
        candidates = [
            self.root / csv_name,
            self.root / 'CheXpert-v1.0-small' / csv_name,
            self.root / 'chexpert-v10-small' / csv_name,
        ]
        csv_path = None
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            raise FileNotFoundError(f'Could not find {csv_name} in {self.root}')

        df = pd.read_csv(csv_path)

        # Frontal only filter
        if self.frontal_only and 'Frontal/Lateral' in df.columns:
            df = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)

        # Handle uncertain labels
        strategy_val = UNCERTAIN_STRATEGIES[self.uncertain_strategy]
        for col in self.TARGET_LABELS:
            if col not in df.columns:
                df[col] = 0.0
                continue
            if self.uncertain_strategy == 'ignore':
                # Mark rows with uncertain labels for this column
                df[col] = df[col].fillna(0)
                # Keep uncertain as -1, will be filtered in __getitem__
            else:
                df[col] = df[col].replace(-1, strategy_val).fillna(0)

        print(f'[CheXpert/{self.split}] Loaded {len(df)} images '
              f'(uncertain strategy: {self.uncertain_strategy})')
        return df

    def _resolve_path(self, raw_path):
        """CheXpert CSV paths start with CheXpert-v1.0-small/... resolve them."""
        # Try as-is relative to root
        p = self.root / raw_path
        if p.exists():
            return str(p)
        # Try stripping leading path component
        parts = Path(raw_path).parts
        for i in range(len(parts)):
            p = self.root / Path(*parts[i:])
            if p.exists():
                return str(p)
        return str(self.root / raw_path)  # return best guess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row['Path'])

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        labels = row[self.TARGET_LABELS].values.astype(np.float32)

        # For 'ignore' strategy: if any label is -1, skip by returning None
        # DataLoader collate_fn should handle None (use custom collate)
        if self.uncertain_strategy == 'ignore' and -1 in labels:
            return None

        return img, labels, img_path


# ── TRANSFORMS ────────────────────────────────────────────────────────────────

def get_transforms(split, image_size=224):
    """
    Standard transforms for training and evaluation.
    Training includes augmentation. Val/test are deterministic.
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    if split == 'train':
        return T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            normalize,
        ])


def skip_none_collate(batch):
    """Custom collate that filters out None samples (from uncertain label ignore strategy)."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return __import__('torch').utils.data.dataloader.default_collate(batch)


# ── DATALOADER FACTORY ────────────────────────────────────────────────────────

def get_kaggle_loaders(kaggle_root, image_size=224, batch_size=32, num_workers=4):
    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = KagglePneumoniaDataset(
            root=kaggle_root,
            split=split,
            transform=get_transforms(split, image_size)
        )
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    return loaders


def get_chexpert_loaders(chexpert_root, image_size=224, batch_size=32,
                         num_workers=4, uncertain_strategy='smooth'):
    loaders = {}
    for split in ('train', 'valid'):
        ds = CheXpertDataset(
            root=chexpert_root,
            split=split,
            transform=get_transforms(split, image_size),
            uncertain_strategy=uncertain_strategy
        )
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=skip_none_collate if uncertain_strategy == 'ignore' else None
        )
    return loaders
