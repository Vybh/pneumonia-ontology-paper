# Cross-Dataset Pneumonia Detection Failure as an Ontological Problem
# A Semantic Label Alignment Study

## Quick Start (Colab)

1. Open `notebooks/00_setup.ipynb` — run once per session
2. Open `notebooks/01_main_experiment.ipynb` — runs all 7 phases

## Project Structure

```
src/
├── data/
│   └── datasets.py          # KagglePneumoniaDataset, CheXpertDataset, DataLoaders
├── models/
│   └── classifier.py        # CheXpertClassifier, KaggleAdaptedClassifier
├── analysis/
│   ├── semantic_alignment.py # BioMedBERT embeddings, distance matrix, correlation
│   └── label_confusion.py    # Failure taxonomy, label confusion matrix
└── training/
    └── ontology_finetuning.py # OntologyGuidedLoss, OntologyFineTuner, RecoveryRate

notebooks/
├── 00_setup.ipynb            # GPU check, Drive mount, dataset download
└── 01_main_experiment.ipynb  # Full 7-phase pipeline
```

## Phases

| Phase | What it does | Paper section |
|-------|-------------|---------------|
| 1 | Train CheXpert ResNet50 | §3.1 |
| 2 | BioMedBERT semantic distance matrix | §3.2 / Fig 1-2 |
| 3 | Zero-shot transfer baseline | §4.1 |
| 4 | Label confusion / failure taxonomy | §4.2 / Fig 3 |
| 5 | Semantic distance ↔ performance correlation | §4.3 / Fig 4 |
| 6 | OGFT vs standard fine-tuning | §5 / Fig 5 |
| 7 | Recovery rate + alpha ablation | §5.2 / Appendix |
