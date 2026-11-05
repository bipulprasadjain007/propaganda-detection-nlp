# Propaganda Detection using NLP

This project investigates **propaganda technique classification** and **propaganda span identification** using classical machine learning and transformer-based natural language processing.

## Tasks

1. **Technique classification:** Given a marked span and its sentence context, predict one of eight propaganda techniques.
2. **Span identification and classification:** Identify propagandistic spans in free text and classify their techniques.

## Methods

- **TF-IDF + Logistic Regression:** Classical classification baseline.
- **DeBERTa-v3-base:** Fine-tuned transformer for technique classification.
- **Soft-voting ensemble:** Combines Logistic Regression and DeBERTa probabilities.
- **DeBERTa + CRF:** BIO sequence labeling for propaganda span identification.
- **Data augmentation:** Back-translation, synonym replacement, mention replacement, and oversampling.

## Reported validation results

| Approach | Metric | Validation score |
| --- | --- | ---: |
| TF-IDF + Logistic Regression | Micro-F1 | 0.5049 |
| DeBERTa-v3-base | Micro-F1 | 0.5761 |
| Soft-voting ensemble | Micro-F1 | **0.5922** |
| Two-stage span pipeline | Span F1 | 0.3793 |
| Binary BIO + CRF | Span F1 | **0.4057** |

These results are from the accompanying project report and have not been independently reproduced in this repository review.

## Repository structure

```text
propaganda-detection-nlp/
├── propaganda_detection.ipynb
├── propaganda_train.tsv
├── propaganda_val.tsv
├── report.docx
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting started

1. Clone the repository.
   ```bash
   git clone https://github.com/bipulprasadjain007/propaganda-detection-nlp.git
   cd propaganda-detection-nlp
   ```
2. Install the listed dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Open `propaganda_detection.ipynb` in Jupyter Notebook or Google Colab.
4. Set notebook dataset paths to `propaganda_train.tsv` and `propaganda_val.tsv`, as appropriate for the execution environment.
5. Execute the notebook cells in order. A GPU is recommended for transformer fine-tuning.

## Project report

See [report.docx](report.docx) for the methodology, hyperparameters, evaluation, error analysis, and future work.

## Notes

- This repository contains experimental code, not a deployed production system.
- Verify the dataset's original licensing and redistribution conditions before further reuse.
- Model weights may need to be downloaded or trained when reproducing the experiments.
