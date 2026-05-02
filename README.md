# Siamese Bi-LSTM Paraphrase Identification

A Deep Learning project for **Semantic Paraphrase Identification** using **Siamese Bidirectional LSTM** networks on the [Quora Question Pairs (QQP)](https://www.kaggle.com/c/quora-question-pairs) dataset. The model leverages GloVe pre-trained word embeddings and a Siamese architecture to detect whether two questions are semantically equivalent (duplicates).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Paraphrase identification is the task of determining whether two sentences convey the same meaning. This project tackles the problem using a **Siamese Network** — a twin neural network that shares weights across both inputs — combined with **Bidirectional LSTM** encoders and **GloVe** word embeddings to capture rich semantic representations.

Key objectives:
- Preprocess raw text (tokenisation, lowercasing, punctuation removal, stop-word filtering)
- Map words to pre-trained 300-dimensional GloVe vectors
- Encode each question independently via a shared Bi-LSTM encoder
- Measure sentence similarity with a cosine / Manhattan distance head
- Classify pairs as *duplicate* (1) or *non-duplicate* (0)

---

## Architecture

```
Question 1 ──► Embedding (GloVe 300d) ──► Bi-LSTM Encoder ──┐
                                                              ├──► Distance Layer ──► Dense ──► Sigmoid ──► Output
Question 2 ──► Embedding (GloVe 300d) ──► Bi-LSTM Encoder ──┘
                     (shared weights)
```

| Component | Details |
|---|---|
| Embedding | GloVe 300-dimensional pre-trained vectors (frozen or fine-tuned) |
| Encoder | 2-layer Bidirectional LSTM, hidden size 128 |
| Distance | Cosine similarity + element-wise absolute difference |
| Classifier | Fully-connected layer → Sigmoid |
| Loss | Binary Cross-Entropy |
| Optimiser | Adam (lr = 1e-3, weight decay = 1e-5) |

---

## Dataset

The **Quora Question Pairs** dataset contains over **400 000** labeled question pairs sourced from Quora:

| Split | Pairs | Duplicate % |
|---|---|---|
| Train | ~323 000 | ~37% |
| Validation | ~40 000 | ~37% |
| Test | ~40 000 | ~37% |

**Download:** [Kaggle – Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs/data)

Place the downloaded CSV files inside the `data/` directory:

```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

**GloVe embeddings** (glove.6B.300d): [Stanford NLP GloVe page](https://nlp.stanford.edu/projects/glove/)

Place the embeddings file at:

```
embeddings/
└── glove.6B.300d.txt
```

---

## Features

- **Text preprocessing** – tokenisation, lowercasing, punctuation & stop-word removal, padding/truncation to a fixed sequence length
- **GloVe embeddings** – pre-trained 300-dimensional word vectors loaded into a PyTorch `nn.Embedding` layer
- **Siamese Bi-LSTM** – shared-weight bidirectional LSTM encoder to produce sentence representations
- **Distance metrics** – cosine similarity and Manhattan distance for measuring representational similarity
- **Training utilities** – learning-rate scheduling, early stopping, model checkpointing
- **Evaluation** – accuracy, F1-score, precision, recall, and confusion matrix via scikit-learn

---

## Project Structure

```
siamese-lstm-paraphrase-identification/
│
├── data/                        # Raw and processed datasets (not tracked by git)
│   ├── train.csv
│   └── test.csv
│
├── embeddings/                  # GloVe embeddings (not tracked by git)
│   └── glove.6B.300d.txt
│
├── src/
│   ├── preprocess.py            # Text cleaning & tokenisation pipeline
│   ├── dataset.py               # PyTorch Dataset / DataLoader wrappers
│   ├── model.py                 # Siamese Bi-LSTM model definition
│   ├── train.py                 # Training loop with validation
│   └── evaluate.py              # Metrics & confusion matrix
│
├── notebooks/
│   └── exploration.ipynb        # EDA and result visualisation
│
├── checkpoints/                 # Saved model weights (not tracked by git)
│
├── requirements.txt             # Python dependencies
├── LICENSE
└── README.md
```

---

## Installation

### Prerequisites

- Python ≥ 3.8
- CUDA-enabled GPU (recommended) or CPU

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/mdshahadathossainit/siamese-lstm-paraphrase-identification.git
   cd siamese-lstm-paraphrase-identification
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset** from Kaggle and place the CSV files in `data/`.

5. **Download GloVe embeddings** and place `glove.6B.300d.txt` in `embeddings/`.

---

## Usage

### Preprocessing

```bash
python src/preprocess.py --data_dir data/ --output_dir data/processed/
```

### Training

```bash
python src/train.py \
    --data_dir data/processed/ \
    --embedding_path embeddings/glove.6B.300d.txt \
    --hidden_size 128 \
    --num_layers 2 \
    --dropout 0.3 \
    --batch_size 64 \
    --epochs 20 \
    --lr 1e-3 \
    --checkpoint_dir checkpoints/
```

### Evaluation

```bash
python src/evaluate.py \
    --data_dir data/processed/ \
    --checkpoint checkpoints/best_model.pt
```

---

## Results

| Metric | Score |
|---|---|
| Accuracy | ~85% |
| F1-Score | ~83% |
| Precision | ~84% |
| Recall | ~82% |

> **Note:** Results may vary depending on hyperparameters, embedding fine-tuning, and hardware.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
