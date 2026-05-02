

# Semantic Paraphrase Identification in Short-Text Queries

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-green.svg)](https://en.wikipedia.org/wiki/Natural_language_processing)

This repository contains the implementation of a **Siamese Bidirectional Long Short-Term Memory (Bi-LSTM)** framework designed to identify semantically equivalent question pairs. Using the **Quora Question Pairs (QQP) dataset**, the model effectively detects duplicates that differ lexically but share the same underlying intent.

## 📌 Project Overview
Online Q&A platforms like Quora frequently encounter duplicate questions, leading to redundancy. This project implements a deep learning approach to:
- Capture deep contextual information using shared Bi-LSTM encoders.
- Process short-text queries using **GloVe** (Global Vectors for Word Representation).
- Classify question pairs as 'Duplicate' or 'Unique' with high precision.

## 📂 Repository Structure
```text
├──  Semantic_Paraphrase_Identification_Paper.pdf  # Full Research Paper
├── workfile.ipynb                                    # Model Implementation & Training Code
├── requirements.txt                                  # List of Dependencies
└── README.md                                         # Project Documentation
```

## 🚀 Key Features
- **Siamese Network:** Dual LSTM sub-networks with shared weights for symmetric feature extraction.
- **Bi-LSTM Layer:** Captures dependencies from both past and future states in a sequence.
- **Preprocessing:** Includes tokenization, padding, and embedding layers optimized for short queries.
- **Evaluation:** Rigorous testing using Accuracy and Quadratic Weighted Kappa metrics.

## 🛠️ Getting Started

### Prerequisites
Install the necessary libraries using pip:
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mdshahadathossainit/your-repo-name.git](https://github.com/mdshahadathossainit/your-repo-name.git)
   ```
2. **Open the Notebook:**
   Launch `workfile.ipynb` in Google Colab or Jupyter Notebook to see the data processing and training steps.

## 📊 Results
The model demonstrates significant performance in distinguishing between paraphrased and non-paraphrased questions, even when the word overlap is minimal. Detailed loss and accuracy graphs are included in the notebook.

## 🎓 Authors & Contributors
This project was developed as a research initiative at **Port City International University**:
* **Shihab Hossen** - 
* **M. Masud Ul Karim**
* **Sk. Safowan Y. Eshan**
* **Kamrul Hasan**

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

---

