# Enhancing DDI Classification via Textual Arguments

This repository contains the implementation for our paper:

**"Enhancing Drug-Drug Interaction Classification by Leveraging Textual Drug Arguments"** (2025)

## 📄 Overview

This study proposes a novel method for Drug-Drug Interaction (DDI) classification that leverages textual arguments extracted from **DrugBank**, going beyond traditional approaches that primarily rely on molecular or chemical features.

It explores the impact of integrating **text embeddings** and **text-based similarity matrices** alongside existing chemical properties. To achieve this, drug-related types and concepts extracted from the **Unified Medical Language System (UMLS)**, as well as identified entities, were used to construct similarity matrices as additional input features.

The effect of these textual features was assessed through a series of experiments using various implementation scenarios with a **deep neural network** model.

---

## 🚀 Getting Started

### 1. Generate `drugbank.db`

Use the DrugBank parser and processor to build a local SQLite database (`drugbank.db`) that contains both textual and structured drug-related data.

> **Note:** In this work, **DrugBank v5.1.10** was used and processed.  
> You must download the full DrugBank XML file from [DrugBank's official website](https://go.drugbank.com/releases/latest) (requires a free academic license). Place the XML file in the appropriate input folder before running the parser.


[Drugbank Preparation Notebook](colab_notebooks/prep-0001-data-preparation-for-DrugBank.ipynb) 

---

### 2. Named Entity Recognition (NER)

The NER step identifies drug mentions in biomedical texts:

- **Preferred**: Load the pre-generated NER results from the provided `.pkl` file.
- **Optional**: To generate NER outputs from scratch:
  - Ensure [Apache cTAKES](https://ctakes.apache.org/) is installed and running.
  - Execute [NER processor](run_ner.py) to process raw text and produce the output.

---

### 3. Textual Drug Embeddings

Text embeddings were generated for textual features of drugs found in DrugBank columns such as **description**, **interactions**, and **indications**.

You can either:

- Download the **precomputed text embeddings** from the link below for quick setup:
📥 **Download from Kaggle**: [DrugBank Text Embeddings (Kaggle)](https://www.kaggle.com/datasets/kivancbayraktar/0295501b-e673-43e9-af91-7d06ec21cb7d)

- or generate your own embeddings using pre-trained models compatible with Sentence Transformer.

[Embedding Generation Notebook](colab_notebooks/embedding_generation.ipynb) 

 
### 4. Run Experiments

To run all of the experiments were carried out of this study, please run [Experiments](colab_notebooks/experiments.ipynb.ipynb).To switch between experiments you should change `experiment_configuration_file`. All experiments can be follow using [MLflow](https://www.mlflow.org).


---

## 📌 Requirements

- Python 3.11+
- TensorFlow 
- spaCy or NLTK
- Apache cTAKES
- Transformers
- SentenceTransformers 

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📝 Citation

If you use this code or our dataset in your research, please cite our paper:
```
@article{bayraktar2025enhancing,
  title={Enhancing drug-drug interaction classification by leveraging textual drug arguments},
  author={Bayraktar, Kivanc and Sezer, Ebru Akcapinar and Mutlu, Begum and {\"O}zdemir, Suat},
  journal={Computers in Biology and Medicine},
  volume={194},
  pages={110467},
  year={2025},
  publisher={Elsevier}
}
```

---

## 📬 Contact

For questions, suggestions, or collaborations, please [open an issue](https://github.com/kivancbayraktar/enhancing-ddi-classification-via-textual-arguments) or contact us at [bayraktarkivanc@gmail.com](mailto:bayraktarkivanc@gmail.com).
