# elastic_net_rnaseq_analysis.py
Reproducible Python pipeline for Elastic Net logistic regression applied to RNA-seq data. Includes preprocessing, cross-validation, model training, and extraction of prioritized genes. Designed for transcriptomic analysis of perturbation experiments.

# Elastic Net–based RNA-seq Analysis Pipeline
This repository provides a reproducible Python pipeline for Elastic Net logistic regression–based analysis of RNA-seq data.  
The workflow is designed to prioritize genes with dominant contributions to transcriptional responses in perturbation experiments, beyond conventional differential expression analysis.

---

## Scientific Context
This pipeline was developed to analyze RNA-seq data from human neural stem cells subjected to pharmacological perturbation.  
In the associated study, Elastic Net regression was used to identify metabolism-related genes downstream of LSD1 inhibition that mediate metabolic reprogramming and stem cell fate decisions.

The approach combines:
- Variance-stabilized RNA-seq expression data
- Elastic Net logistic regression for feature selection
- Cross-validation to assess model generalization
- Coefficient-based gene prioritization

---

## Input Data
The pipeline expects the following input files:

- `X_samples_by_genes.csv`  
  Variance-stabilized RNA-seq expression matrix  
  (rows: samples, columns: genes)

- `y_labels.csv`  
  Sample labels indicating experimental condition  
  (e.g., control vs perturbation)

- `gene_order.csv`  
  Gene order corresponding to columns in the expression matrix

- `gene_map.csv`  
  Mapping table between gene IDs (e.g., Ensembl IDs) and gene symbols

---

## Method Overview
Elastic Net logistic regression is applied to classify samples based on transcriptional profiles while performing feature selection.  
By combining L1 (lasso) and L2 (ridge) regularization, the model accounts for correlated gene expression and prioritizes coordinated gene modules rather than isolated genes.

Model performance is evaluated using leave-one-out cross-validation (LOO-CV), and final gene prioritization is based on standardized regression coefficients obtained from the full dataset.

---

## How to Run
```bash
python elastic_net_rnaseq_analysis.py
