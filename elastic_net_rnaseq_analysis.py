"""
Elastic Net–based RNA-seq analysis pipeline.

This script performs:
1) Loading variance-stabilized RNA-seq data
2) Elastic Net logistic regression with nested cross-validation
3) Model training and evaluation
4) Extraction of prioritized genes based on standardized coefficients

Designed for perturbation-based transcriptomic studies.
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import roc_auc_score

from joblib import dump


# ============================================================
# 1) Load input data
# ============================================================

# Expression matrix (samples × genes, variance-stabilized)
X_df = pd.read_csv(
    "expression_matrix_vst.csv", index_col=0
)

# Sample labels (S2101 or DMSO)
y_df = pd.read_csv(
    "sample_labels.csv"
)

# Gene order used during preprocessing
gene_order = pd.read_csv(
    "gene_order.csv"
)["gene"].tolist()

# Mapping table (Ensembl ID → Gene symbol)
gene_map = pd.read_csv(
    "gene_id_to_symbol.csv"
)

# Sanity check: gene order consistency
assert list(X_df.columns) == gene_order, \
    "Gene order in expression matrix does not match gene_order.csv"

X = X_df.values
y = (y_df["label"] == "S2101").astype(int).values
samples = X_df.index.tolist()
gene_names = X_df.columns.tolist()

print(f"[INFO] Expression matrix shape: {X.shape}")
print(f"[INFO] Labels: {np.unique(y, return_counts=True)}")


# ============================================================
# 2) Elastic Net logistic regression with nested CV
# ============================================================

pipeline = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("classifier", LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=5000,
        class_weight="balanced"
    ))
])

param_grid = {
    "classifier__C": [0.01, 0.03, 0.1, 0.3, 1.0],
    "classifier__l1_ratio": [0.5, 0.7, 0.9]  # L1-heavy models emphasized
}

loo = LeaveOneOut()
predicted_probs = np.zeros(len(y))

for train_idx, test_idx in loo.split(X):
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=2,
        n_jobs=-1
    )
    grid.fit(X[train_idx], y[train_idx])
    predicted_probs[test_idx] = grid.predict_proba(X[test_idx])[:, 1]

auc = roc_auc_score(y, predicted_probs)
print(f"[RESULT] Leave-one-out AUC = {auc:.3f}")

pd.DataFrame({
    "sample": samples,
    "true_label": y,
    "probability_S2101": predicted_probs
}).to_csv(
    "elastic_net_loo_predictions.csv",
    index=False
)

print("[SAVED] elastic_net_loo_predictions.csv")


# ============================================================
# 3) Train final model on all samples
# ============================================================

grid_full = GridSearchCV(
    pipeline,
    param_grid,
    cv=2,
    n_jobs=-1
)
grid_full.fit(X, y)

final_model = grid_full.best_estimator_

dump(final_model, "elastic_net_final_model.joblib")
print("[SAVED] elastic_net_final_model.joblib")


# ============================================================
# 4) Extract and rank Elastic Net coefficients
# ============================================================

classifier = final_model.named_steps["classifier"]
coefficients = classifier.coef_.ravel()

coef_df = pd.DataFrame({
    "gene_id": gene_names,
    "elastic_net_coefficient": coefficients
}).sort_values(
    "elastic_net_coefficient",
    ascending=False
)

coef_df.to_csv(
    "elastic_net_coefficients_ensembl.csv",
    index=False
)

# Merge gene symbols (keep Ensembl ID if symbol unavailable)
coef_with_symbols = coef_df.merge(
    gene_map.rename(columns={
        "Geneid": "gene_id",
        "GeneSymbol": "gene_symbol"
    }),
    on="gene_id",
    how="left"
)

coef_with_symbols["gene"] = coef_with_symbols["gene_symbol"].fillna(
    coef_with_symbols["gene_id"]
)

coef_with_symbols[[
    "gene",
    "elastic_net_coefficient"
]].to_csv(
    "elastic_net_coefficients_with_symbols.csv",
    index=False
)

print("[SAVED] elastic_net_coefficients_with_symbols.csv")
