"""Evaluation module — computes QWK (Quadratic Weighted Kappa) from inference CSV."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

RUBRICS = [
    "task_1",
    "content_1",
    "content_2",
    "content_3",
    "organization_1",
    "organization_2",
    "expression_1",
    "expression_2",
]


def evaluate_results(csv_path: str, save_dir: str, split_name: str = "test") -> dict:
    """Compute QWK metrics from inference CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    total_test = len(df["sample_idx"].unique())

    labels, preds_token, preds_weighted = [], [], []

    for i in range(total_test):
        rows = df[df["sample_idx"] == i]
        label_str = rows["label"].values[0]
        label = [int(s) for s in label_str.strip().split()[:8]]

        preds_t, preds_w = [], []
        for pos in range(1, 17, 2):
            row = rows[rows["gen_pos"] == pos]
            if row.empty:
                preds_t.append(5)  # fallback
                preds_w.append(5.0)
                continue

            try:
                tok = int(row["chosen_token"].values[0])
            except (ValueError, TypeError):
                probs = [row[f"prob_{k}"].values[0] for k in range(1, 10)]
                tok = int(np.argmax(probs) + 1)

            weighted = sum(
                row[f"prob_{k}"].values[0] * k for k in range(1, 10)
            )
            preds_t.append(tok)
            preds_w.append(weighted)

        labels.append(label)
        preds_token.append(preds_t)
        preds_weighted.append(preds_w)

    labels = np.array(labels).flatten()
    preds_token = np.array(preds_token).flatten()
    preds_weighted = np.rint(preds_weighted).flatten()

    res_token = {
        "overall": cohen_kappa_score(labels, preds_token, weights="quadratic")
    }
    res_weighted = {
        "overall": cohen_kappa_score(labels, preds_weighted, weights="quadratic")
    }

    labels_2d = labels.reshape(-1, 8)
    preds_t_2d = preds_token.reshape(-1, 8)
    preds_w_2d = preds_weighted.reshape(-1, 8)

    for i, rubric in enumerate(RUBRICS):
        res_token[rubric] = cohen_kappa_score(
            labels_2d[:, i], preds_t_2d[:, i], weights="quadratic"
        )
        res_weighted[rubric] = cohen_kappa_score(
            labels_2d[:, i], preds_w_2d[:, i], weights="quadratic"
        )

    # Print results
    print(f"\n=== [{split_name}] Token Scores (QWK) ===")
    for k, v in res_token.items():
        print(f"  {k}: {v:.4f}")
    res_token["average"] = np.mean(list(res_token.values()))
    print(f"  Average: {res_token['average']:.4f}")

    print(f"\n=== [{split_name}] Weighted Scores (QWK) ===")
    for k, v in res_weighted.items():
        print(f"  {k}: {v:.4f}")
    res_weighted["average"] = np.mean(list(res_weighted.values()))
    print(f"  Average: {res_weighted['average']:.4f}")

    # Save
    result = {"token": res_token, "weighted": res_weighted}
    out_path = os.path.join(save_dir, f"{split_name}_evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")
    return result
