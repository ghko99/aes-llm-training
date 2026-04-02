import json
import os
import pandas as pd


runs_dir = "runs"
runs = [r for r in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, r)) and r != "backup"]

token_rows = []
weight_rows = []

for run in sorted(runs):
    results_path = os.path.join(runs_dir, run, "evaluation_results.json")
    if not os.path.exists(results_path):
        continue
    with open(results_path, "r") as f:
        metrics = json.load(f)

    # run name without timestamp suffix (e.g. kanana_ce_20260321_033704 -> kanana_ce)
    parts = run.split("_")
    # remove last 2 parts (date, time)
    ablation_name = "_".join(parts[:-2])

    token_row = {"ablation": ablation_name, **metrics["token"]}
    weight_row = {"ablation": ablation_name, **metrics["weighted"]}
    token_rows.append(token_row)
    weight_rows.append(weight_row)

token_df = pd.DataFrame(token_rows).set_index("ablation").T
weight_df = pd.DataFrame(weight_rows).set_index("ablation").T

pd.set_option("display.float_format", "{:.3f}".format)

print("=== Token F1 Results ===")
print(token_df.to_string())
print()
print("=== Weighted F1 Results ===")
print(weight_df.to_string())

token_df.round(3).to_csv("ablation_token_results.csv")
weight_df.round(3).to_csv("ablation_weight_results.csv")
print("\nSaved to ablation_token_results.csv and ablation_weight_results.csv")
