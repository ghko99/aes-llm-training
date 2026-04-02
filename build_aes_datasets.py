"""Build train/test/valid JSONL from aes_datasets raw JSON files.

Converts individual JSON files (14-1, 14-2, 14-3) into the compact
chat-template format matching the existing dataset/ JSONL format.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

_BASE = Path(__file__).resolve().parent
_SRC_DIR = _BASE / "aes_datasets"
_OUT_DIR = _SRC_DIR  # Output to aes_datasets/

SYSTEM_MSG = (
    "에세이 채점기. 8개 루브릭(과제충실성, 설명명료성, 설명구체성, 설명적절성, "
    "문장연결성, 글통일성, 어휘적절성, 어법적절성)별 1-9점 채점 후 피드백을 작성한다."
)

RUBRIC_ORDER = [
    "task_1", "content_1", "content_2", "content_3",
    "organization_1", "organization_2", "expression_1", "expression_2",
]


def convert_sample(data: dict) -> dict:
    """Convert one raw JSON sample to compact chat-template format."""
    question = data["essay_question"]
    answer = data["essay_answer"]
    analytic_scores = data["score"]["personal"]["analytic"]
    rubric = data["rubric"]["analytic"]

    # --- User message ---
    parts = [f"질문: {question['prompt']}", f"\n에세이: {answer['text']}"]
    keyword = question.get("keyword", "")
    if keyword:
        parts.append(f"\n핵심 키워드: {keyword.strip()}")
    user_msg = "".join(parts)

    # --- Grader scores ---
    grader_1 = []
    grader_2 = []
    for key in RUBRIC_ORDER:
        scores = analytic_scores[key]["score"]
        grader_1.append(float(scores[0]))
        grader_2.append(float(scores[1]))

    # --- Assistant output: averaged scores + feedback ---
    avg_scores = []
    for s1, s2 in zip(grader_1, grader_2):
        avg_scores.append(int(math.floor((s1 + s2) / 2 + 0.5)))

    score_line = " ".join(str(s) for s in avg_scores)

    feedback_parts = []
    for key in RUBRIC_ORDER:
        name = rubric[key]["name"]
        feedback = analytic_scores[key]["feedback"]
        feedback_parts.append(f"- {name}:\n {feedback}")

    assistant_msg = f"{score_line}\n\n### Feedback:\n" + "\n".join(feedback_parts)

    return {
        "system": SYSTEM_MSG,
        "user": user_msg,
        "assistant": assistant_msg,
        "grader_1_scores": grader_1,
        "grader_2_scores": grader_2,
    }


DATASET_PREFIXES = ["14-1", "14-2", "14-3"]


def _get_prefix(filename: str) -> str:
    """Extract dataset prefix (14-1, 14-2, 14-3) from filename."""
    for prefix in DATASET_PREFIXES:
        if filename.startswith(prefix):
            return prefix
    return "unknown"


def process_split(split: str) -> tuple[list[dict], dict[str, list[dict]]]:
    """Process all JSON files in a split directory.

    Returns:
        (all_samples, per_prefix_samples) where per_prefix_samples maps
        prefix -> list of samples.
    """
    src_dir = _SRC_DIR / split
    files = sorted(src_dir.glob("*.json"))

    all_results = []
    by_prefix: dict[str, list[dict]] = {p: [] for p in DATASET_PREFIXES}
    errors = []
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            sample = convert_sample(data)
            all_results.append(sample)
            prefix = _get_prefix(fp.name)
            if prefix in by_prefix:
                by_prefix[prefix].append(sample)
        except Exception as e:
            errors.append((fp.name, str(e)))

    if errors:
        print(f"  Errors ({len(errors)}):")
        for name, err in errors[:5]:
            print(f"    {name}: {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    return all_results, by_prefix


def _write_jsonl(path: Path, samples: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    for split in ["train", "valid", "test"]:
        print(f"Processing {split}...")
        samples, by_prefix = process_split(split)

        out_path = _OUT_DIR / f"{split}.jsonl"
        _write_jsonl(out_path, samples)
        print(f"  → {out_path} ({len(samples)} samples)")

        # For test split, also write per-prefix files
        if split == "test":
            for prefix, prefix_samples in by_prefix.items():
                if prefix_samples:
                    safe_name = prefix.replace("-", "_")
                    prefix_path = _OUT_DIR / f"test_{safe_name}.jsonl"
                    _write_jsonl(prefix_path, prefix_samples)
                    print(f"  → {prefix_path} ({len(prefix_samples)} samples)")


if __name__ == "__main__":
    main()
