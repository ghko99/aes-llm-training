"""Build compact dataset from aes_dataset_mtl for Kanana chat-template training.

Transforms the original verbose instruction format into a compact Kanana
chat-template format, reducing ~900 tokens → ~300 tokens per instruction.

Changes from original:
  - Rubric descriptions removed from instruction (model learns from data)
  - Rubric names listed in system message
  - Uses Kanana chat template
  - Only raw keywords included (no match count, requirements, grammar)
  - Output format unchanged: "s1 s2 ... s8\n\n- rubric: feedback ..."
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent
_SRC_DIR = _BASE.parent / "wce_full_dataset_aes" / "aes_dataset_mtl"
_OUT_DIR = _BASE / "dataset"

_KEYWORD_MAP_PATH = _BASE.parent / "korean_essay_rater" / "essay_question_keyword_mapping.json"

SYSTEM_MSG = (
    "에세이 채점기. 8개 루브릭(과제충실성, 설명명료성, 설명구체성, 설명적절성, "
    "문장연결성, 글통일성, 어휘적절성, 어법적절성)별 1-9점 채점 후 피드백을 작성한다."
)


def _parse_question_essay(instruction: str) -> tuple[str, str]:
    """Extract question and essay from the original instruction format."""
    q_match = re.search(
        r"### 에세이 질문:\n(.+?)\n### 학생 에세이:", instruction, re.DOTALL
    )
    e_match = re.search(
        r"### 학생 에세이:\n(.+?)\n### 관련 정보:", instruction, re.DOTALL
    )
    question = q_match.group(1).strip() if q_match else ""
    essay = e_match.group(1).strip() if e_match else ""
    return question, essay


def _parse_raw_keywords(instruction: str) -> str:
    """Extract raw keyword string from original instruction."""
    m = re.search(r"- 핵심 키워드:\s*(.+)", instruction)
    return m.group(1).strip() if m else ""


def build_compact_sample(
    original: dict,
    keyword_map: dict,
) -> dict:
    """Convert one sample to compact chat-template format."""
    question, essay = _parse_question_essay(original["instruction"])

    # Build user message: question + essay + raw keywords only
    parts = [f"질문: {question}", f"\n에세이: {essay}"]

    # Raw keywords from keyword_map (same as original "핵심 키워드" line)
    kw_str = keyword_map.get(question, "")
    if not kw_str:
        # Fallback: parse from original instruction
        kw_str = _parse_raw_keywords(original["instruction"])
    if kw_str:
        parts.append(f"\n핵심 키워드: {kw_str}")

    user_msg = "".join(parts)
    output = original["output"]

    return {
        "system": SYSTEM_MSG,
        "user": user_msg,
        "assistant": output,
        "grader_1_scores": original.get("grader_1_scores"),
        "grader_2_scores": original.get("grader_2_scores"),
    }


def process_split(
    split: str,
    keyword_map: dict,
) -> list[dict]:
    """Process one data split (train/valid/test)."""
    src_path = _SRC_DIR / f"{split}.jsonl"

    with open(src_path, encoding="utf-8") as f:
        base_samples = [json.loads(line) for line in f]

    results = []
    for base in base_samples:
        results.append(build_compact_sample(base, keyword_map))

    return results


def main():
    # Load resources
    keyword_map = json.loads(_KEYWORD_MAP_PATH.read_text(encoding="utf-8"))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        print(f"Processing {split}...")
        samples = process_split(split, keyword_map)

        out_path = _OUT_DIR / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"  → {out_path} ({len(samples)} samples)")

    # Token length analysis
    print("\nToken length analysis (Kanana tokenizer):")
    _analyze_tokens()


def _analyze_tokens():
    """Print token length statistics for the built dataset."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "/home/khko/models/kanana", use_fast=True, trust_remote_code=True
    )

    dataset_path = _OUT_DIR / "train.jsonl"
    with open(dataset_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    prompt_lens = []
    total_lens = []
    for s in samples:
        msgs = [
            {"role": "system", "content": s["system"]},
            {"role": "user", "content": s["user"]},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + s["assistant"] + tok.eos_token

        p_len = len(tok.encode(prompt))
        t_len = len(tok.encode(full))
        prompt_lens.append(p_len)
        total_lens.append(t_len)

    import statistics
    print(f"  Prompt: mean={statistics.mean(prompt_lens):.0f}, "
          f"median={statistics.median(prompt_lens):.0f}, "
          f"max={max(prompt_lens)}, "
          f"p95={sorted(prompt_lens)[int(len(prompt_lens)*0.95)]}")
    print(f"  Total:  mean={statistics.mean(total_lens):.0f}, "
          f"median={statistics.median(total_lens):.0f}, "
          f"max={max(total_lens)}, "
          f"p95={sorted(total_lens)[int(len(total_lens)*0.95)]}")
    print(f"  Samples > 2048: {sum(1 for t in total_lens if t > 2048)}/{len(total_lens)}")
    print(f"  Samples > 1536: {sum(1 for t in total_lens if t > 1536)}/{len(total_lens)}")
    print(f"  Samples > 1024: {sum(1 for t in total_lens if t > 1024)}/{len(total_lens)}")


if __name__ == "__main__":
    main()
