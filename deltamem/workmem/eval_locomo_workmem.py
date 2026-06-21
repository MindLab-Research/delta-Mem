"""WORKMEM gold-evidence LoCoMo evaluation.
Phase 1: write gold evidence (resolved from question["evidence"] dia_ids) into S.
Phase 2: generate the answer; score with the official scorer.
Compare overall/category F1 against the full_history_replay baseline (0.4491).
"""
import json
from pathlib import Path

from deltamem.eval.locomo_delta import load_base_model, attach_delta_adapter_in_place
from deltamem.eval.locomo_protocol import score_locomo_prediction
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.workmem.osam_workmem import populate_osam_from_evidence, answer_with_osam

DATA_FILE = "data/locomo10.json"
BASE_MODEL_PATH = "/data6/rahulsiripur/models/Qwen3-4B-Instruct-2507"
ADAPTER_DIR = "/data6/rahulsiripur/models/delta-mem_qwen3_4b-instruct"
OUTPUT_FILE = "/data6/rahulsiripur/outputs/workmem_locomo_gold.json"
MAX_SAMPLES = None  # bump to None once verified correct


def resolve_evidence_id(sample: dict, evidence_id: str) -> str:
    """Turn 'D1:3' into the actual utterance text. Searches by dia_id -
    the list index does NOT correspond to the dia_id number."""
    session_num = evidence_id.split(":")[0].lstrip("D")
    session_key = f"session_{session_num}"
    turns = sample["conversation"].get(session_key, [])
    for turn in turns:
        if turn.get("dia_id") == evidence_id:
            return f"{turn['speaker']}: {turn['text']}"
    return ""


def gather_gold_evidence(sample: dict, question: dict) -> list[str]:
    out = []
    raw_ids = question.get("evidence", [])
    flat_ids = []
    for eid in raw_ids:
        flat_ids.extend(part.strip() for part in str(eid).split(";"))
    for eid in flat_ids:
        text = resolve_evidence_id(sample, eid)
        if not text:
            print(f"WARNING: could not resolve evidence id {eid}")
        else:
            out.append(text)
    return out

def main():
    model, tokenizer = load_base_model(
        model_path=BASE_MODEL_PATH,
        device="cuda:0",
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
    )
    attach_delta_adapter_in_place(
        model,
        adapter_dir=ADAPTER_DIR,
        rank=8,
        alpha=16.0,
        beta_bias_init=0.0,
        rankwise_gates=True,
        output_init="zero",
        online_gain=1.0,
        load_adapter=True,
    )

    samples = json.load(open(DATA_FILE))
    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]

    results = []
    for sample_idx, sample in enumerate(samples):
        for q_idx, question in enumerate(sample.get("qa", [])):
            gold_evidence = gather_gold_evidence(sample, question)
            if not gold_evidence:
                continue

            session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device="cuda:0")
            populate_osam_from_evidence(session, gold_evidence)
            out = answer_with_osam(session, question["question"], max_new_tokens=50)
            prediction = out["assistant"]
            score = score_locomo_prediction(question, prediction)

            results.append({
                "sample_idx": sample_idx,
                "question": question["question"],
                "gold_answer": question.get("answer", question.get("adversarial_answer", "")),
                "category": question.get("category"),
                "evidence_ids": question.get("evidence", []),
                "prediction": prediction,
                "score": score,
            })
            print(f"[{sample_idx}.{q_idx}] cat={question.get('category')} "
                  f"score={score:.3f} pred={prediction[:60]!r}")

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUTPUT_FILE, "w"), indent=2)

    if results:
        overall = sum(r["score"] for r in results) / len(results)
        print(f"\nOverall avg score: {overall:.4f}  (n={len(results)})")
        by_cat = {}
        for r in results:
            by_cat.setdefault(r["category"], []).append(r["score"])
        for cat, scores in sorted(by_cat.items()):
            print(f"  category {cat}: {sum(scores)/len(scores):.4f}  (n={len(scores)})")


if __name__ == "__main__":
    main()
