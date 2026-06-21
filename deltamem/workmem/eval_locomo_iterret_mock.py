"""WORKMEM + ITERRET integration test, mock LLM (no real generation cost on
the ITERRET side, no SentenceTransformer/network dependency).

Goal of THIS run: prove the data flow works end to end --
  ITERRET.retrieve/reflect -> accumulated_evidence (list[str])
  -> WORKMEM.populate_osam_from_evidence -> WORKMEM.answer_with_osam
  -> score_locomo_prediction
Evidence QUALITY is meaningless with the mock LLM. Only plumbing is tested.

Plain single-process script, no torch.distributed -- matches the working
pattern already proven in eval_locomo_workmem.py.

Each MockLLMClient is FRESH per graph-build and per question, since its
canned replies are stateful by call count (see iterret/llm_client.py's own
docstring: "A FRESH client per stage, deliberately"). Reusing one instance
across the whole run exhausts its early-call-count branches and silently
returns zero evidence for every question.
"""
from __future__ import annotations

import json
from pathlib import Path

from deltamem.eval.locomo_delta import attach_delta_adapter_in_place, load_base_model
from deltamem.eval.locomo_protocol import score_locomo_prediction
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.workmem.iterret_bridge import get_iterret_evidence
from deltamem.workmem.osam_workmem import answer_with_osam, populate_osam_from_evidence
from iterret.llm_client import OpenAICompatibleLLMClient
from iterret.ctc_graph import CueTagContentGraph
from iterret.experience_bank import empty_experience_bank
from iterret.memory_builder import DialogueTurn, build_ctc_graph_from_dialogue

DATA_FILE = "data/locomo10.json"
BASE_MODEL_PATH = "/data6/rahulsiripur/models/Qwen3-4B-Instruct-2507"
ADAPTER_DIR = "/data6/rahulsiripur/models/delta-mem_qwen3_4b-instruct"
OUTPUT_FILE = "/data6/rahulsiripur/outputs/workmem_iterret_mock_smoke.json"

MAX_SAMPLES = 1
MAX_QUESTIONS_PER_SAMPLE = 10
ADVERSARIAL_CATEGORY = 5


def session_keys_sorted(conversation: dict) -> list[str]:
    keys = [
        k for k in conversation
        if k.startswith("session_") and not k.endswith("_date_time")
    ]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def conversation_to_dialogue_turns(conversation: dict) -> list[DialogueTurn]:
    turns: list[DialogueTurn] = []
    for sk in session_keys_sorted(conversation):
        time = conversation.get(f"{sk}_date_time")
        for t in conversation[sk]:
            turns.append(
                DialogueTurn(
                    speaker=t.get("speaker", "Unknown"),
                    text=t.get("text", ""),
                    time=time,
                )
            )
    return turns


def gold_answer_of(question: dict) -> str:
    return str(question.get("answer", question.get("adversarial_answer", "")))


def main() -> None:
    print("Loading WORKMEM model (delta-mem adapter)...")
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
    print("WORKMEM model ready.")

    bank = empty_experience_bank()

    samples = json.load(open(DATA_FILE))[:MAX_SAMPLES]
    results = []

    for sample_idx, sample in enumerate(samples):
        graph_llm = OpenAICompatibleLLMClient(base_url="http://localhost:8000/v1", model="Qwen/Qwen3-4B-Instruct-2507")
        turns = conversation_to_dialogue_turns(sample["conversation"])[:60]  # cap for real-LLM feasibility
        print(f"[sample {sample_idx}] distilling {len(turns)} turn(s) into a CTC graph (mock LLM)...")
        graph: CueTagContentGraph = build_ctc_graph_from_dialogue(turns, graph_llm)
        print(f"[sample {sample_idx}] graph built: {len(graph.cues)} cue(s), {len(graph.contents)} content node(s)")

        questions = [
            q for q in sample.get("qa", [])
            if q.get("category") != ADVERSARIAL_CATEGORY
        ][:MAX_QUESTIONS_PER_SAMPLE]

        for q_idx, question in enumerate(questions):
            question_llm = OpenAICompatibleLLMClient(base_url="http://localhost:8000/v1", model="Qwen/Qwen3-4B-Instruct-2507")
            evidence = get_iterret_evidence(question["question"], graph, bank, question_llm, max_iterations=2) 
            print(f"[sample {sample_idx}.{q_idx}] retrieved {len(evidence)} evidence item(s)")

            if not evidence:
                print(f"[sample {sample_idx}.{q_idx}] no evidence retrieved, skipping")
                continue

            session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device="cuda:0")
            populate_osam_from_evidence(session, evidence)
            out = answer_with_osam(session, question["question"])
            prediction = out["assistant"]

            score = score_locomo_prediction(question, prediction)

            results.append({
                "sample_idx": sample_idx,
                "question": question["question"],
                "gold_answer": gold_answer_of(question),
                "category": question.get("category"),
                "n_evidence_retrieved": len(evidence),
                "prediction": prediction,
                "score": score,
            })
            print(
                f"[sample {sample_idx}.{q_idx}] score={score:.3f} "
                f"n_ev={len(evidence)} pred={prediction[:60]!r}"
            )

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUTPUT_FILE, "w"), indent=2)
    print(f"\nWrote {len(results)} result(s) to {OUTPUT_FILE}")

    if results:
        avg = sum(r["score"] for r in results) / len(results)
        print(f"Overall avg score (MOCK LLM -- plumbing test only, not a real result): {avg:.4f}")
    else:
        print("No results produced -- check evidence retrieval above.")


if __name__ == "__main__":
    main()
