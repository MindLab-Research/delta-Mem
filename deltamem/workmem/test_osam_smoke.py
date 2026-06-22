from deltamem.eval.locomo_delta import load_base_model
from deltamem.eval.locomo_delta import load_base_model, attach_delta_adapter_in_place
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.workmem.osam_workmem import populate_osam_from_evidence, answer_with_osam
model, tokenizer = load_base_model(
model_path="/data6/rahulsiripur/models/Qwen3-4B-Instruct-2507",
device="cuda:0",
dtype="bfloat16",
attn_implementation="flash_attention_2",
)
config = attach_delta_adapter_in_place(
    model,
    adapter_dir="/data6/rahulsiripur/models/delta-mem_qwen3_4b-instruct",
    rank=8,
    alpha=16.0,
    beta_bias_init=0.0,
    rankwise_gates=True,
    output_init="zero",
    online_gain=1.0,
    load_adapter=True,
# beta_bias_init / rankwise_gates / output_init / online_gain / config_override:
# leave at function defaults unless eval/locomo_delta.py passes explicit non-default
# values - confirm with: sed -n '95,140p' deltamem/eval/locomo_delta.py
)
session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device="cuda:0")
evidence = [
"Sarah adopted a golden retriever named Max in March 2023.",
"Sarah moved to Seattle for a software engineering job at a startup.",
]
before = session.state_stats()
populate_osam_from_evidence(session, evidence) # Phase 1
after = session.state_stats()
print("S changed:", before != after) # MUST be True
out = answer_with_osam(session, "What is the name of Sarah's dog?", max_new_tokens=20)
print(out["assistant"]) # expect "Max"
