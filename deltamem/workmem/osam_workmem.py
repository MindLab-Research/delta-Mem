"""WORKMEM: two-phase OSAM population for C-AIMMS.
Phase 1: write retrieved evidence E_T at segment (message_mean) granularity,
reusing the same ingest pattern as eval/locomo_delta.py's build_teacher_forced_snapshot.
Phase 2: generate with token granularity; S persists across both phases.
"""
from deltamem.core.delta_impl import (
    set_delta_mem_write_granularity,
    reset_delta_mem_states,
)
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.eval.locomo_protocol import OFFICIAL_QA_PROMPT, OFFICIAL_MAX_NEW_TOKENS

def populate_osam_from_evidence(session, evidence_list, *, reset=True):
    """Phase 1. evidence_list: list[str], one retrieved unit per string (E_T)."""
    if reset:
        reset_delta_mem_states(session.model)
    
    set_delta_mem_write_granularity(session.model, "message_mean")
    session.messages = [{"role": "system", "content": unit} for unit in evidence_list]
    
    full_ids = session._tokenize_messages(session.messages, add_generation_prompt=False)
    session._ingest_full_ids(full_ids)
    
    return session.state_stats()  # verify S changed vs a fresh reset

def answer_with_osam(session, query, **gen_kwargs):
    """Phase 2. Token-granularity writes; generates on the evidence-populated S."""
    set_delta_mem_write_granularity(session.model, "token")
    formatted_query = OFFICIAL_QA_PROMPT.format(query)
    gen_kwargs.setdefault("max_new_tokens", OFFICIAL_MAX_NEW_TOKENS)
    return session.generate_reply(formatted_query, **gen_kwargs)
