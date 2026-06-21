"""Minimal OpenAI-compatible chat endpoint, backed by the plain (no-adapter)
Qwen3-4B-Instruct model already proven to load in this environment.
Just enough of the API surface for ITERRET's OpenAICompatibleLLMClient."""
import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/data6/rahulsiripur/models/Qwen3-4B-Instruct-2507"

app = FastAPI()
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
).to("cuda:0")
model.eval()
print("Model loaded.")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 1024


@app.get("/v1/models")
def list_models():
    return {"data": [{"id": "Qwen/Qwen3-4B-Instruct-2507", "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids = encoded["input_ids"].to("cuda:0")
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=min(req.max_tokens, 512),   # hard cap, ignore runaway client requests
            do_sample=req.temperature > 0,
            temperature=max(req.temperature, 0.01),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    del out, input_ids
    torch.cuda.empty_cache()
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }
