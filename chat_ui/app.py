import asyncio
import json
import os
from typing import AsyncGenerator, List, Dict
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread
import uvicorn

app = FastAPI(title="Qwen3 Chat")

# ── MODEL LOADING ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.expanduser("~/models/qwen3-1.7b-base")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()
print(f"Model ready — VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB used")

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
SKIP_TOKENS = {"<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|im_sep|>"}
THINK_START = "<think>"
THINK_END = "</think>"


def check_partial_tag(buffer: str, tag: str) -> int:
    """How many chars at end of buffer might be the beginning of tag."""
    for i in range(1, len(tag)):
        if buffer.endswith(tag[:i]):
            return i
    return 0


# ── REQUEST SCHEMA ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 1024
    enable_thinking: bool = True


# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    with open(path, encoding="utf-8") as f:
        return f.read()


@app.get("/vram")
async def vram():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {"used_gb": used, "total_gb": total}
    return {"used_gb": 0.0, "total_gb": 0.0}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    loop = asyncio.get_running_loop()

    async def generate() -> AsyncGenerator[str, None]:
        messages = [{"role": "system", "content": request.system_prompt}] + request.messages

        # Apply chat template (with thinking if supported)
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=request.enable_thinking,
            )
        except Exception:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
            timeout=60.0,
        )

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=int(request.max_new_tokens),
            temperature=max(float(request.temperature), 0.01),
            top_p=float(request.top_p),
            top_k=int(request.top_k),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        streamer_iter = iter(streamer)

        def next_token():
            try:
                return next(streamer_iter)
            except StopIteration:
                return None

        buffer = ""
        in_think = False
        think_done = False

        try:
            while True:
                new_text = await loop.run_in_executor(None, next_token)
                if new_text is None:
                    break
                if new_text in SKIP_TOKENS:
                    continue

                buffer += new_text

                # Parse think tags out of the stream
                while buffer:
                    if not in_think and not think_done:
                        idx = buffer.find(THINK_START)
                        if idx != -1:
                            pre = buffer[:idx]
                            if pre.strip():
                                yield f"data: {json.dumps({'type': 'response', 'content': pre})}\n\n"
                            buffer = buffer[idx + len(THINK_START):]
                            in_think = True
                            yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
                        else:
                            partial = check_partial_tag(buffer, THINK_START)
                            safe_len = len(buffer) - partial
                            if safe_len > 0:
                                safe = buffer[:safe_len]
                                if safe.strip():
                                    yield f"data: {json.dumps({'type': 'response', 'content': safe})}\n\n"
                                buffer = buffer[safe_len:]
                            break

                    elif in_think:
                        idx = buffer.find(THINK_END)
                        if idx != -1:
                            chunk = buffer[:idx]
                            if chunk:
                                yield f"data: {json.dumps({'type': 'thinking', 'content': chunk})}\n\n"
                            buffer = buffer[idx + len(THINK_END):]
                            in_think = False
                            think_done = True
                            yield f"data: {json.dumps({'type': 'thinking_end'})}\n\n"
                        else:
                            partial = check_partial_tag(buffer, THINK_END)
                            safe_len = len(buffer) - partial
                            if safe_len > 0:
                                yield f"data: {json.dumps({'type': 'thinking', 'content': buffer[:safe_len]})}\n\n"
                                buffer = buffer[safe_len:]
                            break

                    else:  # think_done — pure response
                        if buffer:
                            yield f"data: {json.dumps({'type': 'response', 'content': buffer})}\n\n"
                        buffer = ""
                        break

            # Flush any remaining buffer
            if buffer.strip():
                etype = "thinking" if in_think else "response"
                yield f"data: {json.dumps({'type': etype, 'content': buffer})}\n\n"
                if in_think:
                    yield f"data: {json.dumps({'type': 'thinking_end'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
