# HAVOC

A from-scratch ~49M-parameter decoder-only transformer with iterative
self-refinement and a lightweight agent + tool orchestration layer.

The point isn't raw scale — at 49M params HAVOC will never match a 7B
model on knowledge-heavy tasks. The point is to make a *small* model
punch above its weight by combining:

* **A modern, clean core** — RoPE, RMSNorm, SwiGLU, tied embeddings,
  flash attention, gradient checkpointing.
* **Iterative self-refinement at inference time** — up to 10 visible
  passes with calibrated confidence (capped at 95%, never claims 100%);
  early-stops when stable.
* **An additive agent / tool orchestration layer** — retrieval (RAG),
  a critic role on the same model, a tool router (calculator, Python
  sandbox, JSON validator, file reader, text parser, unit converter,
  regex), and a *deterministic* verifier that cross-checks the output.

There are no Qwen, Llama, or other pretrained weights anywhere in this
repo. The tokenizer is trained from scratch too.

## Architecture

| Field | Value |
|---|---|
| `vocab_size` | 16384 |
| `hidden_size` | 512 |
| `num_layers` | 12 |
| `num_heads` | 8 (head_dim 64) |
| `intermediate_size` (SwiGLU) | 1536 |
| `max_seq_len` | 2048 |
| `dropout` | 0.1 |
| Position encoding | RoPE (base 10000) |
| Norm | RMSNorm |
| Activation | SwiGLU |
| Tied embeddings | yes |
| **Total params** | **49,295,872** |

Parameter math (and a CI gate):

```
python scripts/verify_params.py
# -> exits 0 iff |actual - 49,295,872| / 49,295,872 <= 2%
```

## Layout

```
configs/havoc-50m.json          canonical 50M preset
scripts/
  config.py                     HavocConfig dataclass + special-token list
  model.py                      HavocModel, RoPE, RMSNorm, SwiGLU
  tokenizer_havoc.py            byte-level BPE wrapper
  train_tokenizer.py            CLI: train BPE on a corpus
  preprocess.py                 jsonl/docx/txt -> tokenised .bin files
  dataset.py                    DataLoader factory
  pretrain.py                   next-token pretraining
  sft.py                        prompt/completion fine-tune (loss masked on prompt)
  inference.py                  streaming generation engine
  refinement.py                 iterative self-refinement (inference-time only)
  orchestrator.py               retrieval + tools + critic + verifier + refinement
  agents/{retrieval,critic}.py
  tools/{router,calculator,python_exec,json_validator,
         file_reader,text_parser,unit_converter,regex_utility}.py
  verifier.py                   deterministic checks (no LLM)
  verify_params.py              parameter-count CI gate
  gui_app.py                    optional customtkinter desktop UI (legacy)
chat_ui/                        FastAPI web UI (single / refine / orchestrate modes)
data/
  raw/                          your corpus (jsonl / docx / txt)
  processed/                    train.bin + val.bin + tokenizer_info.json
  knowledge/                    documents indexed by RetrievalAgent (BM25)
  sft/                          prompt/completion JSONL for SFT
models/
  tokenizers/havoc_bpe/         trained BPE tokenizer
  checkpoints/                  pretrain + SFT checkpoints
```

## End-to-end pipeline

```bash
# 0. Install
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130   # or cu124

# 1. Verify the model architecture before doing anything else
python scripts/verify_params.py
#   -> 49,295,872 params ; PASS

# 2. Train the BPE tokenizer (5-30s on CPU)
python scripts/train_tokenizer.py \
    --corpus data/raw \
    --vocab_size 16384 \
    --out models/tokenizers/havoc_bpe

# 3. Tokenise the corpus into bin files
python scripts/preprocess.py \
    --data_dir data/raw \
    --tokenizer_dir models/tokenizers/havoc_bpe \
    --out_dir data/processed \
    --block_size 2048

# 4. Pretrain (RunPod recommended for the 4B-token target)
python scripts/pretrain.py \
    --processed_dir data/processed \
    --tokenizer_dir models/tokenizers/havoc_bpe \
    --batch_size 32 --grad_accum 4 --max_epochs 3

# 4b. Resume from any checkpoint
python scripts/pretrain.py --resume models/checkpoints/best.pt

# 5. (optional) Supervised fine-tuning on prompt/completion JSONL
python scripts/sft.py \
    --base_ckpt models/checkpoints/best.pt \
    --sft_data data/sft/your_pairs.jsonl \
    --tokenizer_dir models/tokenizers/havoc_bpe \
    --max_epochs 2 --batch_size 8 --lr 2e-5

# 6. Single-prompt inference (CLI)
python scripts/inference.py \
    --ckpt models/checkpoints/best.pt \
    --prompt "What color is the sky in the daytime?"

# 6a. With iterative refinement (visible passes + confidence)
python scripts/inference.py \
    --ckpt models/checkpoints/best.pt \
    --prompt "What color is the sky in the daytime?" \
    --refine

# 6b. With full orchestration (retrieval + tools + verifier)
python scripts/inference.py \
    --ckpt models/checkpoints/best.pt \
    --prompt "calc: 12.7 * 84 — what is the rounded result?" \
    --orchestrate

# 7. Web UI (FastAPI + SSE, three modes: Single / Refine / Orchestrate)
python chat_ui/app.py
# -> http://localhost:7860
```

## Iterative self-refinement

Refinement is **purely inference-time scaffolding**. The model architecture
is unchanged.

For each pass the model produces three labelled lines: `Reasoning:`,
`Answer:`, `Confidence:`. The wrapping prompt explicitly instructs the
model never to claim 100% certainty (the engine also clamps the parsed
confidence to ≤ 95% as a hard ceiling).

Pass 1 is initial. Pass N≥2 is prefixed with *"Reviewing my previous
answer for missed wording, hidden assumptions, or ambiguity."* and given
the prior passes' answers/confidences as context.

Stop criteria:

* Reached `max_passes` (default 10), **or**
* The last two passes both have confidence ≥ `confidence_threshold`
  (default 0.85) AND their answers have Jaccard similarity ≥
  `similarity_threshold` (default 0.9). This is what counts as "stable".

After the loop, a separate generation produces the **Final Answer**, kept
visually distinct from the refinement trace in the UI.

Toggle on/off via:

* CLI: `--refine`
* Config: `enable_refinement: true|false`
* Web UI: the **Refine** mode tab.

## Orchestration layer

```
User Input
   ↓
HAVOC Orchestrator                       (orchestrator.py)
   ├─ Retrieval Agent       (BM25 over data/knowledge/)
   ├─ Critic Agent          (HAVOC + critic system prompt)
   ├─ Tool Router           (calc, python, json, read_file,
   │                         parse, convert, regex)
   ├─ Verifier (deterministic; not LLM-based)
   └─ Refinement Engine     (refinement.py)
```

Important properties:

* **One model, multiple roles.** All agents share the single HAVOC
  inference engine; "critic" is just a swapped system prompt.
* **Verifier is deterministic.** JSON validity, schema enforcement,
  numeric re-evaluation of `<expr> = <value>` claims, regex matching,
  length bounds. No LLM-checks-LLM circularity.
* **Modular.** Pass `enable_retrieval=False`, `enable_tools=False`, or
  `enable_refinement=False` to disable any layer.

Tool argument formats are documented in `scripts/tools/router.py`. Inline
tool calls in user input are detected with the regex
`(calc|python|json|read_file|parse|convert|regex):\s*(args)`.

## Configuration

Everything lives in one dataclass, `HavocConfig` (`scripts/config.py`).
Save / load via `HavocConfig.to_json(path)` / `HavocConfig.from_json(path)`.
The canonical preset is `configs/havoc-50m.json`.

CLI flags on `pretrain.py` / `sft.py` override individual fields. See
`python scripts/pretrain.py --help` for the full list.

## Resuming, checkpoints, monitoring

Checkpoints are written to `models/checkpoints/` with these tags:

* `best.pt`           — best validation loss seen so far (pretrain)
* `epoch_NN.pt`       — every epoch
* `step_NNNNNNN.pt`   — every `ckpt_interval` optimizer steps
* `best_sft.pt`       — best validation loss during SFT
* `sft_epoch_NN.pt`   — every SFT epoch

Each checkpoint persists model weights, optimizer state, the full
`HavocConfig`, and training metadata (epoch / step / val_loss). Resume
with `--resume <path>`.

Logs:

* `logs/pretrain_log.jsonl` / `logs/sft_log.jsonl` — JSONL records
* `logs/tensorboard/`                              — TensorBoard
* `logs/loss_curves.png` / `logs/sft_loss_curves.png` — matplotlib

## RunPod

The provided `Dockerfile` builds on `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel`.
Paths under `/workspace` are auto-detected by `pretrain.py` / `sft.py`.
See `scripts/setup_runpod.sh` and `scripts/train_runpod.sh`.

## What this project is *not*

* It's not a base LLM you should ship to users — it's a 49M-parameter
  research tinker model.
* It does not include any pretrained weights. You train HAVOC yourself.
* `transformers` is listed as an optional dependency but **HAVOC core
  imports it nowhere** — every code path runs against `HavocModel` /
  `HavocTokenizer`.
