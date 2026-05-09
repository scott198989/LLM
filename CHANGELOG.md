# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
dates are ISO 8601.

## 2026-04-16

Major architectural pivot: dropped Qwen, re-spec'd HAVOC as a
~49M-parameter from-scratch transformer with iterative refinement and a
modular agent + tool orchestration layer.

### Added

- **Model**
  - `scripts/model.py`: `HavocModel`, `RoPE`, `RMSNorm`, `SwiGLU`,
    `HavocBlock`, `HavocAttention`, `count_params`, `build_model`.
  - `scripts/config.py`: `HavocConfig` dataclass (model + training +
    refinement + orchestration), `SPECIAL_TOKENS`, `default_50m_config`.
  - `scripts/verify_params.py`: layer-by-layer parameter breakdown +
    CI-style pass/fail gate. Verified at exactly 49,295,872 params.
- **Tokenizer**
  - `scripts/tokenizer_havoc.py`: byte-level BPE wrapper around
    `tokenizers`, with deterministic special-token IDs and a
    `encode_chat()` chat template helper.
  - `scripts/train_tokenizer.py`: CLI to train BPE from `.txt`/`.jsonl`
    corpora.
- **Pretrain / SFT**
  - `scripts/pretrain.py` (replaces `train.py`): same training loop and
    infra (cosine LR, AdamW, mixed-precision, gradient checkpointing,
    early stop, TB, JSONL logs, matplotlib loss curves) but built on
    `HavocModel` + `HavocTokenizer`.
  - `scripts/sft.py`: prompt/completion fine-tuning with prompt-side
    loss masking (-100), reuses pretrain infra.
- **Inference**
  - `scripts/inference.py` rewritten for HAVOC. Streaming
    `generate_stream()` interface unchanged so `gui_app.py` keeps
    working. New `generate_with_refinement()` and
    `generate_with_orchestration()` wrappers.
  - `scripts/refinement.py`: `RefinementEngine` — up to 10 visible
    passes, calibrated confidence (capped at 95%), early stop when last
    two passes are stable + confident, separate final-synthesis pass.
- **Orchestration**
  - `scripts/orchestrator.py`: `Orchestrator` class wiring retrieval,
    tools, refinement, critic, verifier.
  - `scripts/agents/retrieval.py`: `RetrievalAgent` (BM25 over
    `data/knowledge/`, TF-IDF fallback if `rank_bm25` is missing).
  - `scripts/agents/critic.py`: `CriticAgent` — same HAVOC, critic
    system prompt.
  - `scripts/tools/`: `router`, `calculator`, `python_exec`,
    `json_validator`, `file_reader`, `text_parser`, `unit_converter`,
    `regex_utility`.
  - `scripts/verifier.py`: deterministic verification (JSON validity,
    schema, numeric consistency, regex, length).
- **Config / docs**
  - `configs/havoc-50m.json`: canonical 50M preset (auto-generated from
    `default_50m_config()`).
  - `data/knowledge/`, `data/sft/`: scaffolding directories.
  - `README.md`: full rewrite covering arch, pipeline, refinement,
    orchestration, configuration.
  - `CHANGELOG.md`: this file.

### Changed

- **`scripts/preprocess.py`**: now uses `HavocTokenizer` instead of
  `GPT2TokenizerFast`. Adds `--tokenizer_dir` flag. JSONL ChatML
  formatting + special-token handling preserved.
- **`scripts/dataset.py`**: standalone-test decode path uses
  `HavocTokenizer.from_pretrained(info["tokenizer_dir"])`.
- **`requirements.txt`**: added `fastapi`, `uvicorn`, `pydantic`,
  `rank_bm25`, `pint`, `jsonschema`, `RestrictedPython`, `sympy`.
  Removed nothing — `transformers`/`datasets` remain available for
  ancillary scripts but are no longer used by HAVOC core.
- **`chat_ui/app.py`**: full rewrite. Drops `transformers` and
  `bitsandbytes`; loads `scripts.inference.InferenceEngine` instead.
  Adds `/chat/refine/stream` and `/chat/orchestrate/stream` endpoints
  alongside the existing `/chat/stream`.
- **`chat_ui/templates/index.html`**: new mode toggle (Single /
  Refine / Orchestrate), refinement pass cards with confidence bars,
  separate final-answer block, retrieval / tool / verifier panels for
  orchestrate mode, status pill that reflects whether HAVOC weights are
  loaded.

### Renamed

- `scripts/train.py` → `scripts/pretrain.py`. The inline `GPT` and
  `Config` classes were lifted into dedicated `model.py` / `config.py`
  modules.

### Removed

- `scripts/train.py` (replaced by `pretrain.py`; old inline model
  definitions extracted to dedicated modules).
- All Qwen / `transformers.AutoModelForCausalLM` / `BitsAndBytesConfig`
  imports from `chat_ui/`.

### Notes

- The Qwen models in `~/models/qwen3-*-base` are not removed from disk;
  they're simply no longer referenced by any code path in this repo.
- All training infrastructure preserved: LR schedule, checkpointing,
  resume, validation, early stopping, TensorBoard logging, matplotlib
  loss curves, gradient checkpointing, mixed precision, RunPod path
  detection.
