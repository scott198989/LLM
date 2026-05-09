"""
HAVOC Orchestrator: stitches retrieval, tools, the critic agent, the
deterministic verifier, and the recursive refinement engine into one
streaming pipeline. Every component reuses the *same* underlying HAVOC
inference engine - agents are role prompts, not separate models.

Pipeline (simplified):
    1. Retrieval        - top-K BM25 chunks from data/knowledge/
    2. Tool detection   - look for explicit tool calls in the question
                          (e.g. "calc: 2+2") and pre-execute them
    3. Generation       - either single-pass or refinement (configurable)
    4. Verification     - deterministic checks on the output
    5. Critic           - if verification fails, request a critique and
                          run one more refinement pass
    6. Final answer     - emit the consolidated answer

Streaming events:
    {"type": "retrieval",      "chunks": [...]}
    {"type": "tool_call",      "name": str, "args": str}
    {"type": "tool_result",    "name": str, "ok": bool, "output": str}
    {"type": "pass_*",         (forwarded from RefinementEngine)}
    {"type": "verification",   "passed": bool, "failures": [...]}
    {"type": "critic",         "text": str}
    {"type": "final_complete", "answer": str, "confidence": float, ...}
    {"type": "done"}
"""

from __future__ import annotations

import re
from typing import Iterator

from agents.retrieval import RetrievalAgent, RetrievedChunk
from agents.critic    import CriticAgent
from refinement       import RefinementEngine, parse_answer, parse_confidence
from tools            import ToolRouter
from verifier         import run_checks


# Recognise inline tool calls in user input or model output:
#   "calc: 2+2"        -> ("calc", "2+2")
#   "convert: 5 km in mi"
_INLINE_CALL_RX = re.compile(
    r"\b(calc|python|json|read_file|parse|convert|regex)\s*:\s*([^\n]+?)(?=$|[.;\n])",
    re.IGNORECASE,
)


class Orchestrator:
    def __init__(self,
                 engine,                                  # InferenceEngine
                 retrieval:   RetrievalAgent | None = None,
                 critic:      CriticAgent     | None = None,
                 router:      ToolRouter      | None = None,
                 refiner:     RefinementEngine | None = None,
                 enable_retrieval:   bool = True,
                 enable_tools:       bool = True,
                 enable_refinement:  bool = True,
                 retrieval_top_k:    int = 4):
        self.engine             = engine
        self.retrieval          = retrieval
        self.critic             = critic or CriticAgent(_BareWrapper(engine))
        self.router             = router or ToolRouter.default()
        self.refiner            = refiner
        self.enable_retrieval   = enable_retrieval
        self.enable_tools       = enable_tools
        self.enable_refinement  = enable_refinement
        self.retrieval_top_k    = retrieval_top_k

    # ── factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_engine(cls, engine) -> "Orchestrator":
        cfg = getattr(engine, "cfg", None)
        kdir = getattr(cfg, "knowledge_dir", "data/knowledge") if cfg else "data/knowledge"
        retrieval = RetrievalAgent(knowledge_dir=kdir)
        # Lazy-load the index on first call (load() is idempotent + cheap)
        return cls(
            engine            = engine,
            retrieval         = retrieval,
            critic            = CriticAgent(_BareWrapper(engine)),
            router            = ToolRouter.default(),
            refiner           = RefinementEngine(engine=_BareWrapper(engine)),
            enable_retrieval  = bool(getattr(cfg, "enable_retrieval", True))    if cfg else True,
            enable_tools      = bool(getattr(cfg, "enable_tools", True))         if cfg else True,
            enable_refinement = bool(getattr(cfg, "enable_refinement", True))    if cfg else True,
            retrieval_top_k   = int(getattr(cfg, "retrieval_top_k", 4))          if cfg else 4,
        )

    # ── main streaming entry point ───────────────────────────────────────

    def stream(self, user_question: str, system_prompt: str = "",
               **gen_kwargs) -> Iterator[dict]:

        chunks: list[RetrievedChunk] = []
        if self.enable_retrieval and self.retrieval is not None:
            chunks = self.retrieval.query(user_question, top_k=self.retrieval_top_k)
            yield {
                "type":   "retrieval",
                "chunks": [{"source": c.source, "score": c.score,
                            "text": c.text} for c in chunks],
            }

        tool_outputs: list[dict] = []
        if self.enable_tools:
            for name, args in _INLINE_CALL_RX.findall(user_question):
                yield {"type": "tool_call", "name": name.lower(), "args": args.strip()}
                res = self.router.call(name.lower(), args.strip())
                yield {"type": "tool_result", "name": res.name,
                       "ok": res.ok, "output": res.output}
                tool_outputs.append({"name": res.name, "ok": res.ok,
                                     "output": res.output})

        # Build augmented system prompt: original + retrieval + tool outputs
        augmented_system = self._augment_system(system_prompt, chunks, tool_outputs)

        # Generation phase
        if self.enable_refinement and self.refiner is not None:
            # Use the refinement engine; forward all its events
            refiner = self.refiner
            for ev in refiner.stream(user_question, system_prompt=augmented_system):
                yield ev
                if ev["type"] == "final_complete":
                    answer = ev["answer"]
                    final_event = ev
        else:
            # Single-pass: stream tokens and accumulate for verification
            yield {"type": "single_start"}
            buf = ""
            prior_sys = ""
            try:
                prior_sys = self.engine.get_system_prompt()
            except Exception:
                pass
            try:
                self.engine.set_system_prompt(augmented_system)
            except Exception:
                pass
            try:
                for tok, done, _ in self.engine.generate_stream(
                    prompt          = user_question,
                    max_new_tokens  = gen_kwargs.get("max_new_tokens", 300),
                    temperature     = gen_kwargs.get("temperature", 0.7),
                    top_k           = gen_kwargs.get("top_k", 40),
                    top_p           = gen_kwargs.get("top_p", 0.9),
                ):
                    if tok:
                        buf += tok
                        yield {"type": "single_delta", "delta": tok}
                    if done:
                        break
            finally:
                try:
                    self.engine.set_system_prompt(prior_sys)
                except Exception:
                    pass

            answer = parse_answer(buf) or buf.strip()
            confidence = parse_confidence(buf)
            final_event = {"type": "final_complete", "answer": answer,
                           "confidence": confidence, "raw": buf, "passes_used": 1}
            yield final_event

        # Verification (deterministic)
        v = run_checks(answer, check_numeric=True)
        yield {"type": "verification", "passed": v["passed"], "failures": v["failures"]}

        # If verification flagged something, run one critic pass (also no-LLM safety net)
        if not v["passed"]:
            critique = self.critic.critique(user_question, answer)
            yield {"type": "critic", "text": critique}

        yield {"type": "done"}

    # ── helpers ─────────────────────────────────────────────────────────

    def _augment_system(self, base: str,
                        chunks: list[RetrievedChunk],
                        tool_outputs: list[dict]) -> str:
        parts: list[str] = []
        if base.strip():
            parts.append(base.strip())
        if chunks:
            parts.append("Reference excerpts:")
            for c in chunks:
                parts.append(f"  [{c.source}] {c.text[:300]}")
        if tool_outputs:
            parts.append("Tool outputs:")
            for t in tool_outputs:
                ok = "OK" if t["ok"] else "ERR"
                parts.append(f"  {t['name']} [{ok}]: {t['output'][:200]}")
        if self.router.names():
            parts.append("Available tools (mention by name only if you need them):")
            parts.append(self.router.describe())
        return "\n".join(parts) if parts else ""


class _BareWrapper:
    """Forwards generate_stream w/ wrap_chat=False so refiner/critic scaffolds aren't re-wrapped."""

    def __init__(self, engine):
        self.engine = engine

    def generate_stream(self, prompt: str, **kwargs):
        kwargs.setdefault("wrap_chat", False)
        yield from self.engine.generate_stream(prompt, **kwargs)

    def set_system_prompt(self, text: str) -> None:
        self.engine.set_system_prompt(text)

    def get_system_prompt(self) -> str:
        return self.engine.get_system_prompt()
