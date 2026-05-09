"""
CriticAgent: same HAVOC model, different system prompt.

Used inside the orchestrator to review prior outputs for:
  - factual inconsistencies
  - ambiguity
  - missing assumptions
  - logic gaps
  - unsupported claims

Returns a short critique string. The orchestrator can splice it into
the next refinement pass to give the model targeted feedback.
"""

from __future__ import annotations


_CRITIC_SYSTEM = (
    "You are a strict critic. Read the user question and the candidate answer "
    "below. Produce a short bulleted critique (3-5 bullets max). Look for:\n"
    "  - factual inconsistencies\n"
    "  - ambiguity or vague wording\n"
    "  - missing assumptions or qualifications\n"
    "  - logic gaps\n"
    "  - claims not supported by the question's wording\n"
    "If the answer looks fine, say 'No major issues.' Be concise."
)


class CriticAgent:
    def __init__(self, engine, max_tokens: int = 200):
        self.engine     = engine     # any object with generate_stream(wrap_chat=True)
        self.max_tokens = max_tokens

    def critique(self,
                 user_question: str,
                 candidate_answer: str,
                 ) -> str:
        prompt = (
            f"Question: {user_question}\n\n"
            f"Candidate answer: {candidate_answer}\n\n"
            "Critique:"
        )
        # Save current system prompt, swap to critic prompt, restore
        prior = ""
        try:
            prior = self.engine.get_system_prompt()
        except Exception:
            pass
        try:
            self.engine.set_system_prompt(_CRITIC_SYSTEM)
        except Exception:
            pass

        out_parts: list[str] = []
        try:
            for tok, done, _ in self.engine.generate_stream(
                prompt          = prompt,
                max_new_tokens  = self.max_tokens,
                temperature     = 0.5,
                top_k           = 40,
                top_p           = 0.9,
            ):
                if tok:
                    out_parts.append(tok)
                if done:
                    break
        finally:
            try:
                self.engine.set_system_prompt(prior)
            except Exception:
                pass

        return "".join(out_parts).strip()
