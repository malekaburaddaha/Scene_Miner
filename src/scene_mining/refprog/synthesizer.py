"""LLM-based Python code synthesizer for RefProg.

Sends the system prompt + user query to an LLM (Claude or GPT-4o) and
returns the synthesised Python code string.

Backend selection via the ``backend`` argument:
- ``"claude"`` — Anthropic Claude (default, recommended by RefAV paper)
- ``"openai"`` — OpenAI GPT-4o
"""

from __future__ import annotations

import re

from .prompt_builder import build_system_prompt, build_user_prompt


class RefProgSynthesizer:
    """Synthesise RefProg Python code from a natural language query.

    Args:
        backend: LLM backend — ``"claude"`` or ``"openai"``.
        model: Model identifier. Defaults to ``"claude-opus-4-6"`` for
            Claude or ``"gpt-4o"`` for OpenAI.
        include_visual: Whether to include visual atomic function API in
            the system prompt.
        temperature: Sampling temperature (default 0.0 for determinism).
        max_tokens: Maximum tokens in the LLM response.
    """

    def __init__(
        self,
        backend: str = "claude",
        model: str | None = None,
        include_visual: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.backend = backend
        self.model = model or ("claude-opus-4-6" if backend == "claude" else "gpt-4o")
        self.include_visual = include_visual
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._system_prompt = build_system_prompt(include_visual=include_visual)

    def synthesize(
        self,
        query: str,
        log_id: str | None = None,
        extra_context: str | None = None,
    ) -> str:
        """Generate Python code for a single natural language query.

        Args:
            query: Natural language scene description.
            log_id: Optional AV2 log ID (for context injection).
            extra_context: Optional additional context (e.g. previous
                error message for iterative correction).

        Returns:
            Python source code string (the ``run`` function body).
        """
        user_prompt = build_user_prompt(query, log_id=log_id)
        if extra_context:
            user_prompt += f"\n\nAdditional context / previous error:\n{extra_context}"

        raw_response = self._call_llm(self._system_prompt, user_prompt)
        return _extract_code(raw_response)

    def synthesize_batch(
        self,
        queries: list[str],
        log_id: str | None = None,
    ) -> list[str]:
        """Synthesise code for a batch of queries in one LLM call.

        Returns:
            List of code strings, one per query.
        """
        user_prompt = build_user_prompt("", log_id=log_id, batch_queries=queries)
        raw_response = self._call_llm(self._system_prompt, user_prompt)
        return _extract_code_batch(raw_response, len(queries))

    # ── Backend dispatch ───────────────────────────────────────────────────────

    def _call_llm(self, system: str, user: str) -> str:
        if self.backend == "claude":
            return self._call_claude(system, user)
        elif self.backend == "openai":
            return self._call_openai(system, user)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def _call_claude(self, system: str, user: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text

    def _call_openai(self, system: str, user: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


# ── Code extraction helpers ────────────────────────────────────────────────────


def _extract_code(response: str) -> str:
    """Extract Python code from a markdown-fenced response.

    Handles:
    - ```python ... ``` blocks
    - ``` ... ``` blocks
    - Plain text (returned as-is)
    """
    # Try ```python ... ``` first
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return raw response (may already be code)
    return response.strip()


def _extract_code_batch(response: str, n: int) -> list[str]:
    """Extract multiple code blocks from a batch response.

    Tries to find n ```python...``` blocks; falls back to splitting
    on numbered headers.
    """
    blocks = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if len(blocks) == n:
        return [b.strip() for b in blocks]

    # Fallback: duplicate the single block n times (LLM may have combined them)
    if len(blocks) == 1:
        return [blocks[0].strip()] * n

    # Last resort: return empty strings so error corrector can retry
    return [""] * n
