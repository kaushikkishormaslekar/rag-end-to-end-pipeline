from __future__ import annotations

import re
from typing import Callable

import requests

from .config import ServiceSettings


_CONTEXT_RE = re.compile(r"CITED CONTEXT:\n(?P<context>.*?)(?:\n\nReturn:|\Z)", re.DOTALL)


def _extractive_answer(prompt: str) -> str:
    match = _CONTEXT_RE.search(prompt)
    context = match.group("context").strip() if match else ""
    if not context:
        return "I do not have enough grounded evidence in the retrieved context to answer reliably."

    snippets = []
    for block in context.split("\n\n"):
        cleaned = " ".join(block.split())
        if cleaned:
            snippets.append(cleaned)
        if len(snippets) == 3:
            break

    evidence = "\n".join(f"- {snippet}" for snippet in snippets)
    return (
        "Direct answer: The retrieved context contains relevant evidence, but no external LLM provider "
        "is configured, so this service is returning an extractive grounded response.\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Unknowns: Configure RAG_LLM_PROVIDER=ollama for generated answers."
    )


def _ollama_answer(prompt: str, settings: ServiceSettings) -> str:
    response = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("response", "")).strip()


def build_llm_callable(settings: ServiceSettings) -> Callable[[str], str]:
    if settings.llm_provider == "ollama":
        return lambda prompt: _ollama_answer(prompt, settings)
    if settings.llm_provider == "extractive":
        return _extractive_answer
    raise ValueError(f"Unsupported RAG_LLM_PROVIDER: {settings.llm_provider}")

