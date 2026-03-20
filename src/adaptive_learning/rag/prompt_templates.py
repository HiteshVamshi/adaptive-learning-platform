from __future__ import annotations


def build_grounded_prompt(*, query: str, retrieved_context: str) -> str:
    return (
        "You are a math tutor answering strictly from the retrieved context.\n"
        "If the context is insufficient, say so explicitly.\n\n"
        f"Question: {query}\n\n"
        "Retrieved context:\n"
        f"{retrieved_context}\n\n"
        "Provide a concise, grounded answer."
    )
