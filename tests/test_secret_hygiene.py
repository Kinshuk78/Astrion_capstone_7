from __future__ import annotations

import re
from pathlib import Path


def test_env_example_uses_placeholder_openrouter_key():
    text = Path("config/.env.example").read_text(encoding="utf-8")
    match = re.search(r"^OPENROUTER_API_KEY=(.*)$", text, flags=re.MULTILINE)
    assert match is not None, "config/.env.example must define OPENROUTER_API_KEY"

    value = match.group(1).strip()
    assert value == "sk-or-v1-your-key-here"
    assert not re.fullmatch(r"sk-or-v1-[A-Za-z0-9]{20,}", value), (
        "config/.env.example must not contain a real OpenRouter key"
    )


def test_gitignore_ignores_local_env_files_but_keeps_example():
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "config/.env*" in text
    assert "!config/.env.example" in text
