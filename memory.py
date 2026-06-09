"""
Long-Term Memory Module
=======================
Harness long memroy —— cross-session persistent storage of key information.

Mechanism:
  Session complete → abstract key information → write to memory.json
  Next loading → read memory.json → inject system prompt
"""

import json
import os
from datetime import datetime

from langchain_core.messages import HumanMessage

import config

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.json")
MAX_MEMORIES = 20


def load_memories() -> list[dict]:
    """Load persisted memories, return empty list if file doesn't exist"""
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_memories(memories: list[dict]) -> None:
    """Write persisted memories, truncate oldest entries if limit exceeded"""
    memories = memories[-MAX_MEMORIES:]
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)


def format_memories_for_prompt(memories: list[dict]) -> str:
    """Format memory list as injectable prompt string, taking only the most recent 5 entries"""
    if not memories:
        return ""
    lines = ["## Long-Term Memory (from previous sessions)\n"]
    for m in memories[-5:]:
        lines.append(f"- [{m['date']}] {m['summary']}")
    return "\n".join(lines)


def extract_and_save_memory(messages: list, task: str) -> str:
    """
    After a session is complete, call the model to extract key information from this session and save it to persistent storage.
    Uses config.MEMORY_MODEL, supported by config.get_llm() for all providers.
    """
    history_text = []
    for m in messages:
        role = getattr(m, "type", "unknown")
        content = m.content if isinstance(m.content, str) else str(m.content)
        if content.strip():
            history_text.append(f"[{role}]: {content[:500]}")

    history_str = "\n".join(history_text[-20:])

    extract_prompt = f"""You are a memory extraction assistant.

Given this agent session, extract ONE concise summary sentence (max 80 words) capturing:
- What task was completed
- Key files created or modified
- Any important outcomes or errors

Task: {task}

Session history:
{history_str}

Respond with ONLY the summary sentence, nothing else."""

    llm = config.get_llm(config.MEMORY_MODEL)
    response = llm.invoke([HumanMessage(content=extract_prompt)])
    summary = response.content.strip()

    memories = load_memories()
    memories.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": task[:100],
        "summary": summary,
    })
    save_memories(memories)

    return summary
