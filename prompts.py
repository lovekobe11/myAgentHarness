"""
Prompt Management Module
========================
Harness prompts —— responsible for injecting the correct "onboarding manual" into the model at each startup.

"""

from memory import format_memories_for_prompt, load_memories

SYSTEM_PROMPT_BASE = """
You are a coding assistant operating inside a safe harness.

Rules:
- You may write files and run Python code inside the sandbox
- You MUST NOT run shell commands that delete, move, or overwrite files outside the workspace
- Always explain what you are about to do before doing it
- If a task is unclear, ask for clarification instead of guessing
- Keep code clean, readable, and well-commented

Workspace: /tmp/sandbox/
"""


def get_system_prompt() -> str:
    """
    Assemble the system prompt: base rules + long-term memory (if any).

    Called each time the Agent starts, automatically appending key information from the previous session,
    enabling the model to have cross-session contextual awareness.
    """
    base = SYSTEM_PROMPT_BASE.strip()
    memories = load_memories()
    memory_block = format_memories_for_prompt(memories)

    if memory_block:
        full_prompt = f"{base}\n\n{memory_block}"
        print(f"[HARNESS] Memory loaded: {len(memories)} record(s) injected into prompt.")
        return full_prompt

    return base
