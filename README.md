# myAgentHarness

Minimal Agent Harness built on LangGraph. Multi-provider, safety-guarded, memory-enabled.

## Features

- Multi-provider support (Anthropic, OpenAI, DeepSeek, Qwen, Custom)
- 3-tier tool safety guard (auto-approve / always-confirm / keyword-check)
- Cross-session long-term memory (persisted to `memory.json`)
- Minimal 6-layer architecture

## Quick Start

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# For non-Anthropic providers, also install:
pip install langchain-openai

# 2. Edit .env — set your Provider and API Key
# 3. Run
python harness.py
```

## Usage Examples

### Example 1: Write and run a Python script

```
Task: Write a Python script that prints Fibonacci numbers up to 100, then run it

[HARNESS] Step 1/10 — Agent thinking...
[HARNESS] Step 2/10 — Agent thinking...
=======================================================
  FINAL RESPONSE
=======================================================
I wrote fib.py and ran it. Output: 1 1 2 3 5 8 13 21 34 55 89
=======================================================
  Total steps used: 2/10
=======================================================
[HARNESS] Memory saved: User asked to write and run a Fibonacci script.
```

### Example 2: Resume with long-term memory

On the second run, MicroHarness automatically loads the previous session summary:

```
[HARNESS] Found 1 long-term memory record(s).
          Last: 2024-01-15 — User asked to write and run a Fibonacci script...

Task: Improve the script from last time

[HARNESS] Step 1/10 — Agent thinking...
(Agent recalls previous session context from injected memory)
```

### Example 3: Safety guard blocks a dangerous write

When the agent attempts to write code containing dangerous operations, the guard pauses and requests human confirmation:

```
[HARNESS] Step 1/10 — Agent thinking...

=======================================================
  [HARNESS GUARD] 📝 WRITE OP
  Tool   : write_file
  filename: cleanup.py
  content : import shutil; shutil.rmtree('/tmp/sandbox')
=======================================================
  Approve? (yes / no): no
  ❌ Rejected. Operation cancelled.
```

## Provider Configuration

Edit `.env` to select your provider and supply the API key:

### Anthropic (example)
```bash
PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key
MAIN_MODEL=claude-sonnet-4-20250514
MEMORY_MODEL=claude-haiku-4-5-20251001
```

### DeepSeek
```bash
PROVIDER=deepseek
OPENAI_COMPATIBLE_API_KEY=your_key
MAIN_MODEL=deepseek-chat
MEMORY_MODEL=deepseek-chat
```

### Qwen (Alibaba)
```bash
PROVIDER=qwen
OPENAI_COMPATIBLE_API_KEY=your_key
MAIN_MODEL=qwen-plus
MEMORY_MODEL=qwen-turbo
```

### OpenAI / GPT
```bash
PROVIDER=openai
OPENAI_COMPATIBLE_API_KEY=your_key
MAIN_MODEL=gpt-4o
MEMORY_MODEL=gpt-4o-mini
```

```

### Custom Provider (any OpenAI-compatible API)
```bash
PROVIDER=custom
OPENAI_COMPATIBLE_API_KEY=your_key
OPENAI_COMPATIBLE_BASE_URL=https://your-api-endpoint/v1
MAIN_MODEL=your-model-name
MEMORY_MODEL=your-model-name
```

## Tool Safety Levels

| Tool | Level | Behavior |
|---|---|---|
| `list_files` | AUTO_APPROVE | Automatically allowed |
| `read_file` | AUTO_APPROVE | Automatically allowed |
| `get_file_info` | AUTO_APPROVE | Automatically allowed |
| `write_file` | ALWAYS_CONFIRM | Always requires human approval |
| `delete_file` | ALWAYS_CONFIRM | Always requires human approval |
| `run_python` | KEYWORD_CHECK | Blocked if dangerous keywords detected |

## Architecture

| Layer | File | Responsibility |
|---|---|---|
| Config | `config.py` | Reads `.env`, builds LLM instance by provider |
| Prompts | `prompts.py` | System prompt + long-term memory injection |
| Tools | `tools.py` | 6 tools, sandboxed to `/tmp/sandbox/` |
| Guard | `guard.py` | 3-tier classification, human confirmation |
| Lifecycle | `harness.py` | MAX_STEPS, routing control |
| Memory | `memory.py` | Cross-session persistence, `memory.json` |
