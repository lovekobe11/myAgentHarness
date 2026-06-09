# myAgentHarness Architecture

## Overview

`myAgentHarness` is a minimal LangGraph-based agent harness with the following core responsibilities:

- Load provider configuration and build model clients
- Assemble a safe system prompt with injected long-term memory
- Run the agent in a graph loop that alternates between model reasoning, safety checks, and tool execution
- Persist session summaries for future runs
- Keep tool operations inside an isolated sandbox

## Major Components

### 1. Configuration

File: `config.py`

Responsibilities:
- Load `.env` values into environment variables
- Determine which provider is active
- Create a provider-specific LangChain LLM client
- Validate required API keys and dependencies

### 2. Prompt / Memory

Files: `prompts.py`, `memory.py`

Responsibilities:
- Build the system prompt with safety rules and sandbox instructions
- Load recent session memory from `memory.json`
- Inject long-term memory into the prompt automatically
- After session completion, extract a concise summary from model output
- Append the new summary to `memory.json`

### 3. Agent / Graph Control

File: `harness.py`

Responsibilities:
- Define the agent state and graph flow
- Create `StateGraph` nodes for model reasoning, guard decision, and tool execution
- Manage step count and termination conditions
- Start the interactive user task loop

### 4. Safety Guard

File: `guard.py`

Responsibilities:
- Classify tools into safety categories
- Auto-approve read-only operations
- Always require confirmation for writes and deletes
- Detect dangerous keywords for content-sensitive tools
- Request explicit human approval when needed

### 5. Sandbox Tools

File: `tools.py`

Responsibilities:
- Provide sandboxed tool APIs for the model
- Keep all file operations under `/tmp/sandbox`
- Support listing, reading, inspecting, writing, deleting, and running Python files
- Return normalized text responses for the agent

## Runtime Flow

```text
User input
   ↓
[harness.py] init_state
   ↓
agent_node
   ↓
[prompts.py] system prompt + memory
   ↓
LLM invocation
   ↓
model response
   ↓
if tool call exists → guard_node
   ↓
[guard.py] confirm/deny
   ↓
if approved → tool_node
   ↓
[tools.py] execute in /tmp/sandbox
   ↓
tool output → agent_node
   ↓
repeat until done or max steps
   ↓
[memory.py] extract_and_save_memory
```

## Component Diagram

```text
+-----------------+      +----------------+      +-----------------+
|                 |      |                |      |                 |
|   User Task     | ---> |   harness.py   | ---> |   config.py     |
|                 |      |                |      |                 |
+-----------------+      +----------------+      +-----------------+
                                      |
                                      v
                              +-----------------+
                              |  prompts.py     |
                              |  memory.py      |
                              +-----------------+
                                      |
                                      v
                               +----------------+
                               |   LLM Model    |
                               +----------------+
                                      |
                       +--------------+--------------+
                       |                             |
                       v                             v
                +-------------+               +---------------+
                | guard.py    |               | tools.py      |
                +-------------+               +---------------+
                       |                             |
                       +-------------+---------------+
                                     |
                                     v
                                  Loop back
```

## Safety Zones

- `config.py`: trusted provider setup and validation
- `prompts.py`: safe operation rules, memory injection
- `guard.py`: human approval boundary
- `tools.py`: sandbox enforcement in `/tmp/sandbox`
- `memory.json`: persistent session memory

## Notes

- The harness is designed for experimentation with multi-provider LLMs.
- The loop is bounded by `config.MAX_STEPS` to prevent runaway reasoning.
- All side-effectful file operations are gated by explicit approval.
- `memory.json` keeps up to 20 summaries and injects the 5 newest into the prompt.
