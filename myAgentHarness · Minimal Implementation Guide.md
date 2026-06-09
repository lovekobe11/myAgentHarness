# myAgentHarness · Minimal Implementation Guide

**myAgentHarness** is a minimal Agent framework demo project based on **LangGraph**\. It consists of 9 core files, implements a 5\-layer core architecture, supports switching between 8 LLM providers, and includes built\-in **Human\-in\-the\-Loop** security guard and cross\-session **Long\-Term Memory**\. This document aims to provide the most concise implementation guide for easy understanding and further development\.

# 01 · Background

The value of **myAgentHarness** lies in its modularity and observability\. Unlike out\-of\-the\-box tools like **DeepAgents**, this project separates each core concept \(configuration, tools, guard, memory\) into independent files\. This allows you to clearly see, modify, and replace each layer—the foundation for understanding how **Harness** works and customizing it \(e\.g\., integrating supply chain platform business APIs\)\.



# 02 · Quick Start

Start the project in 3 steps\.

1. **Download and enter project directory**

2. **Install core dependencies*** \(For non\-Anthropic models, additionally install: pip install langchain\-openai\)*

3. **Configure \.env and run**

**Running Example**: After starting, enter a task \(e\.g\., `Write a Python script that prints Fibonacci numbers up to 100, then run it`\)\. Harness will automatically execute the `write file → guard intercepts and requests human confirmation → execute → read results → return summary` workflow\.



# 03 · Project Structure

The project consists of 9 files with clear responsibilities\.

```Plain Text
myAgentHarness/
├── harness.py    # Main entry — LangGraph StateGraph definition, routing & lifecycle control
├── config.py     # Config layer — reads .env, builds LLM instances by PROVIDER
├── prompts.py    # Prompts layer — system prompt injection & long-term memory appending
├── tools.py      # Tools layer — 6 tools classified by security level
├── guard.py      # Guard layer — three-level interception logic for human-in-the-loop
├── memory.py     # Memory layer — session summary extraction & cross-session persistence
├── .env          # User config file — Provider / API Key / Model Name
├── memory.json   # Auto-generated — cross-session memory persistence file
└── requirements.txt # Dependency declaration

```



# 04 · Core Architecture

Harness is split into 5 functional layers \+ 1 config layer, each with a single responsibility\.

|Layer|Core Files|Core Responsibility|
|---|---|---|
|**CONFIG**|`config.py` \+ `.env`|Unified config entry, reads `.env` and builds LLM instances by **PROVIDER**, supports 8 providers\.|
|**LAYER 01**|`prompts.py`|Manages prompts, injects system prompts on each startup, and automatically appends long\-term memory from `memory.json`\.|
|**LAYER 02**|`tools.py`|Defines and schedules 6 tools, classified by security level, executes in sandbox, auto\-truncates output to prevent context explosion\.|
|**LAYER 03**|`guard.py`|Security guard, implements three\-level classification: auto\-approve / force\-confirmation / keyword detection\.|
|**LAYER 04**|`harness.py`|Lifecycle management, defines **MAX\_STEPS** hard limit, controls routing loop, gracefully exits when limit reached\.|
|**LAYER 05**|`memory.py`|Long\-term memory management, calls memory model to extract summary after session, writes to `memory.json`, injects on next startup\.|



# 05 · Execution Flow

The entire Harness is a **LangGraph StateGraph**, with nodes connected via conditional routing\.

1. **User Input**: Task input, initialize State \(`step_count=0`\)\.

2. **agent\_node**: Inject system prompt \(with long\-term memory\) → model reasoning → `step_count +1`\.

3. **guard\_node**: Check tool calls → intercept write operations → wait for human confirmation \(yes/no\)\.

4. **tool\_node**: Execute tools \(e\.g\., `write_file`, `run_python`\) → write results back to State\.

5. **Loop**: Return to `agent_node` to continue reasoning until task completion or **MAX\_STEPS** reached\.

6. **Memory Save**: Session ends → call `memory.extract_and_save()` to extract summary → write to `memory.json`\.



# 06 · Security Guard

The guard layer **guard\.py** takes different strategies for different tools—this is the key difference between Harness and regular Agents\.

|Tool|Interception Strategy|Description|
|---|---|---|
|`list_files`|**✅ Auto\-Approved**|Read\-only operation, no side effects\.|
|`write_file`|**⏸ Force Confirmation**|Write operation, blocked by default, waits for human yes/no confirmation\.|
|`run_python`|**⚠️ Keyword Detection**|Blocked if contains dangerous keywords like `rm`, `DELETE`, `rmtree`, etc\.|

If human selects "no", Harness routes to **END** and will not execute any further tool calls\.



# 07 · Multi\-Model Support

All provider switching logic is centralized in **config\.py**\. Users only need to modify three lines in `.env`\.

**\.env Configuration Example \(DeepSeek\)**:

```Bash
PROVIDER=deepseek
OPENAI_COMPATIBLE_API_KEY=your_deepseek_key
MAIN_MODEL=deepseek-chat

```

**Supported Providers**:

|Provider|PROVIDER=|Common MAIN\_MODEL|Base URL \(Built\-in\)|
|---|---|---|---|
|Anthropic|`anthropic`|claude\-sonnet\-4\-20250514|—|
|**DeepSeek**|`deepseek`|deepseek\-chat|api\.deepseek\.com|
|**Qwen**|`qwen`|qwen\-plus|dashscope\.aliyuncs\.com/\.\.\.|
|OpenAI|`openai`|gpt\-4o|api\.openai\.com/v1|
|Custom|Any value|—|Fill in `OPENAI_COMPATIBLE_BASE_URL`|

*Note: For non\-Anthropic providers, pre\-install: pip install langchain\-openai\.*



# 08 · Expansion Directions

Based on this minimal implementation, you can expand in the following directions:

- **Real Sandbox Isolation**: Replace `/tmp/sandbox` with Docker containers for complete code execution isolation without affecting the host machine\.

- **Async Approval Notifications**: Modify the `request_human_approval` function in `guard.py` to push approval requests to Slack, whatsapp, etc\., supporting mobile approval\.

- **Integrate Business Tools**: Add business APIs to `tools.py` so the Agent can operate real business data\.

- **Upgrade Memory Backend**: Replace `memory.json` with PostgreSQL or vector database for semantic retrieval of historical memory\.



> (Note: The content is generated by AI. Please use with caution.)
