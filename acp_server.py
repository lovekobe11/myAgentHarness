"""
ACP Server — Agent Client Protocol entry point for myAgentHarness
=================================================================
Wraps the existing LangGraph agent harness as an ACP-compliant agent server,
allowing any ACP-compatible editor (Zed, etc.) to connect via JSON-RPC over stdio.

Architecture:
  ┌─────────────┐   stdio / JSON-RPC   ┌──────────────────┐
  │  ACP Client │ ◄──────────────────► │  HarnessAgent    │
  │  (Zed, etc) │                      │  (this module)   │
  └─────────────┘                      └────────┬─────────┘
                                                │
                                   ┌────────────┴────────────┐
                                   │   LangGraph  (manual     │
                                   │   step-through)          │
                                   │                          │
                                   │  agent_node ──► guard    │
                                   │       ▲          │       │
                                   │       │     tools_node   │
                                   │       └──────────┘       │
                                   └──────────────────────────┘

Key design decisions:
  • Manual graph step-through (instead of harness.invoke()) to enable
    per-step streaming via session_update() and async permission requests.
  • Guard approvals routed through ACP request_permission callback,
    replacing the CLI stdin-based approval in guard.py.
  • All existing modules (config, tools, guard, memory, prompts) reused as-is.

Usage:
  # Standalone — test with any ACP client:
  python acp_server.py

  # Or configure in Zed settings.json:
  {
    "agent_servers": {
      "myAgentHarness": {
        "type": "custom",
        "command": "python",
        "args": ["/abs/path/to/acp_server.py"]
      }
    }
  }
"""

import asyncio
import logging
from typing import Any
from uuid import uuid4

# ── ACP SDK imports ──────────────────────────────────────────
from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    start_tool_call,
    text_block,
    tool_content,
    update_agent_message,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import (
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    McpServerStdio,
    ResourceContentBlock,
    SseMcpServer,
    TextContentBlock,
)

# ── LangChain / LangGraph imports ────────────────────────────
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

# ── Harness module imports (reuses all existing logic) ───────
import config
from guard import should_confirm, ALWAYS_CONFIRM_TOOLS, AUTO_APPROVE_TOOLS, is_dangerous
from memory import extract_and_save_memory, load_memories
from prompts import get_system_prompt
from tools import TOOLS

# ── Logger ───────────────────────────────────────────────────
logger = logging.getLogger("harness")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _classify_tool_kind(tool_name: str) -> str:
    """Map tool name to an ACP tool-call kind label for the editor UI."""
    if tool_name in AUTO_APPROVE_TOOLS:
        return "read"
    if tool_name == "delete_file":
        return "delete"
    if tool_name in ALWAYS_CONFIRM_TOOLS:
        return "edit"
    return "execute"


def _risk_label(tool_name: str, tool_input: dict) -> str:
    """Produce a human-readable risk label for the permission dialog."""
    if tool_name == "delete_file":
        return "DELETE operation"
    if is_dangerous(tool_input):
        return "HIGH RISK operation"
    return "Write operation"


def _format_args(tool_input: dict) -> str:
    """Compact display of tool arguments for the permission dialog."""
    parts = []
    for k, v in tool_input.items():
        display_val = str(v)
        if len(display_val) > 200:
            display_val = display_val[:200] + "..."
        parts.append(f"{k}: {display_val}")
    return "\n".join(parts) if parts else "(no arguments)"


# ──────────────────────────────────────────────────────────────
# ACP Agent Implementation
# ──────────────────────────────────────────────────────────────

class HarnessAgent(Agent):
    """
    ACP-compliant wrapper around the myAgentHarness LangGraph agent.

    Lifecycle:
      on_connect  → stores the client connection for sending updates
      initialize  → validates config, initialises the LLM, loads memory
      new_session → creates a new conversation session
      prompt      → runs the full agent loop with streaming & permissions
    """

    _conn: Client
    _llm: Any  # LangChain BaseChatModel bound with tools

    # ── Connection ────────────────────────────────────────────

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    # ── ACP Lifecycle ────────────────────────────────────────

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        """Validate configuration and prepare the LLM + memory."""
        config.validate()
        self._llm = config.get_llm(config.MAIN_MODEL).bind_tools(TOOLS)

        existing = load_memories()
        if existing:
            logger.info(
                f"[ACP] Loaded {len(existing)} memory record(s). "
                f"Last: {existing[-1]['summary'][:60]}..."
            )
        else:
            logger.info("[ACP] No long-term memory found — starting fresh.")

        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        return NewSessionResponse(session_id=uuid4().hex)

    # ── Prompt (main agent loop) ─────────────────────────────

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        """
        Run the LangGraph agent loop, streaming text and tool-call updates
        to the ACP client and routing guard approvals through request_permission.
        """
        # ── 1. Extract user text from ACP content blocks ─────
        user_text = ""
        for block in prompt:
            if isinstance(block, dict):
                user_text += block.get("text", "")
            else:
                user_text += getattr(block, "text", "")
        user_text = user_text.strip()

        if not user_text:
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message(text_block("No task provided.")),
            )
            return PromptResponse(stop_reason="end_turn", user_message_id=message_id)

        # ── 2. Initialise conversation state ─────────────────
        messages: list = [HumanMessage(content=user_text)]
        step_count = 0

        # ── 3. Agent loop (manual graph step-through) ────────
        while step_count < config.MAX_STEPS:
            step_count += 1
            logger.info(f"[ACP] Step {step_count}/{config.MAX_STEPS} — Agent thinking...")

            # ▸ Agent node: system prompt + memory → LLM inference
            system = SystemMessage(content=get_system_prompt())
            response: AIMessage = self._llm.invoke([system] + messages)
            messages.append(response)

            # ▸ Stream the model's text response to the editor
            if response.content and isinstance(response.content, str) and response.content.strip():
                await self._conn.session_update(
                    session_id=session_id,
                    update=update_agent_message(text_block(response.content)),
                )

            # ▸ If no tool calls → task is complete
            if not (hasattr(response, "tool_calls") and response.tool_calls):
                break

            # ▸ Guard + Tool execution for each tool call
            approved = True
            for call in response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                call_id = call.get("id", uuid4().hex)
                kind = _classify_tool_kind(tool_name)

                # ── Stream: tool call started ────────────────
                await self._conn.session_update(
                    session_id=session_id,
                    update=start_tool_call(
                        call_id,
                        tool_name,
                        kind=kind,
                        status="pending",
                    ),
                )

                # ── Guard check ──────────────────────────────
                if should_confirm(tool_name, tool_args):
                    # Route through ACP permission flow (editor shows approval dialog)
                    perm_result = await self._conn.request_permission(
                        session_id=session_id,
                        tool_call={
                            "title": f"{_risk_label(tool_name, tool_args)}: {tool_name}",
                            "raw_input": _format_args(tool_args),
                        },
                        options=[
                            {
                                "kind": "allow_once",
                                "label": "Approve",
                                "description": f"Allow {tool_name} to execute",
                            },
                            {
                                "kind": "reject_once",
                                "label": "Reject",
                                "description": "Deny this operation",
                            },
                        ],
                    )

                    outcome = perm_result.get("outcome", {})
                    if isinstance(outcome, dict):
                        outcome_type = outcome.get("outcome", "cancelled")
                    else:
                        outcome_type = str(outcome)

                    if outcome_type not in ("allow_once", "allow_always", "approved"):
                        # Permission denied
                        await self._conn.session_update(
                            session_id=session_id,
                            update=update_tool_call(
                                call_id,
                                status="failed",
                                content=[tool_content(text_block("Operation rejected by user."))],
                            ),
                        )
                        approved = False
                        break

                # ── Permission granted — execute the tool ────
                await self._conn.session_update(
                    session_id=session_id,
                    update=update_tool_call(call_id, status="in_progress"),
                )

                # Invoke the prebuilt ToolNode with only the current AI message
                tool_node = ToolNode(TOOLS)
                tool_state = tool_node.invoke({"messages": [response]})

                # Accumulate tool result messages
                new_tool_msgs = tool_state.get("messages", [])
                messages.extend(new_tool_msgs)

                # ── Stream: tool call completed ──────────────
                for tm in new_tool_msgs:
                    if isinstance(tm, ToolMessage):
                        result_text = (
                            tm.content
                            if isinstance(tm.content, str)
                            else str(tm.content)
                        )
                        await self._conn.session_update(
                            session_id=session_id,
                            update=update_tool_call(
                                call_id,
                                status="completed",
                                content=[tool_content(text_block(result_text))],
                            ),
                        )

            # If guard rejected, stop the loop
            if not approved:
                break

        # ── 4. Max-steps safeguard ───────────────────────────
        else:
            logger.warning(f"[ACP] Max steps ({config.MAX_STEPS}) reached.")
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message(
                    text_block(f"⚠️ Max steps ({config.MAX_STEPS}) reached. Stopping.")
                ),
            )

        # ── 5. Extract & persist long-term memory ────────────
        try:
            summary = extract_and_save_memory(messages, user_text)
            logger.info(f"[ACP] Memory saved: {summary}")
        except Exception as exc:
            logger.warning(f"[ACP] Memory extraction failed: {exc}")

        return PromptResponse(stop_reason="end_turn", user_message_id=message_id)


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

async def main() -> None:
    """Run the ACP agent server over stdio JSON-RPC."""
    await run_agent(HarnessAgent())


if __name__ == "__main__":
    asyncio.run(main())
