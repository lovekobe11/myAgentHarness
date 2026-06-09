"""
myAgentHarness — Main Entry Point
================================
Based on LangGraph small Agent Harness demo。

Config edit .env file to set provider and API keys, then run.
Supported Provider: anthropic / openai / deepseek / kimi / minimax / qwen / glm

Architecture:
  [User Input]
       ↓
  [agent_node]   ← Prompt Management(including long-term memory) + model inference
       ↓
  [guard_node]   ← Safety Guard (Write Operations/High-Risk Operations Intercept)
       ↓
  [tool_node]    ← Tool Execution
       ↓
  [agent_node]   ← Continue Reasoning (Loop, until task completion or limit reached)
       ↓
  [memory]       ← Long-term Memory Extraction + Persistent Writing to memory.json

Usage:
  python harness.py
"""

from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

import config
from guard import should_confirm, request_human_approval
from memory import extract_and_save_memory, load_memories
from prompts import get_system_prompt
from tools import TOOLS


# ──────────────────────────────────────────────────
# State Definition
# ──────────────────────────────────────────────────

class HarnessState(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int
    approved: bool


# ──────────────────────────────────────────────────
# Model Initialization (Built by config.get_llm according to provider)
# ──────────────────────────────────────────────────

llm = config.get_llm(config.MAIN_MODEL).bind_tools(TOOLS)


# ──────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────

def agent_node(state: HarnessState) -> dict:
    """Model Inference Node: Inject System Prompt (including long-term memory) → Inference → step_count +1"""
    system = SystemMessage(content=get_system_prompt())
    messages = [system] + state["messages"]

    print(f"\n[HARNESS] Step {state['step_count'] + 1}/{config.MAX_STEPS} — Agent thinking...")
    response = llm.invoke(messages)

    return {
        "messages": [response],
        "step_count": state["step_count"] + 1,
    }


def guard_node(state: HarnessState) -> dict:
    """Safety Guard Node: Check tool calls, request human approval for write/delete operations"""
    last = state["messages"][-1]
    approved = True

    if hasattr(last, "tool_calls") and last.tool_calls:
        for call in last.tool_calls:
            if should_confirm(call["name"], call["args"]):
                approved = request_human_approval(call["name"], call["args"])
                if not approved:
                    break

    return {"approved": approved}


tool_node = ToolNode(TOOLS)


# ──────────────────────────────────────────────────
# Routing Functions
# ──────────────────────────────────────────────────

def route_after_agent(state: HarnessState) -> str:
    if state["step_count"] >= config.MAX_STEPS:
        print(f"\n[HARNESS] ⚠️  Max steps ({config.MAX_STEPS}) reached. Stopping.")
        return END
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "guard"
    return END


def route_after_guard(state: HarnessState) -> str:
    return "tools" if state["approved"] else END


# ──────────────────────────────────────────────────
# 构建图
# ──────────────────────────────────────────────────

def build_harness():
    graph = StateGraph(HarnessState)
    graph.add_node("agent", agent_node)
    graph.add_node("guard", guard_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_conditional_edges("guard", route_after_guard)
    graph.add_edge("tools", "agent")
    return graph.compile()
##build_harness().get_graph().draw_mermaid_png(output_file_path="langgraph_human_review.png")

# ──────────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────────

def main():
    config.validate()

    print("=" * 55)
    print("  myAgentHarness  —  LangGraph + Qwen Demo")
    print(f"  Provider    : {config.PROVIDER}")
    print(f"  Main Model  : {config.MAIN_MODEL}")
    print(f"  Memory Model: {config.MEMORY_MODEL}")
    print(f"  Max Steps   : {config.MAX_STEPS}")
    print(f"  Sandbox     : /tmp/sandbox/")
    print("=" * 55)

    existing = load_memories()
    if existing:
        print(f"\n[HARNESS] Found {len(existing)} long-term memory record(s).")
        print(f"          Last: {existing[-1]['date']} — {existing[-1]['summary'][:60]}...")
    else:
        print("\n[HARNESS] No long-term memory found. Starting fresh.")

    harness = build_harness()

    print("\nType your task below. Examples:")
    print("  • Write a Python script that prints Fibonacci numbers up to 100, then run it")
    print("  • Improve the script from last time")
    print()

    user_input = input("Task: ").strip()
    if not user_input:
        print("No task provided. Exiting.")
        return

    init_state: HarnessState = {
        "messages": [HumanMessage(content=user_input)],
        "step_count": 0,
        "approved": True,
    }

    print("\n[HARNESS] Starting...\n")
    final_state = harness.invoke(init_state)

    final_messages = final_state["messages"]
    final_response = next(
        (m for m in reversed(final_messages)
         if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip()),
        None
    )

    print("\n" + "=" * 55)
    print("  FINAL RESPONSE")
    print("=" * 55)
    print(final_response.content if final_response else "(Task completed — see tool outputs above)")
    print("=" * 55)
    print(f"  Total steps used: {final_state['step_count']}/{config.MAX_STEPS}")
    print("=" * 55)

    print("\n[HARNESS] Extracting long-term memory...")
    summary = extract_and_save_memory(final_state["messages"], user_input)
    print(f"[HARNESS] Memory saved: {summary}\n")


if __name__ == "__main__":
    main()



