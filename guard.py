"""
Safety Guard Module
===================
This guard module implements a multi-tier safety mechanism to mitigate risks:

Design Principles:
- AUTO_APPROVE_TOOLS (Read-only, No Side Effects): Automatic Approval
- ALWAYS_CONFIRM_TOOLS (Write/Delete Operations): Force Manual Confirmation Regardless of Content
- Other Tools: Detect Dangerous Keywords, Trigger Manual Confirmation When Found
- After human rejection, Harness will pause and skip executing the tool, preventing potential harm.

Tool Classification:
  AUTO_APPROVE  : list_files, read_file, get_file_info   ← Read-only, No Side Effects
  ALWAYS_CONFIRM: write_file, delete_file                ← Write/Delete Operations
  KEYWORD_CHECK : run_python, run_sql                    ← Content Determines Risk
"""

# High risk keywords —— will trigger manual confirmation if appear in tool_input, regardless of tool_name
DANGEROUS_KEYWORDS = [
    "rm ",
    "shutil.rmtree",
    "os.remove",
    "os.unlink",
    "DROP TABLE",
    "DELETE FROM",
    "format(",          # disk formatting command
    "subprocess.call",  # original shell call
    "> /dev/",          # writing to system devices
]

# Regardless of content —— Force manual confirmation for tools with persistent side effects
ALWAYS_CONFIRM_TOOLS = {
    "write_file",    # Write or overwrite file
    "delete_file",   # Delete file
}

# Auto approve —— Read-only, No Side Effects
AUTO_APPROVE_TOOLS = {
    "list_files",    # List files
    "read_file",     # Read file content
    "get_file_info", # Get file metadata
}


def is_dangerous(tool_input: dict) -> bool:
    """
    Check if the tool call parameters contain dangerous keywords.

    Note: This function only checks the parameter content, not the tool_name.
    The tool_name level judgment is handled by the分级 logic in should_confirm().

    Args:
        tool_input: call tools parameters dictionary

    Returns:
        True if parameters contain dangerous content
    """
    content = str(tool_input).lower()
    return any(kw.lower() in content for kw in DANGEROUS_KEYWORDS)


def should_confirm(tool_name: str, tool_input: dict) -> bool:
    """
    Decide whether manual confirmation is needed.

    Priority (from high to low):
    1. AUTO_APPROVE_TOOLS → Directly approve, do not check content
    2. ALWAYS_CONFIRM_TOOLS → Directly intercept, do not check content
    3. Other tools → Check parameters for dangerous keywords

    Args:
        tool_name: Tool name
        tool_input: Tool call parameters dictionary

    Returns:
        True = Manual confirmation needed
        False = Auto approve
    """
    if tool_name in AUTO_APPROVE_TOOLS:
        return False
    if tool_name in ALWAYS_CONFIRM_TOOLS:
        return True
    return is_dangerous(tool_input)


def request_human_approval(tool_name: str, tool_input: dict) -> bool:
    """
    Pause execution and display the pending operation to the operator for explicit approval.

    Args:
        tool_name: Tool name
        tool_input: Tool call parameters

    Returns:
        True = Approved to continue execution
        False = Rejected, Harness will abort this operation
    """
    # Determine risk label based on tool type and content
    if tool_name in ALWAYS_CONFIRM_TOOLS and tool_name == "delete_file":
        flag = "🗑️  DELETE OP"
    elif is_dangerous(tool_input):
        flag = "⚠️  HIGH RISK"
    else:
        flag = "📝 WRITE OP"

    print(f"\n{'='*55}")
    print(f"  [HARNESS GUARD] {flag}")
    print(f"  Tool   : {tool_name}")

    for k, v in tool_input.items():
        display_val = str(v)
        if len(display_val) > 200:
            display_val = display_val[:200] + "... (truncated)"
        print(f"  {k:8}: {display_val}")

    print(f"{'='*55}")

    while True:
        answer = input("  Approve? (yes / no): ").strip().lower()
        if answer in ("yes", "y"):
            print("  ✅ Approved.\n")
            return True
        elif answer in ("no", "n"):
            print("  ❌ Rejected. Operation cancelled.\n")
            return False
        else:
            print("  Please type 'yes' or 'no'.")
