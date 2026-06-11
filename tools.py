"""
Tool Orchestration Module
=========================
Harness tools —— pre-configured tools for the Agent, with unified call management.

Tool Security Levels (corresponding to guard.py):
  AUTO_APPROVE  : list_files, read_file, get_file_info  ← Read-only
  ALWAYS_CONFIRM: write_file, delete_file               ← Write/Delete
  KEYWORD_CHECK : run_python, run_sql                   ← Content-based risk assessment
"""

import math
import os
import sqlite3
import subprocess
import sys
from langchain_core.tools import tool

from config import SANDBOX_DIR, RUN_TIMEOUT

SANDBOX = SANDBOX_DIR
os.makedirs(SANDBOX, exist_ok=True)


def _safe_path(filename: str) -> str:
    """Prevent path traversal attacks, enforce limitation to sandbox directory."""
    return os.path.join(SANDBOX, os.path.basename(filename))


# ── AUTO_APPROVE─────────────────

@tool
def list_files() -> str:
    """List all files currently in the sandbox directory."""
    files = os.listdir(SANDBOX)
    if not files:
        return "📂 Sandbox is empty."
    return "📂 Files in sandbox:\n" + "\n".join(f"  - {f}" for f in sorted(files))


@tool
def read_file(filename: str) -> str:
    """
    Read and return the content of a file in the sandbox.

    Args:
        filename: Name of the file to read
    """
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"❌ File not found: {filename}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if len(content) > 3000:
        content = content[:3000] + "\n... (truncated)"
    return content


@tool
def get_file_info(filename: str) -> str:
    """
    Return metadata about a file in the sandbox (size, modified time).

    Args:
        filename: Name of the file to inspect
    """
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"❌ File not found: {filename}"
    stat = os.stat(path)
    import datetime
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"📄 {filename}\n"
        f"   Size    : {stat.st_size} bytes\n"
        f"   Modified: {mtime}"
    )


# ── ALWAYS_CONFIRM (with serialize side effect)───────────────

@tool
def write_file(filename: str, content: str) -> str:
    """
    Write content to a file inside the sandbox directory.

    Args:
        filename: Name of the file (e.g. 'main.py')
        content: Full content to write
    """
    path = _safe_path(filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ File written: {path} ({len(content)} chars)"


@tool
def delete_file(filename: str) -> str:
    """
    Delete a file from the sandbox directory.

    Args:
        filename: Name of the file to delete
    """
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"❌ File not found: {filename}"
    os.remove(path)
    return f"🗑️  File deleted: {filename}"


# ── Python run  ────────────────────────────────────────

@tool
def run_python(filename: str) -> str:
    """
    Execute a Python file inside the sandbox directory.

    Args:
        filename: Name of the Python file to run
    """
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"❌ Error: {path} does not exist. Did you write the file first?"
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
            cwd=SANDBOX,
        )
        output = result.stdout or result.stderr or "(no output)"
    except subprocess.TimeoutExpired:
        output = "❌ Timeout: execution exceeded 15 seconds and was terminated."

    if len(output) > 2000:
        output = output[:2000] + "\n... (output truncated)"
    return output


# ── SQLite  ────────────────────────────────────────

_DB_PATH = os.path.join(SANDBOX, "harness.db")


@tool
def run_sql(sql: str) -> str:
    """
    Execute a SQL statement against a local SQLite database and return the result.

    The database file lives at /tmp/sandbox/harness.db and is created automatically.
    Supports SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, DROP TABLE, etc.
    Results are returned as a formatted text table.

    Args:
        sql: The SQL statement to execute (e.g. "SELECT * FROM users")
    """
    if not sql or not sql.strip():
        return "❌ Empty SQL statement."

    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)

        # Detect if this was a query (returns rows) or a write statement
        if cursor.description:
            rows = cursor.fetchall()
            if not rows:
                result = "✅ Query executed — no rows returned."
            else:
                headers = [desc[0] for desc in cursor.description]
                # Build a simple aligned table
                col_widths = [len(h) for h in headers]
                for row in rows:
                    for i, val in enumerate(row):
                        col_widths[i] = max(col_widths[i], len(str(val)))
                # Header row
                lines = []
                header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
                sep = "-+-".join("-" * w for w in col_widths)
                lines.append(header_line)
                lines.append(sep)
                for row in rows:
                    vals = [str(v) if v is not None else "NULL" for v in row]
                    lines.append(" | ".join(v.ljust(col_widths[i]) for i, v in enumerate(vals)))
                result = f"✅ Query returned {len(rows)} row(s):\n" + "\n".join(lines)
        else:
            result = f"✅ SQL executed. ({cursor.rowcount} row(s) affected)"

        conn.commit()
        conn.close()
        return result

    except sqlite3.Error as e:
        return f"❌ SQL error: {e}"



# ── Registry ──────────────────────────────────────────────
TOOLS = [list_files, read_file, get_file_info, write_file, delete_file, run_python, run_sql]
