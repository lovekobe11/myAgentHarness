"""
Config Module
=============
Load config from .env, based on PROVIDER select LangChain LLM factory function.

Supported PROVIDER:
  anthropic  → ChatAnthropic
  openai     → ChatOpenAI (api.openai.com)
  deepseek   → ChatOpenAI + base_url (api.deepseek.com)
  qwen       → ChatOpenAI + base_url (dashscope.aliyuncs.com/...)


Usage:
  from config import get_llm, MAIN_MODEL, MEMORY_MODEL, MAX_STEPS
  llm = get_llm(MAIN_MODEL)
"""

import os
from pathlib import Path

# Suppress noisy third-party warnings early (NotOpenSSLWarning, LangChain pending deprecation)
import warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass
# Suppress messages about `allowed_objects` changing in langgraph/langchain
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

# ── Load .env ──────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key not in os.environ:
                os.environ[key] = value

# ── Load Config ─────────────────────────────────────────────────────
PROVIDER: str = os.environ.get("PROVIDER").lower()

##ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_COMPATIBLE_API_KEY: str = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
OPENAI_COMPATIBLE_BASE_URL: str = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "")

MAIN_MODEL: str = os.environ.get("MAIN_MODEL")
MEMORY_MODEL: str = os.environ.get("MEMORY_MODEL")
MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "10"))

# OpenAI compatible provider base_url
SANDBOX_DIR: str = os.environ.get("SANDBOX_DIR", "/tmp/sandbox")
RUN_TIMEOUT: int = int(os.environ.get("RUN_TIMEOUT", "15"))
_DEFAULT_BASE_URLS: dict[str, str] = {
    "openai":   "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen":     "https://dashscope.aliyuncs.com/compatible-mode/v1",

}


# ── LLM factory function ───────────────────────────────────────────────────
def get_llm(model: str):
    """
    based on PROVIDER select LangChain LLM factory function.

    Args:
        model: name, from MAIN_MODEL or MEMORY_MODEL in .env

    Returns:
        LangChain BaseChatModel instance.
    """

    # Other provider use OpenAI compatible interface, require langchain-openai
    from langchain_openai import ChatOpenAI

    # base_url use .env
    base_url = (
        OPENAI_COMPATIBLE_BASE_URL
        or _DEFAULT_BASE_URLS.get(PROVIDER, "")
    )

    if not base_url:
        raise ValueError(
            f"Unknown provider '{PROVIDER}'. "
            f"Please set OPENAI_COMPATIBLE_BASE_URL in .env, "
            f"or use one of: {list(_DEFAULT_BASE_URLS.keys())}"
        )

    return ChatOpenAI(
        model=model,
        api_key=OPENAI_COMPATIBLE_API_KEY,
        base_url=base_url,
    )


# ── Validate Config ───────────────────────────────────────────────────────
def validate():

    # Check langchain-openai is installed for non-anthropic providers
    if PROVIDER != "anthropic":
        try:
            import langchain_openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "\n❌ langchain-openai is not installed.\n"
                "   Run: pip install langchain-openai\n"
            )
