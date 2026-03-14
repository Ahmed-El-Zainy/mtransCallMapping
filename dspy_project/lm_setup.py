"""
lm_setup.py — Configure DSPy to use Azure OpenAI.

IMPORTANT: dspy.configure() must be called exactly ONCE from the main thread
before any async tasks start. Calling it from async tasks raises RuntimeError.
The _configured guard ensures idempotency.
"""

import dspy
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION

_configured = False
_lm: dspy.LM | None = None


def configure_dspy() -> dspy.LM:
    """
    Configure DSPy global LM. Safe to call multiple times — only runs once.
    Call this at module import time (not inside async functions).
    """
    global _configured, _lm

    if _configured and _lm is not None:
        return _lm  # already done — return cached LM, never call dspy.configure() again

    _lm = dspy.LM(
        model       = "azure/gpt-4o-mini",
        api_key     = AZURE_OPENAI_API_KEY,
        api_base    = AZURE_OPENAI_ENDPOINT,
        api_version = AZURE_OPENAI_API_VERSION,
        cache       = False,
    )
    dspy.configure(lm=_lm)   # called ONCE, from module-load / main thread
    _configured = True
    return _lm


def get_lm() -> dspy.LM:
    """Return the configured LM (configures if not yet done)."""
    return configure_dspy()


# Configure immediately on import so it always runs from the main thread
configure_dspy()
