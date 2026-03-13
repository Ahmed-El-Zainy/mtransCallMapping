# dspy_module/lm_setup.py
import dspy
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
def configure_dspy() -> dspy.LM:
    lm = dspy.LM(
        model       = "azure/gpt-4o-mini",   # LiteLLM Azure format
        api_key     = AZURE_OPENAI_API_KEY,
        api_base    = AZURE_OPENAI_ENDPOINT,
        api_version = AZURE_OPENAI_API_VERSION,
    )
    dspy.configure(lm=lm)   # sets global LM for all DSPy modules
    return lm

if __name__ == "__main__":
    configure_dspy()