"""
refiner.py – Sends raw transcription to Azure OpenAI for Arabic + English refinement.
"""

from __future__ import annotations

import openai

from config import (
    ARABIC_SYSTEM_PROMPT,
    ARABIC_USER_PROMPT_TEMPLATE,
    AZURE_OPENAI_API,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    ENGLISH_SYSTEM_PROMPT,
    ENGLISH_USER_PROMPT_TEMPLATE,
)


def _get_azure_client() -> openai.AzureOpenAI:
    return openai.AzureOpenAI(
        api_key=AZURE_OPENAI_API,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    client = _get_azure_client()
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


def refine_arabic(original_transcription: str, context_info: str) -> str:
    """Return an improved Arabic (Egyptian dialect) version of the transcript."""
    user_prompt = ARABIC_USER_PROMPT_TEMPLATE.format(
        context_info=context_info,
        original_transcription=original_transcription,
    )
    return _call_llm(ARABIC_SYSTEM_PROMPT, user_prompt)


def refine_english(original_transcription: str, context_info: str) -> str:
    """Return an improved English version of the transcript."""
    user_prompt = ENGLISH_USER_PROMPT_TEMPLATE.format(
        context_info=context_info,
        original_transcription=original_transcription,
    )
    return _call_llm(ENGLISH_SYSTEM_PROMPT, user_prompt)



if __name__ == "__main__":
    # Quick test
    import sys

    if len(sys.argv) != 2:
        print("Usage: python refiner.py <transcript.txt>")
        sys.exit(1)

    transcript = Path(sys.argv[1]).read_text(encoding="utf-8")
    context = "Test call between customer and agent at ELAraby Group."

    print("=== Arabic Refinement ===")
    print(refine_arabic(transcript, context))
    print("\n=== English Refinement ===")
    print(refine_english(transcript, context))