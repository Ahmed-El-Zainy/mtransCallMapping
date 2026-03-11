"""
test_refiner.py
===============
Tests for app/refiner.py — covers both refine_arabic() and refine_english().

ALL tests mock the Azure OpenAI HTTP call so:
  - No real API key is needed
  - Tests run offline / in CI
  - We test our own logic (prompt building, output handling), not Azure's model

Run:
    cd transCallMapping
    pytest tests/test_refiner.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from tests.fixtures import (
    ARABIC_LONG,
    ARABIC_SHORT,
    ARABIC_WITH_TYPOS,
    CONTEXT_EMPTY,
    CONTEXT_FULL,
    CONTEXT_MINIMAL,
    ENGLISH_MIXED_ARABIC,
    ENGLISH_SHORT,
    ENGLISH_WITH_ERRORS,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

MOCK_ARABIC_RESPONSE  = "[00:00:01.200] Agent: أهلاً وسهلاً، معك خدمة عملاء العربي جروب"
MOCK_ENGLISH_RESPONSE = "[00:00:01.200] Agent: Hello, you've reached ELAraby Group customer service."


def _mock_completion(content: str) -> MagicMock:
    """Build a fake openai ChatCompletion response object."""
    msg      = MagicMock()
    msg.content = content

    choice       = MagicMock()
    choice.message = msg

    response         = MagicMock()
    response.choices = [choice]
    return response


def _patch_llm(return_value: str):
    """
    Context manager that patches the Azure OpenAI client's
    chat.completions.create method.
    """
    return patch(
        "app.refiner._get_azure_client",
        return_value=MagicMock(
            chat=MagicMock(
                completions=MagicMock(
                    create=MagicMock(return_value=_mock_completion(return_value))
                )
            )
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. refine_arabic() — happy-path tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRefineArabicHappyPath:

    def test_returns_string(self):
        """refine_arabic() must always return a str."""
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC_RESPONSE):
            result = refine_arabic(ARABIC_SHORT, CONTEXT_FULL)
        assert isinstance(result, str)

    def test_returns_llm_content_stripped(self):
        """Return value is the LLM content with leading/trailing whitespace stripped."""
        from app.refiner import refine_arabic
        padded = f"  \n{MOCK_ARABIC_RESPONSE}\n  "
        with _patch_llm(padded):
            result = refine_arabic(ARABIC_SHORT, CONTEXT_FULL)
        assert result == MOCK_ARABIC_RESPONSE

    def test_short_transcript(self):
        """Works with a minimal 5-line Arabic transcript."""
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC_RESPONSE) as mock_client:
            result = refine_arabic(ARABIC_SHORT, CONTEXT_FULL)
        assert result == MOCK_ARABIC_RESPONSE

    def test_long_transcript(self):
        """Works with a 10-line Arabic transcript."""
        from app.refiner import refine_arabic
        with _patch_llm(ARABIC_LONG):
            result = refine_arabic(ARABIC_LONG, CONTEXT_FULL)
        assert len(result.strip()) > 0

    def test_transcript_with_typos(self):
        """Passes typo-laden transcript to the LLM without raising."""
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC_RESPONSE):
            result = refine_arabic(ARABIC_WITH_TYPOS, CONTEXT_FULL)
        assert result == MOCK_ARABIC_RESPONSE

    def test_full_context_info(self):
        """Context with date, agent ID, customer ID, topic is accepted."""
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC_RESPONSE):
            result = refine_arabic(ARABIC_SHORT, CONTEXT_FULL)
        assert result == MOCK_ARABIC_RESPONSE

    def test_minimal_context_info(self):
        """Minimal context (just filename) is accepted."""
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC_RESPONSE):
            result = refine_arabic(ARABIC_SHORT, CONTEXT_MINIMAL)
        assert result == MOCK_ARABIC_RESPONSE

    def test_empty_context_info(self):
        """Empty context_info string should not crash."""
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC_RESPONSE):
            result = refine_arabic(ARABIC_SHORT, CONTEXT_EMPTY)
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. refine_english() — happy-path tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRefineEnglishHappyPath:

    def test_returns_string(self):
        """refine_english() must always return a str."""
        from app.refiner import refine_english
        with _patch_llm(MOCK_ENGLISH_RESPONSE):
            result = refine_english(ENGLISH_SHORT, CONTEXT_FULL)
        assert isinstance(result, str)

    def test_returns_llm_content_stripped(self):
        """Return value is stripped of surrounding whitespace."""
        from app.refiner import refine_english
        padded = f"\n\n  {MOCK_ENGLISH_RESPONSE}  \n"
        with _patch_llm(padded):
            result = refine_english(ENGLISH_SHORT, CONTEXT_FULL)
        assert result == MOCK_ENGLISH_RESPONSE

    def test_short_transcript_with_errors(self):
        """Handles transcript full of spelling/grammar errors."""
        from app.refiner import refine_english
        with _patch_llm(MOCK_ENGLISH_RESPONSE):
            result = refine_english(ENGLISH_WITH_ERRORS, CONTEXT_FULL)
        assert result == MOCK_ENGLISH_RESPONSE

    def test_mixed_arabic_english_transcript(self):
        """Handles a transcript that mixes Arabic and English lines."""
        from app.refiner import refine_english
        with _patch_llm(MOCK_ENGLISH_RESPONSE):
            result = refine_english(ENGLISH_MIXED_ARABIC, CONTEXT_FULL)
        assert isinstance(result, str)

    def test_minimal_context_info(self):
        """Works with only a source-file context."""
        from app.refiner import refine_english
        with _patch_llm(MOCK_ENGLISH_RESPONSE):
            result = refine_english(ENGLISH_SHORT, CONTEXT_MINIMAL)
        assert result == MOCK_ENGLISH_RESPONSE

    def test_empty_context_info(self):
        """Empty context_info is forwarded without error."""
        from app.refiner import refine_english
        with _patch_llm(MOCK_ENGLISH_RESPONSE):
            result = refine_english(ENGLISH_SHORT, CONTEXT_EMPTY)
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Prompt construction tests — verify placeholders are injected correctly
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptConstruction:

    def _capture_user_prompt(self, fn_name: str, transcript: str, context: str) -> str:
        """
        Call refine_arabic or refine_english, intercept the exact user_prompt
        that was passed to _call_llm, and return it.
        """
        captured: list[str] = []

        def fake_call_llm(system_prompt: str, user_prompt: str) -> str:
            captured.append(user_prompt)
            return "mocked"

        with patch("app.refiner._call_llm", side_effect=fake_call_llm):
            import app.refiner as refiner_mod
            getattr(refiner_mod, fn_name)(transcript, context)

        return captured[0]

    def test_arabic_prompt_contains_transcript(self):
        """Arabic user prompt must embed the original transcript verbatim."""
        prompt = self._capture_user_prompt("refine_arabic", ARABIC_SHORT, CONTEXT_FULL)
        assert ARABIC_SHORT in prompt

    def test_arabic_prompt_contains_context(self):
        """Arabic user prompt must embed the context_info block."""
        prompt = self._capture_user_prompt("refine_arabic", ARABIC_SHORT, CONTEXT_FULL)
        assert "AGT001" in prompt
        assert "CUST4892" in prompt

    def test_arabic_prompt_empty_context_no_placeholder_leak(self):
        """When context is empty the literal '{context_info}' must not appear."""
        prompt = self._capture_user_prompt("refine_arabic", ARABIC_SHORT, CONTEXT_EMPTY)
        assert "{context_info}" not in prompt

    def test_english_prompt_contains_transcript(self):
        """English user prompt must embed the original transcript verbatim."""
        prompt = self._capture_user_prompt("refine_english", ENGLISH_SHORT, CONTEXT_FULL)
        assert ENGLISH_SHORT in prompt

    def test_english_prompt_contains_context(self):
        """English user prompt must embed the context_info block."""
        prompt = self._capture_user_prompt("refine_english", ENGLISH_SHORT, CONTEXT_FULL)
        assert "Washing Machine Complaint" in prompt

    def test_english_prompt_empty_context_no_placeholder_leak(self):
        """When context is empty the literal '{context_info}' must not appear."""
        prompt = self._capture_user_prompt("refine_english", ENGLISH_SHORT, CONTEXT_EMPTY)
        assert "{original_transcription}" not in prompt

    def test_arabic_prompt_no_original_transcription_placeholder(self):
        """Ensure {original_transcription} placeholder is fully substituted."""
        prompt = self._capture_user_prompt("refine_arabic", ARABIC_SHORT, CONTEXT_FULL)
        assert "{original_transcription}" not in prompt

    def test_english_and_arabic_prompts_differ(self):
        """Arabic and English prompts must be different (different system instructions)."""
        arabic_prompt  = self._capture_user_prompt("refine_arabic",  ARABIC_SHORT, CONTEXT_FULL)
        english_prompt = self._capture_user_prompt("refine_english", ARABIC_SHORT, CONTEXT_FULL)
        assert arabic_prompt != english_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LLM call parameters — verify model settings passed to Azure
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMCallParameters:

    def _capture_create_kwargs(self, fn_name: str) -> dict:
        """Return the kwargs that were passed to chat.completions.create."""
        captured: list[dict] = []

        mock_response = _mock_completion("mocked output")
        mock_create   = MagicMock(return_value=mock_response,
                                   side_effect=lambda **kw: (captured.append(kw), mock_response)[1])

        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        with patch("app.refiner._get_azure_client", return_value=mock_client):
            import app.refiner as refiner_mod
            getattr(refiner_mod, fn_name)(ARABIC_SHORT, CONTEXT_FULL)

        return captured[0]

    def test_arabic_uses_correct_temperature(self):
        kw = self._capture_create_kwargs("refine_arabic")
        assert kw["temperature"] == pytest.approx(0.3)

    def test_english_uses_correct_temperature(self):
        kw = self._capture_create_kwargs("refine_english")
        assert kw["temperature"] == pytest.approx(0.3)

    def test_arabic_max_tokens(self):
        kw = self._capture_create_kwargs("refine_arabic")
        assert kw["max_tokens"] == 4096

    def test_english_max_tokens(self):
        kw = self._capture_create_kwargs("refine_english")
        assert kw["max_tokens"] == 4096

    def test_messages_have_system_and_user_roles(self):
        kw = self._capture_create_kwargs("refine_arabic")
        roles = [m["role"] for m in kw["messages"]]
        assert "system" in roles
        assert "user"   in roles

    def test_messages_length_is_two(self):
        """Exactly one system + one user message, no extras."""
        kw = self._capture_create_kwargs("refine_english")
        assert len(kw["messages"]) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Error handling tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRefinerErrorHandling:

    def test_arabic_propagates_api_exception(self):
        """If Azure OpenAI raises, refine_arabic must propagate it."""
        from app.refiner import refine_arabic
        import openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )

        with patch("app.refiner._get_azure_client", return_value=mock_client):
            with pytest.raises(openai.APIConnectionError):
                refine_arabic(ARABIC_SHORT, CONTEXT_FULL)

    def test_english_propagates_api_exception(self):
        """If Azure OpenAI raises, refine_english must propagate it."""
        from app.refiner import refine_english
        import openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="rate limit", response=MagicMock(), body={}
        )

        with patch("app.refiner._get_azure_client", return_value=mock_client):
            with pytest.raises(openai.RateLimitError):
                refine_english(ENGLISH_SHORT, CONTEXT_FULL)

    def test_arabic_handles_empty_transcript(self):
        """Empty transcript string should not crash — LLM call is still made."""
        from app.refiner import refine_arabic
        with _patch_llm(""):
            result = refine_arabic("", CONTEXT_FULL)
        assert isinstance(result, str)

    def test_english_handles_empty_transcript(self):
        """Empty transcript string is forwarded without error."""
        from app.refiner import refine_english
        with _patch_llm(""):
            result = refine_english("", CONTEXT_FULL)
        assert isinstance(result, str)

    def test_arabic_handles_very_long_transcript(self):
        """Transcript of 500 lines should not raise before reaching the LLM."""
        from app.refiner import refine_arabic
        big_transcript = "\n".join(
            f"[00:{i:02d}:00.000] {'Agent' if i % 2 == 0 else 'Customer'}: نص تجريبي للسطر رقم {i}"
            for i in range(500)
        )
        with _patch_llm(MOCK_ARABIC_RESPONSE):
            result = refine_arabic(big_transcript, CONTEXT_FULL)
        assert result == MOCK_ARABIC_RESPONSE


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Integration-style: both refiners called in sequence (pipeline simulation)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineSimulation:

    def test_arabic_then_english_on_same_transcript(self):
        """
        Simulates the /process pipeline: same raw transcript fed to both
        refiners. Both must succeed independently.
        """
        from app.refiner import refine_arabic, refine_english

        call_count = 0

        def fake_call_llm(system_prompt: str, user_prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if "Egyptian" in system_prompt or "مساعد" in system_prompt:
                return MOCK_ARABIC_RESPONSE
            return MOCK_ENGLISH_RESPONSE

        with patch("app.refiner._call_llm", side_effect=fake_call_llm):
            arabic  = refine_arabic(ARABIC_SHORT,  CONTEXT_FULL)
            english = refine_english(ARABIC_SHORT, CONTEXT_FULL)

        assert call_count == 2
        assert arabic  == MOCK_ARABIC_RESPONSE
        assert english == MOCK_ENGLISH_RESPONSE

    def test_results_are_independent(self):
        """Arabic and English outputs must not bleed into each other."""
        from app.refiner import refine_arabic, refine_english

        with patch("app.refiner._call_llm", side_effect=lambda sys, usr: (
            MOCK_ARABIC_RESPONSE if "مساعد" in sys else MOCK_ENGLISH_RESPONSE
        )):
            arabic  = refine_arabic(ARABIC_SHORT,  CONTEXT_FULL)
            english = refine_english(ENGLISH_SHORT, CONTEXT_FULL)

        assert arabic  != english

    def test_context_info_appears_in_both_prompts(self):
        """CONTEXT_FULL data must be injected into both Arabic and English prompts."""
        prompts: list[str] = []

        def capture(sys_p: str, usr_p: str) -> str:
            prompts.append(usr_p)
            return "mocked"

        with patch("app.refiner._call_llm", side_effect=capture):
            from app.refiner import refine_arabic, refine_english
            refine_arabic(ARABIC_SHORT,   CONTEXT_FULL)
            refine_english(ENGLISH_SHORT, CONTEXT_FULL)

        for prompt in prompts:
            assert "2024-03-15" in prompt, f"Date missing from prompt"
            assert "AGT001"     in prompt, f"Agent ID missing from prompt"
            assert "CUST4892"   in prompt, f"Customer ID missing from prompt"