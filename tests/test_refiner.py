"""
test_refiner.py
===============
Tests for app/refiner.py — refine_arabic, refine_english, summarise, run_pipeline.

All tests mock the Azure OpenAI HTTP call:
  - No real API key needed
  - Runs fully offline / in CI
  - Tests our logic: prompt building, output parsing, pipeline orchestration

Run:
    cd clean_refiner
    pytest tests/test_refiner.py -v
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch, call

from tests.fixtures import (
    ARABIC_SHORT,
    CONTEXT_EMPTY,
    CONTEXT_FULL,
    CONTEXT_MINIMAL,
    EMPTY_TRANSCRIPT,
    ENGLISH_WITH_ERRORS,
    MIXED_TRANSCRIPT,
    REAL_TRANSCRIPT,
    SINGLE_LINE,
    VERY_LONG_TRANSCRIPT,
    WHITESPACE_ONLY,
)

# ── Test constants ────────────────────────────────────────────────────────────

MOCK_ARABIC  = "[00:00:01.200] Agent: أهلاً وسهلاً، معك خدمة عملاء شركة ميراكوا"
MOCK_ENGLISH = "[00:00:01.200] Agent: Hello, you've reached Miraco Company customer service."
MOCK_SUMMARY = {
    "call_topic":          "AC unit price inquiry",
    "customer_request":    "Customer asked about the price of a 2.25 ton inverter AC",
    "agent_actions":       ["Provided Carrier price", "Provided Midea price", "Explained warranty"],
    "resolution":          "Prices provided, customer advised to visit branch or order online",
    "products_mentioned":  ["Carrier X-Cool Inverter", "Midea ECO Master"],
    "follow_up_required":  False,
    "follow_up_notes":     "",
    "call_sentiment":      "Positive",
    "language":            "Arabic",
}

# ── Patch helpers ─────────────────────────────────────────────────────────────

def _mock_response(content: str) -> MagicMock:
    msg          = MagicMock(); msg.content = content
    choice       = MagicMock(); choice.message = msg
    response     = MagicMock(); response.choices = [choice]
    return response


def _patch_llm(return_value: str):
    """Patch the Azure client so chat.completions.create returns return_value."""
    return patch(
        "app.refiner._get_client",
        return_value=MagicMock(
            chat=MagicMock(
                completions=MagicMock(
                    create=MagicMock(return_value=_mock_response(return_value))
                )
            )
        ),
    )


def _capture_user_prompt(fn_name: str, transcript: str, context: str) -> str:
    """Call a refiner function and return the user_prompt that was built."""
    captured: list[str] = []
    def fake_call_llm(sys_p: str, usr_p: str, temperature: float = 0.3) -> str:
        captured.append(usr_p)
        return "mocked"
    with patch("app.refiner._call_llm", side_effect=fake_call_llm):
        import app.refiner as m
        getattr(m, fn_name)(transcript, context)
    return captured[0]


def _capture_all_prompts(transcript: str, context: str) -> list[str]:
    """Run run_pipeline and capture all 3 user prompts built."""
    captured: list[str] = []
    def fake_call_llm(sys_p: str, usr_p: str, temperature: float = 0.3) -> str:
        captured.append(usr_p)
        if "JSON" in sys_p or "json" in usr_p.lower():
            return json.dumps(MOCK_SUMMARY)
        return "mocked"
    with patch("app.refiner._call_llm", side_effect=fake_call_llm):
        from app.refiner import run_pipeline
        run_pipeline(transcript, context)
    return captured


# ══════════════════════════════════════════════════════════════════════════════
# 1. refine_arabic
# ══════════════════════════════════════════════════════════════════════════════

class TestRefineArabic:

    def test_returns_string(self):
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC):
            assert isinstance(refine_arabic(ARABIC_SHORT, CONTEXT_FULL), str)

    def test_strips_whitespace(self):
        from app.refiner import refine_arabic
        with _patch_llm(f"  \n{MOCK_ARABIC}\n  "):
            assert refine_arabic(ARABIC_SHORT, CONTEXT_FULL) == MOCK_ARABIC

    def test_real_transcript(self):
        """Works on the actual Miraco call transcript."""
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC):
            result = refine_arabic(REAL_TRANSCRIPT, CONTEXT_FULL)
        assert result == MOCK_ARABIC

    def test_no_timestamps_transcript(self):
        """Transcript without timestamps (plain text) is accepted."""
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC):
            result = refine_arabic(REAL_TRANSCRIPT, CONTEXT_EMPTY)
        assert isinstance(result, str)

    def test_empty_context(self):
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC):
            assert isinstance(refine_arabic(ARABIC_SHORT, CONTEXT_EMPTY), str)

    def test_minimal_context(self):
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC):
            assert refine_arabic(ARABIC_SHORT, CONTEXT_MINIMAL) == MOCK_ARABIC

    def test_very_long_transcript(self):
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC):
            assert refine_arabic(VERY_LONG_TRANSCRIPT, CONTEXT_FULL) == MOCK_ARABIC

    def test_single_line_transcript(self):
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC):
            assert isinstance(refine_arabic(SINGLE_LINE, CONTEXT_EMPTY), str)

    def test_empty_transcript_still_calls_llm(self):
        """Even an empty transcript is forwarded — the LLM decides what to do."""
        from app.refiner import refine_arabic
        with _patch_llm("") as mock_client:
            result = refine_arabic(EMPTY_TRANSCRIPT, CONTEXT_FULL)
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# 2. refine_english
# ══════════════════════════════════════════════════════════════════════════════

class TestRefineEnglish:

    def test_returns_string(self):
        from app.refiner import refine_english
        with _patch_llm(MOCK_ENGLISH):
            assert isinstance(refine_english(ENGLISH_WITH_ERRORS, CONTEXT_FULL), str)

    def test_strips_whitespace(self):
        from app.refiner import refine_english
        with _patch_llm(f"\n\n{MOCK_ENGLISH}  "):
            assert refine_english(ENGLISH_WITH_ERRORS, CONTEXT_FULL) == MOCK_ENGLISH

    def test_real_transcript(self):
        """Works on the actual Miraco call (Arabic → English translation)."""
        from app.refiner import refine_english
        with _patch_llm(MOCK_ENGLISH):
            assert isinstance(refine_english(REAL_TRANSCRIPT, CONTEXT_FULL), str)

    def test_mixed_language_transcript(self):
        from app.refiner import refine_english
        with _patch_llm(MOCK_ENGLISH):
            assert isinstance(refine_english(MIXED_TRANSCRIPT, CONTEXT_FULL), str)

    def test_empty_context(self):
        from app.refiner import refine_english
        with _patch_llm(MOCK_ENGLISH):
            assert isinstance(refine_english(ENGLISH_WITH_ERRORS, CONTEXT_EMPTY), str)

    def test_very_long_transcript(self):
        from app.refiner import refine_english
        with _patch_llm(MOCK_ENGLISH):
            assert refine_english(VERY_LONG_TRANSCRIPT, CONTEXT_FULL) == MOCK_ENGLISH


# ══════════════════════════════════════════════════════════════════════════════
# 3. summarise
# ══════════════════════════════════════════════════════════════════════════════

class TestSummarise:

    def test_returns_dict(self):
        from app.refiner import summarise
        with _patch_llm(json.dumps(MOCK_SUMMARY)):
            result = summarise(REAL_TRANSCRIPT, CONTEXT_FULL)
        assert isinstance(result, dict)

    def test_all_expected_keys_present(self):
        from app.refiner import summarise
        with _patch_llm(json.dumps(MOCK_SUMMARY)):
            result = summarise(REAL_TRANSCRIPT, CONTEXT_FULL)
        expected_keys = {
            "call_topic", "customer_request", "agent_actions", "resolution",
            "products_mentioned", "follow_up_required", "follow_up_notes",
            "call_sentiment", "language",
        }
        assert expected_keys.issubset(result.keys())

    def test_strips_markdown_fences(self):
        """Model wraps JSON in ```json ... ``` — must be stripped cleanly."""
        from app.refiner import summarise
        fenced = f"```json\n{json.dumps(MOCK_SUMMARY)}\n```"
        with _patch_llm(fenced):
            result = summarise(REAL_TRANSCRIPT, CONTEXT_FULL)
        assert result["call_topic"] == MOCK_SUMMARY["call_topic"]

    def test_invalid_json_returns_raw_summary_key(self):
        """If LLM returns non-JSON, result dict must contain 'raw_summary'."""
        from app.refiner import summarise
        with _patch_llm("Here is the summary: the customer asked about AC prices."):
            result = summarise(REAL_TRANSCRIPT, CONTEXT_FULL)
        assert "raw_summary" in result
        assert "parse_error" in result

    def test_agent_actions_is_list(self):
        from app.refiner import summarise
        with _patch_llm(json.dumps(MOCK_SUMMARY)):
            result = summarise(REAL_TRANSCRIPT, CONTEXT_FULL)
        assert isinstance(result["agent_actions"], list)

    def test_products_mentioned_is_list(self):
        from app.refiner import summarise
        with _patch_llm(json.dumps(MOCK_SUMMARY)):
            result = summarise(REAL_TRANSCRIPT, CONTEXT_FULL)
        assert isinstance(result["products_mentioned"], list)

    def test_follow_up_required_is_bool(self):
        from app.refiner import summarise
        with _patch_llm(json.dumps(MOCK_SUMMARY)):
            result = summarise(REAL_TRANSCRIPT, CONTEXT_FULL)
        assert isinstance(result["follow_up_required"], bool)

    def test_real_transcript_empty_context(self):
        from app.refiner import summarise
        with _patch_llm(json.dumps(MOCK_SUMMARY)):
            result = summarise(REAL_TRANSCRIPT, CONTEXT_EMPTY)
        assert isinstance(result, dict)

    def test_uses_low_temperature(self):
        """Summary should be called with temperature=0.1 for determinism."""
        captured_temps: list[float] = []
        def fake_call_llm(sys_p, usr_p, temperature=0.3):
            captured_temps.append(temperature)
            return json.dumps(MOCK_SUMMARY)
        with patch("app.refiner._call_llm", side_effect=fake_call_llm):
            from app.refiner import summarise
            summarise(REAL_TRANSCRIPT, CONTEXT_FULL)
        assert captured_temps[0] == pytest.approx(0.1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Prompt construction
# ══════════════════════════════════════════════════════════════════════════════

class TestPromptConstruction:

    def test_arabic_prompt_embeds_transcript(self):
        prompt = _capture_user_prompt("refine_arabic", ARABIC_SHORT, CONTEXT_FULL)
        assert ARABIC_SHORT in prompt

    def test_arabic_prompt_embeds_context(self):
        prompt = _capture_user_prompt("refine_arabic", ARABIC_SHORT, CONTEXT_FULL)
        assert "AGT001"   in prompt
        assert "CUST4892" in prompt

    def test_arabic_prompt_no_stray_placeholder(self):
        prompt = _capture_user_prompt("refine_arabic", ARABIC_SHORT, CONTEXT_EMPTY)
        assert "{context_info}"           not in prompt
        assert "{original_transcription}" not in prompt

    def test_english_prompt_embeds_transcript(self):
        prompt = _capture_user_prompt("refine_english", ENGLISH_WITH_ERRORS, CONTEXT_FULL)
        assert ENGLISH_WITH_ERRORS in prompt

    def test_english_prompt_embeds_context(self):
        prompt = _capture_user_prompt("refine_english", ENGLISH_WITH_ERRORS, CONTEXT_FULL)
        assert "AC Unit Price Inquiry" in prompt

    def test_english_prompt_no_stray_placeholder(self):
        prompt = _capture_user_prompt("refine_english", ENGLISH_WITH_ERRORS, CONTEXT_EMPTY)
        assert "{context_info}"           not in prompt
        assert "{original_transcription}" not in prompt

    def test_summary_prompt_embeds_transcript(self):
        prompt = _capture_user_prompt("summarise", REAL_TRANSCRIPT, CONTEXT_FULL)
        assert REAL_TRANSCRIPT in prompt

    def test_summary_prompt_requests_json(self):
        prompt = _capture_user_prompt("summarise", ARABIC_SHORT, CONTEXT_EMPTY)
        assert "JSON" in prompt or "json" in prompt.lower()

    def test_all_three_prompts_are_different(self):
        prompts = _capture_all_prompts(ARABIC_SHORT, CONTEXT_FULL)
        assert len(prompts) == 3
        assert prompts[0] != prompts[1]
        assert prompts[1] != prompts[2]
        assert prompts[0] != prompts[2]

    def test_real_transcript_in_summary_prompt(self):
        prompt = _capture_user_prompt("summarise", REAL_TRANSCRIPT, CONTEXT_FULL)
        assert "كاريار" in prompt or "كاريير" in prompt or "mirako" in prompt.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 5. LLM call parameters
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMCallParameters:

    def _get_create_kwargs(self, fn_name: str, transcript: str = ARABIC_SHORT,
                           ret: str = "mocked") -> dict:
        captured: list[dict] = []
        mock_resp = _mock_response(ret)
        def fake_create(**kw):
            captured.append(kw)
            return mock_resp
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create
        with patch("app.refiner._get_client", return_value=mock_client):
            import app.refiner as m
            getattr(m, fn_name)(transcript, CONTEXT_FULL)
        return captured[0]

    def test_arabic_temperature(self):
        kw = self._get_create_kwargs("refine_arabic")
        assert kw["temperature"] == pytest.approx(0.3)

    def test_english_temperature(self):
        kw = self._get_create_kwargs("refine_english")
        assert kw["temperature"] == pytest.approx(0.3)

    def test_summary_temperature(self):
        kw = self._get_create_kwargs("summarise", ret=json.dumps(MOCK_SUMMARY))
        assert kw["temperature"] == pytest.approx(0.1)

    def test_max_tokens_4096(self):
        for fn in ("refine_arabic", "refine_english"):
            kw = self._get_create_kwargs(fn)
            assert kw["max_tokens"] == 4096

    def test_messages_has_system_and_user(self):
        kw = self._get_create_kwargs("refine_arabic")
        roles = [m["role"] for m in kw["messages"]]
        assert "system" in roles and "user" in roles

    def test_exactly_two_messages(self):
        kw = self._get_create_kwargs("refine_english")
        assert len(kw["messages"]) == 2


# ══════════════════════════════════════════════════════════════════════════════
# 6. Error handling
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_arabic_propagates_api_error(self):
        import openai
        from app.refiner import refine_arabic
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )
        with patch("app.refiner._get_client", return_value=mock_client):
            with pytest.raises(openai.APIConnectionError):
                refine_arabic(ARABIC_SHORT, CONTEXT_FULL)

    def test_english_propagates_rate_limit(self):
        import openai
        from app.refiner import refine_english
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="rate limit", response=MagicMock(), body={}
        )
        with patch("app.refiner._get_client", return_value=mock_client):
            with pytest.raises(openai.RateLimitError):
                refine_english(ENGLISH_WITH_ERRORS, CONTEXT_FULL)

    def test_summarise_bad_json_does_not_raise(self):
        """summarise() must never raise on bad JSON — it wraps it gracefully."""
        from app.refiner import summarise
        with _patch_llm("This is not JSON at all, just plain text."):
            result = summarise(ARABIC_SHORT, CONTEXT_FULL)
        assert isinstance(result, dict)
        assert "raw_summary" in result

    def test_whitespace_transcript_still_forwarded(self):
        """Whitespace-only transcript passes through to LLM without error."""
        from app.refiner import refine_arabic
        with _patch_llm(MOCK_ARABIC):
            result = refine_arabic(WHITESPACE_ONLY, CONTEXT_EMPTY)
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# 7. run_pipeline — full orchestration
# ══════════════════════════════════════════════════════════════════════════════

class TestRunPipeline:

    def _run(self, transcript: str = ARABIC_SHORT, context: str = CONTEXT_FULL) -> dict:
        call_count = 0
        def fake_call_llm(sys_p: str, usr_p: str, temperature: float = 0.3) -> str:
            nonlocal call_count
            call_count += 1
            if temperature == 0.1:                      # summary call
                return json.dumps(MOCK_SUMMARY)
            if "مساعد" in sys_p or "Egyptian" in sys_p: # arabic call
                return MOCK_ARABIC
            return MOCK_ENGLISH                          # english call
        with patch("app.refiner._call_llm", side_effect=fake_call_llm):
            from app.refiner import run_pipeline
            result = run_pipeline(transcript, context)
        result["_call_count"] = call_count
        return result

    def test_returns_dict(self):
        assert isinstance(self._run(), dict)

    def test_makes_exactly_three_llm_calls(self):
        assert self._run()["_call_count"] == 3

    def test_result_has_all_keys(self):
        result = self._run()
        for key in ("original_transcription", "context_info",
                    "arabic_refined", "english_refined", "summary"):
            assert key in result, f"Missing key: {key}"

    def test_original_transcript_preserved(self):
        result = self._run(ARABIC_SHORT, CONTEXT_FULL)
        assert result["original_transcription"] == ARABIC_SHORT

    def test_context_info_preserved(self):
        result = self._run(ARABIC_SHORT, CONTEXT_FULL)
        assert result["context_info"] == CONTEXT_FULL

    def test_arabic_and_english_differ(self):
        result = self._run()
        assert result["arabic_refined"] != result["english_refined"]

    def test_summary_is_dict(self):
        assert isinstance(self._run()["summary"], dict)

    def test_summary_has_call_topic(self):
        assert "call_topic" in self._run()["summary"]

    def test_real_transcript_full_pipeline(self):
        """End-to-end pipeline on the real Miraco call transcript."""
        result = self._run(REAL_TRANSCRIPT, CONTEXT_FULL)
        assert result["original_transcription"] == REAL_TRANSCRIPT
        assert isinstance(result["summary"], dict)
        assert isinstance(result["arabic_refined"], str)
        assert isinstance(result["english_refined"], str)

    def test_empty_context_allowed(self):
        result = self._run(ARABIC_SHORT, CONTEXT_EMPTY)
        assert result["context_info"] == CONTEXT_EMPTY

    def test_pipeline_order_arabic_english_summary(self):
        """Verify call order: arabic → english → summary."""
        order: list[str] = []
        def fake_call_llm(sys_p: str, usr_p: str, temperature: float = 0.3) -> str:
            if temperature == 0.1:
                order.append("summary"); return json.dumps(MOCK_SUMMARY)
            if "مساعد" in sys_p or "Egyptian" in sys_p:
                order.append("arabic");  return MOCK_ARABIC
            order.append("english"); return MOCK_ENGLISH
        with patch("app.refiner._call_llm", side_effect=fake_call_llm):
            from app.refiner import run_pipeline
            run_pipeline(ARABIC_SHORT, CONTEXT_FULL)
        assert order == ["arabic", "english", "summary"]