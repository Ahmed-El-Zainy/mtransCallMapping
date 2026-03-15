"""
config.py – Settings and prompts for the transCallMapping refiner.

Your role in the system:
  RECEIVE  raw transcript  (from upstream STT API)
  PROCESS  3 LLM calls     (arabic refine → english refine → analysis)
  RETURN   enriched JSON   (to downstream post-call details API)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY",     "")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT",    "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT",  "gpt-4o-mini")


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT 1 — Arabic transcript refinement
# ══════════════════════════════════════════════════════════════════════════════

ARABIC_SYSTEM_PROMPT = """\
أنت محرر متخصص في تنقيح محادثات مراكز خدمة العملاء المصرية لشركة ميراكوا.
قواعدك الثابتة:
- كل سطر يبدأ بـ  Agent:  أو  Customer:  ثم النص مباشرة
- لا تضيف طوابع زمنية أبداً
- اللهجة مصرية عامية مهنية — ليس فصحى أبداً
- لا تضيف أي معلومة لم تكن في النص الأصلي
- لا تضيف مقدمات أو تعليقات — ابدأ مباشرة بالمحادثة
"""

ARABIC_USER_PROMPT_TEMPLATE = """\
هذه محادثة بين عميل وموظف خدمة عملاء من شركة ميراكوا.

── مثال توضيحي ──
الأصلي:
  ألو معك خدمه عملاء ميراكوا مساء الخير
  ايه ده التلاجه بتاعتى مش شغاله

المحسّن:
  Agent: أهلاً وسهلاً، معك خدمة عملاء شركة ميراكوا، كيف أقدر أساعدك؟
  Customer: عندي مشكلة في التلاجة، مش شغالة خالص.
── نهاية المثال ──

{context_info}

قواعد التحويل:
- كل سطر:  Agent: ثم النص  أو  Customer: ثم النص — بدون أي طوابع زمنية
- صحح الإملاء والنحو مع الحفاظ على المعنى الأصلي
- أزل التكرار غير الضروري
- لا تضيف سطوراً جديدة لم تكن في الأصل

المحادثة الأصلية:
{original_transcription}

المحادثة المحسّنة (ابدأ مباشرة بـ Agent: أو Customer:):
"""


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT 2 — English transcript refinement
# ══════════════════════════════════════════════════════════════════════════════

ENGLISH_SYSTEM_PROMPT = """\
You are a professional call center transcript editor for Miraco Company.
Fixed rules:
- Every line starts with exactly  Agent:  or  Customer:  followed by the text
- Never add timestamps
- Translate Arabic naturally — never word-for-word
- Do not add information not in the original
- No introductions or comments — start directly with line 1
"""

ENGLISH_USER_PROMPT_TEMPLATE = """\
Refine this Miraco Company customer service call into professional English.

── Example ──
Original:  ألو معك خدمه عملاء ميراكوا
Refined:   Agent: Hello, you've reached Miraco Company customer service. How may I help you?

Original:  Customer: ايه ده التلاجه بتاعتى مش شغاله
Refined:   Customer: My refrigerator isn't working at all.
── End example ──

{context_info}

Rules:
- Every line: Agent: text  OR  Customer: text — no timestamps
- Same number of lines as original
- Professional but conversational tone
- Product names preserved exactly: Carrier, Midea, Eco Master, Inverter, etc.

Original transcript:
{original_transcription}

Professional English version (start directly with Agent: or Customer:):
"""


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT 3 — Full call analysis
# ══════════════════════════════════════════════════════════════════════════════

ANALYSIS_SYSTEM_PROMPT = """\
You are an expert call center quality analyst for Miraco Company.
You analyse customer service call transcripts and return structured JSON.
You output ONLY valid JSON — no markdown, no backticks, no explanation.
"""

ANALYSIS_USER_PROMPT_TEMPLATE = """\
Analyse the following call transcript and return a single JSON object.

{context_info}

Return this exact JSON structure (no markdown, no backticks, raw JSON only):
{{
  "main_subject": "one sentence — the main reason for this call",

  "call_outcome": "one of: Resolved | Unresolved | Escalated | Follow-up Needed",

  "issue_resolution": "1-2 sentences explaining how the issue was handled or why it was not resolved",

  "call_summary": "2-3 sentence plain-language summary of the full call",

  "keywords": ["up to 5 English keywords", "describing the main topics"],

  "call_category": "exactly one of: Inquiry | Complaint | Technical Support | Billing | Sales | Feedback",

  "service": "the single main service or product discussed (e.g. AC Installation, Refrigerator Repair)",

  "agent_attitude": "exactly one of: Friendly | Neutral | Rude",

  "customer_satisfaction": "exactly one of: Satisfied | Neutral | Dissatisfied",

  "language": "exactly one of: Arabic | English | Mixed",

  "call_score": <integer 0-10 based on the 10 evaluation questions below>,

  "score_breakdown": {{
    "greeted_professionally":       {{"result": "Pass or Fail", "note": "brief reason"}},
    "understood_customer_need":     {{"result": "Pass or Fail", "note": "brief reason"}},
    "answered_all_questions":       {{"result": "Pass or Fail", "note": "list any unanswered questions or say none"}},
    "provided_accurate_info":       {{"result": "Pass or Fail", "note": "brief reason"}},
    "satisfied_customer_need":      {{"result": "Pass or Fail", "note": "brief reason"}},
    "maintained_professional_tone": {{"result": "Pass or Fail", "note": "brief reason"}},
    "showed_empathy":               {{"result": "Pass or Fail", "note": "brief reason"}},
    "demonstrated_product_knowledge":{{"result": "Pass or Fail", "note": "brief reason"}},
    "offered_alternatives_if_needed":{{"result": "Pass or Fail", "note": "N/A if no alternatives were needed"}},
    "proper_closing":               {{"result": "Pass or Fail", "note": "brief reason"}}
  }}
}}

Score calculation rule:
  call_score = round( (number of Pass results / 10) * 10 )
  Example: 8 Pass out of 10 = round(8/10*10) = 8

Transcript:
{original_transcription}

Return only the JSON object:
"""

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
