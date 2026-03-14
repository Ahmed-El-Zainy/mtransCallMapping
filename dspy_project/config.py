"""
config.py – Settings and prompts for the transCallMapping refiner.

Your role in the system:
  RECEIVE  raw transcript  (from upstream STT API)
  PROCESS  3 LLM calls     (arabic refine → english refine → analysis)
  RETURN   enriched JSON   (to downstream post-call details API)

Output contract (matches Scenario 1 user story):
  transcript_arabic       refined Arabic transcript  (Speaker: text per line)
  transcript_english      refined English transcript (Speaker: text per line)
  main_subject            one-line call subject
  call_outcome            resolved / unresolved / escalated / follow-up-needed
  issue_resolution        paragraph explaining how the issue was handled
  call_summary            2-3 sentence human-readable summary
  keywords                up to 5 English keywords
  call_category           Inquiry | Complaint | Technical Support |
                          Billing | Sales | Feedback
  service                 single service name mentioned in the call
  agent_attitude          Friendly | Neutral | Rude
  customer_satisfaction   Satisfied | Neutral | Dissatisfied
  call_score              0-10 numeric quality score
  score_breakdown         per-question pass/fail for the score framework
  language                Arabic | English | Mixed
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
# Keeps the call in its original language, cleans and formats it.
# ══════════════════════════════════════════════════════════════════════════════

ARABIC_SYSTEM_PROMPT = """\
أنت محرر متخصص في تنقيح محادثات مراكز خدمة العملاء المصرية.
قواعدك الثابتة:
- كل سطر يبدأ بـ  Agent:  أو  Customer:  ثم النص
- الطوابع الزمنية تبقى كما هي بالضبط إن وجدت
- اللهجة مصرية عامية مهنية — ليس فصحى أبداً
- لا تضيف أي معلومة لم تكن في النص الأصلي
- لا تضيف مقدمات أو تعليقات — ابدأ مباشرة بالمحادثة
"""

ARABIC_USER_PROMPT_TEMPLATE = """\
هذه محادثة بين عميل وموظف خدمة عملاء من شركة شركة ميراكوا.

── مثال توضيحي ──
الأصلي:
  ألو معك خدمه عملاء العربى مساء الخير
  ايه ده التلاجه بتاعتى مش شغاله

المحسّن:
  Agent: أهلاً وسهلاً، معك خدمة عملاء شركة ميراكوا، كيف أقدر أساعدك؟
  Customer: عندي مشكلة في التلاجة، مش شغالة خالص.
── نهاية المثال ──

{context_info}

قواعد التحويل:
- كل سطر:  [طابع زمني إن وجد]  Agent: أو Customer: ثم النص المصحح
- صحح الإملاء والنحو مع الحفاظ على المعنى الأصلي
- أزل التكرار غير الضروري
- لا تضيف سطوراً جديدة لم تكن في الأصل

المحادثة الأصلية:
{original_transcription}

المحادثة المحسّنة (ابدأ مباشرة):
"""


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT 2 — English transcript refinement
# Translates + cleans into professional call-center English.
# ══════════════════════════════════════════════════════════════════════════════

ENGLISH_SYSTEM_PROMPT = """\
You are a professional call center transcript editor for Miraco Company.
Fixed rules:
- Every line starts with  Agent:  or  Customer:  followed by the text
- Preserve timestamps exactly if present
- Translate Arabic naturally — never word-for-word
- Do not add information not in the original
- No introductions or comments — start directly with line 1
"""

ENGLISH_USER_PROMPT_TEMPLATE = """\
Refine this Miraco Company customer service call into professional English.

── Example ──
Original:  ألو معك خدمه عملاء العربى
Refined:   Agent: Hello, you've reached Miraco Company customer service. How may I help you?

Original:  Customer: ايه ده التلاجه بتاعتى مش شغاله
Refined:   Customer: My refrigerator isn't working at all.

❌ "Hello, this is the service of the Arabic group"  (literal — wrong)
✅ "Hello, you've reached Miraco Company customer service"  (natural — correct)
── End example ──

{context_info}

Rules:
- Every line: Agent: text  OR  Customer: text
- Same number of lines as original
- Professional but conversational tone
- Product names preserved exactly: Carrier, Midea, Eco Master, Inverter, etc.

Original transcript:
{original_transcription}

Professional English version (start directly):
"""


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT 3 — Full call analysis
# Single LLM call that returns ALL analysis fields as one JSON object.
# Maps 1-to-1 with the user story Scenario 1 fields.
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

  "call_score": <integer 0-10 based on the evaluation below>,

  "score_breakdown": {{
    "greeted_professionally":     {{"result": "Pass or Fail", "note": "brief reason"}},
    "identified_customer_need":   {{"result": "Pass or Fail", "note": "brief reason"}},
    "provided_accurate_info":     {{"result": "Pass or Fail", "note": "brief reason"}},
    "maintained_professional_tone":{{"result": "Pass or Fail", "note": "brief reason"}},
    "offered_complete_solution":  {{"result": "Pass or Fail", "note": "brief reason"}},
    "confirmed_resolution":       {{"result": "Pass or Fail", "note": "brief reason"}},
    "proper_closing":             {{"result": "Pass or Fail", "note": "brief reason"}}
  }}
}}

Score calculation rule:
  call_score = round( (number of Pass results / 7) * 10 )
  Example: 5 Pass out of 7 = round(5/7*10) = 7

Transcript:
{original_transcription}

Return only the JSON object:
"""


# ══════════════════════════════════════════════════════════════════════════════
# Supported audio extensions (kept here for any audio-handling utilities)
# ══════════════════════════════════════════════════════════════════════════════
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
