"""
config.py – All settings and LLM prompts for the refiner pipeline.
Edit the prompts here freely — no code changes needed elsewhere.
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
# PROMPTS  —  edit freely, just keep the {placeholders} where they are
# ══════════════════════════════════════════════════════════════════════════════

# ── Arabic refinement ─────────────────────────────────────────────────────────
ARABIC_SYSTEM_PROMPT = """\
أنت مساعد متخصص في تحسين وتنقيح المحادثات العربية المصرية لمراكز خدمة العملاء.
"""

ARABIC_USER_PROMPT_TEMPLATE = """\
هذه المحادثة تمت بين عميل وموظف خدمة عملاء من شركة ميراكوا (Miraco Company).

مهمتك هي إنشاء نسخة محسنة ومنقحة من النسخة الأصلية للمحادثة مع الحفاظ على:
1. التسلسل الزمني الدقيق للمحادثة
2. تسميات المتحدثين (Customer: / Agent:) باللغة الإنجليزية
3. الطوابع الزمنية بنفس الصيغة [HH:MM:SS.mmm] إن وجدت
4. المعنى والسياق الأصلي للمحادثة
5. طبيعة الحوار بين العميل شركة ميراكوا

التحسينات المطلوبة:
- تصحيح الأخطاء الإملائية والنحوية
- تحسين الوضوح والطلاقة مع الحفاظ على اللهجة المصرية الطبيعية
- تنسيق أفضل للجمل والعبارات
- إزالة التكرار غير الضروري
- تحسين التعبيرات المهنية للموظف بما يليق بممثل شركة شركة ميراكوا
- الحفاظ على التعبيرات الطبيعية للعميل

{context_info}

إرشادات مهمة:
- لا تضيف معلومات جديدة لم تكن في النسخة الأصلية
- احتفظ بنفس عدد الأسطر تقريباً
- احتفظ بالطوابع الزمنية كما هي إن وجدت
- استخدم تسميات المتحدثين Customer: للعميل و Agent: للموظف (باللغة الإنجليزية)
- استخدم اللغة العربية المصرية الطبيعية والمهنية للمحتوى
- لا تضيف أي جمل تمهيدية أو تعريفية
- ابدأ مباشرة بالمحادثة المحسنة

النسخة الأصلية للمحادثة:
{original_transcription}

ابدأ مباشرة بالمحادثة المحسنة دون أي مقدمة:
"""

# ── English refinement ────────────────────────────────────────────────────────
ENGLISH_SYSTEM_PROMPT = """\
You are a professional call center conversation editor specializing in creating clean, \
professional English versions of customer service calls.
"""

ENGLISH_USER_PROMPT_TEMPLATE = """\
This conversation took place between a customer and a customer service agent \
from Miraco Company (شركة ميراكوا).

Your task is to create a revised, clean English version of the original conversation \
while maintaining:
1. The exact chronological order of the conversation
2. Speaker labels (Customer: / Agent:)
3. Timestamps in the same format [HH:MM:SS.mmm] if present
4. The original meaning and context of the conversation
5. The natural dialogue flow between customer and  Miraco Company agent

Required improvements:
- Correct all spelling and grammatical errors
- Improve clarity and fluency with professional English
- Better sentence structure and phrasing
- Remove unnecessary repetition
- Enhance professional expressions for the agent as a representative of Miraco Company
- Maintain natural customer expressions while correcting errors
- Translate Arabic content to natural English while preserving meaning

{context_info}

Important guidelines:
- Do not add new information that wasn't in the original
- Keep approximately the same number of lines
- Preserve timestamps exactly as they are if present
- Keep speaker labels (Customer: / Agent:)
- Use clear, professional English appropriate for call center context
- Do not add any introductory sentences or headers
- Start directly with the improved conversation

Original conversation:
{original_transcription}

Start directly with the improved conversation without any introduction:
"""



# # ── Call summary ──────────────────────────────────────────────────────────────
# SUMMARY_SYSTEM_PROMPT = """\
# You are an expert call center quality analyst for Miraco Company. \
# You produce concise, structured summaries of customer service calls.
# """

# SUMMARY_USER_PROMPT_TEMPLATE = """\
# Analyse the following customer service call transcript and produce a structured summary.

# {context_info}

# Return your response as valid JSON — no markdown, no backticks, just raw JSON — \
# using exactly this structure:
# {{
#   "call_topic": "one-line description of the main reason for the call",
#   "customer_request": "what the customer asked for or complained about",
#   "agent_actions": ["list", "of", "key", "actions", "taken", "by", "agent"],
#   "resolution": "how the call was resolved or what the next step is",
#   "products_mentioned": ["list", "of", "products", "or", "models", "mentioned"],
#   "follow_up_required": true or false,
#   "follow_up_notes": "any pending actions or callbacks needed, empty string if none",
#   "call_sentiment": "Positive | Neutral | Negative",
#   "language": "Arabic | English | Mixed",
#   "score": 1-10 quality score of the call handling (10 is best)"
#   "agent attatude": "Generated in English only , Only one attitude per call (Friendly | Neutral | Rude)"
#   "category": "classifiy based on those (Inquiry , Complaint , Technical Support, Billing , Sales, Feedback)"

# }}

# Transcript:
# {original_transcription}

# Return only the JSON object:
# """



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
  call_score = round( (number of Pass results / 7) * 100 )
  Example: 5 Pass out of 7 = round(5/7*100) = 71
 
Transcript:
{original_transcription}
 
Return only the JSON object:
"""
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Supported audio extensions (kept here for any audio-handling utilities)
# ══════════════════════════════════════════════════════════════════════════════
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
