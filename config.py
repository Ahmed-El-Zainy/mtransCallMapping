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
5. طبيعة الحوار بين العميل وموظف العربي جروب

التحسينات المطلوبة:
- تصحيح الأخطاء الإملائية والنحوية
- تحسين الوضوح والطلاقة مع الحفاظ على اللهجة المصرية الطبيعية
- تنسيق أفضل للجمل والعبارات
- إزالة التكرار غير الضروري
- تحسين التعبيرات المهنية للموظف بما يليق بممثل شركة العربي جروب
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
5. The natural dialogue flow between customer and ELAraby Group agent

Required improvements:
- Correct all spelling and grammatical errors
- Improve clarity and fluency with professional English
- Better sentence structure and phrasing
- Remove unnecessary repetition
- Enhance professional expressions for the agent as a representative of ELAraby Group
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

# ── Call summary ──────────────────────────────────────────────────────────────
SUMMARY_SYSTEM_PROMPT = """\
You are an expert call center quality analyst for ELAraby Group. \
You produce concise, structured summaries of customer service calls.
"""

SUMMARY_USER_PROMPT_TEMPLATE = """\
Analyse the following customer service call transcript and produce a structured summary.

{context_info}

Return your response as valid JSON — no markdown, no backticks, just raw JSON — \
using exactly this structure:
{{
  "call_topic": "one-line description of the main reason for the call",
  "customer_request": "what the customer asked for or complained about",
  "agent_actions": ["list", "of", "key", "actions", "taken", "by", "agent"],
  "resolution": "how the call was resolved or what the next step is",
  "products_mentioned": ["list", "of", "products", "or", "models", "mentioned"],
  "follow_up_required": true or false,
  "follow_up_notes": "any pending actions or callbacks needed, empty string if none",
  "call_sentiment": "Positive | Neutral | Negative",
  "language": "Arabic | English | Mixed"
}}

Transcript:
{original_transcription}

Return only the JSON object:
"""