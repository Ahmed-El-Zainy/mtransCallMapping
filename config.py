import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_OPENAI_KEY= os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

# ── OpenAI Whisper ────────────────────────────────────────────────────────────
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── App ───────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}


# ── Filename → context_info parser ───────────────────────────────────────────
# Expected filename patterns (flexible):
#   <date>_<agent_id>_<customer_id>_<topic>.<ext>
#   <call_id>_<agent>_<topic>.<ext>
#   <any readable tokens separated by _ or ->
#
# The parser extracts whatever it can and builds a human-readable context string.


def extract_context_from_filename(filename: str) -> str:
    """
    Parse an audio filename and return a context_info string
    to be injected into the LLM prompt.

    Supports patterns like:
      20240315_AGT001_CUST4892_complaint_washing_machine.mp3
      call-20240315-john-refund-request.wav
      IVR_2024_03_15_agent_ahmed_issue_delivery.mp3
    """
    stem = Path(filename).stem  # strip extension

    # Normalise separators
    normalised = re.sub(r"[-]+", "_", stem)
    tokens = [t.strip() for t in normalised.split("_") if t.strip()]

    context_parts: list[str] = []

    date_str = None
    agent_id = None
    customer_id = None
    topic_words = []

    date_pattern = re.compile(r"^\d{4}[\-\d]{4,7}$|^\d{8}$")
    agent_pattern = re.compile(r"^(agt|agent|emp|staff|rep)\w*$", re.IGNORECASE)
    customer_pattern = re.compile(r"^(cust|customer|client|clnt)\w*$", re.IGNORECASE)
    numeric_id = re.compile(r"^\d{3,}$")

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if date_pattern.match(tok) and date_str is None:
            # Try to format nicely
            digits = re.sub(r"\D", "", tok)
            if len(digits) == 8:
                date_str = f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
            else:
                date_str = tok

        elif agent_pattern.match(tok):
            # Next token might be the agent name/id
            if i + 1 < len(tokens):
                agent_id = f"{tok}_{tokens[i + 1]}"
                i += 1
            else:
                agent_id = tok

        elif customer_pattern.match(tok):
            if i + 1 < len(tokens):
                customer_id = f"{tok}_{tokens[i + 1]}"
                i += 1
            else:
                customer_id = tok

        elif numeric_id.match(tok):
            # Bare numeric id – could be agent or customer; treat as ID
            if agent_id is None:
                agent_id = tok
            elif customer_id is None:
                customer_id = tok

        else:
            # Generic word → part of topic
            topic_words.append(tok)

        i += 1

    if date_str:
        context_parts.append(f"Call Date: {date_str}")
    if agent_id:
        context_parts.append(f"Agent ID: {agent_id}")
    if customer_id:
        context_parts.append(f"Customer ID: {customer_id}")
    if topic_words:
        topic = " ".join(topic_words).replace("_", " ").title()
        context_parts.append(f"Call Topic / Notes: {topic}")

    if not context_parts:
        context_parts.append(f"Source file: {filename}")

    return "\n".join(context_parts)


# ── Prompts ───────────────────────────────────────────────────────────────────
ARABIC_SYSTEM_PROMPT = """\
أنت مساعد متخصص في تحسين وتنقيح المحادثات العربية المصرية لمراكز خدمة العملاء.
"""

ARABIC_USER_PROMPT_TEMPLATE = """\
هذه المحادثة تمت بين عميل وموظف خدمة عملاء من شركة  ميراكوا (Miraco).

مهمتك هي إنشاء نسخة محسنة ومنقحة من النسخة الأصلية للمحادثة مع الحفاظ على:
1. التسلسل الزمني الدقيق للمحادثة
2. تسميات المتحدثين (Customer: / Agent:) باللغة الإنجليزية
3. الطوابع الزمنية بنفس الصيغة [HH:MM:SS.mmm]
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
- احتفظ بالطوابع الزمنية كما هي
- استخدم تسميات المتحدثين Customer: للعميل و Agent: للموظف (باللغة الإنجليزية)
- استخدم اللغة العربية المصرية الطبيعية والمهنية للمحتوى
- لا تضيف أي جمل تمهيدية أو تعريفية
- ابدأ مباشرة بالمحادثة المحسنة

النسخة الأصلية للمحادثة:
{original_transcription}

ابدأ مباشرة بالمحادثة المحسنة دون أي مقدمة:
"""

ENGLISH_SYSTEM_PROMPT = """\
You are a professional call center conversation editor specializing in creating clean, \
professional English versions of customer service calls.
"""

ENGLISH_USER_PROMPT_TEMPLATE = """\
This conversation took place between a customer and a customer service agent from Miraco (ميراكوا).

Your task is to create a revised, clean English version of the original conversation while maintaining:
1. The exact chronological order of the conversation
2. Speaker labels (Customer: / Agent:)
3. Timestamps in the same format [HH:MM:SS.mmm]
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
- Preserve timestamps exactly as they are
- Keep speaker labels (Customer: / Agent:)
- Use clear, professional English appropriate for call center context
- Maintain the conversational tone and customer service nature
- Do not add any introductory sentences or headers
- Start directly with the improved conversation

Original conversation:
{original_transcription}

Start directly with the improved conversation without any introduction:
"""




if __name__ == "__main__":
    # Quick test
    # test_filename = "20240315_AGT001_CUST4892_complaint_washing_machine.mp3"
    # print(extract_context_from_filename(test_filename))
    print(f"English prompt preview:\n{ENGLISH_USER_PROMPT_TEMPLATE.format(context_info='[Context info here]', original_transcription='[Original transcription here]')[:500]}...")