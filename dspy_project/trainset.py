"""trainset.py — Training + validation examples for DSPy optimization."""
from __future__ import annotations
import dspy
from fixtures import (
    ARABIC_SHORT, ENGLISH_WITH_ERRORS, MIXED_TRANSCRIPT,
    REAL_TRANSCRIPT, CONTEXT_FULL, CONTEXT_EMPTY,
)

# ── Shared score_breakdown template for 10 questions ─────────────────────────
def _bd(
    greeted="Pass", understood="Pass", answered="Pass",
    accurate="Pass", satisfied="Pass", tone="Pass",
    empathy="Pass", knowledge="Pass", alternatives="Pass", closing="Pass",
    notes=None,
):
    n = notes or {}
    def note(k, default): return n.get(k, default)
    return (
        '{"greeted_professionally":{"result":"' + greeted + '","note":"' + note("greeted","Agent greeted warmly") + '"},'
        '"understood_customer_need":{"result":"' + understood + '","note":"' + note("understood","Need identified clearly") + '"},'
        '"answered_all_questions":{"result":"' + answered + '","note":"' + note("answered","All questions addressed") + '"},'
        '"provided_accurate_info":{"result":"' + accurate + '","note":"' + note("accurate","Information was correct") + '"},'
        '"satisfied_customer_need":{"result":"' + satisfied + '","note":"' + note("satisfied","Customer need was met") + '"},'
        '"maintained_professional_tone":{"result":"' + tone + '","note":"' + note("tone","Tone was professional throughout") + '"},'
        '"showed_empathy":{"result":"' + empathy + '","note":"' + note("empathy","Agent showed empathy") + '"},'
        '"demonstrated_product_knowledge":{"result":"' + knowledge + '","note":"' + note("knowledge","Good product knowledge shown") + '"},'
        '"offered_alternatives_if_needed":{"result":"' + alternatives + '","note":"' + note("alternatives","N/A or alternatives offered") + '"},'
        '"proper_closing":{"result":"' + closing + '","note":"' + note("closing","Call closed properly") + '"}}'
    )


# ══════════════════════════════════════════════════════════════════════════════
# ARABIC TRAINING EXAMPLES — no timestamps in refined output
# ══════════════════════════════════════════════════════════════════════════════

ARABIC_TRAINSET = [
    dspy.Example(
        raw_transcript=ARABIC_SHORT,
        context_info="Topic: Refrigerator complaint",
        refined_transcript=(
            "Agent: أهلاً وسهلاً، معك خدمة عملاء شركة ميراكوا، هقدر أساعدك إزاي؟\n"
            "Customer: عندي مشكلة، التلاجة بتاعتي بقالها أسبوعين مش شغالة خالص.\n"
            "Agent: حضرتك تقدر تقولي موديل التلاجة إيه؟\n"
            "Customer: موديل نو فروست، وعندي الفاتورة معايا.\n"
            "Agent: تمام، هنبعت ليك تقني متخصص خلال 48 ساعة."
        ),
    ).with_inputs("raw_transcript", "context_info"),

    dspy.Example(
        raw_transcript=REAL_TRANSCRIPT,
        context_info=CONTEXT_FULL,
        refined_transcript=(
            "Agent: أهلاً وسهلاً، معك هبة من خدمة عملاء شركة ميراكوا، كيف أقدر أساعدك؟\n"
            "Customer: مساء الخير، عايزة أستفسر عن تكييف كاريير اتنين وربع حصن، التبريد بس.\n"
            "Agent: تمام، ممكن توضحي اسم حضرتك أولاً؟\n"
            "Customer: اسمي زهيبة.\n"
            "Agent: أهلاً يا أستاذة زهيبة، وعنوان التركيب؟\n"
            "Customer: أربعة أبراج المصري، عمرانية.\n"
            "Agent: والتركيب هيكون في دور متكرر ولا دور أخير؟\n"
            "Customer: دور متكرر.\n"
            "Agent: التكييف اتنين وربع حصن في الدور المتكرر بيغطي من 16 لـ 20 متر."
        ),
    ).with_inputs("raw_transcript", "context_info"),

    dspy.Example(
        raw_transcript="ألو احنا معاك خدمة عملاء ميراكوا مساء النور\nCustomer اه مساء النور انا عايز اشتكي على الغسالة بتاعتي",
        context_info="Topic: Washing machine complaint",
        refined_transcript=(
            "Agent: أهلاً وسهلاً، معك خدمة عملاء شركة ميراكوا، مساء النور.\n"
            "Customer: مساء النور، عندي شكوى بخصوص الغسالة بتاعتي."
        ),
    ).with_inputs("raw_transcript", "context_info"),
]


# ══════════════════════════════════════════════════════════════════════════════
# ENGLISH TRAINING EXAMPLES — no timestamps in refined output
# ══════════════════════════════════════════════════════════════════════════════

ENGLISH_TRAINSET = [
    dspy.Example(
        raw_transcript=ENGLISH_WITH_ERRORS,
        context_info="Topic: Defective product complaint",
        refined_transcript=(
            "Agent: Hello, thank you for calling Miraco Company customer service. How may I assist you today?\n"
            "Customer: I'd like to file a complaint about a product I purchased from you — it was defective.\n"
            "Agent: I'm very sorry to hear that. Could you please describe the problem in detail?\n"
            "Customer: The refrigerator I bought yesterday isn't cooling at all."
        ),
    ).with_inputs("raw_transcript", "context_info"),

    dspy.Example(
        raw_transcript=MIXED_TRANSCRIPT,
        context_info="Topic: Order tracking",
        refined_transcript=(
            "Agent: Hello, you've reached Miraco Company customer service. How can I help you?\n"
            "Customer: I'd like to check on my order status.\n"
            "Agent: Of course. Could you please provide your order number?\n"
            "Customer: The order number is 98765. I bought an inverter AC unit."
        ),
    ).with_inputs("raw_transcript", "context_info"),

    dspy.Example(
        raw_transcript=REAL_TRANSCRIPT,
        context_info=CONTEXT_FULL,
        refined_transcript=(
            "Agent: Good evening, you've reached Miraco Company customer service. My name is Heba. How can I assist you?\n"
            "Customer: Good evening. I'd like to inquire about the price of a 2.25-ton Carrier air conditioner, cooling only.\n"
            "Agent: Of course. May I have your name please?\n"
            "Customer: My name is Zuhaiba.\n"
            "Agent: Welcome, Ms. Zuhaiba. What's the installation address?\n"
            "Customer: 4 Abraj Al-Masri, Umraniya.\n"
            "Agent: Will the installation be on an intermediate floor or the top floor?\n"
            "Customer: Intermediate floor.\n"
            "Agent: The 2.25-ton unit on an intermediate floor covers 16 to 20 square meters."
        ),
    ).with_inputs("raw_transcript", "context_info"),
]


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS TRAINING EXAMPLES — updated to 10-question score_breakdown
# ══════════════════════════════════════════════════════════════════════════════

ANALYSIS_TRAINSET = [
    dspy.Example(
        raw_transcript=ARABIC_SHORT,
        context_info="Topic: Refrigerator complaint",
        main_subject="Customer reporting non-functional refrigerator for two weeks",
        call_outcome="Resolved",
        issue_resolution="Agent collected the refrigerator model and scheduled a technician visit within 48 hours.",
        call_summary="Customer called to report that her refrigerator had not been working for two weeks. The agent collected the model details and arranged a technician visit within 48 hours, resolving the complaint.",
        keywords="refrigerator, malfunction, technician, repair, no-frost",
        call_category="Complaint",
        service="Refrigerator Repair",
        agent_attitude="Friendly",
        customer_satisfaction="Satisfied",
        language="Arabic",
        call_score="8",
        score_breakdown=_bd(
            greeted="Pass", understood="Pass", answered="Pass",
            accurate="Pass", satisfied="Pass", tone="Pass",
            empathy="Pass", knowledge="Pass", alternatives="Fail",
            closing="Pass",
            notes={
                "alternatives": "Agent did not mention any alternative solutions",
                "closing": "Agent confirmed next steps and closed politely",
            }
        ),
    ).with_inputs("raw_transcript", "context_info"),

    dspy.Example(
        raw_transcript=REAL_TRANSCRIPT,
        context_info=CONTEXT_FULL,
        main_subject="Customer inquiring about 2.25-ton inverter AC unit prices and purchase options",
        call_outcome="Resolved",
        issue_resolution="Agent provided pricing for Carrier (44,025 EGP) and Midea ECO Master (39,715 EGP), explained features, installation coverage, warranty, and purchase channels.",
        call_summary="Customer called to inquire about the price of a 2.25-ton cooling-only inverter AC. The agent provided detailed pricing for two brands, explained technical specs, installation floor coverage, warranty, and available payment and purchase methods including branches and distributors.",
        keywords="AC unit, Carrier, Midea, inverter, price inquiry",
        call_category="Inquiry",
        service="AC Unit Sales",
        agent_attitude="Friendly",
        customer_satisfaction="Satisfied",
        language="Arabic",
        call_score="9",
        score_breakdown=_bd(
            greeted="Pass", understood="Pass", answered="Pass",
            accurate="Pass", satisfied="Pass", tone="Pass",
            empathy="Pass", knowledge="Pass", alternatives="Pass",
            closing="Fail",
            notes={
                "answered": "Customer asked about price, floor coverage, warranty, payment — all answered",
                "knowledge": "Agent demonstrated detailed product knowledge for both Carrier and Midea",
                "alternatives": "Agent proactively offered Midea as a lower-cost alternative to Carrier",
                "closing": "Call ended without explicitly confirming customer satisfaction",
            }
        ),
    ).with_inputs("raw_transcript", "context_info"),

    dspy.Example(
        raw_transcript=ENGLISH_WITH_ERRORS,
        context_info="Topic: Defective product complaint",
        main_subject="Customer filing complaint about a defective refrigerator purchased the previous day",
        call_outcome="Follow-up Needed",
        issue_resolution="Agent acknowledged the complaint and started collecting details but did not provide a resolution or schedule a follow-up action.",
        call_summary="Customer called to complain about a refrigerator purchased the day before that was not cooling at all. The agent acknowledged the issue and asked for details, but the call ended without a concrete resolution or next step being offered.",
        keywords="complaint, refrigerator, defective, cooling, return",
        call_category="Complaint",
        service="Refrigerator After-Sales",
        agent_attitude="Neutral",
        customer_satisfaction="Dissatisfied",
        language="English",
        call_score="4",
        score_breakdown=_bd(
            greeted="Fail", understood="Pass", answered="Fail",
            accurate="Fail", satisfied="Fail", tone="Pass",
            empathy="Pass", knowledge="Fail", alternatives="Fail",
            closing="Pass",
            notes={
                "greeted": "Greeting contained multiple spelling errors — unprofessional",
                "answered": "Customer's question about what to do was not answered",
                "accurate": "No accurate information or solution was provided",
                "satisfied": "Customer need was not satisfied — no resolution offered",
                "knowledge": "Agent did not demonstrate product knowledge or policy knowledge",
                "alternatives": "No alternatives (replacement, refund, technician) were offered",
                "closing": "Call ended appropriately even without resolution",
            }
        ),
    ).with_inputs("raw_transcript", "context_info"),
]


# ── Split helpers ─────────────────────────────────────────────────────────────

def get_arabic_trainset():
    return ARABIC_TRAINSET[:2], ARABIC_TRAINSET[2:]

def get_english_trainset():
    return ENGLISH_TRAINSET[:2], ENGLISH_TRAINSET[2:]

def get_analysis_trainset():
    return ANALYSIS_TRAINSET[:2], ANALYSIS_TRAINSET[2:]

def get_all_trainsets():
    return {
        "arabic":   get_arabic_trainset(),
        "english":  get_english_trainset(),
        "analysis": get_analysis_trainset(),
    }
