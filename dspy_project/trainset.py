"""trainset.py — Training + validation examples for DSPy optimization."""
from __future__ import annotations
import dspy
from fixtures import (
    ARABIC_SHORT, ENGLISH_WITH_ERRORS, MIXED_TRANSCRIPT,
    REAL_TRANSCRIPT, CONTEXT_FULL, CONTEXT_EMPTY,
)

ARABIC_TRAINSET = [
    dspy.Example(
        raw_transcript=ARABIC_SHORT,
        context_info="Topic: Refrigerator complaint",
        refined_transcript=(
            "Agent: أهلاً وسهلاً، معك خدمة عملاء شركة ميراكوا، هقدر أساعدك إزاي؟\n"
            "Customer: عندي مشكلة، التلاجة بتاعتي بقالها أسبوعين مش شغالة خالص.\n"
            "Agent: حضرتك تقدر تقولي موديل التلاجة إيه؟\n"
            "[00:00:12.300] Customer: موديل نو فروست، وعندي الفاتورة معايا.\n"
            "[00:00:15.600] Agent: تمام، هنبعت ليك تقني متخصص خلال 48 ساعة."
        ),
    ).with_inputs("raw_transcript","context_info"),
    dspy.Example(
        raw_transcript=REAL_TRANSCRIPT,
        context_info=CONTEXT_FULL,
        refined_transcript=(
            "Agent: أهلاً وسهلاً، معك هبة من خدمة عملاء شركة ميراكوا، كيف أقدر أساعدك؟\n"
            "Customer: مساء الخير، عايزة أستفسر عن تكييف كاريير اتنين وربع حصن، التبريد بس.\n"
            "Agent: تمام، هكد مع حضرتك على السعر. بس ممكن توضحي لي الاسم أولاً؟\n"
            "Customer: اسمي زهيبة.\n"
            "Agent: أهلاً يا أستاذة زهيبة، وعنوان التركيب؟\n"
            "Customer: أربعة أبراج المصري، عمرانية.\n"
        ),
    ).with_inputs("raw_transcript","context_info"),
    dspy.Example(
        raw_transcript="ألو احنا معاك خدمة عملاء ميراكوا مساء النور\nCustomer اه مساء النور انا عايز اشتكي على الغسالة بتاعتي",
        context_info="Topic: Washing machine complaint",
        refined_transcript=(
            "Agent: أهلاً وسهلاً، معك خدمة عملاء شركة ميراكوا، مساء النور.\n"
            "Customer: مساء النور، عندي شكوى بخصوص الغسالة بتاعتي."
        ),
    ).with_inputs("raw_transcript","context_info"),
]

ENGLISH_TRAINSET = [
    dspy.Example(
        raw_transcript=ENGLISH_WITH_ERRORS,
        context_info="Topic: Defective product complaint",
        refined_transcript=(
            "[00:00:00.000] Agent: Hello, thank you for calling Miraco Company customer service. How may I assist you today?\n"
            "[00:00:05.100] Customer: I'd like to file a complaint about a product I purchased from you — it was defective.\n"
            "[00:00:09.800] Agent: I'm very sorry to hear that. Could you please describe the problem in detail?\n"
            "[00:00:13.200] Customer: The refrigerator I bought yesterday isn't cooling at all."
        ),
    ).with_inputs("raw_transcript","context_info"),
    dspy.Example(
        raw_transcript=MIXED_TRANSCRIPT,
        context_info="Topic: Order tracking",
        refined_transcript=(
            "[00:00:01.000] Agent: Hello, you've reached Miraco Company customer service. How can I help you?\n"
            "[00:00:04.200] Customer: I'd like to check on my order status.\n"
            "[00:00:07.500] Agent: Of course. Could you please provide your order number?\n"
            "[00:00:10.800] Customer: The order number is 98765. I bought an inverter AC unit."
        ),
    ).with_inputs("raw_transcript","context_info"),
    dspy.Example(
        raw_transcript=REAL_TRANSCRIPT,
        context_info=CONTEXT_FULL,
        refined_transcript=(
            "Agent: Good evening, you've reached Miraco Company customer service. My name is Heba. How can I assist you?\n"
            "Customer: Good evening. I'd like to inquire about the price of a 2.25-ton Carrier air conditioner — cooling only.\n"
            "Agent: Of course, I'll look that up for you. May I have your name please?\n"
            "Customer: My name is Zuhaiba.\n"
        ),
    ).with_inputs("raw_transcript","context_info"),
]

ANALYSIS_TRAINSET = [
    dspy.Example(
        raw_transcript=ARABIC_SHORT,
        context_info="Topic: Refrigerator complaint",
        main_subject="Customer reporting non-functional refrigerator for two weeks",
        call_outcome="Resolved",
        issue_resolution="Agent scheduled a technician visit within 48 hours to inspect the faulty refrigerator.",
        call_summary="Customer called to report that her refrigerator has not been working for two weeks. The agent collected the model details and arranged for a technician to visit within 48 hours.",
        keywords="refrigerator, malfunction, technician, repair, no-frost",
        call_category="Complaint",
        service="Refrigerator Repair",
        agent_attitude="Friendly",
        customer_satisfaction="Satisfied",
        language="Arabic",
        call_score="86",
        score_breakdown='{"greeted_professionally":{"result":"Pass","note":"Agent greeted warmly"},"identified_customer_need":{"result":"Pass","note":"Need identified immediately"},"provided_accurate_info":{"result":"Pass","note":"Correct 48h timeframe given"},"maintained_professional_tone":{"result":"Pass","note":"Professional throughout"},"offered_complete_solution":{"result":"Pass","note":"Technician scheduled"},"confirmed_resolution":{"result":"Fail","note":"Did not confirm with customer explicitly"},"proper_closing":{"result":"Pass","note":"Call closed politely"}}',
    ).with_inputs("raw_transcript","context_info"),
    dspy.Example(
        raw_transcript=REAL_TRANSCRIPT,
        context_info=CONTEXT_FULL,
        main_subject="Customer inquiring about 2.25-ton inverter AC unit price and purchase options",
        call_outcome="Resolved",
        issue_resolution="Agent provided pricing for both Carrier and Midea 2.25-ton inverter AC units, explained features, installation coverage, and available purchase channels.",
        call_summary="Customer called to ask about the price of a 2.25-ton cooling-only inverter air conditioner. The agent provided prices for Carrier (44,025 EGP) and Midea ECO Master (39,715 EGP), explained warranty terms and payment methods.",
        keywords="AC unit, Carrier, Midea, inverter, price inquiry",
        call_category="Inquiry",
        service="AC Unit Sales",
        agent_attitude="Friendly",
        customer_satisfaction="Satisfied",
        language="Arabic",
        call_score="86",
        score_breakdown='{"greeted_professionally":{"result":"Pass","note":"Warm greeting with name"},"identified_customer_need":{"result":"Pass","note":"AC price inquiry identified"},"provided_accurate_info":{"result":"Pass","note":"Accurate prices and specs provided"},"maintained_professional_tone":{"result":"Pass","note":"Professional and helpful throughout"},"offered_complete_solution":{"result":"Pass","note":"Multiple options and channels offered"},"confirmed_resolution":{"result":"Fail","note":"Customer satisfaction not explicitly confirmed"},"proper_closing":{"result":"Pass","note":"WhatsApp contact shared, polite close"}}',
    ).with_inputs("raw_transcript","context_info"),
    dspy.Example(
        raw_transcript=ENGLISH_WITH_ERRORS,
        context_info="Topic: Defective product complaint",
        main_subject="Customer filing complaint about defective refrigerator purchased the previous day",
        call_outcome="Follow-up Needed",
        issue_resolution="Agent acknowledged the complaint and began gathering details, but the call did not reach a full resolution.",
        call_summary="Customer called to complain about a refrigerator that was not cooling at all, purchased the day before. The agent acknowledged the issue and started collecting details for further action.",
        keywords="complaint, refrigerator, defective, return, quality",
        call_category="Complaint",
        service="Refrigerator After-Sales",
        agent_attitude="Neutral",
        customer_satisfaction="Dissatisfied",
        language="English",
        call_score="57",
        score_breakdown='{"greeted_professionally":{"result":"Fail","note":"Greeting had spelling errors"},"identified_customer_need":{"result":"Pass","note":"Complaint identified"},"provided_accurate_info":{"result":"Fail","note":"No resolution info given"},"maintained_professional_tone":{"result":"Pass","note":"Tone was acceptable"},"offered_complete_solution":{"result":"Fail","note":"No solution offered"},"confirmed_resolution":{"result":"Fail","note":"No resolution to confirm"},"proper_closing":{"result":"Pass","note":"Call ended appropriately"}}',
    ).with_inputs("raw_transcript","context_info"),
]

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
