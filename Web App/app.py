"""
Leeds General Infirmary — MTS Triage Assistant
================================================
Implements the exact 8-step decision tree:

  STEP 1 · Demographics      — collect age + gender before anything else
  STEP 2 · AI Background     — BERT dept + ICD-11 lookup (runs on first message)
  STEP 3 · Instant Red Gate  — hard regex scan, zero LLM latency
  STEP 4 · Complaint         — ask if not yet stated
  STEP 5 · Red Screen        — Ollama Q1: confirm / deny red emergency signs
  STEP 6 · Orange Screen     — Ollama Q2: orange discriminators + hard regex assist
  STEP 7 · Pain + Yellow     — Ollama Q3-4: pain scale and yellow criteria
  STEP 8 · Decide            — forced final JSON output (Green / Blue fallback)

`collected` keys used as state:
  age, gender, complaint, pain_score         — clinical facts
  bert_dept, icd_result, bert_demo           — AI background results
  bert_done                                  — flag: BERT+ICD already ran
  triage_step                                — current position in the tree
  red_cleared, orange_cleared                — flags: those screens passed
  question_count                             — total Ollama questions asked
"""

# High-level overview:
# - This Flask app implements a step-by-step triage assistant.
# - Incoming patient messages are POSTed to `/assess` with a small
#   `collected` state dict and `history` of earlier messages. The server
#   extracts demographics and clinical cues, uses a small BERT model and
#   the WHO ICD-11 API for background hints, applies fast regex-based
#   checks for immediate emergencies, and otherwise asks focused LLM
#   questions (via local Ollama) to reach a final JSON triage decision.
# - The code is organised into: constants/mappings, model + API helpers,
#   simple extractor utilities, regex gates for Red/Orange, prompt
#   builders for the LLM, triage flow control, and Flask endpoints.


from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification, logging
import requests
import re
import os
import json
from dotenv import load_dotenv

load_dotenv()
logging.set_verbosity_error()

app = Flask(__name__)

# Configuration notes:
# - Store `ICD_CLIENT_ID` and `ICD_CLIENT_SECRET` in a `.env` file or
#   environment before running so the ICD-11 lookup can authenticate.
# - The app expects a local Ollama instance at http://localhost:11434 for
#   focused LLM questions. If Ollama is unavailable the service falls back
#   to a Yellow (urgent) default decision.
# - The BERT model file is expected at `../BERT_L/bert_model_fold_0.pth`.

# Map numeric model labels (BERT output) to human-facing department names.
# The BERT classifier predicts one of 20 labels; we translate that to a
# department suggestion used when building the LLM prompt and final output.
LABEL_TO_DEPT = {
    0:  "Maternity",          1:  "Maternity",
    2:  "HIV Outpatient Service",
    3:  "Respiratory Medicine", 4:  "General Medicine",
    5:  "General Medicine",   6:  "General Medicine",
    7:  "General Medicine",   8:  "General Medicine",
    9:  "General Medicine",   10: "A&E",
    11: "A&E",                12: "Audiology",
    13: "Cardiology",         14: "Neurosciences",
    15: "Abdominal Medicine & Surgery",
    16: "A&E",                17: "Trauma & Orthopaedics",
    18: "General Medicine",   19: "General Medicine",
}


# Human readable descriptions for each MTS colour used in final responses.
MTS_DESCRIPTIONS = {
    "Red":    "IMMEDIATE — Life-threatening. Treatment within 0 minutes.",
    "Orange": "VERY URGENT — Potentially life-threatening. Treatment within 10 minutes.",
    "Yellow": "URGENT — Serious but stable. Treatment within 60 minutes.",
    "Green":  "STANDARD — Minor problem. Treatment within 120 minutes.",
    "Blue":   "NON-URGENT — No immediate danger. Treatment within 240 minutes.",
}

# Short labels for MTS levels used in the JSON response (for UI display).
MTS_LEVELS = {
    "Red": "Immediate", "Orange": "Very Urgent",
    "Yellow": "Urgent", "Green": "Standard", "Blue": "Non-Urgent",
}

# Mapping of ICD codes (or code prefixes) to trustable departments. Used
# to convert an ICD-11 match into a department suggestion.
CODE_TO_DEPARTMENT = {
    "MD3": "Cardiology",  "MD4": "Cardiology",
    "MC8": "Cardiology",  "MC9": "Cardiology",
    "MD1": "Respiratory Medicine", "MD2": "Respiratory Medicine",
    "CB":  "Respiratory Medicine", "CA":  "Respiratory Medicine",
    "MB4": "Neurosciences",        "8A":  "Neurosciences",
    "MD8": "Abdominal Medicine & Surgery",
    "DA":  "Abdominal Medicine & Surgery",
    "ME8": "Trauma & Orthopaedics", "FB": "Trauma & Orthopaedics",
    "3A":  "Clinical Haematology",
    "GA":  "Maternity", "GB": "Maternity", "HA": "Maternity",
    "2B":  "Leeds Cancer Centre",  "2C": "Leeds Cancer Centre",
    "AB":  "Audiology",
}

# Fallback mapping from ICD chapter numbers to departments when a code
# prefix doesn't match directly.
CHAPTER_TO_DEPARTMENT = {
    "11": "Cardiology",    "12": "Respiratory Medicine",
    "13": "Abdominal Medicine & Surgery",
    "15": "Trauma & Orthopaedics",
    "16": "Maternity",     "8": "Neurosciences",
    "10": "Audiology",     "2": "Leeds Cancer Centre",
    "3":  "Clinical Haematology",
    "21": "A&E",           "22": "Trauma & Orthopaedics",
}


# Maximum number of LLM questions before forcing a decision
MAX_QUESTIONS = 10


# model loading
MODEL_PATH   = "../BERT_L/bert_model_fold_0.pth"
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer    = BertTokenizer.from_pretrained("bert-base-uncased")
_bert_model  = None
model_loaded = False


def load_bert():
    global _bert_model, model_loaded
    if os.path.exists(MODEL_PATH):
        try:
            _bert_model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=20,
                output_attentions=False, output_hidden_states=False,
            )
            _bert_model.load_state_dict(
                torch.load(MODEL_PATH, map_location=device, weights_only=False)
            )
            _bert_model.to(device)
            _bert_model.eval()
            model_loaded = True
            print(f"BERT loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"BERT load error: {e}")
    else:
        print(f"BERT model not found at {MODEL_PATH} — running in demo mode.")


load_bert()


def run_bert(text: str) -> dict:
    # If the BERT weights failed to load, use a simple heuristics-based fallback so the system can still provide department suggestions.
    if not model_loaded:
        t = text.lower()
        if any(w in t for w in ["pregnant", "contraction", "labour", "period"]):
            label = 0
        elif any(w in t for w in ["chest", "heart", "blood pressure"]):
            label = 13
        elif any(w in t for w in ["stomach", "abdomen", "nausea"]):
            label = 15
        elif any(w in t for w in ["anxiety", "stress", "depressed"]):
            label = 6
        else:
            label = 11
        return {"label": label, "department": LABEL_TO_DEPT.get(label, "A&E"), "demo": True}

    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128, padding="max_length",
    )
    with torch.no_grad():
        out = _bert_model(
            inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
        )
    label = torch.argmax(out.logits, dim=1).item()
    return {"label": label, "department": LABEL_TO_DEPT.get(label, "A&E"), "demo": False}


# WHO ICD API

_icd_token = None


def refresh_icd_token():
    global _icd_token
    try:
        resp = requests.post(
            "https://icdaccessmanagement.who.int/connect/token",
            data={
                "client_id":     os.getenv("ICD_CLIENT_ID"),
                "client_secret": os.getenv("ICD_CLIENT_SECRET"),
                "scope":         "icdapi_access",
                "grant_type":    "client_credentials",
            },
            timeout=10,
        )
        _icd_token = resp.json().get("access_token")
        print("ICD-11 token obtained.")
    except Exception as e:
        print(f"ICD-11 token error: {e}")
        _icd_token = None


refresh_icd_token()


# Return a dict with keys: 'matched_condition', 'icd_code', 'department', 'score'.
# Falls back to Unknown if the ICD token or lookup is unavailable.
def icd_lookup(text: str) -> dict:
    if not _icd_token:
        return {"matched_condition": "Unknown", "icd_code": None, "department": "A&E", "score": 0}
    try:
        resp = requests.get(
            "https://id.who.int/icd/release/11/2024-01/mms/search",
            params={"q": text, "flatResults": True, "useFlexisearch": True},
            headers={
                "Authorization": f"Bearer {_icd_token}",
                "API-Version": "v2",
                "Accept-Language": "en",
            },
            timeout=10,
        )
        for entity in resp.json().get("destinationEntities", []):
            if entity.get("isResidualUnspecified") or entity.get("isResidualOther"):
                continue
            code  = entity.get("theCode")
            chap  = entity.get("chapter")
            title = re.sub(r"<[^>]+>", "", entity.get("title", ""))
            score = entity.get("score", 0)
            if not code:
                continue
            dept = "A&E"
            for ln in [3, 2, 1]:
                if code[:ln] in CODE_TO_DEPARTMENT:
                    dept = CODE_TO_DEPARTMENT[code[:ln]]
                    break
            else:
                dept = CHAPTER_TO_DEPARTMENT.get(str(chap), "A&E")
            return {
                "matched_condition": title,
                "icd_code": code,
                "department": dept,
                "score": round(score, 3),
            }
    except Exception as e:
        print(f"ICD-11 lookup error: {e}")
    return {"matched_condition": "Unknown", "icd_code": None, "department": "A&E", "score": 0}



# STEP 1: Demographic extractors (run on every message)
# Helpers to extract age, gender, and pain score from free text.

# Parse an age from free-form text. Returns integer age or None.
def extract_age(text: str, context_is_age_question: bool = False):
    for pattern in [
        r"\b(\d{1,3})\s*(?:years?\s*old|yo)\b",
        r"\bage\s*:?\s*(\d{1,3})\b",
        r"\bi[' ]?m\s+(\d{1,3})\b",
        r"\bi am\s+(\d{1,3})\b",
        r"\b(\d{1,3})\s*year",
        r"^\s*(\d{1,3})\s*$",          # bare number as entire message
        r"^\s*it['\s]*s\s*(\d{1,3})\s*$",  # "it's 21"
        r"^\s*just\s*(\d{1,3})\s*$",  # "just 21"
    ]:
        m = re.search(pattern, text.strip(), re.IGNORECASE)
        if m:
            age = int(m.group(1))
            if 0 < age < 130:
                return age
    # Context-aware fallback: if we just asked "How old are you?" accept any
    # standalone number in the reply, even surrounded by minor filler words
    if context_is_age_question:
        m = re.search(r"\b(\d{1,3})\b", text)
        if m:
            age = int(m.group(1))
            if 0 < age < 130:
                return age
    return None


# Return 'male', 'female', or None. Uses simple keyword patterns.
def extract_gender(text: str, context_is_gender_question: bool = False):
    t = text.strip().lower()
    for p in [r"\bfemale\b", r"\bwoman\b", r"\bgirl\b", r"\bshe\b", r"\bher\b",
              r"\bpregnant\b", r"\bpregnancy\b", r"\bperiod\b", r"\bmrs\b", r"\bms\b"]:
        if re.search(p, t):
            return "female"
    for p in [r"\bmale\b", r"\bman\b", r"\bboy\b", r"\bhe\b", r"\bhis\b", r"\bmr\b"]:
        if re.search(p, t):
            return "male"
    # Context-aware fallback: single-letter or short answer when we just asked gender
    if context_is_gender_question:
        if re.match(r"^\s*f\s*$", t):
            return "female"
        if re.match(r"^\s*m\s*$", t):
            return "male"
    return None


# Parse a 0–10 pain score from free text. Returns int 0–10 or None.
def extract_pain_score(text: str):
    word_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    t = text.lower()
    for p in [
        r"\b(\d{1,2})\s*/\s*10\b",
        r"\b(\d{1,2})\s+out\s+of\s+10\b",
        r"pain\s+(?:is|of|about|around|like|at)?\s*(?:a\s+)?(\d{1,2})\b",
        r"(?:score|level|rate|rating)\s+(?:is|of)?\s*(\d{1,2})\b",
        r"\babout\s+a\s+(\d{1,2})\b",
        r"\baround\s+(\d{1,2})\b",
    ]:
        m = re.search(p, t)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 10:
                return v
    for p in [
        r"\b(" + "|".join(word_map) + r")\s+out\s+of\s+ten\b",
        r"\b(" + "|".join(word_map) + r")\s*/\s*10\b",
    ]:
        m = re.search(p, t)
        if m:
            return word_map[m.group(1)]
    if re.search(r"\bno\s+pain\b|pain\s*free\b", t):
        return 0
    if re.search(r"\bworst\s+(?:pain|ever)\b|unbearable\b|excruciating\b", t):
        return 10
    if re.search(r"\bsevere\s+pain\b|very\s+bad\s+pain\b", t):
        return 8
    if re.search(r"\bmoderate\s+pain\b|moderate\b", t):
        return 5
    if re.search(r"\bmild\s+pain\b|mild\b|slight\b", t):
        return 2
    return None


# STEP 3 — Immediate red-flag regexes (checked without calling the LLM)

IMMEDIATE_RED_PATTERNS = [
    (r"\bnot\s+breath\w*|cannot\s+breath\w*|can'?t\s+breath\w*|stopped\s+breath\w*|not\s+breathing\b",
     "Not breathing — airway emergency"),
    (r"\bno\s+pulse\b|cardiac\s+arrest\b|heart\s+stopped\b|collapsed\b.{0,30}unresponsive\b",
     "Cardiac arrest"),
    (r"\bcurrently\s+(seiz\w+|fit\w+)|actively\s+(seiz\w+|fit\w+)|seizing\s+now\b|fitting\s+now\b",
     "Active seizure"),
    (r"\bfacial?\s+burn\w*|inhalation\s+burn\w*|burn\w*.{0,20}\bmouth\b|smoke\s+inhal\w*|steam\s+inhal\w*",
     "Facial / inhalation burn"),
    (r"\bunresponsive\b|unconscious\b|not\s+waking\b|won'?t\s+wake\b|won'?t\s+respond\b",
     "Unresponsive / unconscious"),
    (r"\bvomiting\s+blood\b|vomit\w*\s+blood\b|coughing\s+up\s+blood\b|hematemesis\b",
     "Vomiting / coughing blood"),
    (r"\barterial\s+bleed\w*|bleed\w*\s+won'?t\s+stop\b|uncontrolled\s+bleed\w*|bleed\w*\s+everywhere\b",
     "Uncontrolled haemorrhage"),
    (r"\bglucose\s+(?:is\s+|of\s+|was\s+)?[012]\b|glucose\s+less\s+than\s+3\b|sugar\s+(?:is\s+)?[012]\b|hypoglycaemi\w+",
     "Hypoglycaemia — glucose below 3 mmol/L"),
]


# Return a short reason if any immediate Red sign is matched, else None.
def hard_red_check(text: str):
    for pattern, reason in IMMEDIATE_RED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return reason
    return None


# These immediate-check functions are designed to be fast and high-specificity.
# They catch very obvious, life-threatening phrases (Red) or clear Very Urgent
# features (Orange) so we can short-circuit the LLM when rapid action is needed.


# STEP 6 assist — Immediate orange-flag regexes (quick assist before LLM)

IMMEDIATE_ORANGE_PATTERNS = [
    # Chest pain (standalone — Orange regardless of accompanying features)
    (r"\bchest\s+(?:pain|tightness|pressure|discomfort)\b",
     "Chest pain"),
    # Cardiac features alongside chest pain
    (r"\bchest\s+pain\b.{0,80}(?:sweat\w*|nausea\w*|arm\s+pain|jaw\b|radiat\w+)",
     "Chest pain with cardiac features"),
    # Stroke / focal neurology
    (r"\bstroke\b|face.{0,15}droop\w*|arm\s+(?:gone\s+)?weak\w*|speech\s+(?:difficult\w*|slur\w*)|sudden\s+(?:confusion|weakness|numbness)\b",
     "Stroke / focal neurology (FAST)"),
    # Level of consciousness reduced / confused
    (r"\breduced\s+(?:level\s+of\s+)?consciousness\b|confused\s+and\s+(?:drowsy|disoriented|not\s+alert)\b|not\s+(?:fully\s+)?alert\b|level\s+of\s+consciousness",
     "Reduced level of consciousness / confused"),
    # Acute shortness of breath
    (r"\bcan'?t\s+catch\s+(?:my\s+)?breath\b|acutely?\s+short\s+of\s+breath\b|severe\s+(?:breathing\s+difficulty|dyspnoea|shortness\s+of\s+breath)\b|struggling\s+to\s+breathe\b|breathing\s+is\s+very\s+(?:fast|laboured|difficult)\b",
     "Acute shortness of breath"),
    # Coughing blood
    (r"\bcoughing\s+(?:up\s+)?blood\b|haemoptysis\b|blood\s+when\s+(?:I\s+)?cough\b",
     "Coughing blood"),
    # Stabbed neck
    (r"\bstabb?(?:ed|ing)\b.{0,30}\bneck\b|\bneck\b.{0,20}\bstabb?(?:ed|ing)\b|knife\b.{0,30}\bneck\b",
     "Stabbed neck"),
    # Seizure post-ictal
    (r"\bpost[\s\-]?ictal\b|just\s+had\s+a\s+(?:fit|seizure)\b|seizure\s+(?:just\s+)?(?:stopped|finished|ended)\b|had\s+a\s+fit\b",
     "Post-ictal / seizure just ended"),
    # Threatened limb (painful, pale, pulseless)
    (r"\bpulseless\b|no\s+pulse\s+in\b|limb\b.{0,30}(?:pale|cold|numb|pulseless)\b|threatened\s+limb\b|loss\s+of\s+(?:sensation|feeling)\s+in\s+(?:arm|leg|hand|foot)\b",
     "Threatened limb"),
    # Eye injury
    (r"\beye\s+injur\w+\b|injured\s+(?:my\s+)?eye\b|something\s+in\s+(?:my\s+)?eye\b.{0,30}(?:chemical|acid|burn|bleed\w*)\b|chemical\b.{0,20}\beye\b",
     "Eye injury"),
    # Dislocation of larger joint
    (r"\bdislocat\w+\b.{0,30}(?:shoulder|hip|knee|elbow|ankle|wrist)\b|(?:shoulder|hip|knee|elbow|ankle|wrist)\b.{0,20}\bdislocat\w+\b|dislocated\s+(?:my\s+)?(?:shoulder|hip|knee|elbow)\b",
     "Dislocation of larger joint"),
    # Compound / open fracture
    (r"\bcompound\s+fracture\b|open\s+fracture\b|bone\s+(?:is\s+)?(?:through|sticking\s+out|exposed|visible|coming\s+out)\b|fracture\b.{0,20}\bopen\s+wound\b",
     "Compound fracture (bone exposed)"),
    # Burns - major (over 20%, electrical, circumferential, chemical)
    (r"\bburn\w*\s+(?:over|more\s+than|greater\s+than)\s+(?:20|25|30|40|50)\s*(?:percent|%)\b",
     "Burn over 20% body surface area"),
    (r"\belectrical\s+burn\w*\b|burn\w*\s+from\s+electricit\w+\b|electrocution\b",
     "Electrical burn"),
    (r"\bcircumferential\s+burn\w*\b|burn\w*\s+all\s+(?:the\s+way\s+)?around\b",
     "Circumferential burn"),
    (r"\bchemical\s+burn\w*\b|burn\w*\s+from\s+(?:acid|alkali|bleach|chemical)\b|acid\s+burn\b",
     "Chemical burn"),
    # Poisoning / overdose
    (r"\bpoisoning\b|overdose\b|ingested\b.{0,40}(?:drug|tablet|medication|chemical|bleach|pill|substance)",
     "Poisoning / overdose"),
    # High energy transfer / severe mechanism of injury
    (r"\bhigh\s+(?:speed|velocity|energy)\s+(?:accident|crash|collision|impact)\b|road\s+traffic\s+(?:accident|collision)\b|RTA\b|hit\s+by\s+a\s+(?:car|vehicle|truck|bus)\b|fall\s+from\s+(?:height|roof|ladder|building|more\s+than\s+[23456789])\b|gunshot\b|high\s+fall\b",
     "High energy transfer / severe mechanism of injury"),
    # Aggression / threat to self or others
    (r"\bvery\s+aggressive\b|threatening\s+(?:to\s+hurt|violence)\b|violent\s+behaviour\b|attacking\s+(?:people|staff|others)\b",
     "Aggression / threatening behaviour"),
    # Diabetic with glucose >11 AND ketonuria
    (r"\bdiabet\w+\b.{0,60}(?:glucose|sugar|ketone)\b.{0,60}(?:11|12|13|14|15|16|17|18|19|20|2[0-9]|over\s+11|over\s+1[0-9])\b",
     "Diabetic with high glucose"),
    (r"\bketone\w*\b.{0,40}\bpositive\b|ketoacidosis\b|DKA\b|glucose\b.{0,40}ketone\w*\b",
     "Diabetic ketoacidosis / ketonuria"),
    # Pregnancy complications (Orange level)
    (r"\bpregnant\b.{0,80}(?:abdominal\s+(?:pain|trauma)|trauma|accident|fell|bleed\w*\s+heavily|bleeding\s+(?:a\s+lot|heavily))\b",
     "Pregnancy with abdominal pain/trauma"),
    (r"\b(?:20|2[1-9]|3[0-9]|4[0-9])\s+weeks\s+pregnant\b.{0,80}(?:pain|bleed|trauma)\b",
     "Later-stage pregnancy with complications"),
    # Sepsis / temperature >41
    (r"\bsepsis\b|temp\w*\s+(?:of\s+)?4[1-9]\b|temperature\s+(?:of\s+)?4[1-9]\b",
     "Sepsis / temperature above 41 degrees"),
    # Severe pain (8-10/10 stated explicitly)
    (r"\bpain\b.{0,30}(?:10|9|8)\s*(?:/\s*10|out\s+of\s+10)\b|(?:10|9|8)\s*/\s*10\s+pain\b|unbearable\s+pain\b|worst\s+pain\b|excruciating\b",
     "Severe pain (8-10/10)"),
]


# Return a short reason if any obvious Orange sign is matched, else None.
def hard_orange_check(text: str):
    for pattern, reason in IMMEDIATE_ORANGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return reason
    return None


# Ollama (LLM) — step-specific system prompts used for focused questions

_SHARED_RULES = """You are a triage nurse at Leeds General Infirmary.
RULES (non-negotiable):
- Never mention triage colours, MTS, or AI systems to the patient.
- Never ask more than ONE question per response.
- Speak plain English. Be brief and calm.
- CRITICAL DEFAULT: The overwhelming majority of patients are NOT emergencies. Always start from the
  assumption that the patient is Green (minor) and only escalate if there is specific, unambiguous
  clinical evidence of a higher priority. Do NOT infer severity from vague language.
- NEVER assign Red or Orange based on a single ambiguous word like "bleeding", "pain", or "breathing
  difficulty" without clinical specificity. Bleeding from a period, a small cut, or normal post-surgical
  oozing is NOT a Red emergency. Mild breathing difficulty from anxiety or a cold is NOT Orange."""

_FINAL_JSON_SCHEMA = """When you are confident, output ONLY this JSON (no other text before or after it):
{
    "confirmed": true,
    "colour": "Red|Orange|Yellow|Green|Blue",
    "department": "<department name>",
    "reason": "<one sentence clinical justification>"
}"""


def _known_summary(collected: dict) -> str:
    parts = []
    if collected.get("age"):
        parts.append(f"Age: {collected['age']}")
    if collected.get("gender"):
        parts.append(f"Gender: {collected['gender']}")
    if collected.get("complaint"):
        parts.append(f"Complaint: {collected['complaint']}")
    if collected.get("pain_score") is not None:
        parts.append(f"Pain: {collected['pain_score']}/10")
    return ", ".join(parts) if parts else "Nothing yet"


# Build a narrow, step-specific system prompt for a single turn.
def build_ollama_prompt(step: str, collected: dict, bert_dept: str, icd_result: dict) -> str:
    known = _known_summary(collected)
    icd_note = (
        f"ICD-11 matched: {icd_result['matched_condition']} (suggesting {icd_result['department']})"
        if icd_result.get("score", 0) > 0.3
        else "ICD-11 match low confidence — rely on BERT suggestion."
    )
    preamble = f"""{_SHARED_RULES}

AI pre-screening suggests department: {bert_dept}
{icd_note}
Known so far: {known}
"""

    # Step 4: get the complaint
    if step == "complaint":
        return preamble + """
YOUR ONLY JOB THIS TURN:
The presenting complaint is not clear. Ask ONE short open question to find out what has brought the patient in today. Do not ask about symptoms or history yet.
Example: "What has brought you in today?"
"""

    # Step 5: red screen
    if step == "red_screen":
        complaint = collected.get("complaint", "not stated")
        return preamble + f"""
YOUR ONLY JOB THIS TURN:
RULE OUT an immediate life-threatening (Red) emergency. The DEFAULT assumption is that this patient
is NOT Red. Only assign Red if the patient explicitly confirms one of the signs below.

Complaint to consider: {complaint}

WHAT COUNTS AS RED (clinical precision — these are narrow definitions):
- Airway/breathing: NOT breathing at all, or complete airway obstruction. NOT shortness of breath,
  anxiety, or mild difficulty — only total respiratory arrest or complete obstruction.
- Cardiac arrest: No pulse, heart stopped, collapsed and not responding. NOT chest pain on its own.
- Active seizure: Currently convulsing RIGHT NOW. NOT a history of seizures or post-seizure drowsiness.
- Facial/inhalation burn: Burns to the face/mouth/airway from fire or hot gas. NOT a skin burn elsewhere.
- Unresponsive: AVPU score of P or U — does not respond to voice OR pain. NOT just drowsy or confused.
- Uncontrolled haemorrhage: Arterial spurting blood that cannot be stopped. NOT: period blood, minor
  wound oozing, a small cut that is still bleeding slowly, or any normal menstrual bleeding.
- Vomiting blood: Large volume of fresh red blood vomited up. NOT blood-tinged mucus or specks of blood.

WHAT IS NEVER RED (do not ask about these as if they might be Red):
- Period cramps, even heavy periods or heavy menstrual bleeding
- Small cuts, grazes, or superficial wounds
- Mild to moderate pain of any kind
- Anxiety or panic attacks causing breathing difficulty
- Nausea or vomiting without blood

HOW TO PROCEED:
Based on the complaint above, ask the single most clinically relevant question to RULE OUT Red.
If the complaint is obviously minor (period cramps, small cut, headache, cold), skip directly to
asking about pain level — do NOT ask about breathing or cardiac arrest for an unrelated complaint.

If the patient's answer clearly and unambiguously confirms one of the Red signs above (not just any
mention of bleeding or difficulty), output the final JSON with colour "Red".
If the patient's answer does not confirm a Red sign, do NOT output JSON — ask your one question.

{_FINAL_JSON_SCHEMA}
"""

    # Step 6: orange screen
    if step == "orange_screen":
        complaint = collected.get("complaint", "not stated")
        gender    = collected.get("gender", "unknown")
        preg_note = (
            "\nPregnancy note: only relevant if the patient is known to be 20+ weeks pregnant "
            "AND has severe abdominal pain or significant vaginal bleeding — NOT for routine "
            "period pain or early pregnancy."
            if gender == "female" else ""
        )
        return preamble + f"""
YOUR ONLY JOB THIS TURN:
Red signs have been ruled out. Now RULE OUT Very Urgent (Orange) signs.
Complaint: {complaint}{preg_note}

WHAT COUNTS AS ORANGE — full SATS Adult Very Urgent list (narrow definitions):
- Chest pain: ANY chest pain is Orange. NOT heartburn, musculoskeletal, or referred pain clearly not cardiac.
- Acute shortness of breath: struggling to speak full sentences, laboured breathing. NOT anxiety or a cold.
- Reduced level of consciousness / confused: not fully alert, responding only to voice or pain (AVPU V/P).
- Coughing blood (haemoptysis): frank blood coughed up. NOT blood-streaked mucus from a cough.
- Stabbed neck: any penetrating injury to the neck.
- Uncontrolled haemorrhage: arterial bleed, spurting blood. NOT slow oozing or a minor wound.
- Post-ictal: seizure just finished, patient is not yet alert or recovering. NOT a history of epilepsy hours ago.
- Stroke / focal neurology: NEW sudden face drooping, arm weakness, slurred speech (FAST). NOT pre-existing.
- Aggression / violent behaviour: patient is actively violent or threatening harm to others.
- Threatened limb: limb is painful, pale, pulseless — signs of vascular compromise.
- Eye injury: any significant injury to the eye or orbit.
- Dislocation of larger joint: shoulder, hip, knee, elbow (NOT finger or toe — those are Yellow).
- Compound / open fracture: bone visible through skin, or bone breaking through wound.
- Burn over 20% body surface area.
- Electrical burn (any extent).
- Circumferential burn (wraps all the way around a limb or the trunk).
- Chemical burn.
- Poisoning / overdose: known ingestion of toxic substance right now.
- Diabetic with glucose over 11 mmol/L AND ketonuria / ketones present.
- Vomiting fresh blood.
- Pregnancy with abdominal trauma or abdominal pain (especially 20+ weeks).
- Severe pain: 8-10 out of 10.

WHAT IS NEVER ORANGE:
- Period cramps (even severe) without pregnancy complications
- Small cuts, grazes, or minor lacerations
- Mild-to-moderate pain (below 8/10) without other Orange features
- Anxiety or panic attacks
- A headache without trauma or neurological signs
- Minor burns (small area, no electrical/chemical/circumferential)

HOW TO PROCEED:
Ask ONE targeted question about the most plausible Orange criterion for this specific complaint.
If the complaint is clearly minor (period cramps, small cut), ask about pain level and prepare to
assign Green or Yellow — do not ask about cardiac or stroke symptoms for an unrelated complaint.

If the answer CLEARLY confirms an Orange sign, output the final JSON with colour "Orange".
If the answer does not confirm Orange, do NOT output JSON — the next step handles Yellow/Green.

{_FINAL_JSON_SCHEMA}
"""

    # Step 7: pain + yellow
    if step == "pain_yellow":
        complaint  = collected.get("complaint", "not stated")
        pain_score = collected.get("pain_score")
        pain_note  = (
            f"Pain score already captured: {pain_score}/10. Do NOT ask for it again."
            if pain_score is not None
            else "Pain score not yet captured — ask for it now (ask: \'On a scale of 0 to 10, how bad is your pain?\')."
        )
        return preamble + f"""
YOUR ONLY JOB THIS TURN:
Red and Orange have already been ruled out. Your ONLY task now is to decide between Yellow, Green, and Blue.
You MUST NOT assign Red or Orange at this step — those screens are done.

Complaint: {{complaint}}
{{pain_note}}

SATS URGENT (Yellow — within 60 min) — assign if ANY of these apply:
- Controlled haemorrhage (bleeding that has been stopped by pressure, but was active)
- Dislocation of a finger or toe
- Closed fracture (bone broken but skin intact — patient refusing to move the limb)
- Minor burn (small area, not electrical/chemical/circumferential)
- Abdominal pain (any new abdominal pain that is not trivially explained)
- Diabetic with glucose over 17 mmol/L but NO ketones
- Vomiting persistently (3 or more times, cannot keep fluids down)
- Pregnancy with minor trauma (not abdominal trauma — that is Orange)
- Pregnancy with PV (vaginal) bleeding (light, early pregnancy — heavy or later is Orange)
- Moderate pain: 4-7/10
- Fever 38-41°C
- Moderate breathing difficulty (able to speak in sentences but noticeably breathless)
- Recent head injury without confusion, vomiting, or loss of consciousness

GREEN (within 120 min) — the DEFAULT for most minor presentations:
- Mild pain: 1-3/10
- Period cramps (even severe) without pregnancy complications
- Small cuts, grazes, superficial wounds with minor or stopped bleeding
- Cold, sore throat, earache, mild infection
- Minor sprain, bruise, or musculoskeletal pain
- UTI symptoms, toothache, mild rash
- Mild headache without trauma

BLUE (within 240 min):
- No pain or minimal discomfort
- Chronic condition unchanged
- Administrative / repeat prescription

ADDITIONAL INVESTIGATION FLAGS — include these in your reason field if relevant:
- If the patient has chest pain: note "ECG required before assessment"
- If the patient has diabetes or high blood sugar: note "finger prick glucose test required"
- If the patient has reduced consciousness or confusion: note "finger prick glucose test required"
- If the patient has a high respiratory rate or breathing difficulty: note "SpO2 check required"
- If glucose is above 11 mmol/L: note "urine dipstick for ketones required"

IMPORTANT EXAMPLES:
- "Period cramps, pain 6/10" → Yellow (moderate pain)
- "Period cramps, pain 3/10" → Green (mild pain)
- "Small cut on the arm, controlled bleeding" → Green
- "Dislocated finger" → Yellow
- "Closed fracture of wrist, pain 5/10" → Yellow
- "Abdominal pain, no vomiting, pain 4/10" → Yellow
- "Sore throat, mild headache" → Green or Blue

After this question (or if pain is already known), output the final JSON with Yellow, Green, or Blue ONLY.

{_FINAL_JSON_SCHEMA}
"""

    # Step 8: force decision
    if step == "decide":
        pain_score = collected.get("pain_score")
        complaint  = collected.get("complaint", "not stated")
        return preamble + f"""
YOUR ONLY JOB THIS TURN:
Maximum questions reached. Output the final triage JSON NOW based on all information collected.
Do not ask another question.

Complaint: {complaint}
Pain: {f"{pain_score}/10" if pain_score is not None else "not stated — treat as mild if no indication of severity"}

CRITICAL: Assign the LOWEST (least urgent) acuity colour that is consistent with the evidence.
Do NOT escalate to a higher priority without specific clinical justification from what the patient
has actually said. When in doubt, assign Green.

Examples of correct conservative assignment:
- Period cramps, moderate pain → Yellow (if pain 4-7) or Green (if pain 1-3)
- Small cut, minor bleeding → Green
- Cold/flu symptoms → Green or Blue
- Headache without red flags → Green or Blue

{_FINAL_JSON_SCHEMA}
"""

    # Fallback
    return preamble + f"\nOutput the final triage JSON now.\n{_FINAL_JSON_SCHEMA}"


# Call local Ollama with `system_prompt` and `history`; return assistant text or None.
def call_ollama(history: list, system_prompt: str):
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model":    "llama3",
                "messages": [{"role": "system", "content": system_prompt}] + history,
                "stream":   False,
            },
            timeout=60,
        )
        return resp.json()["message"]["content"]
    except Exception as e:
        print(f"Ollama error: {e}")
        return None


# Parse the final triage JSON block produced by the LLM; return dict or None.
def extract_final_json(text: str):
    match = re.search(r'\{[^{}]*"confirmed"\s*:\s*true[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


# Complaint pre-classifier — map initial complaint to a severity band.
# Skip red/orange for clearly minor presentations (go to pain_yellow).

# Complaints that are almost never Red or Orange — skip directly to pain_yellow
_MINOR_COMPLAINT_PATTERNS = [
    r"\bperiod\s+(?:pain|cramp\w*|ache|bleed\w*)\b",
    r"\bmenstrual\s+(?:pain|cramp\w*|bleed\w*)\b",
    r"\bdysmenorrh\w+",
    r"\bsmall\s+cut\b",
    r"\bminor\s+cut\b",
    r"\bsuperficial\s+(?:cut|wound|lacerat\w+)\b",
    r"\bgraze\b",
    r"\bpaper\s+cut\b",
    r"\bsore\s+throat\b",
    r"\bcold\s+(?:and|or|symptoms?|and\s+flu)\b",
    r"\bcommon\s+cold\b",
    r"\brunny\s+nose\b",
    r"\bcough\b(?!.*blood)",           # cough but NOT coughing blood
    r"\bearache\b|\bear\s+(?:pain|ache|infection)\b",
    r"\bmild\s+headache\b",
    r"\btension\s+headache\b",
    r"\binsect\s+bite\b|\bbug\s+bite\b",
    r"\bsplinter\b",
    r"\bblister\b",
    r"\bsprain\w*\b",
    r"\btwisted\s+(?:ankle|knee|wrist)\b",
    r"\bstub\w*\s+(?:toe|finger)\b",
    r"\bbruise\b",
    r"\bhayfever\b|\ballerg\w+\s+(?:symptom|rhinitis)\b",
    r"\bconstipat\w+\b",
    r"\bdiarrhoea\b(?!.{0,30}(?:blood|severe|extreme|dehydrat))",
    r"\bnausea\b(?!.{0,30}(?:blood|severe|collapn|faint))",
    r"\bindigestion\b|\bheartburn\b|\bacid\s+reflux\b",
    r"\bUTI\b|\burinary\s+tract\s+infection\b|\bstinging\s+when\s+(?:peeing|urinating)\b",
    r"\bpiles\b|\bhaemorrhoid\w*\b",
    r"\bmouth\s+ulcer\b",
    r"\btoothache\b(?!.{0,30}(?:jaw|face\s+swelling|neck))",
    r"\bskin\s+rash\b(?!.{0,20}(?:purpur|non.{0,5}blanch|spreading\s+fast))",
    r"\beczema\b|\bpsoriasis\b|\bacne\b",
    r"\bcut\s+(?:my|the|a|his|her)\s+\w+\s+(?:slightly|a\s+bit|a\s+little|small\w*|minor\w*)\b",
    r"\bsmall\s+(?:wound|bleed\w*|injur\w*)\b",
    r"\bnicked\b|\bnick\s+(?:my|the)\b",
    r"\bgot\s+a\s+cut\b|\bcutting\s+(?:my|the)\b",
]

# Complaints where Red/Orange is plausible — run the full screen
_CRITICAL_COMPLAINT_PATTERNS = [
    r"\bchest\s+pain\b",
    r"\bbreath\w*\s+(?:difficult\w*|problem\w*|trouble\w*)\b|difficult\w*\s+breath\w*",
    r"can'?t\s+breath\w*|cannot\s+breath\w*|trouble\s+breath\w*|struggling\s+to\s+breath\w*",
    r"\bseiz\w+\b|\bfit\w*\b",
    r"\bstroke\b|face\s+droop",
    r"\bcollapn?\w*\b",
    r"\bfaint\w*\b|\blockedout\b",
    r"\bpalpitation\w*\b|\bheart\s+(?:racing|pounding)\b",
    r"\bsevere\s+(?:headache|pain|bleed\w*)\b",
    r"\bhead\s+injur\w+\b",
    r"\bburn\w*(?!.{0,10}heartburn)\b",
    r"\boverdo\w+\b|\bpoisoning\b",
    r"\bbleeding\s+(?:heavily|badly|a\s+lot)\b",
]


# Classify an initial complaint into 'minor', 'critical', or 'moderate'.
def classify_complaint(complaint: str) -> str:
    if not complaint:
        return "moderate"
    t = complaint.lower()
    for pattern in _CRITICAL_COMPLAINT_PATTERNS:
        if re.search(pattern, t, re.IGNORECASE):
            return "critical"
    for pattern in _MINOR_COMPLAINT_PATTERNS:
        if re.search(pattern, t, re.IGNORECASE):
            return "minor"
    return "moderate"




# Decide the next `triage_step` based on collected data and current step.
def advance_step(current_step: str, collected: dict) -> str:
    if not collected.get("age") or not collected.get("gender"):
        return "demographics"

    complaint_band = classify_complaint(collected.get("complaint", ""))

    if current_step in ("demographics", None):
        if not collected.get("complaint"):
            return "complaint"
        if complaint_band == "minor":
            collected["red_cleared"]    = True
            collected["orange_cleared"] = True
            return "pain_yellow"
        return "red_screen"

    if current_step == "complaint":
        # Re-classify now that the complaint is known
        complaint_band = classify_complaint(collected.get("complaint", ""))
        if complaint_band == "minor":
            collected["red_cleared"]    = True
            collected["orange_cleared"] = True
            return "pain_yellow"
        return "red_screen"

    if current_step == "red_screen":
        return "orange_screen" if collected.get("red_cleared") else "red_screen"

    if current_step == "orange_screen":
        return "pain_yellow" if collected.get("orange_cleared") else "orange_screen"

    if current_step == "pain_yellow":
        return "decide"

    return "decide"


# Response helpers: build final or question responses
# `complete_response` returns the final JSON payload when triage is finished.
# Fields:
# - `status`: 'complete' for final decisions
# - `colour`, `department`, `reason`: the triage result visible to downstream
#   systems or a UI
# - `collected`, `history`: debugging/state useful for logs or the frontend
def complete_response(colour, reason, dept, collected, history, note=None):
    r = {
        "status":      "complete",
        "mts_level":   MTS_LEVELS.get(colour, "Urgent"),
        "colour":      colour,
        "department":  dept,
        "description": MTS_DESCRIPTIONS.get(colour, reason),
        "reason":      reason,
        "demo_mode":   collected.get("bert_demo", False),
        "collected":   collected,
        "history":     history,
        "triage_step": "done",
    }
    if note:
        r["note"] = note
    return r


def question_response(question, collected, history):
    # `question_response` is returned when we need another user answer.
    # The front-end should display `question` to the user and post the reply
    # back to `/assess` with the same `collected` and updated `history`.
    return {
        "status":      "needs_info",
        "question":    question,
        "collected":   collected,
        "history":     history,
        "triage_step": collected.get("triage_step"),
    }


# Flask routes / HTTP endpoints
#
# Endpoints:
# - `GET /` serves a small web UI (templates/index.html).
# - `POST /assess` is the core API. Expected JSON body:
#     { "transcript": "user text", "collected": {...}, "history": [...] }
#   The server returns either a `needs_info` response (ask one more question)
#   or a `complete` triage JSON. The function follows the 8-step triage flow
#   described in the module header: extract demographics → background hints
#   → instant regex gates → LLM questions → final decision.

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/assess", methods=["POST"])
def assess():
    data       = request.json
    transcript = data.get("transcript", "").strip()
    collected  = data.get("collected", {})
    history    = data.get("history", [])

    if not transcript:
        return jsonify({"error": "No transcript provided."}), 400

    # `transcript` is the newest user message (string).
    # `collected` is a small dict tracking captured facts and flags between
    # turns (age, gender, complaint, pain_score, triage_step, etc.). The
    # client should persist and return it on each request so the server can
    # continue the triage conversation statefully.
    # `history` is the conversation history (list of role/content dicts)
    # used to reconstruct recent assistant questions and prior answers.

    # Step 1 — Demographics (passive extraction)

    # Detect what the last assistant question was about, so bare answers work
    _last_assistant = next(
        (m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"), ""
    ).lower()
    _asked_age    = any(p in _last_assistant for p in ["how old are you", "what is your age",
                                                        "your age", "old are you"])
    _asked_gender = any(p in _last_assistant for p in ["male or female", "gender", "are you male",
                                                        "are you female", "your sex"])
    _asked_both   = any(p in _last_assistant for p in ["age and whether", "age and if",
                                                        "age and are you"])

    if not collected.get("age"):
        age = extract_age(transcript, context_is_age_question=(_asked_age or _asked_both))
        if age:
            collected["age"] = age

    if not collected.get("gender"):
        gender = extract_gender(transcript, context_is_gender_question=(_asked_gender or _asked_both))
        if not gender and history:
            # Also scan prior user messages without context flag
            gender = extract_gender(
                " ".join(m.get("content", "") for m in history if m.get("role") == "user")
            )
        if gender:
            collected["gender"] = gender

    # Pain score: extract passively from any message
    if collected.get("pain_score") is None:
        ps = extract_pain_score(transcript)
        if ps is not None:
            collected["pain_score"] = ps

    # Step 2 — AI background (first message)
    if not collected.get("bert_done"):
        bert_result = run_bert(transcript)
        icd_result  = icd_lookup(transcript)
        collected.update({
            "bert_dept":      bert_result["department"],
            "bert_demo":      bert_result["demo"],
            "icd_result":     icd_result,
            "bert_done":      True,
            "question_count": 0,
        })
        # Try to grab the complaint from the first message (strip pure demographics)
        non_demo = re.sub(
            r"\b(\d{1,3}\s*(?:year|yo)\w*|male|female|man|woman|boy|girl|"
            r"i'?m|my\s+name\s*is|age\s*:?\s*\d+)\b",
            "", transcript, flags=re.IGNORECASE,
        ).strip(" ,.")
        if len(non_demo) > 10:
            collected["complaint"] = transcript

        # Initialise step position
        if not collected.get("triage_step"):
            if not collected.get("age") or not collected.get("gender"):
                collected["triage_step"] = "demographics"
            elif not collected.get("complaint"):
                collected["triage_step"] = "complaint"
            else:
                _band = classify_complaint(collected.get("complaint", ""))
                if _band == "minor":
                    collected["red_cleared"]    = True
                    collected["orange_cleared"] = True
                    collected["triage_step"]    = "pain_yellow"
                else:
                    collected["triage_step"] = "red_screen"

        history = [{"role": "user", "content": transcript}]
    else:
        # Subsequent messages
        history.append({"role": "user", "content": transcript})

        # If we were waiting for the complaint, capture this reply
        if collected.get("triage_step") == "complaint" and not collected.get("complaint"):
            collected["complaint"] = transcript

    bert_dept  = collected["bert_dept"]
    icd_result = collected["icd_result"]

    # Step 3 — Instant Red gate (regex checks)
    all_user_text = " ".join(
        m.get("content", "") for m in history if m.get("role") == "user"
    )
    red_reason = hard_red_check(all_user_text)
    if red_reason:
        collected["triage_step"] = "done"
        return jsonify(complete_response(
            "Red", red_reason, bert_dept, collected, history,
            note="⚠️ Immediate emergency detected. Please go to the resuscitation area NOW.",
        ))

    # Step 6 — Instant Orange gate (assist; active after demographics/complaint)
    current_step = collected.get("triage_step", "demographics")
    if current_step not in ("demographics", "complaint"):
        orange_reason = hard_orange_check(all_user_text)
        if orange_reason:
            collected["triage_step"] = "done"
            return jsonify(complete_response(
                "Orange", orange_reason, bert_dept, collected, history,
            ))

    # Step 1 (gate) — Ask for missing demographics directly (no LLM)

    # Return True if the most recent assistant message contains this phrase.
    def _last_assistant_said(phrase: str) -> bool:
        last = next(
            (m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"), ""
        )
        return phrase.lower() in last.lower()

    # Append question to history only if it differs from the last assistant message.
    def _ask(q: str) -> object:
        last = next(
            (m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"), ""
        )
        if last.strip() != q.strip():
            history.append({"role": "assistant", "content": q})
        collected["triage_step"] = "demographics"
        return jsonify(question_response(q, collected, history))

    if not collected.get("age") and not collected.get("gender"):
        return _ask("Before I ask about your symptoms, could you tell me your age and whether you're male or female?")

    if not collected.get("age"):
        return _ask("How old are you?")

    if not collected.get("gender"):
        return _ask("Are you male or female?")

    # Advance step now that demographics are confirmed
    collected["triage_step"] = advance_step(current_step, collected)

    # Check if a prior assistant message already contains a final decision
    last_assistant_content = next(
        (m["content"] for m in reversed(history) if m.get("role") == "assistant"), ""
    )
    if last_assistant_content:
        last_final = extract_final_json(last_assistant_content)
        if last_final:
            colour = last_final.get("colour", "Yellow")
            collected["triage_step"] = "done"
            note = "⚠️ Immediate emergency. Please proceed to resuscitation NOW." if colour == "Red" else None
            return jsonify(complete_response(
                colour,
                last_final.get("reason", MTS_DESCRIPTIONS.get(colour, "")),
                last_final.get("department", bert_dept),
                collected, history, note=note,
            ))

    # Question cap — force decision after MAX_QUESTIONS
    q_count = collected.get("question_count", 0)
    if q_count >= MAX_QUESTIONS:
        collected["triage_step"] = "decide"

    # Steps 4-8 — Ollama conversation (build a step-specific prompt and call LLM)
    ollama_step   = collected["triage_step"]
    system_prompt = build_ollama_prompt(ollama_step, collected, bert_dept, icd_result)
    ollama_text   = call_ollama(history, system_prompt)

    # Ollama unavailable: Yellow fallback
    if ollama_text is None:
        collected["triage_step"] = "done"
        return jsonify(complete_response(
            "Yellow",
            "Ollama unavailable — defaulting to Urgent. Please see a clinician.",
            bert_dept, collected, history,
            note="Clinical assessment system unavailable. A nurse will see you shortly.",
        ))

    # Did Ollama reach a final decision?
    final = extract_final_json(ollama_text)
    if final:
        colour = final.get("colour", "Yellow")
        # Mark cleared flags so state is consistent for any subsequent calls
        if colour not in ("Red", "Orange"):
            if ollama_step == "red_screen":
                collected["red_cleared"] = True
            if ollama_step == "orange_screen":
                collected["orange_cleared"] = True
        collected["triage_step"] = "done"
        note = "⚠️ Immediate emergency. Please proceed to resuscitation NOW." if colour == "Red" else None
        return jsonify(complete_response(
            colour,
            final.get("reason", MTS_DESCRIPTIONS.get(colour, "")),
            final.get("department", bert_dept),
            collected, history, note=note,
        ))

    # Ollama asked a follow-up question — update state and loop back
    if ollama_step == "red_screen":
        collected["red_cleared"] = True       # asking means no Red confirmed
    if ollama_step == "orange_screen":
        collected["orange_cleared"] = True    # asking means no Orange confirmed

    collected["question_count"] = q_count + 1
    history.append({"role": "assistant", "content": ollama_text})
    collected["triage_step"] = advance_step(ollama_step, collected)

    return jsonify(question_response(ollama_text, collected, history))


if __name__ == "__main__":
    app.run(debug=True, port=5000)