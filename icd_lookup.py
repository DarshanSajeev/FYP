import requests
import re
from dotenv import load_dotenv
import os

load_dotenv()

# ICD-11 code prefix → LGI Department
CODE_TO_DEPARTMENT = {
    # Cardiovascular
    "MD3": "Cardiology",
    "MD4": "Cardiology",
    "MC8": "Cardiology",
    "MC9": "Cardiology",
    "BA8": "Cardiology",
    "BA9": "Cardiology",

    # Respiratory
    "MD1": "Respiratory Medicine",
    "MD2": "Respiratory Medicine",
    "CB":  "Respiratory Medicine",
    "CA":  "Respiratory Medicine",

    # Neurosciences
    "MB4": "Neurosciences",
    "MB5": "Neurosciences",
    "MB6": "Neurosciences",
    "8A":  "Neurosciences",
    "8B":  "Neurosciences",
    "8C":  "Neurosciences",

    # Abdominal/GI
    "MD8": "Abdominal Medicine & Surgery",
    "MD9": "Abdominal Medicine & Surgery",
    "ME0": "Abdominal Medicine & Surgery",
    "DA":  "Abdominal Medicine & Surgery",
    "DB":  "Abdominal Medicine & Surgery",
    "DC":  "Abdominal Medicine & Surgery",

    # Musculoskeletal/Trauma
    "ME8": "Trauma & Orthopaedics",
    "ME9": "Trauma & Orthopaedics",
    "FB":  "Trauma & Orthopaedics",
    "FC":  "Trauma & Orthopaedics",
    "FA":  "Trauma & Orthopaedics",

    # Haematology
    "3A":  "Clinical Haematology",
    "3B":  "Clinical Haematology",

    # Genitourinary
    "MF4": "General Medicine",
    "MF5": "General Medicine",
    "MF6": "General Medicine",

    # Gynaecology/Maternity
    "GA":  "Maternity",
    "GB":  "Maternity",
    "HA":  "Maternity",
    "JA":  "Maternity",
    "JB":  "Maternity",

    # Cancer
    "2B":  "Leeds Cancer Centre",
    "2C":  "Leeds Cancer Centre",
    "2D":  "Leeds Cancer Centre",

    # Ear/Audiology
    "AB":  "Audiology",
    "AA":  "Audiology",

    # Endocrine
    "5A":  "General Medicine",
    "5B":  "General Medicine",
    "5C":  "General Medicine",
    "5D":  "General Medicine",
}

# Chapter-level fallback
CHAPTER_TO_DEPARTMENT = {
    "1":  "A&E",
    "2":  "Leeds Cancer Centre",
    "3":  "Clinical Haematology",
    "4":  "General Medicine",
    "5":  "General Medicine",
    "6":  "General Medicine",
    "7":  "General Medicine",
    "8":  "Neurosciences",
    "9":  "General Medicine",
    "10": "Audiology",
    "11": "Cardiology",
    "12": "Respiratory Medicine",
    "13": "Abdominal Medicine & Surgery",
    "14": "General Medicine",
    "15": "Trauma & Orthopaedics",
    "16": "Maternity",
    "17": "General Medicine",
    "18": "Maternity",
    "19": "General Medicine",
    "20": "General Medicine",
    "21": "A&E",
    "22": "Trauma & Orthopaedics",
    "23": "A&E",
}

# Map ICD-11 code and chapter to LGI department.
def get_department(code, chapter):
    # Try prefix match from longest to shortest
    for length in [3, 2, 1]:
        prefix = code[:length]
        if prefix in CODE_TO_DEPARTMENT:
            return CODE_TO_DEPARTMENT[prefix]
    
    # Fall back to chapter
    return CHAPTER_TO_DEPARTMENT.get(str(chapter), "A&E")

# Query the WHO ICD-11 API and return matched condition, code, chapter, score, and department.
def get_triage_result(symptom_text, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "API-Version": "v2",
        "Accept-Language": "en"
    }

    url = "https://id.who.int/icd/release/11/2024-01/mms/search"
    params = {
        "q": symptom_text,
        "flatResults": True,
        "useFlexisearch": True
    }
    
    response = requests.get(url, params=params, headers=headers)
    results = response.json()

    for entity in results.get("destinationEntities", []):
        if entity.get("isResidualUnspecified") or entity.get("isResidualOther"):
            continue

        code = entity.get("theCode")
        chapter = entity.get("chapter")
        title = re.sub(r'<[^>]+>', '', entity.get("title", ""))
        score = entity.get("score", 0)

        if not code:
            continue

        department = get_department(code, chapter)

        return {
            "symptom_input":     symptom_text,
            "matched_condition": title,
            "icd_code":          code,
            "chapter":           chapter,
            "score":             round(score, 3),
            "department":        department
        }

    # Nothing matched
    return {
        "symptom_input":     symptom_text,
        "matched_condition": "Unknown",
        "icd_code":          None,
        "chapter":           None,
        "score":             0,
        "department":        "A&E"
    }