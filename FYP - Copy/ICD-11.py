import requests
from dotenv import load_dotenv
import os
from icd_lookup import get_triage_result

load_dotenv()

# Get token
token_url = "https://icdaccessmanagement.who.int/connect/token"
token_response = requests.post(token_url, data={
    "client_id": os.getenv("ICD_CLIENT_ID"),
    "client_secret": os.getenv("ICD_CLIENT_SECRET"),
    "scope": "icdapi_access",
    "grant_type": "client_credentials"
})
token = token_response.json()["access_token"]

# Test with a few symptoms
test_symptoms = [
    "chest pain",
    "broken arm",
    "difficulty breathing",
    "seizure",
    "stomach ache"
]

for symptom in test_symptoms:
    result = get_triage_result(symptom, token)
    print(f"\nSymptom   : {result['symptom_input']}")
    print(f"Condition : {result['matched_condition']}")
    print(f"ICD Code  : {result['icd_code']}")
    print(f"Department: {result['department']}")
    print(f"Score     : {result['score']}")