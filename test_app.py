import unittest
import json

# Import the functions to test from your main app file
# (Assuming your main file is named app.py)
from app import (
    extract_age, 
    extract_gender, 
    extract_pain_score, 
    hard_red_check, 
    hard_orange_check,
    classify_complaint,
    extract_final_json
)

class TestTriageLogic(unittest.TestCase):

    # Demographic extraction tests
    def test_age(self):
        self.assertEqual(extract_age("I am 25 years old"), 25)
        self.assertEqual(extract_age("age: 42"), 42)
        self.assertEqual(extract_age("I'm 9yo"), 9)
        # Test context awareness (bare number when asked)
        self.assertEqual(extract_age("55", context_is_age_question=True), 55)
        # Test failure case
        self.assertIsNone(extract_age("I am feeling sick today"))

    def test_gender(self):
        self.assertEqual(extract_gender("I am a pregnant woman"), "female")
        self.assertEqual(extract_gender("he has been coughing"), "male")
        # Test context awareness
        self.assertEqual(extract_gender("f", context_is_gender_question=True), "female")
        self.assertEqual(extract_gender("m", context_is_gender_question=True), "male")
        # Test failure case
        self.assertIsNone(extract_gender("My stomach hurts"))

    def test_pain_score(self):
        self.assertEqual(extract_pain_score("My pain is 8/10"), 8)
        self.assertEqual(extract_pain_score("about a 6"), 6)
        self.assertEqual(extract_pain_score("worst pain ever"), 10)
        self.assertEqual(extract_pain_score("mild pain"), 2)
        self.assertIsNone(extract_pain_score("I just feel dizzy"))

    # Safety gate tests (Red/Orange screens)
    def test_hard_red(self):
        # Should trigger immediate Red
        self.assertEqual(hard_red_check("my dad has no pulse"), "Cardiac arrest")
        self.assertEqual(hard_red_check("they are not breathing"), "Not breathing — airway emergency")
        # Should NOT trigger Red (safely bypasses to next step)
        self.assertIsNone(hard_red_check("I have a bad cough"))
        self.assertIsNone(hard_red_check("I cut my finger, it is bleeding")) 

    def test_hard_orange(self):
        # Should trigger immediate Orange
        self.assertEqual(hard_orange_check("I have severe chest pain"), "Chest pain")
        self.assertEqual(hard_orange_check("my face is drooping"), "Stroke / focal neurology (FAST)")
        # Should NOT trigger Orange
        self.assertIsNone(hard_orange_check("I have a mild headache"))

    # Complaint pre-classification tests
    def test_classify_complaint(self):
        # Critical complaints
        self.assertEqual(classify_complaint("I can't breathe"), "critical")
        self.assertEqual(classify_complaint("seizure"), "critical")
        
        # Minor complaints (Should bypass Red/Orange screens)
        self.assertEqual(classify_complaint("I have period cramps"), "minor")
        self.assertEqual(classify_complaint("just a common cold"), "minor")
        self.assertEqual(classify_complaint("small cut on my arm"), "minor")
        
        # Moderate complaints (Default pathway)
        self.assertEqual(classify_complaint("my stomach hurts a lot"), "moderate")

    # LLM output parsing tests
    def test_extract_final_json(self):
        # Test ideal JSON extraction
        clean_json = '{"confirmed": true, "colour": "Green", "department": "A&E", "reason": "test"}'
        self.assertIsNotNone(extract_final_json(clean_json))
        
        # Test messy LLM output (hallucinated conversational filler before/after JSON)
        messy_llm_output = '''
        Based on the patient's symptoms, here is the final triage result:
        {
            "confirmed": true,
            "colour": "Yellow",
            "department": "General Medicine",
            "reason": "Moderate pain with vomiting."
        }
        Please let me know if you need anything else!
        '''
        extracted = extract_final_json(messy_llm_output)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["colour"], "Yellow")
        
        # Test failure case (LLM completely fails to provide valid JSON)
        self.assertIsNone(extract_final_json("I think the patient is Green."))

if __name__ == '__main__':
    unittest.main()