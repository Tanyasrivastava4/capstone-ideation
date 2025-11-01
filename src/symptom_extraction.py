#src/symptom_extraction.py
#simple keyword-based extraction for demo

SYMPTOMS = [
    "chest pain", "shortness of breath", "breathlessness", "fever","cough","headache","dizziness", "nausea","vomiting", "throat",
    "sore throat","rash", "diarrhea", "fatigue", "weakness", "loss of consciousness"
]

def extract_symptoms(text: str):
    txt = (text or "").lower()
    found = []
    for k in SYMPTOMS:
        if k in txt:
            found.append(k)

    #unique preserving order
    out = []
    seen = set()
    for s in found:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out