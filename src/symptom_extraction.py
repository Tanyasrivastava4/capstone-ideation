"""
symptom_extraction.py
---------------------------------
Extracts medical symptoms or entities from patient text using DistilBERT (NER).
"""

from transformers import pipeline

# ✅ Load pretrained DistilBERT NER pipeline
# You can change the model to 'distilbert-base-cased' or a better one later.
# For medical domain, you could use 'Jean-Baptiste/camembert-ner-with-dates' or similar.
ner_model = pipeline("ner", model="distilbert-base-cased", grouped_entities=True)

def extract_symptoms(text):
    """
    Extracts symptoms or key entities from a given text using DistilBERT NER.

    Args:
        text (str): The patient's report text.

    Returns:
        list: A list of extracted symptom/entity strings.
    """
    if not text or not isinstance(text, str):
        return []

    # Run the model
    try:
        ner_results = ner_model(text)
    except Exception as e:
        print(f"⚠️ NER model failed: {e}")
        return []

    # Extract only entity words that look like potential medical terms
    symptoms = []
    for entity in ner_results:
        word = entity.get("word", "").strip()
        entity_group = entity.get("entity_group", "").upper()

        # Keep only named entities that might represent medical terms or symptoms
        if entity_group in ["ORG", "MISC", "PER", "LOC"]:
            # These are general groups — not always useful
            continue
        if word and word.lower() not in symptoms:
            symptoms.append(word.lower())

    return symptoms


# ✅ Example test (can be removed later)
if __name__ == "__main__":
    sample_text = "The patient reports fever, cough, and chest pain for the past three days."
    extracted = extract_symptoms(sample_text)
    print("🩺 Extracted symptoms:", extracted)










##src/symptom_extraction.py
##simple keyword-based extraction for demo
#
#SYMPTOMS = [
#    "chest pain", "shortness of breath", "breathlessness", "fever","cough","headache","dizziness", "nausea","vomiting", "throat",
#    "sore throat","rash", "diarrhea", "fatigue", "weakness", "loss of consciousness"
#]
#
#def extract_symptoms(text: str):
#    txt = (text or "").lower()
#    found = []
#    for k in SYMPTOMS:
#        if k in txt:
#            found.append(k)
#
#    #unique preserving order
#    out = []
#    seen = set()
#    for s in found:
#        if s not in seen:
#            out.append(s)
#            seen.add(s)
#    return out