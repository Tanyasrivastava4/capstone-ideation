#src/report_generator.py

import os
import pandas as pd
from datetime import datetime
from src.rag_summary import get_similar_cases

KB_PATH = "data/external/medical_kb.csv"
OUTPUT_PATH = "outputs/results.csv"


def load_kb(path=KB_PATH):
    """Load the medical knowledge base from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledge base not found at {path}")
    return pd.read_csv(path)


def lookup_kb(symptoms, kb_df):
    """
    Match symptoms to departments and actions from KB.
    Returns most common department and suggested actions.
    """
    matched_rows = kb_df[kb_df["symptom"].isin(symptoms)]
    if matched_rows.empty:
        return ("General Medicine", "Consult general physician for further advice.")
    
    # Pick most frequent department and combine recommendations
    department = matched_rows["department"].mode().iloc[0]
    recommendations = " ".join(matched_rows["recommended_action"].tolist())
    return department, recommendations


def generate_report(patient_id, report_text, predicted_urgency, symptoms, confidence=0.9):
    """
    Create a structured triage report by combining:
    - Model predictions
    - Knowledge base
    - RAG summaries
    """
    kb_df = load_kb()
    department, recommendations = lookup_kb(symptoms, kb_df)

    # Retrieve similar past cases using RAG
    similar_cases = get_similar_cases(report_text, top_k=2)
    similar_text = " | ".join(similar_cases) if similar_cases else "No similar past cases found."

    # Compose final summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id": patient_id,
        "urgency_level": predicted_urgency,
        "confidence": round(confidence, 2),
        "department": department,
        "symptoms_detected": ", ".join(symptoms),
        "recommendations": recommendations,
        "similar_cases": similar_text
    }

    # Save to CSV (append mode)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = pd.DataFrame([summary])
    if os.path.exists(OUTPUT_PATH):
        df.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Report generated and saved for Patient {patient_id}")
    return summary
