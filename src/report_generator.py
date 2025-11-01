import os
import pandas as pd
from datetime import datetime

def generate_report(
    patient_data: pd.DataFrame,
    symptoms_list: list,
    urgency_preds: list,
    department_preds: list,
    similar_cases: list,
    flagged_cases: list,
    output_dir: str = "outputs"
):
    """
    Generate a final triage report combining all predictions and saving to CSV.

    Args:
        patient_data (pd.DataFrame): Original patient reports (from patient_reports.csv)
        symptoms_list (list): Extracted symptoms from DistilBERT NER
        urgency_preds (list): Urgency predictions from DistilBERT classifier
        department_preds (list): Department predictions from Mistral LLM
        similar_cases (list): Retrieved similar cases (from RAG summary)
        flagged_cases (list): IDs or indexes of low-confidence cases
        output_dir (str): Directory where report will be saved
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Combine all results into a DataFrame
    results_df = pd.DataFrame({
        "Patient_ID": patient_data["Patient_ID"],
        "Patient_Report": patient_data["Report"],
        "Extracted_Symptoms": symptoms_list,
        "Urgency_Prediction": urgency_preds,
        "Suggested_Department": department_preds,
        "Similar_Cases": similar_cases,
        "Flagged_for_Human_Review": [
            "Yes" if idx in flagged_cases else "No" for idx in range(len(patient_data))
        ]
    })

    # Add timestamp for version tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"results_{timestamp}.csv")

    # Save results
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ Triage report successfully generated and saved at: {output_path}")
    print("\n📄 Summary of results:")
    print(results_df.head())

    return results_df


# Example usage (only when running this file directly)
if __name__ == "__main__":
    # Dummy example to test output structure
    df = pd.DataFrame({
        "Patient_ID": [1, 2],
        "Report": [
            "Patient complains of severe chest pain and breathlessness.",
            "Patient reports mild fever and body ache."
        ]
    })
    symptoms = [["chest pain", "breathlessness"], ["fever", "body ache"]]
    urgency = ["High", "Low"]
    dept = ["Cardiology", "General Medicine"]
    similar = [
        "Matched with case ID 101 - similar chest pain symptom.",
        "Matched with case ID 210 - mild viral fever case."
    ]
    flagged = [0]  # Index of first patient

    generate_report(df, symptoms, urgency, dept, similar, flagged)























##src/report_generator.py
#
#import os
#import pandas as pd
#from datetime import datetime
#from src.rag_summary import get_similar_cases
#
#KB_PATH = "data/external/medical_kb.csv"
#OUTPUT_PATH = "outputs/results.csv"
#
#
#def load_kb(path=KB_PATH):
#    """Load the medical knowledge base from CSV."""
#    if not os.path.exists(path):
#        raise FileNotFoundError(f"Knowledge base not found at {path}")
#    return pd.read_csv(path)
#
#
#def lookup_kb(symptoms, kb_df):
#    """
#    Match symptoms to departments and actions from KB.
#    Returns most common department and suggested actions.
#    """
#    matched_rows = kb_df[kb_df["symptom"].isin(symptoms)]
#    if matched_rows.empty:
#        return ("General Medicine", "Consult general physician for further advice.")
#    
#    # Pick most frequent department and combine recommendations
#    department = matched_rows["department"].mode().iloc[0]
#    recommendations = " ".join(matched_rows["recommended_action"].tolist())
#    return department, recommendations
#
#
#def generate_report(patient_id, report_text, predicted_urgency, symptoms, confidence=0.9):
#    """
#    Create a structured triage report by combining:
#    - Model predictions
#    - Knowledge base
#    - RAG summaries
#    """
#    kb_df = load_kb()
#    department, recommendations = lookup_kb(symptoms, kb_df)
#
#    # Retrieve similar past cases using RAG
#    similar_cases = get_similar_cases(report_text, top_k=2)
#    similar_text = " | ".join(similar_cases) if similar_cases else "No similar past cases found."
#
#    # Compose final summary
#    summary = {
#        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#        "patient_id": patient_id,
#        "urgency_level": predicted_urgency,
#        "confidence": round(confidence, 2),
#        "department": department,
#        "symptoms_detected": ", ".join(symptoms),
#        "recommendations": recommendations,
#        "similar_cases": similar_text
#    }
#
#    # Save to CSV (append mode)
#    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
#    df = pd.DataFrame([summary])
#    if os.path.exists(OUTPUT_PATH):
#        df.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
#    else:
#        df.to_csv(OUTPUT_PATH, index=False)
#
#    print(f"✅ Report generated and saved for Patient {patient_id}")
#    return summary
#