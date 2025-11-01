"""
Runs the full medical triage system pipeline:
1. Load and clean data
2. Extract symptoms
3. Predict urgency & department
4. Retrieve similar cases (RAG)
5. Generate final report (Mistral)
6. Save outputs to CSV
"""

import pandas as pd
from src.data_loader import load_data
from src.preprocessing import clean_text
from src.symptom_extraction import extract_symptoms
from src.model_predictor import predict_case
from src.rag_summary import get_similar_cases
from src.report_generator import generate_report
from src.human_review import flag_for_review

OUTPUT_RESULTS = "outputs/results.csv"
OUTPUT_TICKETS = "outputs/tickets.csv"

def run_full_pipeline(data_path="data/raw/patient_reports.csv"):
    # Step 1: Load and clean
    df = load_data(data_path)
    df["cleaned_text"] = df["report_text"].apply(clean_text)

    # Step 2: Extract symptoms
    df["symptoms"] = df["cleaned_text"].apply(extract_symptoms)

    # Step 3: Predict urgency, department, and confidence
    preds = df["cleaned_text"].apply(predict_case)
    df["urgency"] = preds.apply(lambda x: x["urgency"])
    df["department"] = preds.apply(lambda x: x["department"])
    df["confidence"] = preds.apply(lambda x: x["confidence"])

    # Step 4: Retrieve similar cases
    df["similar_cases"] = df["cleaned_text"].apply(lambda t: get_similar_cases(t, top_k=3))

    # Step 5: Generate reports (using Mistral for text generation)
    reports = []
    for idx, row in df.iterrows():
        rep = generate_report(
            patient_id=row.get("patient_id", idx),
            report_text=row["report_text"],
            predicted_urgency=row["urgency"],
            symptoms=row["symptoms"],
            confidence=row["confidence"]
        )
        reports.append(rep)
    df["final_report"] = reports

    # Step 6: Flag for human review if confidence is low
    tickets = []
    for idx, row in df.iterrows():
        flag = flag_for_review(
            row["patient_id"],
            row["confidence"],
            row["final_report"]
        )
        if flag:
            tickets.append(flag)

    # Save outputs
    df.to_csv(OUTPUT_RESULTS, index=False)
    if tickets:
        pd.DataFrame(tickets).to_csv(OUTPUT_TICKETS, index=False)

    print(f"✅ Pipeline completed. Results saved to {OUTPUT_RESULTS}")
    if tickets:
        print(f"⚠️ {len(tickets)} cases sent for human review -> {OUTPUT_TICKETS}")

    return df


if __name__ == "__main__":
    # Run full pipeline on real dataset
    run_full_pipeline()

    # --- Example single test case ---
    example_symptoms = ["chest pain", "shortness of breath"]
    example_text = "Patient complains of chest pain and shortness of breath since morning."

    result = generate_report(
        patient_id=1,
        report_text=example_text,
        predicted_urgency="High",
        symptoms=example_symptoms,
        confidence=0.88
    )
    print("\n🧠 Example Generated Report:\n", result)












##scripts/run_pipeline.py
#
#import os
#import pandas as pd
#from src.data_loader import load_dataset
#from src.preprocessing import preprocess_text
#from src.symptom_extraction import extract_symptoms
#from src.model_predictor import predict_record
#from src.rag_summary import get_similar_cases, build_embeddings
#from src.report_generator import suggested_next_steps, save_result
#from src.human_review import maybe_create_ticket
#from datetime import datetime
#
#def run_pipeline(input_csv="data/raw/patient_reports.csv", save_cleaned=True):
#    os.makedirs("outputs", exist_ok=True)
#    os.makedirs("data/processed", exist_ok=True)
#
#
#    df =  load_dataset(input_csv)
#    #Preprocess and save cleaned
#    df["cleaned_text"] = df["report_text"].apply(preprocess_text)
#    if save_cleaned:
#        os.makedirs("data/processed",exist_ok=True)
#        df.to_csv("data/processed/cleaned_data.csv", index=False)
#        print("Saved cleaned data to data/processed/cleaned_data.csv")
#
#    results = []
#
#    #iterate records
#    for idx,row in df.iterrows():
#        pid = row.get("patient_id",idx)
#        raw = row.get("cleaned_text","")
#        symptoms = extract_symptoms(raw)
#        pred = predict_record(raw, symptoms, age = row.get("age"), gender= row.get("gender"))
#        #get similar cases (may be empty)
#        similar = get_similar_cases(raw, top_k=3)
#        similar_join = " || ".join(similar) if similar else ""
#        next_steps = suggested_next_steps(",".join(symptoms) or raw[:200], pred["department"], pred["urgency"])
#
#        rec = {
#            "patient_id": pid,
#            "age": row.get("age"),
#            "gender":row.get("gender"),
#            "report_text": raw,
#            "symptoms": ": ".join(symptoms),
#            "urgency": pred["department"],
#            "department": pred["department"],
#            "next_steps": next_steps,
#            "confidence": pred["confidence"],
#            "similar_cases":similar_join,
#            "processed_at": datetime.now().isoformat()
#        }
#
#        # save per-record summary
#        save_result(rec)
#        maybe_create_ticket(rec, threshold=0.6)
#        results.append(rec)
#
#        #After all, ensure embeddings built for future RAG
#        try:
#            build_embeddings()
#            print("Built/updated embeddings for RAG.")
#        except Exception:
#            print("could not build embeddings(sentence-transformers not installed).")
#
#        print(f"Processed {len(results)} records. Results saved too outputs/results.csv")
#
#
#if __name__ == "__main__":
#    run.pipeline()