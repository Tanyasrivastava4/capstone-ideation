"""
Run the full Medical Triage System pipeline.
--------------------------------------------------
Execution flow:
1. Load and validate input data
2. Preprocess patient reports
3. Extract symptoms (DistilBERT NER)
4. Predict urgency (DistilBERT classifier)
5. Predict department (Mistral LLM)
6. Retrieve similar cases (TF-IDF + Mistral)
7. Generate triage report
8. Flag low-confidence cases for human review
9. Save results to outputs/results.csv
"""
import os
import pandas as pd

# ✅ Import from src modules (FIXED NAMES)
from src.data_loader import load_dataset
from src.input_validation import validate_input_file
from src.preprocessing import preprocess_text
from src.symptom_extraction import extract_symptoms
from src.model_predictor import predict_record  # ✅ FIXED: was predict_case
from src.department_predictor import predict_department
from src.rag_summary import get_similar_cases  # ✅ FIXED: was retrieve_similar_cases
from src.human_review import flag_ticket_if_low_conf, notify_admin_summary

# ✅ File paths
DATA_PATH = "data/raw/patient_reports.csv"
OUTPUT_PATH = "outputs/results.csv"
TICKETS_PATH = "outputs/tickets.csv"

def main():
    print("🚀 Starting Medical Triage System Pipeline...")
    
    # Step 1: Validate input
    print("\n✅ Step 1: Validating input file...")
    if not validate_input_file(DATA_PATH):
        print("❌ Validation failed. Exiting.")
        return
    
    # Step 2: Load dataset
    print("\n✅ Step 2: Loading dataset...")
    df = load_dataset(DATA_PATH)
    
    # Step 3: Preprocess text
    print("\n✅ Step 3: Preprocessing text...")
    df["cleaned_text"] = df["report_text"].apply(preprocess_text)
    
    # Step 4: Symptom Extraction
    print("\n✅ Step 4: Extracting symptoms using DistilBERT (NER)...")
    df["symptoms"] = df["cleaned_text"].apply(extract_symptoms)
    
    # Step 5: Urgency + Confidence Prediction
    print("\n✅ Step 5: Predicting urgency using DistilBERT (Classifier)...")
    predictions = df.apply(
        lambda row: predict_record(row["cleaned_text"], row["symptoms"]), 
        axis=1
    )
    df["urgency_predicted"] = predictions.apply(lambda x: x["urgency"])
    df["confidence"] = predictions.apply(lambda x: x["confidence"])
    
    # Step 6: Department Prediction
    print("\n✅ Step 6: Predicting department using Mistral LLM...")
    df["department_predicted"] = df.apply(
        lambda row: predict_department(row["cleaned_text"], row["symptoms"]),
        axis=1
    )
    
    # Step 7: Retrieve similar cases (RAG)
    print("\n✅ Step 7: Retrieving similar cases using TF-IDF + Mistral...")
    df["similar_cases"] = df["cleaned_text"].apply(
        lambda text: get_similar_cases(text, top_k=3)
    )
    
    # Step 8: Flag low-confidence cases for human review
    print("\n✅ Step 8: Flagging low-confidence cases for human review...")
    df.apply(lambda row: flag_ticket_if_low_conf(row, threshold=0.6), axis=1)
    
    # Step 9: Save final results
    print("\n✅ Step 9: Saving results to CSV...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Select columns for final output
    output_cols = [
        "patient_id", "age", "gender", "report_text", 
        "symptoms", "urgency_predicted", "department_predicted", 
        "confidence", "similar_cases"
    ]
    df[output_cols].to_csv(OUTPUT_PATH, index=False)
    
    print("\n🎉 Pipeline execution complete!")
    print(f"📄 Results saved to: {OUTPUT_PATH}")
    
    # Step 10: Show summary of flagged cases
    notify_admin_summary()

if __name__ == "__main__":
    main()
    notify_admin_summary()













#"""
#Run the full Medical Triage System pipeline.
#--------------------------------------------------
#Execution flow:
#1 Load and validate input data
#2 Preprocess patient reports
#3 Extract symptoms (DistilBERT NER)
#4 Predict urgency (DistilBERT classifier)
#5 Predict department (Mistral LLM)
#6 Retrieve similar cases (TF-IDF + Mistral)
#7 Generate triage report (combine KB + predictions)
#8 Flag low-confidence cases for human review
#9 Save results to outputs/results.csv
#"""

#import os
#import pandas as pd

# ✅ Step 1: Import from src modules
#from src.data_loader import load_dataset
#from src.input_validation import validate_input_file
#from src.preprocessing import preprocess_text
#from src.symptom_extraction import extract_symptoms
#from src.model_predictor import predict_urgency
#from src.department_predictor import predict_department
#from src.rag_summary import retrieve_similar_cases
#from src.report_generator import generate_report
#from src.human_review import flag_ticket_if_low_conf, summarize_flagged_cases
#from src.human_review import notify_admin_summary

# ✅ File paths
#DATA_PATH = "data/raw/patient_reports.csv"
#OUTPUT_PATH = "outputs/results.csv"


#def main():
 #   print("🚀 Starting Medical Triage System Pipeline...")

    # Step 1: Validate input
  #  print("\n✅ Step 1: Validating input file...")
   # validate_input_file(DATA_PATH)

    # Step 2: Load dataset
    #print("\n✅ Step 2: Loading dataset...")
    #df = load_dataset(DATA_PATH)

    # Step 3: Preprocess text
    #print("\n✅ Step 3: Preprocessing text...")
    #df["cleaned_text"] = df["report_text"].apply(preprocess_text)

    # Step 4: Symptom Extraction
    #print("\n✅ Step 4: Extracting symptoms using DistilBERT (NER)...")
    #df["symptoms"] = df["cleaned_text"].apply(extract_symptoms)

    # Step 5: Urgency Prediction
    #print("\n✅ Step 5: Predicting urgency using DistilBERT (Classifier)...")
    #urgency_results = df["cleaned_text"].apply(predict_urgency)
    #urgency_df = pd.DataFrame(urgency_results.tolist())
    #df = pd.concat([df, urgency_df], axis=1)

    # Step 6: Department Prediction
    #print("\n✅ Step 6: Predicting department using Mistral LLM...")
    #df["department_predicted"] = df["cleaned_text"].apply(predict_department)

    # Step 7: Retrieve similar cases (RAG)
    #print("\n✅ Step 7: Retrieving similar cases using TF-IDF + Mistral...")
    #df["similar_cases"] = df["cleaned_text"].apply(lambda text: retrieve_similar_cases(text, top_k=2))

    # Step 8: Generate structured triage reports
    #print("\n✅ Step 8: Generating triage reports...")
    #reports = []
    #for _, row in df.iterrows():
     #   report = generate_report(
      #      patient_id=row.get("patient_id", 0),
       #     report_text=row["report_text"],
        #    predicted_urgency=row.get("urgency", "Unknown"),
         #   symptoms=row.get("symptoms", []),
          #  department=row.get("department_predicted", "General Medicine"),
          #  confidence=row.get("confidence", 0.9)
        #)
        #reports.append(report)

    # Step 9: Flag low-confidence cases
    #print("\n✅ Step 9: Flagging low-confidence cases for human review...")
    #for _, row in df.iterrows():
     #   flag_ticket_if_low_conf(row, threshold=0.6)

    # Step 10: Summarize flagged cases
    #print("\n✅ Step 10: Summarizing flagged cases...")
    #summarize_flagged_cases()

    # Step 11: Save final output
    #print("\n✅ Step 11: Saving final output to CSV...")
    #os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    #pd.DataFrame(reports).to_csv(OUTPUT_PATH, index=False)

    #print("\n🎉 Pipeline execution complete!")
    #print(f"📄 Results saved to: {OUTPUT_PATH}")


#if __name__ == "__main__":
 #   main()
 #   notify_admin_summary()
