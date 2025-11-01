"""
department_predictor.py
---------------------------------
Predicts the medical department for a patient case using Mistral LLM.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ✅ Load Mistral model from Hugging Face
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

try:
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
except Exception as e:
    print(f"⚠️ Could not load Mistral model: {e}")
    generator = None


def predict_department(report_text, symptoms):
    """
    Use Mistral LLM to predict which medical department the case belongs to.

    Args:
        report_text (str): The cleaned text of patient report.
        symptoms (list): Extracted symptom list.

    Returns:
        str: Predicted department (e.g., 'Cardiology', 'Neurology', 'Orthopedics', etc.)
    """
    if not report_text:
        return "General Medicine"

    if generator is None:
        print("⚠️ Model not loaded. Returning default department.")
        return "General Medicine"

    # 🧠 Construct a clear prompt for Mistral
    prompt = f"""
    You are a medical triage assistant.
    Based on the following patient report and symptoms, predict the most relevant hospital department.

    Report: {report_text}
    Symptoms: {', '.join(symptoms) if symptoms else 'Not provided'}

    Choose one department from this list:
    [Cardiology, Neurology, Orthopedics, Pulmonology, Gastroenterology, Dermatology, Psychiatry, General Medicine]

    Respond with only the department name.
    """

    try:
        response = generator(
            prompt,
            max_new_tokens=50,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )[0]["generated_text"]

        # 🧩 Clean and extract department from model output
        department = extract_department_name(response)
        return department

    except Exception as e:
        print(f"⚠️ Department prediction failed: {e}")
        return "General Medicine"


def extract_department_name(response_text):
    """
    Clean and extract department name from the generated LLM output.
    """
    departments = [
        "Cardiology", "Neurology", "Orthopedics", "Pulmonology",
        "Gastroenterology", "Dermatology", "Psychiatry", "General Medicine"
    ]
    for dept in departments:
        if dept.lower() in response_text.lower():
            return dept
    return "General Medicine"


# ✅ Example test
if __name__ == "__main__":
    sample_report = "Patient complains of chest tightness and shortness of breath while walking."
    sample_symptoms = ["chest tightness", "shortness of breath"]
    result = predict_department(sample_report, sample_symptoms)
    print("🏥 Predicted Department:", result)
