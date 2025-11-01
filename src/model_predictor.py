# src/model_predictor.py
"""
Prediction module. If a fine-tuned HF model is available, use it.
Otherwise use a simple rule-based fallback.
"""
import os
import random

HF_MODEL_PATH = os.environ.get("TRIAGE_MODEL_PATH", None)  # e.g. path or HF repo id

# Try to import transformers pipeline if available
try:
    from transformers import pipeline
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

# Rule-based fallback
def _rule_based_predict(text, symptoms, age=None, gender=None):
    txt = (text or "").lower()
    urgency = "Low"
    department = "General Medicine"
    next_steps_hint = "Monitor symptoms; seek care if condition worsens."

    if any(k in txt for k in ["chest pain", "shortness of breath", "loss of consciousness", "severe bleed", "blood in vomit"]):
        urgency = "Emergency"
        department = "Cardiology" if "chest" in txt else "Emergency"
        next_steps_hint = "Seek emergency care immediately."
    elif any(k in txt for k in ["fever", "cough", "dizziness", "vomiting", "nausea", "abdominal pain"]):
        urgency = "Moderate"
        department = "General Medicine"
        next_steps_hint = "Consult a physician within 24 hours."
    else:
        urgency = "Routine"
        department = "General Medicine"
        next_steps_hint = "Schedule a routine check-up if persists."

    # confidence heuristic
    base = 0.5 + min(len(symptoms) * 0.08, 0.4)
    noise = random.uniform(-0.08, 0.06)
    confidence = round(max(0.01, min(0.99, base + noise)), 2)

    return {
        "urgency": urgency,
        "department": department,
        "next_steps_hint": next_steps_hint,
        "confidence": confidence
    }

# HF wrapper
_hf_pipeline = None
if _HF_AVAILABLE and HF_MODEL_PATH:
    try:
        _hf_pipeline = pipeline("text-classification", model=HF_MODEL_PATH, top_k=5)
    except Exception:
        _hf_pipeline = None

def predict_record(text: str, symptoms: list, age=None, gender=None):
    """
    Returns dict: urgency, department, next_steps_hint, confidence
    If a HF model is available, returns model output (user needs to fine-tune model to return desired labels).
    Otherwise returns rule-based output.
    """
    if _hf_pipeline:
        # Example: your fine-tuned model should return labels like "Urgency_Emergency|Dept_Cardiology"
        try:
            # Use model to classify — this depends on how model was trained.
            preds = _hf_pipeline(text[:512])
            # pick top label and score; mapping logic here depends on label scheme
            top = preds[0]
            label = top.get("label", "")
            score = float(top.get("score", 0.0))
            # naive parsing: expecting "EMERGENCY__CARDIOLOGY" or similar
            parts = label.replace("-", "_").split("__")
            urgency = parts[0] if parts else "Routine"
            department = parts[1] if len(parts) > 1 else "General Medicine"
            return {
                "urgency": urgency,
                "department": department,
                "next_steps_hint": "See provided guidance",
                "confidence": round(score, 2)
            }
        except Exception:
            return _rule_based_predict(text, symptoms, age, gender)
    else:
        return _rule_based_predict(text, symptoms, age, gender)























##src/model_predictor.py
#import randoms
#
#def predict_for_record(clean_text: str, symptoms: list, age=None, gender= None):
#    """
#    Rule-based predictor :
#    - High urgency when chest pain or shortness of breath exists
#    - Medium when fever/cough/dizziness/vomiting present
#    - Low otherwise
#    Returns dict with urgency , department, next_steps, confidence
#    """
#
#    txt = (clean_text or "").lower()
#    urgency = "Low"
#    department = "General Medicine"
#    next_steps = "Monitor symptoms; seek care if worsens."
#
#    #rules for high urgency
#    if any(k in txt for k in ["chest pain","shortness of breath","loss of consciousness"]):
#        urgency = "High"
#        department = "Cardiology" if "chest pain"  in txt else "Emergency"
#        next_steps = "Seek emergency care immediately."
#
#    elif any(k in txt for k in ["fever", "cough", "dizziness","vomiting", "nausea"]):
#        urgency = "Medium"
#        department = "General Medicine"
#        next_steps = "Consult a physician within 24 hours"
#
#    #Confidense heuristic : base 0.5+0.08 per extracted symptom (max 0.9), + small noise
#    base = 0.5 + min(len(symptoms) * 0.08, 0.4)
#    noise = random.uniform(-0.08, 0.06)
#    confidence = round(max(0.01, min(0.99, base + noise)), 2)
#
#    return {
#        "urgency" : urgency,
#        "department": department,
#        "next_steps": next_steps,
#        "confidence": confidence
#    }