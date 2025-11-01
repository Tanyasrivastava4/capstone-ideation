# src/rag_summary.py
"""
Full RAG: Retriever (embeddings) + Generator (Mistral or DistilGPT2)
"""
import os
import pandas as pd
import joblib
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    _SENT_AVAILABLE = True
except Exception:
    _SENT_AVAILABLE = False

EMB_PATH = "data/processed/embeddings.pkl"
CLEANED_PATH = "data/processed/cleaned_data.csv"

# --------------- RETRIEVER PART ---------------
def build_embeddings(source_csv=CLEANED_PATH, model_name="all-MiniLM-L6-v2"):
    """Create and save embeddings for the knowledge base."""
    if not _SENT_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed.")
    df = pd.read_csv(source_csv)
    texts = df.get("cleaned_text", df.get("report_text", pd.Series([""]))).fillna("").tolist()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    joblib.dump({"texts": texts, "embeddings": embeddings}, EMB_PATH)
    return {"texts": texts, "embeddings": embeddings}

def get_similar_cases(query_text: str, top_k: int = 3):
    """Retrieve similar cases."""
    if not os.path.exists(EMB_PATH):
        build_embeddings()
    data = joblib.load(EMB_PATH)
    texts, embeddings = data["texts"], data["embeddings"]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_vec = model.encode([query_text], convert_to_numpy=True)
    scores = util.cos_sim(q_vec, embeddings)[0].cpu().numpy()
    best_idx = np.argsort(scores)[-top_k:][::-1]
    return [texts[i] for i in best_idx]

# --------------- GENERATOR PART ---------------
def generate_summary(query_text: str, retrieved_cases: list, model_name="mistralai/Mistral-7B-Instruct", max_new_tokens=128):
    """
    Uses a small open-source model to summarize retrieved info.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception:
        return "Summary model not available in offline mode."

    context = "\n".join(retrieved_cases)
    prompt = (
        f"Patient report: {query_text}\n\n"
        f"Similar past cases:\n{context}\n\n"
        f"Summarize common findings and suggest next steps:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# --------------- PIPELINE WRAPPER ---------------
def rag_pipeline(query_text: str):
    """Full RAG pipeline: retrieve + summarize."""
    similar = get_similar_cases(query_text)
    summary = generate_summary(query_text, similar)
    return {"similar_cases": similar, "summary": summary}







#def generate_summary(query_text: str, retrieved_cases:list, model_name="mistralia/Mistral-7B-Instruct", max_new_tokens= 128):
#    """
#    Uses a small open-source model to summarize retrieved info
#    """
#    try:
#        tokenizer = AutoTokenizer.form_pretrained(model_name)
#        model = AutoModelForCausalLM.form_pretrained(model_name)
#    except Exception:
#        return "Summary model not available in offline mode."
#    
#    context = "\n".join(retrieved_cases)
#    prompts = (
#        f"Patient report: {query_text}\n\n"
#        f"Similar past cases:\n{context}\n\n"
#        f"Summarize common findings and suggest next steps:"
#    )
#    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
#    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
#    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
#    return summary
#
#
#def rag_pipeline(query_text: str):
#    """ Full RAG pipeline: retrieve + summarize"""
#    similar = get_similar_cases(query_text)
#    summary = generate_summary(query_text, similar)
#    return {"similar_cases": similar, "summary": summary}
#











##src/rag_summary.py
#
#"""
#Simple retrieval using SentenceTransformers embeddings.
#Saves embeddings to data/processed/embeddings.pkl
#"""
#
#import os:
#import pandas as pd
#import joblib
#import numpy as np
#
#EMB_PATH = "data/processed/embeddings.pkl" 
#CLEANED_PATH = "data/processed/cleaned_data.csv"
#
#try:
#    from sentence_transformers import SentenceTransformer , util
#    _SENT_AVAILABLE = True
#except Exception:
#    _SENT_AVAILABLE = False
#
#def build_embeddings(source_csv=CLEANED_PATH, model_name="all-MiniLM-L6-v2"):
#    """
#    Read cleaned CSV and build embeddings for 'report_text' (or cleaned_text).
#    Saves a dict with {'texts' : [..], 'embeddings':np.array(...)}
#    """
#
#    if not _SENT_AVAILABLE:
#        raise RuntimeError("sentence-transformers not installed. Install to use embeddiing-based RAG")
#    if not os.path.exists(source_csv):
#        return None
#    df = pd.read_csv(source_csv)
#    texts = df.get("cleaned_text", df.get("report_text", pd.Series([""]))).fillna("").tolist()
#    model = SentenceTransformer(model_name)
#    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
#    obj = {"texts": texts, "embeddings":embeddings}
#    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
#    joblib.dump(obj, EMB_PATH)
#    RETURN obj
#
#
#def get_similar_cases(query_text:str, top_k: int=3):
#    """
#    Return tok_k similar texts from saved embeddings. If embeddings not found, attempt to build from cleaned CSV
#    """
#    if not _SENT_AVAILABLE:
#        #fallback: return empty list(or we can implement TF-IDF fallback)
#        return []
#    if not os.path.exists(EMB_PATH):
#        build_embeddings()
#        if not os.path.exists(EMB_PATH):
#            return []
#    data = joblib.load(EMB_PATH)
#    texts = data["texts"]
#    embeddings - data["embeddings"]
#    model = SentenceTransformer("all-MiniLM-L6-v2")
#    q_vec = model.encoder([query_text], convert_to_numpy=True)
#    scores = util.cos_sim(q_vec,embeddings)[0].cpu().numpy()
#    best_idx = np.argsort(scores)[-top_k:][::-1]
#    results = []
#    for i in best_idx:
#        if texts[i].strip():
#            results.append(texts[i])
#    return results