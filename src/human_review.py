# src/human_review.py
"""
Human Review Module
--------------------
Automatically checks model confidence and flags low-confidence cases
for manual review. Also prints a summary of all flagged cases at the end.
"""

import os
import pandas as pd
from datetime import datetime

TICKET_PATH = "outputs/tickets.csv"


def flag_ticket_if_low_conf(row, threshold: float = 0.6):
    """
    Flags a record for human review if the model confidence is below a threshold.
    """
    confidence = row.get("confidence", 1.0)

    if confidence >= threshold:
        return False

    ticket = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id": row.get("patient_id", "Unknown"),
        "urgency_level": row.get("urgency", "Unknown"),
        "predicted_department": row.get("department", "Unknown"),
        "confidence": round(confidence, 2),
        "reason": f"Low model confidence ({confidence:.2f}) below threshold {threshold}.",
        "status": "Pending Review"
    }

    os.makedirs(os.path.dirname(TICKET_PATH), exist_ok=True)
    df = pd.DataFrame([ticket])

    if os.path.exists(TICKET_PATH):
        df.to_csv(TICKET_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(TICKET_PATH, index=False)

    print(f"⚠️ Ticket created for Patient {ticket['patient_id']} (Confidence: {ticket['confidence']})")
    return True


def notify_admin_summary():
    """
    Prints a summary of all low-confidence tickets after pipeline execution.
    (You can later extend this to send email notifications.)
    """
    if not os.path.exists(TICKET_PATH):
        print("✅ No low-confidence tickets found. All predictions are confident!")
        return

    df = pd.read_csv(TICKET_PATH)
    if df.empty:
        print("✅ No low-confidence tickets found. All predictions are confident!")
        return

    print("\n📢 SUMMARY: Low-Confidence Cases for Human Review:")
    for _, row in df.iterrows():
        print(f" - Patient {row['patient_id']} | Dept: {row['predicted_department']} | Confidence: {row['confidence']} | Status: {row['status']}")

    print(f"\n🗂️  Total {len(df)} cases need manual review. Details saved in '{TICKET_PATH}'.")



















## src/human_review.py
#
#import pandas as pd
#import os
#
#def create_ticket(patient_id, report_text, confidence, threshold=0.5):
#    """
#    Creates a ticket if confidence is below threshold.
#    """
#
#    if confidence < threshold:
#        ticket = {
#            "patient_id": patient_id,
#            "report": report_text[:60],  # short preview
#            "model_confidence": confidence,
#            "status": "Needs Clinician Review"
#        }
#
#        # Save or append to tickets.csv
#        ticket_path = "outputs/tickets.csv"
#        if os.path.exists(ticket_path):
#            df = pd.read_csv(ticket_path)
#            df = pd.concat([df, pd.DataFrame([ticket])], ignore_index=True)
#        else:
#            df = pd.DataFrame([ticket])
#
#        df.to_csv(ticket_path, index=False)
#        print(f"⚠️ Ticket created for patient {patient_id}")
#
#
## src/human_review.py
#import os
#import pandas as pd
#
#TICKETS_PATH = "outputs/tickets.csv"
#os.makedirs(os.path.dirname(TICKETS_PATH), exist_ok=True)
#
#def maybe_create_ticket(result_record, threshold=0.6):
#    """
#    If confidence < threshold, append ticket entry.
#    Returns True if ticket created.
#    """
#    conf = result_record.get("confidence", 1.0)
#    if conf is None:
#        conf = 0.0
#    if conf < threshold:
#        ticket = {
#            "patient_id": result_record.get("report_id"),
#            "report": (result_record.get("symptoms") or "")[:200],
#            "model_confidence": conf,
#            "status": "Needs Clinician Review"
#        }
#        if os.path.exists(TICKETS_PATH):
#            df = pd.read_csv(TICKETS_PATH)
#            df = pd.concat([df, pd.DataFrame([ticket])], ignore_index=True)
#        else:
#            df = pd.DataFrame([ticket])
#        df.to_csv(TICKETS_PATH, index=False)
#        return True
#    return False
#
#
#
#
## src/human_review.py
#import os
#import pandas as pd
#
#TICKETS_PATH = "outputs/tickets.csv"
#os.makedirs(os.path.dirname(TICKETS_PATH), exist_ok=True)
#
#def flag_ticket_if_low_conf(record, threshold=0.6):
#    """
#    If record['confidence'] < threshold -> append to outputs/tickets.csv and return True
#    """
#    conf = record.get("confidence")
#    try:
#        conf = float(conf)
#    except Exception:
#        conf = 0.0
#    if conf < threshold:
#        ticket = {
#            "patient_id": record.get("patient_id"),
#            "symptoms": record.get("symptoms"),
#            "model_confidence": conf,
#            "status": "Needs Clinician Review"
#        }
#        if os.path.exists(TICKETS_PATH):
#            df = pd.read_csv(TICKETS_PATH)
#            df = pd.concat([df, pd.DataFrame([ticket])], ignore_index=True)
#        else:
#            df = pd.DataFrame([ticket])
#        df.to_csv(TICKETS_PATH, index=False)
#        return True
#    return False
#