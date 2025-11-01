#src/preprocessing.py

import re


def preprocess_text(text:str) -> str:
    """
    Basic cleaning:
    - normalize newlines and whitespace
    - remove control chars
    - strip leading/trailing spaces
    """

    if text is None :
        return ""
    t = str(text)
    t = t.replace("\r"," ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    # remove non-printable/control characters
    t = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", t)
    t = t.strip()
    return t

