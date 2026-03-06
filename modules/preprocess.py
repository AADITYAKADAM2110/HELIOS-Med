import re

def preprocess_text(text):

    text = re.sub(r"Page \d+ of \d+", "", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"-\s+", "", text)

    text = re.sub(r"\n+", "\n", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()