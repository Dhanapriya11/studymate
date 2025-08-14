import re

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50):
    """Naive whitespace 'token' chunking. max_tokens≈words."""
    words = text.split()
    chunks = []
    i = 0
    step = max(1, max_tokens - overlap)
    while i < len(words):
        chunk = words[i:i+max_tokens]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += step
    return chunks
