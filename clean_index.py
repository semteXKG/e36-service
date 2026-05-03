#!/usr/bin/env python3
"""
clean_index.py -- Deep-clean OCR artifacts from index.json in-place.
"""
import json, re, pathlib

INDEX_PATH = pathlib.Path("index.json")


def deep_clean(text: str) -> str:
    # 1. Watermark (do first — it's long and easy to identify)
    text = re.sub(r'Versi[oó]n electr[oó]nica licenciada[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Buenos Aires\s*//\s*Argentina', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'(?i)tel:\s*[\d\s\(\)\-\+]+', '', text)
    text = re.sub(r'free download from[^\n]*', '', text, flags=re.IGNORECASE)

    # 2. Long runs of repeated/symbolic chars — BEFORE replacing garbage chars
    #    so we catch runs like "~~~~~~" that would get fragmented by step 3.
    text = re.sub(r'[~=]{4,}', ' ', text)                 # dividers
    text = re.sub(r'_{4,}', ' ', text)                    # underscores
    text = re.sub(r'-{7,}', ' ', text)                    # long dash runs
    text = re.sub(r'\.{5,}', ' ', text)                   # dot leaders (TOC)
    text = re.sub(r'\|{2,}', ' ', text)                   # repeated pipes
    text = re.sub(r'([A-Z])\1{2,}', ' ', text)            # LLLL, EEE etc.
    text = re.sub(r'([a-z])\1{4,}', ' ', text)            # rrrrr, sssss etc.

    # 3. Unicode garbage chars (after long-run cleanup)
    text = text.replace('�', ' ')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)

    # 4. Repeated short token pairs/triples: "vsvs vsvs", "oo oo oo"
    text = re.sub(r'\b(\w{1,4})\s+\1\s+\1\b', ' ', text)
    text = re.sub(r'\b(\w{1,2})\s+\1\b', ' ', text)

    # 5. Consecutive isolated 1-2 char tokens (4+ in a row = column gutter)
    text = re.sub(r'(\b[a-zA-Z]{1,2}\b\s+){4,}', ' ', text)

    # 6. Stray isolated symbols not bordering a word or digit
    text = re.sub(r'(?<!\w)[~|\\{}@#$%^*]+(?!\w)', ' ', text)

    # 7. Slash noise
    text = re.sub(r'(\s*/\s*)+', ' ', text)

    # 8. Chunks that are mostly non-alphabetic (figure noise, border text)
    #    Split on sentence endings and drop noise chunks
    parts = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for chunk in parts:
        stripped = chunk.strip()
        if not stripped:
            continue
        alpha = sum(1 for c in stripped if c.isalpha())
        if len(stripped) > 20 and alpha / len(stripped) < 0.30:
            continue
        cleaned.append(chunk)
    text = ' '.join(cleaned) if cleaned else text

    # 9. Normalize whitespace
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def main():
    print("Loading index...", end=" ", flush=True)
    with open(INDEX_PATH, encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    print(f"{len(records)} records")

    total_before = sum(len(r["text"]) for r in records)
    for r in records:
        r["text"] = deep_clean(r["text"])
    total_after = sum(len(r["text"]) for r in records)

    removed = total_before - total_after
    print(f"Removed {removed:,} chars ({100*removed/total_before:.1f}% of index)")

    tmp = INDEX_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(INDEX_PATH)
    print("Saved index.json")


if __name__ == "__main__":
    main()
