#!/usr/bin/env python3
"""
query.py  --  Search the BMW E36 service manual.

Usage:
    python query.py "brake bleeding"
    python query.py "torque spec cylinder head" --top 15
    python query.py "cooling system" --no-browser
    python query.py "oil change" --llm        # Phase 2: requires openai + MOONSHOT_API_KEY
"""

import argparse
import html as html_lib
import json
import os
import pathlib
import re
import sys
import webbrowser

try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).parent / ".env")
except ImportError:
    pass

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit(
        "ERROR: rank-bm25 not installed.\n"
        "Run:  pip install pymupdf rank-bm25"
    )

BASE_DIR   = pathlib.Path(__file__).parent.resolve()
PDF_PATH   = BASE_DIR / "BMW - E36 - 3 Series Service Manual (1992 - 1998) EN.pdf"
INDEX_PATH = BASE_DIR / "index.json"
TAGS_PATH  = BASE_DIR / "tags.json"
TOP_K      = 10
TAG_BOOST  = 5.0   # weight added per unit of tag score on top of BM25


def load_tags() -> dict:
    if not TAGS_PATH.exists():
        return {}
    with open(TAGS_PATH, encoding="utf-8") as f:
        return json.load(f)


def compute_tag_score(page_tags: list, query_tokens: list) -> float:
    """
    Score tags against query tokens. A tag matching ALL tokens gets full weight;
    partial matches are scaled by (matched/total)^2, heavily discounting pages
    where only one token of a multi-word query appears in their tags.
    """
    if not query_tokens:
        return 0.0
    n = len(query_tokens)
    score = 0.0
    for entry in page_tags:
        tag_text = entry["tag"].lower()
        matched = sum(1 for t in query_tokens if t in tag_text)
        if matched == 0:
            continue
        score += entry["weight"] * (matched / n) ** 2
    return score


PHRASE_BONUS = 4.0   # added when all query tokens appear as a phrase in the OCR text

def compute_phrase_bonus(text: str, query_tokens: list) -> float:
    """Return PHRASE_BONUS if the full query phrase appears verbatim in the text."""
    if len(query_tokens) < 2:
        return 0.0
    phrase = " ".join(query_tokens)
    return PHRASE_BONUS if phrase in text.lower() else 0.0


def deduplicate_by_chapter(results: list, tags: dict, gap: int = 3) -> list:
    """
    Group consecutive results that share the same primary section tag.
    Returns the first page (lowest number) of each group, scored by the
    group's maximum score. Fetch 3x top-k before calling this.
    """
    if not results:
        return results

    def primary_tag(page_num):
        page_tags = tags.get(str(page_num), [])
        if not page_tags:
            return None
        best = max(page_tags, key=lambda t: t["weight"])
        return best["tag"] if best["weight"] >= 0.8 else None

    by_page  = sorted(results, key=lambda r: r["page"])
    groups   = []
    group    = [by_page[0]]

    for r in by_page[1:]:
        prev = group[-1]
        if (r["page"] - prev["page"] <= gap and
                primary_tag(r["page"]) == primary_tag(prev["page"])):
            group.append(r)
        else:
            groups.append(group)
            group = [r]
    groups.append(group)

    deduped = []
    for g in groups:
        best = max(g, key=lambda r: r["score"])
        deduped.append(best)

    return sorted(deduped, key=lambda r: r["score"], reverse=True)


# ── Phase 2 LLM hook ──────────────────────────────────────────────────────────

def llm_answer(query: str, top_records: list, base_dir: pathlib.Path) -> str:
    """
    Send top-5 pages (OCR text) to moonshot.ai (Kimi) for a synthesized answer.
    To enable:
        pip install openai
        set MOONSHOT_API_KEY=sk-...
        python query.py "your question" --llm
    """
    try:
        from openai import OpenAI
    except ImportError:
        return "ERROR: openai package not installed. Run: pip install openai"

    api_key  = os.environ.get("LLM_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.moonshot.ai/v1")
    model    = os.environ.get("LLM_MODEL",    "kimi-k2.6")

    if not api_key:
        return "ERROR: LLM_API_KEY environment variable not set."

    client = OpenAI(api_key=api_key, base_url=base_url)

    context_parts = []
    for r in top_records[:5]:
        text = clean_text(r["text"])
        if text:
            context_parts.append(f"[Page {r['page']}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a BMW E36 service manual assistant. "
        "Answer the question using only the provided manual excerpts. "
        "Format your response in Markdown: use **bold** for critical values (torque specs, fluid types, part numbers, temperatures), "
        "and numbered lists for procedures. "
        "Cite page numbers in your answer (e.g. Page 73).\n\n"
        f"{context}\n\n"
        f"Question: {query}"
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def clean_text(text: str) -> str:
    """Remove OCR watermarks and common artifacts from page text."""
    # Watermark
    text = re.sub(r'Versi[oó]n electr[oó]nica licenciada[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Buenos Aires\s*//\s*Argentina', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'tel:\s*[\d\s\(\)\-\+]+', '', text, flags=re.IGNORECASE)
    # Long runs of repeated punctuation (dividers, dot leaders, etc.)
    text = re.sub(r'[~=]{4,}', ' ', text)
    text = re.sub(r'_{4,}', ' ', text)
    text = re.sub(r'\.{5,}', ' ', text)
    text = re.sub(r'\|{2,}', ' ', text)
    text = re.sub(r'([A-Z])\1{2,}', ' ', text)
    # Unicode garbage
    text = text.replace('�', ' ')
    # Repeated short OCR noise tokens
    text = re.sub(r'\b(\w{1,2})\s+\1\b', ' ', text)
    text = re.sub(r'(\b[a-zA-Z]{1,2}\b\s+){4,}', ' ', text)
    # Stray symbols
    text = re.sub(r'(?<!\w)[~|\\{}]+(?!\w)', ' ', text)
    text = re.sub(r'(\s*/\s*)+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── Excerpt ───────────────────────────────────────────────────────────────────

def make_excerpt(text: str, query_tokens: list, window: int = 400) -> str:
    text     = clean_text(text)
    lower    = text.lower()
    best_pos = len(text)
    # Prefer position of the full phrase over individual tokens
    phrase = " ".join(query_tokens)
    p = lower.find(phrase)
    if p >= 0:
        best_pos = p
    else:
        for tok in query_tokens:
            p = lower.find(tok)
            if 0 <= p < best_pos:
                best_pos = p
    start   = max(0, best_pos - window // 3)
    end     = min(len(text), start + window)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


# ── HTML ──────────────────────────────────────────────────────────────────────

def highlight(text: str, tokens: list) -> str:
    for tok in sorted(tokens, key=len, reverse=True):
        text = re.sub(f"(?i)({re.escape(tok)})", r"<mark>\1</mark>", text)
    return text


CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* BMW brand colours */
/* Blue: #1C69D4  |  Dark: #1C1C1C  |  Light grey bg: #F2F2F2  |  Red: #CC0000 */

body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  background: #F2F2F2;
  color: #1C1C1C;
  min-height: 100vh;
}

/* ── Top bar ── */
header {
  background: #1C1C1C;
  padding: 0 32px;
  display: flex;
  align-items: center;
  gap: 16px;
  height: 56px;
  border-bottom: 3px solid #1C69D4;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
}

.logo-ring {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: 2px solid #fff;
  display: grid;
  grid-template-columns: 1fr 1fr;
  overflow: hidden;
  flex-shrink: 0;
}

.logo-ring span:nth-child(1) { background: #fff; }
.logo-ring span:nth-child(2) { background: #1C69D4; }
.logo-ring span:nth-child(3) { background: #1C69D4; }
.logo-ring span:nth-child(4) { background: #fff; }

header h1 {
  font-size: 0.95rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.subtitle {
  margin-left: auto;
  font-size: 0.85rem;
  color: #999;
}

/* ── Results area ── */
main {
  max-width: 1100px;
  margin: 0 auto;
  padding: 28px 24px;
}

.results-header {
  font-size: 0.85rem;
  color: #666;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid #ddd;
}

.results-header strong {
  color: #1C1C1C;
}

/* ── LLM answer box ── */
.llm-box {
  background: #fff;
  border-left: 4px solid #1C69D4;
  border-radius: 4px;
  padding: 20px 24px;
  margin-bottom: 24px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  line-height: 1.7;
}

.llm-box h2 {
  font-size: 0.75rem;
  color: #1C69D4;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 10px;
  font-weight: 700;
}

/* ── Result card ── */
.card {
  display: flex;
  gap: 0;
  background: #fff;
  border-radius: 4px;
  margin-bottom: 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  overflow: hidden;
  transition: box-shadow 0.15s, transform 0.15s;
  border-top: 3px solid transparent;
}

.card:hover {
  box-shadow: 0 4px 16px rgba(0,0,0,0.13);
  transform: translateY(-1px);
  border-top-color: #1C69D4;
}

/* Left column — thumbnail */
.card-left {
  flex: 0 0 200px;
  background: #F2F2F2;
  border-right: 1px solid #e8e8e8;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 16px 12px 10px;
  gap: 10px;
}

.card-left a img {
  width: 100%;
  height: auto;
  display: block;
  border: 1px solid #ddd;
  border-radius: 2px;
  cursor: pointer;
  transition: box-shadow 0.15s;
}

.card-left a:hover img {
  box-shadow: 0 4px 12px rgba(28,105,212,0.25);
}

.img-links {
  display: flex;
  gap: 10px;
  font-size: 0.75rem;
}

.img-links a {
  color: #1C69D4;
  text-decoration: none;
  font-weight: 500;
}

.img-links a:hover {
  text-decoration: underline;
}

/* Right column — content */
.card-right {
  flex: 1;
  padding: 18px 22px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.page-header {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}

.page-num {
  font-size: 0.78rem;
  font-weight: 700;
  color: #fff;
  background: #1C69D4;
  padding: 3px 12px;
  border-radius: 2px;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.score-label {
  font-size: 0.72rem;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.score-bar-wrap {
  flex: 0 0 100px;
  height: 4px;
  background: #e8e8e8;
  border-radius: 2px;
  overflow: hidden;
}

.score-bar {
  height: 100%;
  background: #1C69D4;
  border-radius: 2px;
}

.score-val {
  font-size: 0.78rem;
  color: #666;
  font-variant-numeric: tabular-nums;
}

.excerpt {
  font-size: 0.9rem;
  line-height: 1.7;
  color: #444;
  background: #F8F8F8;
  padding: 12px 14px;
  border-radius: 2px;
  border-left: 3px solid #e8e8e8;
  white-space: pre-wrap;
  word-break: break-word;
}

mark {
  background: transparent;
  color: #CC0000;
  font-weight: 700;
  padding: 0;
}

@media (max-width: 680px) {
  .card { flex-direction: column; }
  .card-left { flex: none; border-right: none; border-bottom: 1px solid #e8e8e8; }
  header { flex-wrap: wrap; height: auto; padding: 10px 16px; }
  .subtitle { margin-left: 0; width: 100%; }
}
"""


def build_html(query_str: str, results: list, query_tokens: list, llm_text: str = "") -> str:
    max_score = results[0]["score"] if results else 1.0
    pdf_uri   = PDF_PATH.as_uri()

    llm_section = ""
    if llm_text:
        llm_section = f"""
    <div class="llm-box">
      <h2>Kimi AI Answer</h2>
      <p>{html_lib.escape(llm_text).replace(chr(10), "<br>")}</p>
    </div>"""

    cards = []
    for r in results:
        excerpt_raw = make_excerpt(r["text"], query_tokens)
        excerpt_esc = html_lib.escape(excerpt_raw)
        excerpt_hl  = highlight(excerpt_esc, [html_lib.escape(t) for t in query_tokens])
        img_src     = r["img_path"].replace("\\", "/")
        score_pct   = int(min(100, (r["score"] / max_score) * 100)) if max_score > 0 else 0
        pdf_page_uri = f"{pdf_uri}#page={r['page']}"

        cards.append(f"""
        <div class="card">
          <div class="card-left">
            <a href="{pdf_page_uri}" target="_blank" title="Open page {r['page']} in PDF">
              <img src="{img_src}" alt="Page {r['page']}" loading="lazy"
                   onerror="this.parentElement.style.display='none'">
            </a>
            <div class="img-links">
              <a href="{pdf_page_uri}" target="_blank">Open in PDF</a>
              &nbsp;|&nbsp;
              <a href="{img_src}" target="_blank">Full image</a>
            </div>
          </div>
          <div class="card-right">
            <div class="page-header">
              <span class="page-num">Page {r['page']}</span>
              <span class="score-label">relevance</span>
              <div class="score-bar-wrap">
                <div class="score-bar" style="width:{score_pct}%"></div>
              </div>
              <span class="score-val">{r['score']:.2f}</span>
            </div>
            <p class="excerpt">{excerpt_hl}</p>
          </div>
        </div>""")

    cards_html = "\n".join(cards)
    q_esc      = html_lib.escape(query_str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BMW E36 Manual &mdash; {q_esc}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-ring"><span></span><span></span><span></span><span></span></div>
    <h1>BMW E36 &mdash; Service Manual</h1>
  </div>
  <span class="subtitle">{len(results)} result(s) for &ldquo;{q_esc}&rdquo;</span>
</header>
<main>
  <p class="results-header">Showing top {len(results)} pages matching <strong>&ldquo;{q_esc}&rdquo;</strong></p>
{llm_section}
{cards_html}
</main>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Search the BMW E36 service manual.")
    ap.add_argument("query",              help="Search terms, e.g. \"brake bleeding\"")
    ap.add_argument("--top",  "-n",       type=int, default=TOP_K,
                    help=f"Number of results to show (default: {TOP_K})")
    ap.add_argument("--no-browser",       action="store_true",
                    help="Do not open results.html automatically")
    ap.add_argument("--llm",              action="store_true",
                    help="(Phase 2) Send top pages to Kimi API for a synthesized answer (needs MOONSHOT_API_KEY)")
    args = ap.parse_args()

    if not INDEX_PATH.exists():
        sys.exit(
            "ERROR: index.json not found.\n"
            "Run `python index.py` first to build the index."
        )

    print(f'Searching for: "{args.query}"')

    with open(INDEX_PATH, encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    tags    = load_tags()

    corpus       = [tokenize(r["text"]) for r in records]
    bm25         = BM25Okapi(corpus)
    query_tokens = tokenize(args.query)

    if not query_tokens:
        sys.exit("ERROR: query is empty after tokenization.")

    bm25_scores = bm25.get_scores(query_tokens)
    final_scores = []
    for i, r in enumerate(records):
        page_tags    = tags.get(str(r["page"]), [])
        tag_score    = compute_tag_score(page_tags, query_tokens)
        phrase_bonus = compute_phrase_bonus(r["text"], query_tokens)
        final_scores.append(float(bm25_scores[i]) + TAG_BOOST * tag_score + phrase_bonus)

    ranked_all = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
    ranked_all = [i for i in ranked_all if final_scores[i] > 0][: args.top * 3]

    if not ranked_all:
        print(f'\nNo results found for "{args.query}".')
        print("Try different keywords, e.g. individual words rather than phrases.")
        sys.exit(0)

    pre_dedup      = [{**records[i], "score": final_scores[i]} for i in ranked_all]
    result_records = deduplicate_by_chapter(pre_dedup, tags)[: args.top]

    print(f"Found {len(result_records)} matching page(s)\n")
    sep = "-" * 60

    for r in result_records:
        excerpt = make_excerpt(r["text"], query_tokens, window=300)
        print(sep)
        print(f"Page {r['page']:>4}  [score: {r['score']:.2f}]")
        print(excerpt[:300])

    print(sep)

    # Phase 2 LLM
    llm_text = ""
    if args.llm:
        print("\nQuerying Kimi AI...")
        llm_text = llm_answer(args.query, result_records, BASE_DIR)
        print("\n=== Kimi AI Answer ===")
        print(llm_text)
        print("=" * 60)

    # Write results.html
    html_path = BASE_DIR / "results.html"
    try:
        html_content = build_html(args.query, result_records, query_tokens, llm_text)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        if not args.no_browser:
            print(f"\nOpening results -> {html_path}")
            try:
                webbrowser.open(html_path.as_uri())
            except Exception:
                print(f"(Could not open browser automatically. Open manually: {html_path})")
    except Exception as e:
        print(f"WARNING: Could not write results.html: {e}")


if __name__ == "__main__":
    main()
