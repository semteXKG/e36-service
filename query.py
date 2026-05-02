#!/usr/bin/env python3
"""
query.py  --  Search the BMW E36 service manual.

Usage:
    python query.py "brake bleeding"
    python query.py "torque spec cylinder head" --top 15
    python query.py "cooling system" --no-browser
    python query.py "oil change" --llm        # Phase 2: requires anthropic + API key
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
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit(
        "ERROR: rank-bm25 not installed.\n"
        "Run:  pip install pymupdf rank-bm25"
    )

BASE_DIR   = pathlib.Path(__file__).parent.resolve()
PDF_PATH   = BASE_DIR / "BMW - E36 - 3 Series Service Manual (1992 - 1998) EN.pdf"
INDEX_PATH = BASE_DIR / "index.json"
TOP_K      = 10


# ── Phase 2 LLM hook ──────────────────────────────────────────────────────────

def llm_answer(query: str, top_records: list, base_dir: pathlib.Path) -> str:
    """
    Phase 2 entry point. Send top-k page images to Claude API for a synthesized answer.
    To enable:
        pip install anthropic
        set ANTHROPIC_API_KEY=sk-ant-...
        python query.py "your question" --llm
    """
    try:
        import anthropic
        import base64
    except ImportError:
        return "ERROR: anthropic package not installed. Run: pip install anthropic"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "ERROR: ANTHROPIC_API_KEY environment variable not set."

    client  = anthropic.Anthropic(api_key=api_key)
    content = []

    for r in top_records[:5]:
        img_path = base_dir / r["img_path"]
        if img_path.exists():
            with open(img_path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })
            content.append({
                "type": "text",
                "text": f"[Page {r['page']} of the BMW E36 3 Series service manual]",
            })

    content.append({
        "type": "text",
        "text": (
            f"Using the BMW E36 service manual pages shown above, answer this question "
            f"as specifically as possible and cite page numbers:\n\n{query}"
        ),
    })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# ── Excerpt ───────────────────────────────────────────────────────────────────

def make_excerpt(text: str, query_tokens: list, window: int = 400) -> str:
    lower    = text.lower()
    best_pos = len(text)
    for tok in query_tokens:
        p = lower.find(tok)
        if 0 <= p < best_pos:
            best_pos = p
    start   = max(0, best_pos - window // 3)
    end     = min(len(text), start + window)
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
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
      <h2>Claude API Answer</h2>
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
                    help="(Phase 2) Send top pages to Claude API for a synthesized answer")
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

    corpus       = [tokenize(r["text"]) for r in records]
    bm25         = BM25Okapi(corpus)
    query_tokens = tokenize(args.query)

    if not query_tokens:
        sys.exit("ERROR: query is empty after tokenization.")

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked = [i for i in ranked if scores[i] > 0][: args.top]

    if not ranked:
        print(f'\nNo results found for "{args.query}".')
        print("Try different keywords, e.g. individual words rather than phrases.")
        sys.exit(0)

    print(f"Found {len(ranked)} matching page(s)\n")
    sep = "-" * 60

    result_records = []
    for idx in ranked:
        r       = records[idx]
        score   = float(scores[idx])
        excerpt = make_excerpt(r["text"], query_tokens, window=300)
        print(sep)
        print(f"Page {r['page']:>4}  [score: {score:.2f}]")
        print(excerpt[:300])
        result_records.append({**r, "score": score})

    print(sep)

    # Phase 2 LLM
    llm_text = ""
    if args.llm:
        print("\nQuerying Claude API...")
        llm_text = llm_answer(args.query, result_records, BASE_DIR)
        print("\n=== Claude API Answer ===")
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
