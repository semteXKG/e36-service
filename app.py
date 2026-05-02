#!/usr/bin/env python3
"""
app.py  --  Flask web interface for the BMW E36 service manual search.

Usage:
    pip install flask
    python app.py
    # Open http://localhost:5000
"""

import html as html_lib
import json
import pathlib
import sys

try:
    from flask import Flask, jsonify, request, send_file, send_from_directory
except ImportError:
    sys.exit("ERROR: flask not installed.\nRun: pip install flask")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("ERROR: rank-bm25 not installed.\nRun: pip install rank-bm25")

from query import tokenize, make_excerpt, highlight, CSS

BASE_DIR   = pathlib.Path(__file__).parent.resolve()
INDEX_PATH = BASE_DIR / "index.json"
PDF_PATH   = BASE_DIR / "BMW - E36 - 3 Series Service Manual (1992 - 1998) EN.pdf"
DEFAULT_TOP = 10

# ── Load index once at startup ────────────────────────────────────────────────

if not INDEX_PATH.exists():
    sys.exit("ERROR: index.json not found.\nRun `python index.py --ocr` first.")

print("Loading index...", end=" ", flush=True)
with open(INDEX_PATH, encoding="utf-8") as f:
    _data = json.load(f)
_records = _data["records"]
_corpus  = [tokenize(r["text"]) for r in _records]
_bm25    = BM25Okapi(_corpus)
print(f"done ({len(_records)} pages indexed)")

# ── HTML page ─────────────────────────────────────────────────────────────────

_EXTRA_CSS = """
/* Webapp: header search input */
header { height: 64px; }

.search-wrap {
  flex: 1;
  max-width: 520px;
  margin-left: auto;
}
.search-wrap input[type=search] {
  width: 100%;
  padding: 9px 18px;
  font-size: 0.92rem;
  font-family: inherit;
  border: none;
  border-radius: 2px;
  background: #2a2a2a;
  color: #fff;
  outline: none;
  -webkit-appearance: none;
  transition: background 0.15s, box-shadow 0.15s;
}
.search-wrap input[type=search]:focus {
  background: #333;
  box-shadow: 0 0 0 2px #1C69D4;
}
.search-wrap input[type=search]::placeholder { color: #888; }

/* Webapp: status bar */
#status-bar {
  font-size: 0.85rem;
  color: #666;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid #ddd;
}
#status-bar strong { color: #1C1C1C; }

/* Webapp: empty state */
.empty-state {
  text-align: center;
  padding: 64px 0;
  color: #aaa;
  font-size: 1rem;
}
.empty-state span {
  display: block;
  font-size: 2.5rem;
  margin-bottom: 12px;
}
"""

HTML_PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BMW E36 &mdash; Service Manual Search</title>
<style>{CSS}{_EXTRA_CSS}</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-ring"><span></span><span></span><span></span><span></span></div>
    <h1>BMW E36 &mdash; Service Manual</h1>
  </div>
  <div class="search-wrap">
    <input id="q" type="search" placeholder="Search {len(_records)} pages..." autofocus autocomplete="off">
  </div>
</header>
<main>
  <p id="status-bar">Type a query to search the manual.</p>
  <div id="cards">
    <div class="empty-state"><span>&#128269;</span>Enter keywords above &mdash; results appear as you type.</div>
  </div>
</main>
<script>
const qInput   = document.getElementById('q');
const statusEl = document.getElementById('status-bar');
const cardsEl  = document.getElementById('cards');
let debounceTimer;

qInput.addEventListener('input', () => {{
  clearTimeout(debounceTimer);
  const q = qInput.value.trim();
  if (!q) {{
    statusEl.textContent = 'Type a query to search the manual.';
    cardsEl.innerHTML = '<div class="empty-state"><span>&#128269;</span>Enter keywords above &mdash; results appear as you type.</div>';
    return;
  }}
  statusEl.textContent = 'Searching…';
  debounceTimer = setTimeout(() => runSearch(q), 300);
}});

async function runSearch(q) {{
  try {{
    const resp = await fetch('/search?q=' + encodeURIComponent(q) + '&top={DEFAULT_TOP}');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    render(data);
  }} catch (e) {{
    statusEl.textContent = 'Search error: ' + e.message;
  }}
}}

function render(data) {{
  const {{ query, results }} = data;
  if (!results.length) {{
    statusEl.innerHTML = 'No results for <strong>"' + esc(query) + '"</strong>.';
    cardsEl.innerHTML = '<div class="empty-state"><span>&#128270;</span>No pages matched that query. Try different keywords.</div>';
    return;
  }}
  statusEl.innerHTML = 'Showing top <strong>' + results.length + '</strong> page(s) matching <strong>&ldquo;' + esc(query) + '&rdquo;</strong>';
  cardsEl.innerHTML = results.map(r => `
    <div class="card">
      <div class="card-left">
        <a href="/pdf#page=${{r.page}}" target="_blank" title="Open page ${{r.page}} in PDF">
          <img src="/${{r.img_path}}" alt="Page ${{r.page}}" loading="lazy"
               onerror="this.parentElement.style.display='none'">
        </a>
        <div class="img-links">
          <a href="/pdf#page=${{r.page}}" target="_blank">Open in PDF</a>
          &nbsp;|&nbsp;
          <a href="/${{r.img_path}}" target="_blank">Full image</a>
        </div>
      </div>
      <div class="card-right">
        <div class="page-header">
          <span class="page-num">Page ${{r.page}}</span>
          <span class="score-label">relevance</span>
          <div class="score-bar-wrap">
            <div class="score-bar" style="width:${{r.score_pct}}%"></div>
          </div>
          <span class="score-val">${{r.score.toFixed(2)}}</span>
        </div>
        <p class="excerpt">${{r.excerpt_html}}</p>
      </div>
    </div>
  `).join('');
}}

function esc(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}
</script>
</body>
</html>"""

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index_page():
    return HTML_PAGE


@app.route("/pages/<path:filename>")
def serve_page_image(filename):
    return send_from_directory(BASE_DIR / "pages", filename)


@app.route("/pdf")
def serve_pdf():
    if not PDF_PATH.exists():
        return "PDF not found", 404
    return send_file(PDF_PATH, mimetype="application/pdf")


@app.route("/search")
def search():
    q   = request.args.get("q", "").strip()
    top = min(int(request.args.get("top", DEFAULT_TOP)), 50)

    if not q:
        return jsonify({"query": q, "results": [], "total": 0})

    query_tokens = tokenize(q)
    if not query_tokens:
        return jsonify({"query": q, "results": [], "total": 0})

    scores = _bm25.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked = [i for i in ranked if scores[i] > 0][:top]

    max_score = float(scores[ranked[0]]) if ranked else 1.0
    results   = []
    for idx in ranked:
        r     = _records[idx]
        score = float(scores[idx])
        excerpt_raw  = make_excerpt(r["text"], query_tokens)
        excerpt_esc  = html_lib.escape(excerpt_raw)
        excerpt_html = highlight(excerpt_esc, [html_lib.escape(t) for t in query_tokens])
        results.append({
            "page":         r["page"],
            "img_path":     r["img_path"].replace("\\", "/"),
            "score":        round(score, 2),
            "score_pct":    int(min(100, (score / max_score) * 100)),
            "excerpt_html": excerpt_html,
        })

    return jsonify({"query": q, "results": results, "total": len(ranked)})


if __name__ == "__main__":
    print("Starting server at http://localhost:5000")
    app.run(debug=False, port=5000)
