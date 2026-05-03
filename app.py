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
import os
import pathlib
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).parent / ".env")
except ImportError:
    pass

try:
    from flask import Flask, Response, jsonify, request, send_file, send_from_directory, stream_with_context
except ImportError:
    sys.exit("ERROR: flask not installed.\nRun: pip install flask")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("ERROR: rank-bm25 not installed.\nRun: pip install rank-bm25")

from query import tokenize, make_excerpt, highlight, CSS, compute_tag_score, TAG_BOOST, compute_phrase_bonus, deduplicate_by_chapter, llm_answer

BASE_DIR   = pathlib.Path(__file__).parent.resolve()
INDEX_PATH = BASE_DIR / "index.json"
TAGS_PATH  = BASE_DIR / "tags.json"
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

_tags = {}
if TAGS_PATH.exists():
    with open(TAGS_PATH, encoding="utf-8") as f:
        _tags = json.load(f)
    print(f"Tags loaded ({len(_tags)} pages)")

# ── HTML page ─────────────────────────────────────────────────────────────────

_EXTRA_CSS = """
/* Webapp: header search input */
header { height: 64px; }

/* Full-height two-column layout */
body {
  height: 100vh;
  overflow: hidden;
  display: flex;
  flex-direction: row;
}
#left-col {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
main {
  flex: 1;
  overflow-y: auto;
  padding: 28px 24px;
  max-width: none;
  margin: 0;
  display: block;
}
#viewer-panel {
  position: fixed;
  top: 0;
  right: 0;
  height: 100vh;
  width: 100vw;
  transform: translateX(100%);
  transition: transform 0.28s cubic-bezier(.4,0,.2,1);
  display: flex;
  flex-direction: row;
  z-index: 200;
  background: #fff;
  box-shadow: -4px 0 32px rgba(0,0,0,0.28);
}
#viewer-panel.open {
  transform: translateX(0);
}
#viewer-header {
  width: 28px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 0 10px;
  gap: 16px;
  background: #1a1a1a;
  border-radius: 4px 0 0 4px;
}
#viewer-close {
  background: none;
  border: none;
  color: #888;
  font-size: 1.1rem;
  cursor: pointer;
  line-height: 1;
  padding: 3px;
  border-radius: 2px;
  flex-shrink: 0;
}
#viewer-close:hover { color: #fff; background: #444; }
#viewer-page-badge {
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  font-size: 0.68rem;
  font-weight: 700;
  color: #fff;
  background: #1C69D4;
  padding: 8px 4px;
  border-radius: 3px;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  white-space: nowrap;
}
#viewer-pdf-link {
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  font-size: 0.68rem;
  color: #7fb3f5;
  text-decoration: none;
  white-space: nowrap;
}
#viewer-pdf-link:hover { color: #fff; }
#viewer-img-wrap {
  flex: 1;
  overflow: hidden;
  min-width: 0;
}
#viewer-img-wrap iframe {
  width: 100%;
  height: 100%;
  border: none;
  display: block;
}

/* Thumbnail — clickable div replacing anchor */
.thumb-wrap {
  width: 100%;
  cursor: zoom-in;
  position: relative;
}
.thumb-wrap img {
  width: 100%;
  height: auto;
  display: block;
  border: 1px solid #ddd;
  border-radius: 2px;
  transition: box-shadow 0.15s;
}
.thumb-wrap:hover img {
  box-shadow: 0 4px 12px rgba(28,105,212,0.25);
}
.thumb-wrap::after {
  content: "view";
  position: absolute;
  bottom: 6px;
  right: 6px;
  background: rgba(28,105,212,0.85);
  color: #fff;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.5px;
  padding: 2px 7px;
  border-radius: 2px;
  opacity: 0;
  transition: opacity 0.15s;
  text-transform: uppercase;
}
.thumb-wrap:hover::after { opacity: 1; }

/* Active card highlight */
.card-active { border-left: 4px solid #1C69D4 !important; }


.search-wrap {
  flex: 1;
  max-width: 580px;
  margin-left: auto;
  display: flex;
  gap: 8px;
  align-items: center;
}
.search-wrap input[type=search] {
  flex: 1;
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
#ask-btn {
  padding: 9px 16px;
  font-size: 0.88rem;
  font-family: inherit;
  font-weight: 600;
  border: none;
  border-radius: 2px;
  background: #1C69D4;
  color: #fff;
  cursor: pointer;
  white-space: nowrap;
  transition: background 0.15s;
  letter-spacing: 0.3px;
}
#ask-btn:hover:not(:disabled) { background: #1557b0; }
#ask-btn:disabled { background: #555; cursor: not-allowed; }

/* LLM answer panel */
#llm-panel {
  background: #fff;
  border-left: 4px solid #1C69D4;
  border-radius: 4px;
  padding: 20px 24px;
  margin-bottom: 24px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  line-height: 1.7;
}
#llm-panel h2 {
  font-size: 0.75rem;
  color: #1C69D4;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 10px;
  font-weight: 700;
}
#llm-meta {
  margin-top: 10px;
  font-size: 0.78rem;
  color: #999;
  border-top: 1px solid #eee;
  padding-top: 8px;
}
#llm-panel p, #llm-panel li { margin-bottom: 6px; }
#llm-panel strong { color: #1C1C1C; }
#llm-panel ul, #llm-panel ol { padding-left: 20px; }
#llm-panel code { background: #f2f2f2; padding: 1px 5px; border-radius: 2px; font-size: 0.88em; }

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

/* Tag chips */
.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 2px;
}
.tag {
  font-size: 0.72rem;
  font-weight: 600;
  padding: 2px 9px;
  border-radius: 2px;
  letter-spacing: 0.3px;
  text-transform: lowercase;
  border: 1px solid transparent;
}
.tag-high   { background: #1C69D4; color: #fff; }
.tag-mid    { background: #e8f0fc; color: #1C69D4; border-color: #b8cef5; }
.tag-low    { background: #f2f2f2; color: #888;    border-color: #ddd; }
"""

HTML_PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BMW E36 &mdash; Service Manual Search</title>
<style>{CSS}{_EXTRA_CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<div id="left-col">
<header>
  <div class="logo">
    <div class="logo-ring"><span></span><span></span><span></span><span></span></div>
    <h1>BMW E36 &mdash; Service Manual</h1>
  </div>
  <div class="search-wrap">
    <input id="q" type="search" placeholder="Search {len(_records)} pages..." autofocus autocomplete="off">
    <button id="ask-btn" onclick="askAI()" title="Ask AI for a synthesized answer">Ask AI</button>
  </div>
</header>
<main>
  <p id="status-bar">Type a query to search the manual.</p>
  <div id="llm-panel" style="display:none"></div>
  <div id="cards">
    <div class="empty-state"><span>&#128269;</span>Enter keywords above &mdash; results appear as you type.</div>
  </div>
</main>
</div>
<div id="viewer-panel">
  <div id="viewer-header">
    <button id="viewer-close" onclick="closeViewer()" title="Close (Esc)">&times;</button>
    <span id="viewer-page-badge">Page —</span>
    <a id="viewer-pdf-link" href="#" target="_blank">PDF &#8599;</a>
  </div>
  <div id="viewer-img-wrap">
    <iframe id="viewer-frame" src="" title="Manual page"></iframe>
  </div>
</div>
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
    <div class="card" data-page="${{r.page}}">
      <div class="card-left">
        <div class="thumb-wrap" onclick="openViewer(${{r.page}}, '/${{r.img_path}}')">
          <img src="/${{r.img_path}}" alt="Page ${{r.page}}" loading="lazy"
               onerror="this.parentElement.style.display='none'">
        </div>
        <div class="img-links">
          <a href="/pdf#page=${{r.page}}" target="_blank">PDF &#8599;</a>
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
        <div class="tags">${{(r.tags || []).map(t => '<span class="tag ' + (t.weight >= 0.8 ? 'tag-high' : t.weight >= 0.4 ? 'tag-mid' : 'tag-low') + '">' + esc(t.tag) + '</span>').join('')}}</div>
        <p class="excerpt">${{r.excerpt_html}}</p>
      </div>
    </div>
  `).join('');
}}

function esc(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

let _viewerPage = null;

function openViewer(page, imgPath) {{
  if (_viewerPage === page) {{ closeViewer(); return; }}
  _viewerPage = page;
  document.getElementById('viewer-frame').src = '/pdf#page=' + page;
  document.getElementById('viewer-page-badge').textContent = 'Page ' + page;
  document.getElementById('viewer-pdf-link').href = '/pdf#page=' + page;
  document.getElementById('viewer-panel').classList.add('open');
  document.body.classList.add('viewer-open');
  document.querySelectorAll('.card-active').forEach(el => el.classList.remove('card-active'));
  const card = document.querySelector('.card[data-page="' + page + '"]');
  if (card) card.classList.add('card-active');
}}

function closeViewer() {{
  _viewerPage = null;
  document.getElementById('viewer-panel').classList.remove('open');
  document.getElementById('viewer-frame').src = '';
  document.body.classList.remove('viewer-open');
  document.querySelectorAll('.card-active').forEach(el => el.classList.remove('card-active'));
}}

document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeViewer(); }});

async function askAI() {{
  const q = qInput.value.trim();
  if (!q) return;
  const btn   = document.getElementById('ask-btn');
  const panel = document.getElementById('llm-panel');
  btn.disabled    = true;
  btn.textContent = 'Thinking…';
  panel.style.display = 'block';
  panel.innerHTML = '<h2 id="llm-title">AI Answer</h2><p id="llm-text" style="color:#888">Connecting…</p><div id="llm-meta"></div>';
  const textEl = document.getElementById('llm-text');
  const metaEl = document.getElementById('llm-meta');
  let buffer = '';
  try {{
    const resp = await fetch('/llm/stream?q=' + encodeURIComponent(q));
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let raw = '';
    while (true) {{
      const {{done, value}} = await reader.read();
      if (done) break;
      raw += decoder.decode(value, {{stream: true}});
      const parts = raw.split('\\n\\n');
      raw = parts.pop();
      for (const part of parts) {{
        if (!part.startsWith('data: ')) continue;
        const msg = JSON.parse(part.slice(6));
        if (msg.type === 'chunk') {{
          if (!buffer) textEl.style.color = '';
          buffer += msg.text;
          textEl.innerHTML = marked.parse(buffer);
        }} else if (msg.type === 'done') {{
          document.getElementById('llm-title').textContent = 'AI Answer — ' + msg.model;
          metaEl.innerHTML = msg.elapsed + 's'
            + ' &nbsp;&middot;&nbsp; ~' + msg.completion_tokens + ' tokens'
            + ' &nbsp;&middot;&nbsp; pages: ' + msg.pages.join(', ');
        }} else if (msg.type === 'error') {{
          document.getElementById('llm-title').textContent = 'AI Answer';
          textEl.style.color = '#CC0000';
          textEl.textContent = msg.error;
        }}
      }}
    }}
  }} catch(e) {{
    textEl.style.color = '#CC0000';
    textEl.textContent = 'Error: ' + e.message;
  }} finally {{
    btn.disabled    = false;
    btn.textContent = 'Ask AI';
  }}
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


@app.route("/llm")
def llm_route():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "No query provided"})

    query_tokens = tokenize(q)
    if not query_tokens:
        return jsonify({"error": "Empty query after tokenization"})

    bm25_scores  = _bm25.get_scores(query_tokens)
    final_scores = []
    for i, r in enumerate(_records):
        page_tags    = _tags.get(str(r["page"]), [])
        tag_score    = compute_tag_score(page_tags, query_tokens)
        phrase_bonus = compute_phrase_bonus(r["text"], query_tokens)
        final_scores.append(float(bm25_scores[i]) + TAG_BOOST * tag_score + phrase_bonus)

    ranked = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
    ranked = [i for i in ranked if final_scores[i] > 0][:15]

    pre_dedup = [{**_records[i], "score": final_scores[i]} for i in ranked]
    top5      = deduplicate_by_chapter(pre_dedup, _tags)[:5]

    answer = llm_answer(q, top5, BASE_DIR)
    if answer.startswith("ERROR:"):
        return jsonify({"error": answer})
    return jsonify({"query": q, "answer": answer})


def _top_records(q, n=5):
    query_tokens = tokenize(q)
    if not query_tokens:
        return [], query_tokens
    bm25_scores  = _bm25.get_scores(query_tokens)
    final_scores = []
    for i, r in enumerate(_records):
        page_tags    = _tags.get(str(r["page"]), [])
        tag_score    = compute_tag_score(page_tags, query_tokens)
        phrase_bonus = compute_phrase_bonus(r["text"], query_tokens)
        final_scores.append(float(bm25_scores[i]) + TAG_BOOST * tag_score + phrase_bonus)
    ranked = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
    ranked = [i for i in ranked if final_scores[i] > 0][:n * 3]
    pre    = [{**_records[i], "score": final_scores[i]} for i in ranked]
    return deduplicate_by_chapter(pre, _tags)[:n], query_tokens


@app.route("/llm/stream")
def llm_stream():
    from query import clean_text
    q = request.args.get("q", "").strip()

    def event(obj):
        return f"data: {json.dumps(obj)}\n\n"

    def generate(q):
        if not q:
            yield event({"type": "error", "error": "No query provided"})
            return

        top5, query_tokens = _top_records(q)
        if not top5:
            yield event({"type": "error", "error": "No matching pages found"})
            return

        api_key  = os.environ.get("LLM_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        base_url = os.environ.get("LLM_BASE_URL", "https://api.moonshot.ai/v1")
        model    = os.environ.get("LLM_MODEL",    "kimi-k2.6")

        if not api_key:
            yield event({"type": "error", "error": "LLM_API_KEY not set"})
            return

        try:
            from openai import OpenAI
        except ImportError:
            yield event({"type": "error", "error": "openai package not installed"})
            return

        parts   = ["[Page " + str(r["page"]) + "]\n" + clean_text(r["text"]) for r in top5]
        context = "\n\n---\n\n".join(parts)
        prompt  = (
            "You are a BMW E36 service manual assistant. "
            "Answer the question using only the provided manual excerpts. "
            "Format your response in Markdown: use **bold** for critical values (torque specs, fluid types, part numbers, temperatures), "
            "and numbered lists for procedures. "
            "When you refer to a manual page, always link it like this: [Page 73](/pdf#page=73). "
            "Be concise and specific.\n\n"
            + context + "\n\nQuestion: " + q
        )

        client = OpenAI(api_key=api_key, base_url=base_url)
        t0 = time.time()
        completion_tokens = 0
        try:
            stream = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    completion_tokens += 1
                    yield event({"type": "chunk", "text": text})
        except Exception as e:
            yield event({"type": "error", "error": str(e)})
            return

        yield event({
            "type":              "done",
            "elapsed":           round(time.time() - t0, 1),
            "completion_tokens": completion_tokens,
            "model":             model,
            "pages":             [r["page"] for r in top5],
        })

    return Response(
        stream_with_context(generate(q)),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Transfer-Encoding": "chunked"},
    )


@app.route("/search")
def search():
    q   = request.args.get("q", "").strip()
    top = min(int(request.args.get("top", DEFAULT_TOP)), 50)

    if not q:
        return jsonify({"query": q, "results": [], "total": 0})

    query_tokens = tokenize(q)
    if not query_tokens:
        return jsonify({"query": q, "results": [], "total": 0})

    bm25_scores  = _bm25.get_scores(query_tokens)
    final_scores = []
    for i, r in enumerate(_records):
        page_tags    = _tags.get(str(r["page"]), [])
        tag_score    = compute_tag_score(page_tags, query_tokens)
        phrase_bonus = compute_phrase_bonus(r["text"], query_tokens)
        final_scores.append(float(bm25_scores[i]) + TAG_BOOST * tag_score + phrase_bonus)
    ranked_all = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
    ranked_all = [i for i in ranked_all if final_scores[i] > 0][:top * 3]

    pre_dedup = [{**_records[i], "score": final_scores[i]} for i in ranked_all]
    deduped   = deduplicate_by_chapter(pre_dedup, _tags)[:top]

    max_score = deduped[0]["score"] if deduped else 1.0
    results   = []
    for r in deduped:
        score = r["score"]
        excerpt_raw  = make_excerpt(r["text"], query_tokens)
        excerpt_esc  = html_lib.escape(excerpt_raw)
        excerpt_html = highlight(excerpt_esc, [html_lib.escape(t) for t in query_tokens])
        page_tags = sorted(
            _tags.get(str(r["page"]), []),
            key=lambda t: t["weight"], reverse=True
        )
        results.append({
            "page":         r["page"],
            "img_path":     r["img_path"].replace("\\", "/"),
            "score":        round(score, 2),
            "score_pct":    int(min(100, (score / max_score) * 100)),
            "excerpt_html": excerpt_html,
            "tags":         page_tags,
        })

    return jsonify({"query": q, "results": results, "total": len(deduped)})


if __name__ == "__main__":
    print("Starting server at http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True)
