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

from query import tokenize, make_excerpt, highlight, clean_text, CSS, compute_tag_score, TAG_BOOST, compute_phrase_bonus, deduplicate_by_chapter

BASE_DIR   = pathlib.Path(__file__).parent.resolve()
INDEX_PATH = BASE_DIR / "index.json"
TAGS_PATH  = BASE_DIR / "tags.json"
PDF_PATH   = pathlib.Path(os.environ.get("PDF_PATH",  BASE_DIR / "BMW - E36 - 3 Series Service Manual (1992 - 1998) EN.pdf"))
PAGES_DIR  = pathlib.Path(os.environ.get("PAGES_DIR", BASE_DIR / "pages"))
DEFAULT_TOP = 10

# ── Load index once at startup ────────────────────────────────────────────────

if not INDEX_PATH.exists():
    sys.exit("ERROR: index.json not found.\nRun `python index.py --ocr` first.")

print("Loading index...", end=" ", flush=True)
with open(INDEX_PATH, encoding="utf-8") as f:
    _data = json.load(f)
_records = _data["records"]
for _r in _records:
    _r["text"] = clean_text(_r["text"])
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
  flex-direction: column;
  z-index: 200;
  background: #fff;
  box-shadow: -4px 0 32px rgba(0,0,0,0.28);
}
#viewer-panel.open {
  transform: translateX(0);
}
#viewer-toolbar {
  flex-shrink: 0;
  height: 44px;
  background: #1a1a1a;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 0 8px;
  border-bottom: 1px solid #333;
}
.tb-btn {
  background: none;
  border: none;
  color: #aaa;
  font-size: 1rem;
  cursor: pointer;
  padding: 5px 8px;
  border-radius: 3px;
  line-height: 1;
  flex-shrink: 0;
}
.tb-btn:hover { color: #fff; background: #333; }
.tb-btn:disabled { color: #555; cursor: default; }
#viewer-close { color: #fff; }
#viewer-close:hover { background: #c0392b !important; }
.tb-sep {
  width: 1px;
  height: 20px;
  background: #333;
  margin: 0 4px;
  flex-shrink: 0;
}
#viewer-page-badge {
  font-size: 0.75rem;
  font-weight: 600;
  color: #ddd;
  min-width: 70px;
  text-align: center;
  white-space: nowrap;
}
#viewer-zoom-label {
  font-size: 0.75rem;
  color: #aaa;
  min-width: 36px;
  text-align: center;
  white-space: nowrap;
}
#viewer-pdf-link {
  margin-left: auto;
  font-size: 0.75rem;
  font-weight: 600;
  color: #fff;
  background: #1C69D4;
  text-decoration: none;
  padding: 5px 10px;
  border-radius: 3px;
  white-space: nowrap;
}
#viewer-pdf-link:hover { background: #1557b0; }
#viewer-img-wrap {
  flex: 1;
  overflow: hidden;
  min-width: 0;
  position: relative;
}
#viewer-canvas-wrap {
  width: 100%;
  height: 100%;
  overflow: auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #525659;
  position: relative;
}
.pdf-page {
  flex-shrink: 0;
  margin: 6px 0;
  background: #fff;
  box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}
.pdf-page canvas { display: block; }
#viewer-loading {
  display: none;
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%,-50%);
  color: #fff;
  font-size: 0.85rem;
  background: rgba(0,0,0,0.55);
  padding: 6px 16px;
  border-radius: 4px;
  pointer-events: none;
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

@media (max-width: 640px) {
  header {
    height: auto;
    padding: 10px 14px;
    flex-wrap: wrap;
    gap: 8px;
  }
  .logo h1 { font-size: 0.85rem; }
  .search-wrap {
    width: 100%;
    max-width: none;
    margin-left: 0;
    flex: none;
  }
  .search-wrap input[type=search] {
    font-size: 1rem;
    padding: 10px 14px;
  }
  #ask-btn {
    font-size: 0.85rem;
    padding: 10px 14px;
  }
  main { padding: 14px 12px; }
  .card-right { padding: 12px 14px; }
  .score-label, .score-bar-wrap { display: none; }
  .empty-state { padding: 32px 0; }

  /* PDF link above image on mobile */
  .thumb-wrap { order: 1; }
  .img-links  { order: 0; margin-bottom: 4px; }

  /* Alternating card colors — one solid tone per card */
  .card {
    border-top: 3px solid #1C69D4;
    box-shadow: 0 2px 8px rgba(0,0,0,0.13);
  }
  .card:nth-child(odd)  { background: #eef1f8; }
  .card:nth-child(even) { background: #dde3f0; }
  .card-left  { background: transparent; border-bottom: 1px solid rgba(0,0,0,0.08); }
  .card-right { background: transparent; }
  .excerpt    { background: rgba(0,0,0,0.04); border-left-color: rgba(0,0,0,0.12); }
}
"""

HTML_PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BMW E36 &mdash; Service Manual Search</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="apple-touch-icon" href="/favicon.svg">
<style>{CSS}{_EXTRA_CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/marked@14/marked.min.js"
        integrity="sha384-lqPzN0kmFw9t2syAMwVPM4VbAyqsz/lPyYWbb2Xt6nSPM0WPNrpSWCUBgdcAdgnC"
        crossorigin="anonymous"
        onerror="window._markedFailed=true"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
<script>
if (!window._markedFailed) {{
  marked.use({{ mangle: false, headerIds: false }});
}}
function renderMarkdown(text) {{
  if (window._markedFailed || typeof marked === 'undefined') return '<pre>' + text.replace(/</g,'&lt;') + '</pre>';
  return DOMPurify.sanitize(marked.parse(text));
}}
</script>
<script src="https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js"></script>
<script>
pdfjsLib.GlobalWorkerOptions.workerSrc =
  'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js';
</script>
</head>
<body>
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
  <div id="viewer-toolbar">
    <button class="tb-btn" id="viewer-close" onclick="closeViewer()" title="Close (Esc)">&#10005;</button>
    <div class="tb-sep"></div>
    <button class="tb-btn" id="btn-prev" onclick="prevPage()" title="Previous page (&#8592;)">&#9664;</button>
    <span id="viewer-page-badge">— / —</span>
    <button class="tb-btn" id="btn-next" onclick="nextPage()" title="Next page (&#8594;)">&#9654;</button>
    <div class="tb-sep"></div>
    <button class="tb-btn" onclick="zoomOut()" title="Zoom out">&#8722;</button>
    <span id="viewer-zoom-label">fit</span>
    <button class="tb-btn" onclick="zoomIn()" title="Zoom in">&#43;</button>
    <button class="tb-btn" onclick="zoomFit()" title="Reset to fit">&#8596;</button>
    <button class="tb-btn" id="btn-fs" onclick="toggleFullscreen()" title="Fullscreen (F)">&#x26F6;</button>
    <a id="viewer-pdf-link" href="#" target="_blank" title="Open PDF">PDF &#8599;</a>
  </div>
  <div id="viewer-img-wrap">
    <div id="viewer-canvas-wrap"></div>
    <div id="viewer-loading">Loading…</div>
  </div>
</div>
<script>
const qInput   = document.getElementById('q');
const statusEl = document.getElementById('status-bar');
const cardsEl  = document.getElementById('cards');
let debounceTimer;

qInput.addEventListener('keydown', e => {{ if (e.key === 'Enter') qInput.blur(); }});

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
    <div class="card" data-page="${{r.page}}" onclick="openViewer(${{r.page}}, '/${{r.img_path}}')" style="cursor:pointer">
      <div class="card-left">
        <div class="thumb-wrap">
          <img src="/${{r.img_path}}" alt="Page ${{r.page}}" loading="lazy"
               onerror="this.parentElement.style.display='none'">
        </div>
        <div class="img-links">
          <a href="/pdf#page=${{r.page}}" target="_blank" onclick="event.stopPropagation()">PDF &#8599;</a>
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
let _pdfDoc     = null;
let _scale      = null;   // null = fit-to-width
let _fitScale   = 1;
let _pageEls    = [];     // [{{div, canvas, rendered, rendering}}]
let _renderObs  = null;
let _rafId      = null;

async function _loadPdf() {{
  if (_pdfDoc) return _pdfDoc;
  _pdfDoc = await pdfjsLib.getDocument('/pdf').promise;
  return _pdfDoc;
}}

async function _renderPageEl(pageNum) {{
  const el = _pageEls[pageNum - 1];
  if (!el || el.rendering || el.rendered) return;
  el.rendering = true;
  try {{
    const pdf      = await _loadPdf();
    const pg       = await pdf.getPage(pageNum);
    const scale    = _scale === null ? _fitScale : _scale;
    const viewport = pg.getViewport({{scale}});
    el.canvas.width  = viewport.width;
    el.canvas.height = viewport.height;
    await pg.render({{canvasContext: el.canvas.getContext('2d'), viewport}}).promise;
    el.rendered = true;
  }} catch(e) {{
    if (e && e.name !== 'RenderingCancelledException') console.error(e);
  }} finally {{
    el.rendering = false;
  }}
}}

async function _initScrollViewer(scrollToPage) {{
  const loading = document.getElementById('viewer-loading');
  loading.style.display = 'block';
  try {{
    const pdf  = await _loadPdf();
    const wrap = document.getElementById('viewer-canvas-wrap');
    const firstPg = await pdf.getPage(1);
    const natVp   = firstPg.getViewport({{scale: 1}});
    _fitScale     = wrap.clientWidth / natVp.width;
    const scale   = _scale === null ? _fitScale : _scale;
    const pgW     = Math.floor(natVp.width  * scale);
    const pgH     = Math.floor(natVp.height * scale);
    const stride  = pgH + 12;  // page height + top/bottom margin (6px each)

    if (_renderObs) {{ _renderObs.disconnect(); _renderObs = null; }}
    wrap.innerHTML = '';
    _pageEls = [];

    const frag = document.createDocumentFragment();
    for (let i = 1; i <= pdf.numPages; i++) {{
      const div    = document.createElement('div');
      div.className = 'pdf-page';
      div.id        = 'pdf-p' + i;
      div.style.width  = pgW + 'px';
      div.style.height = pgH + 'px';
      const canvas = document.createElement('canvas');
      div.appendChild(canvas);
      frag.appendChild(div);
      _pageEls.push({{div, canvas, rendered: false, rendering: false}});
    }}
    wrap.appendChild(frag);

    _updateZoomLabel();
    document.getElementById('viewer-page-badge').textContent =
      (scrollToPage || 1) + ' / ' + pdf.numPages;
    document.getElementById('btn-prev').disabled = ((scrollToPage || 1) <= 1);
    document.getElementById('btn-next').disabled = ((scrollToPage || 1) >= pdf.numPages);

    if (scrollToPage && scrollToPage > 1) {{
      wrap.scrollTop = (scrollToPage - 1) * stride;
    }}

    _renderObs = new IntersectionObserver(entries => {{
      entries.forEach(e => {{
        if (e.isIntersecting) _renderPageEl(parseInt(e.target.id.slice(5)));
      }});
    }}, {{root: wrap, rootMargin: '400px 0px', threshold: 0}});
    _pageEls.forEach(({{div}}) => _renderObs.observe(div));

    wrap.onscroll = () => {{
      if (_rafId) return;
      _rafId = requestAnimationFrame(() => {{
        _rafId = null;
        const mid     = wrap.scrollTop + wrap.clientHeight / 2;
        const pageNum = Math.min(pdf.numPages, Math.max(1, Math.floor(mid / stride) + 1));
        if (pageNum === _viewerPage) return;
        _viewerPage = pageNum;
        document.getElementById('viewer-page-badge').textContent = pageNum + ' / ' + pdf.numPages;
        document.getElementById('btn-prev').disabled = (pageNum <= 1);
        document.getElementById('btn-next').disabled = (pageNum >= pdf.numPages);
        document.getElementById('viewer-pdf-link').href = '/pdf#page=' + pageNum;
        document.querySelectorAll('.card-active').forEach(el => el.classList.remove('card-active'));
        const card = document.querySelector('.card[data-page="' + pageNum + '"]');
        if (card) card.classList.add('card-active');
      }});
    }};
  }} finally {{
    loading.style.display = 'none';
  }}
}}

function _updateZoomLabel() {{
  const el = document.getElementById('viewer-zoom-label');
  if (el) el.textContent = _scale === null ? 'fit' : Math.round((_scale / _fitScale) * 100) + '%';
}}

function _scrollToPage(page, smooth) {{
  const wrap = document.getElementById('viewer-canvas-wrap');
  if (!wrap || !_pageEls.length) return;
  const stride = _pageEls[0].div.offsetHeight + 12;
  const top    = (page - 1) * stride;
  if (smooth) wrap.scrollTo({{top, behavior: 'smooth'}});
  else        wrap.scrollTop = top;
}}

async function openViewer(page, imgPath) {{
  const panel = document.getElementById('viewer-panel');
  if (panel.classList.contains('open') && _viewerPage === page) {{ closeViewer(); return; }}
  panel.classList.add('open');
  document.getElementById('viewer-pdf-link').href = '/pdf#page=' + page;
  if (_pageEls.length === 0) {{
    _viewerPage = page;
    await _initScrollViewer(page);
  }} else {{
    _viewerPage = page;
    _scrollToPage(page, false);
  }}
  document.querySelectorAll('.card-active').forEach(el => el.classList.remove('card-active'));
  const card = document.querySelector('.card[data-page="' + page + '"]');
  if (card) card.classList.add('card-active');
}}

function closeViewer() {{
  _viewerPage = null;
  document.getElementById('viewer-panel').classList.remove('open');
  document.querySelectorAll('.card-active').forEach(el => el.classList.remove('card-active'));
}}

function prevPage() {{
  if (!_viewerPage || _viewerPage <= 1) return;
  _scrollToPage(_viewerPage - 1, true);
}}

async function nextPage() {{
  if (!_viewerPage) return;
  const pdf = await _loadPdf();
  if (_viewerPage >= pdf.numPages) return;
  _scrollToPage(_viewerPage + 1, true);
}}

async function zoomIn() {{
  const cur = _viewerPage || 1;
  if (_scale === null) _scale = _fitScale;
  _scale = Math.min(_scale * 1.25, _fitScale * 5);
  await _initScrollViewer(cur);
}}

async function zoomOut() {{
  const cur = _viewerPage || 1;
  if (_scale === null) _scale = _fitScale;
  _scale = Math.max(_scale / 1.25, _fitScale * 0.2);
  await _initScrollViewer(cur);
}}

async function zoomFit() {{
  _scale = null;
  await _initScrollViewer(_viewerPage || 1);
}}

function toggleFullscreen() {{
  const panel = document.getElementById('viewer-panel');
  if (!document.fullscreenElement) {{
    panel.requestFullscreen && panel.requestFullscreen();
  }} else {{
    document.exitFullscreen && document.exitFullscreen();
  }}
}}

document.addEventListener('fullscreenchange', () => {{
  const btn = document.getElementById('btn-fs');
  if (btn) btn.textContent = document.fullscreenElement ? '⧅' : '⛶';
}});

document.addEventListener('keydown', e => {{
  if (!document.getElementById('viewer-panel').classList.contains('open')) return;
  if      (e.key === 'Escape')                                {{ closeViewer(); }}
  else if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')    {{ e.preventDefault(); prevPage(); }}
  else if (e.key === 'ArrowRight' || e.key === 'ArrowDown')  {{ e.preventDefault(); nextPage(); }}
  else if (e.key === '+' || e.key === '=')                   {{ zoomIn(); }}
  else if (e.key === '-')                                    {{ zoomOut(); }}
  else if (e.key === 'f' || e.key === 'F')                   {{ toggleFullscreen(); }}
}});

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
          textEl.innerHTML = renderMarkdown(buffer);
        }} else if (msg.type === 'done') {{
          document.getElementById('llm-title').textContent = 'AI Answer — ' + msg.model;
          metaEl.innerHTML = msg.elapsed + 's'
            + ' &nbsp;&middot;&nbsp; ~' + msg.chunk_count + ' chunks'
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

# ── Favicon ───────────────────────────────────────────────────────────────────

_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="50" fill="#1a1a1a"/>
  <circle cx="50" cy="50" r="43" fill="white"/>
  <clipPath id="c"><circle cx="50" cy="50" r="43"/></clipPath>
  <rect x="7"  y="7"  width="43" height="43" fill="#1C69D4" clip-path="url(#c)"/>
  <rect x="50" y="50" width="43" height="43" fill="#1C69D4" clip-path="url(#c)"/>
  <rect x="50" y="7"  width="43" height="43" fill="white"   clip-path="url(#c)"/>
  <rect x="7"  y="50" width="43" height="43" fill="white"   clip-path="url(#c)"/>
  <line x1="50" y1="7"  x2="50" y2="93" stroke="#1a1a1a" stroke-width="2"/>
  <line x1="7"  y1="50" x2="93" y2="50" stroke="#1a1a1a" stroke-width="2"/>
  <circle cx="50" cy="50" r="43" fill="none" stroke="#c0c0c0" stroke-width="2.5"/>
  <circle cx="50" cy="50" r="47" fill="none" stroke="#1a1a1a" stroke-width="6"/>
</svg>"""

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index_page():
    return HTML_PAGE


@app.route("/favicon.svg")
@app.route("/favicon.ico")
def favicon():
    return Response(_FAVICON_SVG, mimetype="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.route("/pages/<path:filename>")
def serve_page_image(filename):
    return send_from_directory(PAGES_DIR, filename)


@app.route("/pdf")
def serve_pdf():
    if not PDF_PATH.exists():
        return "PDF not found", 404
    resp = send_file(PDF_PATH, mimetype="application/pdf", conditional=True)
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


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
        base_url = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        model    = os.environ.get("LLM_MODEL",    "llama-3.3-70b-versatile")

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
        chunk_count = 0
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
                    chunk_count += 1
                    yield event({"type": "chunk", "text": text})
        except Exception as e:
            yield event({"type": "error", "error": str(e)})
            return

        yield event({
            "type":        "done",
            "elapsed":     round(time.time() - t0, 1),
            "chunk_count": chunk_count,
            "model":       model,
            "pages":       [r["page"] for r in top5],
        })

    return Response(
        stream_with_context(generate(q)),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Transfer-Encoding": "chunked"},
    )


@app.route("/search")
def search():
    q   = request.args.get("q", "").strip()
    try:
        top = min(int(request.args.get("top", DEFAULT_TOP)), 50)
    except (ValueError, TypeError):
        top = DEFAULT_TOP

    if not q:
        return jsonify({"query": q, "results": [], "total": 0})

    deduped, query_tokens = _top_records(q, n=top)
    if not deduped:
        return jsonify({"query": q, "results": [], "total": 0})

    max_score = deduped[0]["score"]
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
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--host", default="127.0.0.1")
    _p.add_argument("--port", type=int, default=5000)
    _args = _p.parse_args()
    print(f"Starting server at http://{_args.host}:{_args.port}")
    app.run(debug=False, host=_args.host, port=_args.port, threaded=True)
