# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A local search tool for the BMW E36 3 Series Service Manual (1992–1998). The PDF is scanned (image-based, no embedded text), so the pipeline renders each page as a PNG, OCRs it with Tesseract, and builds a BM25 keyword index. Queries return matching pages with text excerpts and links back to the original PDF.

## Setup (one-time)

```powershell
pip install pymupdf rank-bm25 pytesseract pillow
# Tesseract is installed at C:\Program Files\Tesseract-OCR\tesseract.exe via winget
python index.py        # render 759 pages to pages\ (~2 min)
python index.py --ocr  # OCR all pages with 12 parallel workers (~4 min)
```

## Commands

```powershell
# Search
python query.py "brake bleeding"
python query.py "torque spec" --top 15
python query.py "cooling system" --no-browser

# Re-index (if PDF or OCR settings change)
python index.py --force --ocr

# LLM-augmented answers (Phase 2, requires ANTHROPIC_API_KEY)
python query.py "brake bleeding" --llm
```

## Architecture

Two scripts, no server, no database:

**`index.py`** — run once. Two modes:
- Default: opens PDF with PyMuPDF (`fitz`), extracts embedded text per page, renders each page to `pages/page_NNNN.png` at 150 DPI. Resumable — skips pages whose PNG already exists.
- `--ocr`: reads existing PNGs, runs Tesseract via `pytesseract` across all CPU cores using `ThreadPoolExecutor` (threads, not processes — pytesseract spawns external subprocesses so there's no GIL bottleneck). Resumable — skips pages with >20 chars of text already in the index. Writes atomically via `.json.tmp` → rename.

**`query.py`** — run per search. Loads `index.json` into memory, builds `BM25Okapi` from the page corpus, scores and ranks pages, prints excerpts to terminal, writes `results.html`, opens it in the browser.

**`index.json`** — the entire index as a flat JSON array. Each record: `{page, page_index, img_path, text}`. ~15 MB for 759 pages.

**`results.html`** — ephemeral, regenerated on every query. Images are served via relative paths (`pages/page_NNNN.png`). PDF links use `file:///...pdf#page=N` so clicking a page thumbnail opens the original PDF at that exact page.

## Key design decisions

- `_ocr_worker()` in `index.py` is a module-level function (not a closure) so it remains picklable if the executor is ever switched from threads to processes.
- BM25 index is rebuilt in memory on every `query.py` run (~<1s for 759 pages). No persistence of the BM25 model needed.
- `llm_answer()` in `query.py` is a fully-wired stub — imports `anthropic` lazily inside the function so Phase 1 (no LLM) works without the package installed. Activated with `--llm` flag + `ANTHROPIC_API_KEY` env var.
- Windows terminal encoding: avoid non-ASCII characters in `print()` calls (cp1252 codec). HTML files use UTF-8 and are fine.
- Tesseract path is hardcoded to `C:\Program Files\Tesseract-OCR\tesseract.exe` with a fallback to `shutil.which("tesseract")`.
