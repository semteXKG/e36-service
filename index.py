#!/usr/bin/env python3
"""
index.py  --  One-time indexing for the BMW E36 service manual.

Usage:
    python index.py            # render pages + extract embedded text
    python index.py --ocr      # OCR the existing page images (for scanned PDFs)
    python index.py --force    # delete existing index/pages and rebuild from scratch
    python index.py --dpi 200  # higher resolution when rendering (default: 150)
    python index.py --workers 8  # parallel OCR workers (default: all CPU cores)

For scanned PDFs (no embedded text):
    1. python index.py            # renders page PNGs — run this first
    2. Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
    3. python index.py --ocr      # OCR the PNGs, update index.json
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import shutil
import sys
import threading
import time

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("ERROR: PyMuPDF not installed.\nRun:  pip install pymupdf rank-bm25 pytesseract pillow")

BASE_DIR   = pathlib.Path(__file__).parent.resolve()
PDF_PATH   = BASE_DIR / "BMW - E36 - 3 Series Service Manual (1992 - 1998) EN.pdf"
PAGES_DIR  = BASE_DIR / "pages"
INDEX_PATH = BASE_DIR / "index.json"

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def parse_args():
    p = argparse.ArgumentParser(description="Index the BMW E36 service manual PDF.")
    p.add_argument("--ocr",     action="store_true",
                   help="OCR existing page PNGs using Tesseract (for scanned PDFs)")
    p.add_argument("--force",   action="store_true",
                   help="Delete existing index/pages and rebuild from scratch")
    p.add_argument("--dpi",     type=int, default=150,
                   help="Render resolution in DPI (default: 150)")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel OCR workers (default: all CPU cores)")
    return p.parse_args()


def load_existing_index():
    if not INDEX_PATH.exists():
        return {}
    try:
        with open(INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {r["page_index"]: r for r in data.get("records", [])}
    except Exception:
        return {}


def setup_tesseract():
    try:
        import pytesseract
    except ImportError:
        sys.exit("ERROR: pytesseract not installed.\nRun:  pip install pytesseract pillow")

    tess_exe = pathlib.Path(TESSERACT_PATH)
    if tess_exe.exists():
        pytesseract.pytesseract.tesseract_cmd = str(tess_exe)
    else:
        found = shutil.which("tesseract")
        if not found:
            sys.exit(
                "ERROR: Tesseract executable not found.\n"
                f"Expected: {TESSERACT_PATH}\n"
                "Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "Use the default install path during setup."
            )
        pytesseract.pytesseract.tesseract_cmd = found

    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        sys.exit(f"ERROR: Tesseract found but failed to run: {e}")

    return pytesseract


def _ocr_worker(img_path_str: str) -> tuple:
    """Top-level function so it's picklable. Returns (page_index, page_num, img_name, text)."""
    import pytesseract
    from PIL import Image

    img_path = pathlib.Path(img_path_str)
    page_num = int(img_path.stem.split("_")[1])

    tess_exe = pathlib.Path(TESSERACT_PATH)
    if tess_exe.exists():
        pytesseract.pytesseract.tesseract_cmd = str(tess_exe)

    try:
        img  = Image.open(img_path)
        raw  = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
        text = " ".join(raw.split())
    except Exception:
        text = ""

    return (page_num - 1, page_num, img_path.name, text)


def run_ocr_mode(args, existing: dict) -> list:
    """OCR all page PNGs in parallel, skipping pages already indexed."""
    setup_tesseract()  # validate Tesseract before spawning workers

    png_files = sorted(PAGES_DIR.glob("page_*.png"))
    if not png_files:
        sys.exit(
            "ERROR: No page PNGs found in pages/\n"
            "Run `python index.py` first to render the pages."
        )

    total   = len(png_files)
    workers = args.workers or os.cpu_count() or 4

    # Resumability: skip pages that already have meaningful OCR text
    already_done = {
        idx for idx, r in existing.items()
        if len(r.get("text", "")) > 20
    }
    to_process = [f for f in png_files
                  if int(f.stem.split("_")[1]) - 1 not in already_done]
    skipped = total - len(to_process)

    print(f"OCR mode — {total} pages, {workers} parallel workers")
    if skipped:
        print(f"Resuming: {skipped} pages already done, {len(to_process)} remaining")
    print()

    results   = {}
    done      = skipped
    lock      = threading.Lock()
    t0        = time.time()

    # Seed results with already-completed pages
    for idx, r in existing.items():
        if idx in already_done:
            results[idx] = r

    def print_progress():
        elapsed = time.time() - t0
        eta_str = ""
        if done > skipped:
            rate    = (done - skipped) / elapsed
            remain  = total - done
            eta     = remain / rate if rate > 0 else 0
            eta_str = f"  ETA {eta:.0f}s"
        print(f"\r  {done}/{total} pages done  ({elapsed:.0f}s elapsed){eta_str}   ",
              end="", flush=True)

    print_progress()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_ocr_worker, str(f)): f for f in to_process
        }
        for future in concurrent.futures.as_completed(future_map):
            try:
                page_index, page_num, img_name, text = future.result()
            except Exception as e:
                img_name = future_map[future].name
                page_num = int(pathlib.Path(img_name).stem.split("_")[1])
                page_index = page_num - 1
                text = ""

            results[page_index] = {
                "page":       page_num,
                "page_index": page_index,
                "img_path":   f"pages/{img_name}",
                "text":       text,
            }

            with lock:
                done += 1
                print_progress()

    print()
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({len(to_process)} pages OCR'd, {skipped} skipped)")

    return [results[i] for i in sorted(results)]


def run_render_mode(args) -> list:
    """Render PDF pages to PNGs and extract embedded text."""
    existing = load_existing_index()

    try:
        doc = fitz.open(str(PDF_PATH))
    except Exception as e:
        sys.exit(f"ERROR: Could not open PDF: {e}")

    total = len(doc)
    mat   = fitz.Matrix(args.dpi / 72, args.dpi / 72)

    print(f"PDF:        {PDF_PATH.name}")
    print(f"Pages:      {total}")
    print(f"Resolution: {args.dpi} DPI")
    print(f"Output:     {PAGES_DIR}")
    print()

    records = []
    skipped = 0
    t0      = time.time()

    for i in range(total):
        img_path = PAGES_DIR / f"page_{i + 1:04d}.png"
        elapsed  = time.time() - t0
        print(f"\r  Page {i + 1:>4}/{total}  ({elapsed:.0f}s elapsed)  ", end="", flush=True)

        if i in existing and img_path.exists():
            records.append(existing[i])
            skipped += 1
            continue

        try:
            page = doc[i]
            raw  = page.get_text("text")
            text = " ".join(raw.split())
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(str(img_path))
        except Exception as e:
            print(f"\n  WARNING: page {i + 1} failed ({e}), skipping.")
            text = ""

        records.append({
            "page":       i + 1,
            "page_index": i,
            "img_path":   f"pages/page_{i + 1:04d}.png",
            "text":       text,
        })

    doc.close()
    print()

    new_pages = total - skipped
    elapsed   = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({new_pages} new pages rendered, {skipped} skipped)")

    with_text = sum(1 for r in records if len(r["text"]) > 20)
    if with_text < total * 0.3:
        print()
        print(f"WARNING: Only {with_text}/{total} pages have readable text.")
        print("This PDF appears to be scanned. Run OCR for better results:")
        print("  python index.py --ocr")

    return records


def write_index(records: list):
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {"version": 1, "pdf": PDF_PATH.name, "records": records},
            f,
            ensure_ascii=False,
        )
    tmp.replace(INDEX_PATH)


def main():
    args = parse_args()

    if not PDF_PATH.exists():
        sys.exit(f"ERROR: PDF not found at:\n  {PDF_PATH}")

    if args.force:
        print("--force: removing existing index and pages...")
        if PAGES_DIR.exists():
            shutil.rmtree(PAGES_DIR)
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()

    PAGES_DIR.mkdir(exist_ok=True)

    if args.ocr:
        existing = load_existing_index()
        records  = run_ocr_mode(args, existing)
    else:
        records = run_render_mode(args)

    write_index(records)

    print(f"Index: {INDEX_PATH}")
    print(f"Pages: {PAGES_DIR}")
    print()
    print('Ready to search:  python query.py "your question here"')


if __name__ == "__main__":
    main()
