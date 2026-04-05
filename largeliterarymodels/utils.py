"""Utility functions for largeliterarymodels."""

import os

from .providers import check_api_keys


def available_models(verbose=False):
    """Return a list of model suggestions based on which API keys are set."""
    keys = check_api_keys(verbose=verbose)
    models = []
    if "ANTHROPIC_API_KEY" in keys:
        models.extend(["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"])
    if "OPENAI_API_KEY" in keys:
        models.extend(["gpt-4o-mini", "gpt-4o"])
    if "GEMINI_API_KEY" in keys:
        models.extend(["gemini-2.5-flash", "gemini-2.5-pro"])
    return models


def pdf_to_images(pdf_path, output_dir=None, dpi=200, fmt="png", pages=None):
    """Convert a PDF to one image per page.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory for output images. Defaults to a sibling
                    directory named ``{stem}_pages/`` next to the PDF.
        dpi: Resolution in dots per inch. 200 is good for text extraction
             (sharp enough for OCR/VLM, ~1600x2100 per letter-size page).
        fmt: Image format ("png" or "jpg"). JPG is smaller but lossy.
        pages: Optional iterable of 0-based page indices to extract.
               None means all pages.

    Returns:
        list[str]: Paths to the generated image files, in page order.
    """
    import fitz  # PyMuPDF

    if output_dir is None:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(os.path.dirname(pdf_path), f"{stem}_pages")
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    zoom = dpi / 72  # fitz default is 72 dpi
    mat = fitz.Matrix(zoom, zoom)

    if pages is None:
        page_indices = range(n_pages)
    else:
        page_indices = [p for p in pages if 0 <= p < n_pages]

    ext = "jpg" if fmt.lower() in ("jpg", "jpeg") else "png"
    paths = []

    for i in page_indices:
        out_path = os.path.join(output_dir, f"page_{i:04d}.{ext}")
        if not os.path.exists(out_path):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat)
            if ext == "jpg":
                pix.save(out_path, jpg_quality=85)
            else:
                pix.save(out_path)
        paths.append(out_path)

    doc.close()
    print(f"  {len(paths)} pages → {output_dir} ({dpi} dpi, {ext})")
    return paths
