"""Render selected PDF pages to PNGs with PyMuPDF.

Output layout: ``{images_dir}/{doc}/page_{n}.png`` at the configured DPI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .config import Config, load_config


def render_page(pdf_path: str, page_number: int, out_path: Path, dpi: int) -> Path:
    """Render a single 1-indexed PDF page to ``out_path`` as PNG."""
    import fitz  # PyMuPDF, lazy import

    out_path.parent.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        index = page_number - 1
        if index < 0 or index >= doc.page_count:
            raise IndexError(
                f"Page {page_number} out of range for {pdf_path} "
                f"({doc.page_count} pages)"
            )
        page = doc[index]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(str(out_path))
    return out_path


def render_document(
    doc: str,
    pdf_path: str,
    pages: List[int],
    images_dir: Path,
    dpi: int,
    overwrite: bool = False,
) -> List[Path]:
    """Render all manifest pages of one document. Returns the PNG paths."""
    out_paths: List[Path] = []
    for n in pages:
        out_path = Path(images_dir) / doc / f"page_{n}.png"
        if out_path.exists() and not overwrite:
            out_paths.append(out_path)
            continue
        render_page(pdf_path, n, out_path, dpi)
        out_paths.append(out_path)
    return out_paths


def _pdf_path_for(doc: str, pdfs_dir: Path) -> Optional[Path]:
    for ext in (".pdf", ".PDF"):
        candidate = pdfs_dir / f"{doc}{ext}"
        if candidate.exists():
            return candidate
    return None


def render_all(
    cfg: Optional[Config] = None, overwrite: bool = False
) -> Dict[str, List[Path]]:
    """Render every document/page listed in the config manifests."""
    cfg = cfg or load_config()
    pdfs_dir = cfg.input_pdfs()
    images_dir = cfg.images_dir()
    dpi = cfg.dpi

    rendered: Dict[str, List[Path]] = {}
    for doc, pages in cfg.manifests.items():
        pdf_path = _pdf_path_for(doc, pdfs_dir)
        if pdf_path is None:
            raise FileNotFoundError(
                f"No PDF found for document '{doc}' in {pdfs_dir}"
            )
        rendered[doc] = render_document(
            doc, str(pdf_path), pages, images_dir, dpi, overwrite=overwrite
        )
    return rendered


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render manifest pages to PNG.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-render existing PNGs"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    rendered = render_all(cfg, overwrite=args.overwrite)
    total = sum(len(v) for v in rendered.values())
    print(f"Rendered {total} page(s) across {len(rendered)} document(s).")


if __name__ == "__main__":
    main()
