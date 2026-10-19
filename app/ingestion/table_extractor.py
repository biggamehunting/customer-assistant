"""Table extraction from digitally-generated PDF pages using PyMuPDF."""
from __future__ import annotations

from typing import Any

import fitz  # PyMuPDF


class TableExtractor:
    """Extracts tables from fitz.Page objects (digital PDFs only).

    For scanned pages, table extraction is handled by the OCR backend.
    """

    def extract_from_page(self, page: fitz.Page, page_number: int) -> list[dict[str, Any]]:
        """Return a list of table dicts found on this page."""
        tables: list[dict[str, Any]] = []

        try:
            # PyMuPDF 1.23+ has find_tables()
            found = page.find_tables()
            for table in found.tables:
                extracted = table.extract()
                if not extracted or len(extracted) < 2:
                    continue  # need at least a header row + one data row

                headers = [str(cell or "").strip() for cell in extracted[0]]
                rows = [
                    [str(cell or "").strip() for cell in row]
                    for row in extracted[1:]
                    if any(cell for cell in row)
                ]

                if not headers or not rows:
                    continue

                tables.append({
                    "headers": headers,
                    "rows": rows,
                    "page_start": page_number,
                    "page_end": page_number,
                    "caption": None,
                    "section_context": None,
                    "extraction_method": "pymupdf",
                    "confidence": 1.0,
                })
        except AttributeError:
            # Older PyMuPDF without find_tables() — gracefully skip
            pass
        except Exception:
            pass

        return tables
