"""OCR routing: Tesseract (dev) or AWS Textract (prod).

Only called for pages where digital text extraction is insufficient.
Never called indiscriminately for every page.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.config import OCRProvider, get_settings
from app.core.exceptions import OCRError
from app.core.logging import get_logger

logger = get_logger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)


class OCRRouter:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def extract_text(self, image_bytes: bytes) -> tuple[str, float]:
        """Return (text, confidence).  Routes to configured OCR backend."""
        if self._settings.ocr_provider == OCRProvider.textract:
            return await self._textract_extract(image_bytes)
        return await self._tesseract_extract(image_bytes)

    async def extract_tables(self, image_bytes: bytes) -> list[dict]:
        """Extract structured tables from an image."""
        if self._settings.ocr_provider == OCRProvider.textract:
            return await self._textract_tables(image_bytes)
        # Tesseract doesn't support structured table extraction; return empty
        return []

    # ── Tesseract ─────────────────────────────────────────────────────────────

    async def _tesseract_extract(self, image_bytes: bytes) -> tuple[str, float]:
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(_executor, self._run_tesseract, image_bytes)
            return result
        except Exception as exc:
            raise OCRError(f"Tesseract OCR failed: {exc}") from exc

    @staticmethod
    def _run_tesseract(image_bytes: bytes) -> tuple[str, float]:
        import io

        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        words = [w for w in data["text"] if w.strip()]
        confs = [c for c, w in zip(data["conf"], data["text"]) if w.strip() and c != -1]
        text = " ".join(words)
        confidence = sum(confs) / len(confs) if confs else 0.0
        return text, confidence

    # ── Textract ──────────────────────────────────────────────────────────────

    async def _textract_extract(self, image_bytes: bytes) -> tuple[str, float]:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                _executor, self._run_textract_text, image_bytes
            )
        except Exception as exc:
            logger.warning("ocr.textract_failed_fallback", error=str(exc))
            return await self._tesseract_extract(image_bytes)

    def _run_textract_text(self, image_bytes: bytes) -> tuple[str, float]:
        import boto3

        settings = self._settings
        client = boto3.client("textract", region_name=settings.textract_region)
        response = client.detect_document_text(Document={"Bytes": image_bytes})

        lines: list[str] = []
        confidences: list[float] = []
        for block in response.get("Blocks", []):
            if block["BlockType"] == "LINE":
                lines.append(block.get("Text", ""))
                confidences.append(block.get("Confidence", 0.0))

        text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return text, avg_conf

    async def _textract_tables(self, image_bytes: bytes) -> list[dict]:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                _executor, self._run_textract_tables, image_bytes
            )
        except Exception as exc:
            logger.warning("ocr.textract_tables_failed", error=str(exc))
            return []

    def _run_textract_tables(self, image_bytes: bytes) -> list[dict]:
        import boto3

        settings = self._settings
        client = boto3.client("textract", region_name=settings.textract_region)
        response = client.analyze_document(
            Document={"Bytes": image_bytes},
            FeatureTypes=["TABLES"],
        )

        # Parse Textract block structure into simple dicts
        blocks = {b["Id"]: b for b in response.get("Blocks", [])}
        tables: list[dict] = []

        for block in blocks.values():
            if block["BlockType"] != "TABLE":
                continue
            cells: dict[tuple[int, int], str] = {}
            max_row = max_col = 0
            for rel in block.get("Relationships", []):
                if rel["Type"] == "CHILD":
                    for cell_id in rel["Ids"]:
                        cell = blocks.get(cell_id, {})
                        if cell.get("BlockType") == "CELL":
                            row = cell.get("RowIndex", 1)
                            col = cell.get("ColumnIndex", 1)
                            max_row = max(max_row, row)
                            max_col = max(max_col, col)
                            cell_text = " ".join(
                                blocks[w]["Text"]
                                for w_rel in cell.get("Relationships", [])
                                if w_rel["Type"] == "CHILD"
                                for w in w_rel["Ids"]
                                if blocks.get(w, {}).get("BlockType") == "WORD"
                            )
                            cells[(row, col)] = cell_text

            if not cells:
                continue

            headers = [cells.get((1, c), "") for c in range(1, max_col + 1)]
            rows = [
                [cells.get((r, c), "") for c in range(1, max_col + 1)]
                for r in range(2, max_row + 1)
            ]
            tables.append({
                "headers": headers,
                "rows": rows,
                "extraction_method": "textract",
                "confidence": block.get("Confidence", 1.0) / 100.0,
                "page_start": 0,
                "page_end": 0,
            })

        return tables
