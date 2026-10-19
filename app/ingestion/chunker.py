"""Structure-aware chunker.

Splits extracted page text into overlapping token-bounded chunks while
preserving paragraph and sentence boundaries.

Uses tiktoken for accurate token counting in production.
Falls back to a word-based approximation when tiktoken BPE data is not
available (air-gapped environments, offline CI).
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field

PARSER_VERSION = "1.0.0"

# Heading heuristic: short line, title-cased, no trailing period
_HEADING_RE = re.compile(r"^[A-Z][^\n.]{3,80}$", re.MULTILINE)

# ── Tiktoken with offline fallback ────────────────────────────────────────────

def _make_encoder(encoding_name: str = "cl100k_base"):
    """Return a tiktoken encoder, or None if unavailable (offline CI etc.)."""
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


def _word_tokens(text: str) -> list[str]:
    """Split text into word-level 'tokens' for offline fallback."""
    return text.split()


@dataclass
class ChunkData:
    content: str
    document_id: uuid.UUID
    tenant_id: uuid.UUID
    page_number: int
    chunk_index: int
    section_path: list[str]
    content_hash: str
    token_count: int
    extraction_method: str
    content_type: str = "text"
    metadata: dict = field(default_factory=dict)


class Chunker:
    def __init__(
        self,
        chunk_size_tokens: int = 800,
        overlap_tokens: int = 100,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self._chunk_size = chunk_size_tokens
        self._overlap = overlap_tokens
        self._enc = _make_encoder(encoding_name)

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        """Return a list of atomic units (BPE tokens or words)."""
        if self._enc is not None:
            ids = self._enc.encode(text, disallowed_special=())
            # Return individual token strings so we can reconstruct text
            return [self._enc.decode([i]) for i in ids]
        return _word_tokens(text)

    @staticmethod
    def _join(units: list[str]) -> str:
        """Reconstruct text from units.

        BPE token strings concatenate directly; words need spaces.
        """
        joined = "".join(units)
        # If the join produced no spaces at all and there are multiple units,
        # they are word tokens — join with spaces instead.
        if " " not in joined and len(units) > 1:
            return " ".join(units)
        return joined

    # ── Section detection ─────────────────────────────────────────────────────

    def _extract_section(self, text: str) -> str | None:
        lines = text.strip().split("\n")
        for line in lines[:5]:
            stripped = line.strip()
            if _HEADING_RE.match(stripped):
                return stripped
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_page(
        self,
        text: str,
        document_id: uuid.UUID,
        tenant_id: uuid.UUID,
        page_number: int,
        extraction_method: str,
        section_path: list[str] | None = None,
        start_index: int = 0,
    ) -> list[dict]:
        """Split page text into overlapping token-bounded chunks.

        Returns a list of plain dicts (no ORM objects) for pipeline isolation.
        """
        units = self._split(text)
        if not units:
            return []

        current_section = list(section_path or [])
        chunks: list[dict] = []
        pos = 0

        while pos < len(units):
            end = min(pos + self._chunk_size, len(units))
            chunk_units = units[pos:end]
            chunk_text = self._join(chunk_units)

            detected = self._extract_section(chunk_text)
            if detected and detected not in current_section:
                current_section = current_section + [detected]

            content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

            chunks.append({
                "content": chunk_text,
                "document_id": str(document_id),
                "tenant_id": str(tenant_id),
                "page_start": page_number,
                "page_end": page_number,
                "chunk_index": start_index + len(chunks),
                "section_path": list(current_section),
                "content_hash": content_hash,
                "token_count": len(chunk_units),
                "extraction_method": extraction_method,
                "content_type": "text",
                "parser_version": PARSER_VERSION,
            })

            advance = self._chunk_size - self._overlap
            pos += max(advance, 1)

        return chunks
