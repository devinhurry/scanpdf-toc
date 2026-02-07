from __future__ import annotations

import base64
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Sequence

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PyMuPDF is required for scanpdf-toc. Install with: pip install pymupdf"
    ) from exc

try:
    import pymupdf  # type: ignore
except ImportError:  # pragma: no cover - optional fallback
    pymupdf = None  # type: ignore


logger = logging.getLogger(__name__)


def _fitz_attr(name: str):
    attr = getattr(fitz, name, None)
    if attr is not None:
        return attr
    if pymupdf is not None:
        attr = getattr(pymupdf, name, None)
    if attr is None:
        raise AttributeError(f"PyMuPDF installation is missing fitz.{name}.")
    return attr


def _fitz_open(*args, **kwargs):
    open_attr = getattr(fitz, "open", None)
    if callable(open_attr):
        return open_attr(*args, **kwargs)
    document_cls = getattr(fitz, "Document", None)
    if document_cls is not None:
        return document_cls(*args, **kwargs)
    if pymupdf is not None:
        pymupdf_open = getattr(pymupdf, "open", None)
        if callable(pymupdf_open):
            return pymupdf_open(*args, **kwargs)
    raise AttributeError("PyMuPDF installation is missing fitz.open / fitz.Document.")


Matrix = _fitz_attr("Matrix")


try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai is optional at import time.
    OpenAI = None  # type: ignore

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

if Image is not None:  # pragma: no cover - depends on Pillow availability
    if hasattr(Image, "Resampling"):
        _PIL_RESAMPLING = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    else:
        _PIL_RESAMPLING = Image.LANCZOS  # type: ignore[attr-defined]
else:
    _PIL_RESAMPLING = None


@dataclass
class TocEntry:
    """Represents a single table-of-contents entry."""

    title: str
    page_number: int
    children: List["TocEntry"] = field(default_factory=list)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "TocEntry":
        title = str(payload.get("title") or "").strip()
        if not title:
            raise ValueError(f"title is required, payload={payload!r}")

        page_number = payload.get("page_number")
        if page_number is None:
            raise ValueError(f"page_number is required, payload={payload!r}")

        page_number = int(page_number)
        if page_number <= 0:
            raise ValueError(
                f"page_number must be greater than 0, payload={payload!r}"
            )

        children_payload = payload.get("children") or []
        if not isinstance(children_payload, list):
            raise ValueError(f"children must be a list, payload={payload!r}")

        children = [TocEntry.from_dict(item) for item in children_payload]
        return TocEntry(title=title, page_number=page_number, children=children)


class LlmClient:
    """Minimal LLM interface for TOC recognition and page matching."""

    def is_toc_page(self, image_bytes: bytes) -> bool:
        raise NotImplementedError

    def extract_outline(self, toc_images: Sequence[bytes]) -> List[TocEntry]:
        raise NotImplementedError

    def page_matches_entry(self, entry: TocEntry, image_bytes: bytes) -> bool:
        raise NotImplementedError

    def detect_printed_page_number(self, image_bytes: bytes) -> Optional[int]:
        return None

    def get_token_usage(self) -> Optional[dict[str, int]]:
        return None


class OpenAiVisionClient(LlmClient):
    """OpenAI-compatible multimodal client."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        use_responses: Optional[bool] = None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("The `openai` package is required for OpenAiVisionClient.")

        resolved_model = model or os.getenv("OPENAI_MODEL")
        if not resolved_model:
            raise RuntimeError("Set --model or OPENAI_MODEL to a vision-capable model.")

        self.model = resolved_model
        self.temperature = temperature
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY.")

        base_url = api_base or os.getenv("OPENAI_BASE_URL")

        default_headers: dict[str, str] = {}
        http_referer = os.getenv("OPENAI_HTTP_REFERER")
        client_title = os.getenv("OPENAI_CLIENT_TITLE")
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if client_title:
            default_headers["X-Title"] = client_title

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._usage_requests = 0

        has_responses_api = hasattr(self.client, "responses")
        if use_responses is None:
            self._use_responses = has_responses_api
            self._force_responses = False
        elif use_responses:
            if not has_responses_api:
                raise RuntimeError(
                    "Installed `openai` package does not support Responses API. "
                    "Upgrade `openai` or disable responses mode."
                )
            self._use_responses = True
            self._force_responses = True
        else:
            self._use_responses = False
            self._force_responses = False

    @staticmethod
    def _encode_image(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("ascii")

    @staticmethod
    def _as_responses_content(
        user_content: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for part in user_content:
            part_type = part.get("type")
            if part_type == "text":
                converted.append(
                    {"type": "input_text", "text": str(part.get("text") or "")}
                )
                continue

            if part_type == "image_url":
                image_payload = part.get("image_url")
                image_url: str = ""
                detail: Optional[str] = None
                if isinstance(image_payload, dict):
                    image_url = str(image_payload.get("url") or "")
                    detail = image_payload.get("detail")
                elif image_payload is not None:
                    image_url = str(image_payload)

                item: dict[str, Any] = {
                    "type": "input_image",
                    "image_url": image_url,
                }
                if detail:
                    item["detail"] = detail
                converted.append(item)
                continue

            converted.append(part)
        return converted

    @staticmethod
    def _get_value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _as_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _record_usage(self, response: Any) -> None:
        usage = self._get_value(response, "usage")
        if usage is None:
            return

        input_tokens = self._as_int(
            self._get_value(usage, "input_tokens", self._get_value(usage, "prompt_tokens"))
        )
        output_tokens = self._as_int(
            self._get_value(usage, "output_tokens", self._get_value(usage, "completion_tokens"))
        )
        total_tokens = self._as_int(self._get_value(usage, "total_tokens"))

        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        if input_tokens is None and output_tokens is None and total_tokens is None:
            return

        self._usage_requests += 1
        self._input_tokens += input_tokens or 0
        self._output_tokens += output_tokens or 0
        self._total_tokens += total_tokens or 0

    def get_token_usage(self) -> Optional[dict[str, int]]:
        if self._usage_requests == 0:
            return None
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self._total_tokens,
            "requests": self._usage_requests,
        }

    def _complete(self, system_prompt: str, user_content: list[dict[str, Any]]) -> str:
        logger.info("Calling LLM: model=%s use_responses=%s", self.model, self._use_responses)
        if self._use_responses:
            try:
                return self._complete_responses(system_prompt, user_content)
            except Exception:
                if self._force_responses:
                    raise
                logger.debug("Responses API call failed, falling back to Chat Completions.", exc_info=True)
                self._use_responses = False
        return self._complete_chat(system_prompt, user_content)

    def _complete_responses(
        self, system_prompt: str, user_content: list[dict[str, Any]]
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._as_responses_content(user_content)},
            ],
            temperature=self.temperature,
        )
        self._record_usage(response)

        text_chunks: List[str] = []
        for block in getattr(response, "output", []):
            if self._get_value(block, "type") != "message":
                continue
            for content in self._get_value(block, "content", []):
                if self._get_value(content, "type") == "output_text":
                    text_chunks.append(self._get_value(content, "text", "") or "")

        if not text_chunks and hasattr(response, "output_text"):
            output_text = response.output_text  # type: ignore[attr-defined]
            if isinstance(output_text, str):
                text_chunks.append(output_text)
            else:
                text_chunks.append("".join(output_text))

        return "\n".join(text_chunks).strip()

    def _complete_chat(self, system_prompt: str, user_content: list[dict[str, Any]]) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        self._record_usage(response)

        if not getattr(response, "choices", None):
            return ""

        message = response.choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text") or "")
            return "\n".join(texts).strip()
        return ""

    @staticmethod
    def _json_from_completion(text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return {}
        return {}

    def is_toc_page(self, image_bytes: bytes) -> bool:
        prompt = (
            "Decide whether this scanned PDF page belongs to the document table of contents. "
            'Respond as JSON: {"is_toc": true|false}.'
        )
        response = self._complete(
            system_prompt="You classify scanned PDF pages.",
            user_content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image(image_bytes)}"
                    },
                },
            ],
        )
        data = self._json_from_completion(response)
        return bool(data.get("is_toc"))

    def extract_outline(self, toc_images: Sequence[bytes]) -> List[TocEntry]:
        prompt = (
            "The provided scanned images are TOC pages. Extract structured JSON: "
            '{"entries": [{"title": str, "page_number": int, "children": [...] }]}. '
            "Keep hierarchy when implied. Ignore decorative lines and non-TOC text."
        )
        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in toc_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image(image)}"
                    },
                }
            )

        response = self._complete(
            system_prompt="You convert scanned TOC pages into valid JSON outlines.",
            user_content=user_content,
        )

        data = self._json_from_completion(response)
        entries = data.get("entries") or []
        if not entries:
            logger.warning("LLM response did not contain entries.")
            return []

        return [TocEntry.from_dict(entry_payload) for entry_payload in entries]

    def page_matches_entry(self, entry: TocEntry, image_bytes: bytes) -> bool:
        prompt = (
            "Determine whether this scanned page matches the TOC entry "
            f"'{entry.title}' expected at page {entry.page_number}. "
            'Respond as JSON: {"match": true|false}.'
        )
        response = self._complete(
            system_prompt="You detect whether a scanned page matches a TOC entry.",
            user_content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image(image_bytes)}"
                    },
                },
            ],
        )
        data = self._json_from_completion(response)
        return bool(data.get("match"))

    def detect_printed_page_number(self, image_bytes: bytes) -> Optional[int]:
        prompt = (
            "Read the printed page number visible on this scanned page. "
            "Use the page number shown on the page itself (not the PDF index). "
            'Respond as JSON: {"printed_page_number": int|null}.'
        )
        response = self._complete(
            system_prompt="You detect printed page numbers on scanned document pages.",
            user_content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image(image_bytes)}"
                    },
                },
            ],
        )
        data = self._json_from_completion(response)
        value = data.get("printed_page_number")
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None


class PdfTocGenerator:
    """Generate TOC outlines for scanned PDFs using an LLM."""

    def __init__(
        self,
        llm_client: Optional[LlmClient] = None,
        max_detection_pages: int = 20,
        offset_search_window: int = 40,
        render_zoom: float = 2.0,
        max_image_size: int = 1600,
        start_page: int = 1,
    ) -> None:
        self.llm = llm_client or OpenAiVisionClient()
        self.max_detection_pages = max_detection_pages
        self.offset_search_window = offset_search_window
        self.render_zoom = render_zoom
        self.max_image_size = max_image_size
        self.start_page = max(start_page, 1) - 1

    def generate(
        self,
        pdf_path: Path | str,
        output_path: Optional[Path | str] = None,
    ) -> Path:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

        if output_path is None:
            output_path = pdf_path.with_name(f"{pdf_path.stem}-with-toc.pdf")
        output_path = Path(output_path)

        logger.info("Starting PDF TOC generation for %s", pdf_path)
        doc = _fitz_open(pdf_path)
        try:
            toc_pages = self._detect_toc_pages(doc)
            if not toc_pages:
                raise RuntimeError("No TOC pages were detected.")

            toc_images = [self._render_page(doc, index) for index in toc_pages]
            entries = self.llm.extract_outline(toc_images)
            if not entries:
                raise RuntimeError("Failed to extract TOC entries from LLM output.")

            offset = self._estimate_offset(doc, entries, toc_pages)
            toc_table = self._build_toc_table(doc.page_count, entries, offset)
            doc.set_toc(toc_table)
            doc.save(output_path)
            logger.info("Saved output PDF with bookmarks: %s", output_path)
        finally:
            doc.close()

        return output_path

    def get_token_usage(self) -> Optional[dict[str, int]]:
        return self.llm.get_token_usage()

    def _render_page(self, doc: fitz.Document, page_index: int) -> bytes:
        zoom_matrix = Matrix(self.render_zoom, self.render_zoom)
        page = doc.load_page(page_index)
        pixmap = page.get_pixmap(matrix=zoom_matrix, alpha=False)
        png_bytes = pixmap.tobytes("png")

        if Image is None:
            return png_bytes

        try:
            with Image.open(BytesIO(png_bytes)) as image:
                original_size = image.size
                if image.mode != "RGB":
                    image = image.convert("RGB")

                if (
                    self.max_image_size
                    and max(original_size) > self.max_image_size
                    and _PIL_RESAMPLING is not None
                ):
                    image.thumbnail(
                        (self.max_image_size, self.max_image_size),
                        resample=_PIL_RESAMPLING,
                    )

                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=60, optimize=True)
                return buffer.getvalue()
        except Exception:
            logger.debug("Image compression failed, using PNG bytes.", exc_info=True)

        return png_bytes

    def _detect_toc_pages(self, doc: fitz.Document) -> List[int]:
        limit = min(self.start_page + self.max_detection_pages, doc.page_count)
        toc_indexes: List[int] = []
        in_block = False

        for index in range(self.start_page, limit):
            image_bytes = self._render_page(doc, index)
            if self.llm.is_toc_page(image_bytes):
                toc_indexes.append(index)
                in_block = True
            elif in_block:
                # Stop after the first contiguous TOC block.
                break

        return toc_indexes

    def _pick_reference_entry(self, entries: Sequence[TocEntry]) -> Optional[TocEntry]:
        queue: List[TocEntry] = list(entries)
        fallback: Optional[TocEntry] = None

        while queue:
            entry = queue.pop(0)
            if entry.page_number <= 0:
                queue.extend(entry.children)
                continue
            if not entry.children:
                return entry
            if fallback is None:
                fallback = entry
            queue.extend(entry.children)

        return fallback

    def _estimate_offset(
        self,
        doc: fitz.Document,
        entries: Sequence[TocEntry],
        toc_pages: Sequence[int],
    ) -> int:
        offset_from_numbers = self._estimate_offset_from_printed_numbers(doc, toc_pages)
        if offset_from_numbers is not None:
            logger.info("Using page offset from printed page numbers: %s", offset_from_numbers)
            return offset_from_numbers

        reference = self._pick_reference_entry(entries)
        if not reference:
            logger.warning("Could not estimate offset from printed page numbers and no reference entry found.")
            return 0

        start_index = toc_pages[-1] + 1 if toc_pages else 0
        search_end = min(doc.page_count, start_index + self.offset_search_window)

        for page_index in range(start_index, search_end):
            image_bytes = self._render_page(doc, page_index)
            if self.llm.page_matches_entry(reference, image_bytes):
                offset = (page_index + 1) - reference.page_number
                logger.info(
                    "Using page offset from entry matching: %s (entry=%r, pdf_page=%s)",
                    offset,
                    reference.title,
                    page_index + 1,
                )
                return offset
        logger.warning("Offset estimation failed, defaulting to 0.")
        return 0

    def _estimate_offset_from_printed_numbers(
        self,
        doc: fitz.Document,
        toc_pages: Sequence[int],
    ) -> Optional[int]:
        start_index = toc_pages[-1] + 1 if toc_pages else 0
        search_end = min(doc.page_count, start_index + self.offset_search_window)
        deltas: List[int] = []

        for page_index in range(start_index, search_end):
            image_bytes = self._render_page(doc, page_index)
            printed_page = self.llm.detect_printed_page_number(image_bytes)
            if not printed_page:
                continue
            deltas.append((page_index + 1) - printed_page)

        if not deltas:
            return None

        counts = Counter(deltas)
        best_offset, frequency = counts.most_common(1)[0]
        logger.info(
            "Printed page offset candidates: best=%s count=%s total=%s",
            best_offset,
            frequency,
            len(deltas),
        )
        return int(best_offset)

    def _build_toc_table(
        self,
        page_count: int,
        entries: Sequence[TocEntry],
        offset: int,
    ) -> List[List[int | str]]:
        toc_rows: List[List[int | str]] = []

        def walk(entry: TocEntry, level: int) -> None:
            actual_page = entry.page_number + offset
            actual_page = min(max(actual_page, 1), page_count)
            toc_rows.append([level, entry.title, actual_page])
            for child in entry.children:
                walk(child, level + 1)

        for entry in entries:
            walk(entry, level=1)

        return toc_rows
