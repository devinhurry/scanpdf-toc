"""CLI entry point for scanpdf-toc."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from scanpdf_toc.generator import OpenAiVisionClient, PdfTocGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanpdf-toc",
        description=(
            "Generate PDF bookmarks for scanned documents by detecting TOC pages "
            "with a multimodal LLM."
        ),
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PDF path (default: <input>-with-toc.pdf)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum pages to scan for TOC pages (default: 20)",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="1-based page number where TOC detection starts (default: 1)",
    )
    parser.add_argument(
        "--offset-window",
        type=int,
        default=40,
        help="Search window for page-offset alignment (default: 40)",
    )
    parser.add_argument(
        "--render-zoom",
        type=float,
        default=2.0,
        help="Rasterization zoom when rendering PDF pages (default: 2.0)",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Maximum image edge length sent to LLM in pixels (default: 1600)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Multimodal model name (required unless OPENAI_MODEL is set)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or set OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        help="Optional API base URL (e.g. https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level (default: INFO)",
    )
    return parser


def create_generator(args: argparse.Namespace) -> PdfTocGenerator:
    model = args.model or os.getenv("OPENAI_MODEL")
    if not model:
        raise ValueError("Set --model or OPENAI_MODEL to a vision-capable model.")

    llm = OpenAiVisionClient(
        model=model,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    return PdfTocGenerator(
        llm_client=llm,
        max_detection_pages=args.max_pages,
        start_page=args.start_page,
        offset_search_window=args.offset_window,
        render_zoom=args.render_zoom,
        max_image_size=args.max_image_size,
    )


def format_token_usage_line(usage: dict[str, int]) -> str:
    return (
        "tokens: "
        f"total={usage.get('total_tokens', 0)} "
        f"input={usage.get('input_tokens', 0)} "
        f"output={usage.get('output_tokens', 0)} "
        f"requests={usage.get('requests', 0)}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    generator = create_generator(args)
    try:
        output = generator.generate(args.pdf, args.output)
    except Exception as exc:  # pragma: no cover - CLI error path.
        parser.error(str(exc))
        return 1
    print(Path(output))
    usage = generator.get_token_usage()
    if usage:
        print(format_token_usage_line(usage))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
