# scanpdf-toc

[中文说明](README.zh-CN.md)

`scanpdf-toc` generates bookmark outlines for scanned PDFs by combining PyMuPDF rendering with multimodal LLM analysis.

## Features

- Detects TOC pages from a configurable start page.
- Extracts hierarchical entries (`title`, `page_number`, `children`) from TOC images.
- Estimates printed-page to PDF-page offset automatically.
- Writes final bookmarks into PDF outline metadata.
- Keeps image payload small by resizing/compressing before sending to the model.

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Set API credentials

Use `OPENAI_API_KEY`. If your provider is OpenAI-compatible, set `OPENAI_BASE_URL` to the provider endpoint.

```bash
export OPENAI_API_KEY="your_openai_key"
export OPENAI_BASE_URL="https://your-provider.example/v1" # optional
```

### 3. Run

```bash
scanpdf-toc ./book.pdf -o ./book-with-toc.pdf
```

The tool prints the output PDF path after completion.

## CLI Usage

```bash
scanpdf-toc INPUT_PDF [options]
```

Common options:

- `-o, --output`: output path (default: `<input>-with-toc.pdf`)
- `--start-page`: first page to start TOC detection (default: `1`)
- `--max-pages`: max pages scanned for TOC block (default: `20`)
- `--offset-window`: pages scanned for offset alignment (default: `40`)
- `--render-zoom`: render zoom factor (default: `2.0`)
- `--max-image-size`: max image edge length in pixels (default: `1600`)
- `--model`: multimodal model name (default: `gpt-4o-mini`)
- `--api-key`: override env API key
- `--api-base`: custom API base URL
- `--log-level`: `CRITICAL|ERROR|WARNING|INFO|DEBUG`

## Python API

```python
from scanpdf_toc import PdfTocGenerator

generator = PdfTocGenerator(max_detection_pages=30, start_page=3)
output_path = generator.generate("book.pdf")
print(output_path)
```

## Security Notes

- Do not commit API keys.
- Use environment variables for credentials.
- Input page images are sent to your configured model provider.
- This repository includes `.gitignore` rules for `.env` and common secrets.

## Development

```bash
pip install -e .[dev]
pytest
```

## License

MIT License.
