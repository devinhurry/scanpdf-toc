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

Use `OPENAI_API_KEY`. For OpenAI-compatible providers, set `OPENAI_BASE_URL`.

```bash
export OPENAI_API_KEY="your_openai_key"
export OPENAI_BASE_URL="https://your-provider.example/v1" # optional
export OPENAI_MODEL="your_vision_model_id"                # required if --model is not set
```

Common providers (OpenAI-style):

| Provider | `OPENAI_BASE_URL` | OpenAI-style support | Recommended vision model for this project | Docs |
|---|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | Native | `gpt-4o-mini` | [GPT-4o mini](https://platform.openai.com/docs/models/gpt-4o-mini) |
| OpenRouter | `https://openrouter.ai/api/v1` | Yes | `openai/gpt-4o-mini` | [OpenRouter model page](https://openrouter.ai/openai/gpt-4o-mini), [Models API](https://openrouter.ai/docs/api-reference/list-available-models) |
| Gemini (Google) | `https://generativelanguage.googleapis.com/v1beta/openai/` | Yes | `gemini-3-flash-preview` | [OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai), [Gemini models](https://ai.google.dev/gemini-api/docs/models/gemini) |
| Kimi (Moonshot AI) | `https://api.moonshot.cn/v1` | Yes | `kimi-thinking-preview` | [Kimi thinking model](https://platform.moonshot.cn/blog/posts/kimi-thinking) |
| Qwen (DashScope) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Yes | `qwen3-vl-plus` | [Qwen-VL OpenAI compatibility](https://www.alibabacloud.com/help/en/model-studio/qwen-vl-compatible-with-openai) |
| MiniMax | `https://api.minimax.io/v1` | Yes (OpenAI format) | N/A (vision model not documented on OpenAI-compatible text endpoint) | [Compatible OpenAI API](https://platform.minimax.io/docs/api-reference/text-openai-api) |

Notes:

- Qwen international endpoint: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`.
- MiniMax China endpoint: `https://api.minimax.chat/v1`.
- Provider model IDs change over time; verify in each provider's model list.
- `scanpdf-toc` has no hardcoded default model.
- You must choose a vision-capable model (image input support).

Recommended copy-paste presets:

```bash
# OpenAI
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"

# OpenRouter
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_MODEL="openai/gpt-4o-mini"

# Gemini
export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export OPENAI_MODEL="gemini-3-flash-preview"

# Kimi
export OPENAI_BASE_URL="https://api.moonshot.cn/v1"
export OPENAI_MODEL="kimi-thinking-preview"

# Qwen (intl)
export OPENAI_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL="qwen3-vl-plus"
```

Free API by default?

- Not by default. The project does not hardcode a keyless/free provider.
- You can use free-tier models by setting your own provider key and base URL.
- Example (OpenRouter free variant, if available):

```bash
export OPENAI_API_KEY="your_openrouter_key"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
scanpdf-toc ./book.pdf --model openai/gpt-4o-mini:free
```

- OpenRouter free-model guidance: [FAQ: Free Models and Rate Limits](https://openrouter.ai/docs/faq#free-models-and-rate-limits).

### 3. Run

```bash
scanpdf-toc ./book.pdf --model <vision_model_id> -o ./book-with-toc.pdf
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
- `--model`: multimodal model name (required unless `OPENAI_MODEL` is set)
- `--api-key`: override env API key
- `--api-base`: custom API base URL
- `--log-level`: `CRITICAL|ERROR|WARNING|INFO|DEBUG`

## Python API

```python
from scanpdf_toc import PdfTocGenerator
from scanpdf_toc.generator import OpenAiVisionClient

llm = OpenAiVisionClient(model="your_vision_model_id")
generator = PdfTocGenerator(llm_client=llm, max_detection_pages=30, start_page=3)
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
