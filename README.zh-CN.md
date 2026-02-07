# scanpdf-toc（中文说明）

[English README](README.md)

`scanpdf-toc` 是一个扫描版 PDF 目录（书签）生成工具：它使用 PyMuPDF 渲染页面，并借助多模态 LLM 自动识别目录结构，再写回 PDF Outline。

## 功能

- 从指定起始页开始检测目录页。
- 从目录页图片提取分层条目（`title`、`page_number`、`children`）。
- 自动估算纸书页码与 PDF 实际页码的偏移量。
- 将最终目录写入 PDF 书签。
- 对发送给模型的图片做缩放和压缩，降低成本。

## 快速开始

### 1. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 配置 API 密钥

统一使用 `OPENAI_API_KEY`。如果你使用兼容 OpenAI 接口的其他提供商，可通过 `OPENAI_BASE_URL` 指定地址。

```bash
export OPENAI_API_KEY="your_openai_key"
export OPENAI_BASE_URL="https://your-provider.example/v1" # 可选
```

### 3. 运行

```bash
scanpdf-toc ./book.pdf -o ./book-with-toc.pdf
```

执行完成后会输出生成文件路径。

## 命令行参数

常用参数：

- `-o, --output`：输出文件路径（默认 `<输入名>-with-toc.pdf`）
- `--start-page`：从第几页开始识别目录（默认 `1`）
- `--max-pages`：最多扫描多少页查找目录（默认 `20`）
- `--offset-window`：估算偏移时向后搜索页数（默认 `40`）
- `--render-zoom`：页面渲染倍率（默认 `2.0`）
- `--max-image-size`：发送给模型的图片最长边像素（默认 `1600`）
- `--model`：多模态模型名称（默认 `gpt-4o-mini`）
- `--api-key`：直接指定 API Key（覆盖环境变量）
- `--api-base`：自定义 API Base
- `--log-level`：日志级别

## Python 调用示例

```python
from scanpdf_toc import PdfTocGenerator

generator = PdfTocGenerator(max_detection_pages=30, start_page=3)
output_path = generator.generate("book.pdf")
print(output_path)
```

## 安全说明

- 不要把 API Key 提交到仓库。
- 建议只用环境变量传递密钥。
- 目录识别时会把页面图片发送到你配置的模型服务商。
- 本仓库已在 `.gitignore` 中忽略 `.env` 等常见敏感文件。

## 开发

```bash
pip install -e .[dev]
pytest
```

## 许可证

MIT License。
