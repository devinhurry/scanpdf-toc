import pytest

from scanpdf_toc import generator as generator_mod
from scanpdf_toc.generator import OpenAiVisionClient, PdfTocGenerator, TocEntry


class FakeLlm:
    def is_toc_page(self, image_bytes: bytes) -> bool:
        return False

    def extract_outline(self, toc_images):
        return []

    def page_matches_entry(self, entry, image_bytes: bytes) -> bool:
        return False

    def detect_printed_page_number(self, image_bytes: bytes):
        return None


def test_toc_entry_from_dict_valid():
    entry = TocEntry.from_dict(
        {
            "title": "Chapter 1",
            "page_number": 5,
            "children": [{"title": "Section 1.1", "page_number": 6}],
        }
    )
    assert entry.title == "Chapter 1"
    assert entry.page_number == 5
    assert len(entry.children) == 1
    assert entry.children[0].title == "Section 1.1"


def test_build_toc_table_with_offset_and_bounds():
    generator = PdfTocGenerator(llm_client=FakeLlm())
    rows = generator._build_toc_table(
        page_count=10,
        entries=[
            TocEntry(
                title="Root",
                page_number=1,
                children=[TocEntry(title="Child", page_number=20)],
            )
        ],
        offset=3,
    )

    assert rows == [[1, "Root", 4], [2, "Child", 10]]


def test_pick_reference_entry_prefers_leaf():
    generator = PdfTocGenerator(llm_client=FakeLlm())
    entries = [
        TocEntry(
            title="Part",
            page_number=1,
            children=[TocEntry(title="Leaf", page_number=2)],
        )
    ]
    reference = generator._pick_reference_entry(entries)
    assert reference is not None
    assert reference.title == "Leaf"


class _FakeDoc:
    def __init__(self, page_count: int):
        self.page_count = page_count


class _OffsetLlm(FakeLlm):
    def __init__(self, page_number_map=None, match_pdf_page=None):
        self.page_number_map = page_number_map or {}
        self.match_pdf_page = match_pdf_page

    def detect_printed_page_number(self, image_bytes: bytes):
        page_index = int(image_bytes.decode("ascii"))
        return self.page_number_map.get(page_index)

    def page_matches_entry(self, entry, image_bytes: bytes) -> bool:
        page_index = int(image_bytes.decode("ascii"))
        return (page_index + 1) == self.match_pdf_page


def test_estimate_offset_uses_printed_page_numbers(monkeypatch):
    llm = _OffsetLlm(page_number_map={5: 1, 6: 2, 7: 3})
    generator = PdfTocGenerator(llm_client=llm, offset_search_window=10)
    monkeypatch.setattr(generator, "_render_page", lambda doc, page_index: str(page_index).encode("ascii"))

    offset = generator._estimate_offset(
        doc=_FakeDoc(page_count=30),
        entries=[],
        toc_pages=[0, 1, 2, 3, 4],
    )

    assert offset == 5


def test_estimate_offset_falls_back_to_entry_matching(monkeypatch):
    llm = _OffsetLlm(page_number_map={}, match_pdf_page=9)
    generator = PdfTocGenerator(llm_client=llm, offset_search_window=10)
    monkeypatch.setattr(generator, "_render_page", lambda doc, page_index: str(page_index).encode("ascii"))

    offset = generator._estimate_offset(
        doc=_FakeDoc(page_count=30),
        entries=[TocEntry(title="Chapter", page_number=3)],
        toc_pages=[0, 1, 2],
    )

    assert offset == 6


def test_estimate_offset_defaults_to_zero_when_unresolved(monkeypatch):
    llm = _OffsetLlm(page_number_map={}, match_pdf_page=None)
    generator = PdfTocGenerator(llm_client=llm, offset_search_window=10)
    monkeypatch.setattr(generator, "_render_page", lambda doc, page_index: str(page_index).encode("ascii"))

    offset = generator._estimate_offset(
        doc=_FakeDoc(page_count=30),
        entries=[TocEntry(title="Chapter", page_number=3)],
        toc_pages=[0, 1, 2],
    )

    assert offset == 0


class _FakeResponses:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return type(
            "FakeResponse",
            (),
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            },
        )()


class _FakeChatCompletions:
    def __init__(self, text: str = "chat-ok"):
        self.text = text

    def create(self, **kwargs):
        message = type("FakeMessage", (), {"content": self.text})()
        choice = type("FakeChoice", (), {"message": message})()
        return type(
            "FakeChatResponse",
            (),
            {
                "choices": [choice],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            },
        )()


class _FailingResponses:
    def create(self, **kwargs):
        raise RuntimeError("responses failed")


class _FakeOpenAIClient:
    def __init__(self, responses=None, chat_text: str = "chat-ok"):
        if responses is not None:
            self.responses = responses
        self.chat = type("FakeChat", (), {"completions": _FakeChatCompletions(chat_text)})()


class _FakeOpenAIWithResponsesFactory:
    def __init__(self, responses):
        self.responses = responses
        self.last_client = None

    def __call__(self, **kwargs):
        self.last_client = _FakeOpenAIClient(responses=self.responses)
        return self.last_client


class _FakeOpenAINoResponsesFactory:
    def __call__(self, **kwargs):
        return _FakeOpenAIClient(responses=None)


def test_complete_falls_back_to_chat_when_responses_fail(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        generator_mod,
        "OpenAI",
        _FakeOpenAIWithResponsesFactory(responses=_FailingResponses()),
    )

    client = OpenAiVisionClient(model="test-model")

    result = client._complete(
        system_prompt="system",
        user_content=[{"type": "text", "text": "hello"}],
    )

    assert result == "chat-ok"
    assert client._use_responses is False


def test_complete_responses_raises_when_forced(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        generator_mod,
        "OpenAI",
        _FakeOpenAIWithResponsesFactory(responses=_FailingResponses()),
    )

    client = OpenAiVisionClient(model="test-model", use_responses=True)

    with pytest.raises(RuntimeError, match="responses failed"):
        client._complete(
            system_prompt="system",
            user_content=[{"type": "text", "text": "hello"}],
        )


def test_complete_responses_converts_content_for_responses_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_responses = _FakeResponses()
    factory = _FakeOpenAIWithResponsesFactory(responses=fake_responses)
    monkeypatch.setattr(generator_mod, "OpenAI", factory)

    client = OpenAiVisionClient(model="test-model", use_responses=True)
    result = client._complete_responses(
        system_prompt="You are helpful.",
        user_content=[
            {"type": "text", "text": "classify"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ],
    )

    assert result == "ok"
    assert fake_responses.last_kwargs is not None

    payload = fake_responses.last_kwargs["input"]
    assert payload[0] == {"role": "system", "content": "You are helpful."}
    assert payload[1]["role"] == "user"
    assert payload[1]["content"] == [
        {"type": "input_text", "text": "classify"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAA"},
    ]


def test_init_raises_when_forcing_responses_without_sdk_support(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(generator_mod, "OpenAI", _FakeOpenAINoResponsesFactory())

    with pytest.raises(RuntimeError, match="does not support Responses API"):
        OpenAiVisionClient(model="test-model", use_responses=True)


def test_token_usage_accumulates_from_chat(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(generator_mod, "OpenAI", _FakeOpenAINoResponsesFactory())
    client = OpenAiVisionClient(model="test-model")

    client._complete("system", [{"type": "text", "text": "hello"}])
    client._complete("system", [{"type": "text", "text": "hello"}])

    assert client.get_token_usage() == {
        "input_tokens": 6,
        "output_tokens": 8,
        "total_tokens": 14,
        "requests": 2,
    }


def test_token_usage_accumulates_from_responses(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        generator_mod,
        "OpenAI",
        _FakeOpenAIWithResponsesFactory(responses=_FakeResponses()),
    )
    client = OpenAiVisionClient(model="test-model", use_responses=True)

    client._complete("system", [{"type": "text", "text": "hello"}])
    usage = client.get_token_usage()
    assert usage == {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
        "requests": 1,
    }
