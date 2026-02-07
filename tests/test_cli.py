from pathlib import Path

from scanpdf_toc import cli


class _FakeGenerator:
    def __init__(self, usage):
        self._usage = usage

    def generate(self, pdf, output):
        return Path("/tmp/book-with-toc.pdf")

    def get_token_usage(self):
        return self._usage


def test_main_prints_token_usage_when_available(monkeypatch, capsys):
    monkeypatch.setattr(cli, "create_generator", lambda args: _FakeGenerator(
        {"total_tokens": 12, "input_tokens": 8, "output_tokens": 4, "requests": 2}
    ))
    exit_code = cli.main(["/tmp/in.pdf"])

    assert exit_code == 0
    output = capsys.readouterr().out.strip().splitlines()
    assert output[0] == "/tmp/book-with-toc.pdf"
    assert output[1] == "tokens: total=12 input=8 output=4 requests=2"


def test_main_skips_token_usage_when_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(cli, "create_generator", lambda args: _FakeGenerator(None))
    exit_code = cli.main(["/tmp/in.pdf"])

    assert exit_code == 0
    output = capsys.readouterr().out.strip().splitlines()
    assert output == ["/tmp/book-with-toc.pdf"]
