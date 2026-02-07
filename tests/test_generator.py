from scanpdf_toc.generator import PdfTocGenerator, TocEntry


class FakeLlm:
    def is_toc_page(self, image_bytes: bytes) -> bool:
        return False

    def extract_outline(self, toc_images):
        return []

    def page_matches_entry(self, entry, image_bytes: bytes) -> bool:
        return False


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
