from html.parser import HTMLParser
from types import SimpleNamespace

import pytest

from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import NavToPages
from vectordb_bench.frontend.components.welcome import welcomePrams


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs)["href"])


@pytest.mark.parametrize(
    ("base_path", "prefix"),
    [
        ("", "/"),
        ("bench", "/bench/"),
        ("/bench/", "/bench/"),
        ("tools/bench", "/tools/bench/"),
        ("/tools/bench/", "/tools/bench/"),
    ],
)
@pytest.mark.parametrize(
    ("renderer", "pages"),
    [
        (
            welcomePrams.welcomePrams,
            [
                "results",
                "quries_per_dollar",
                "tables",
                "full_text_search",
                "concurrent",
                "label_filter",
                "int_filter",
                "streaming",
                "run_test",
                "custom",
            ],
        ),
        (
            NavToPages,
            [
                "run_test",
                "results",
                "qps_recall",
                "full_text_search",
                "quries_per_dollar",
                "concurrent",
                "label_filter",
                "int_filter",
                "streaming",
                "tables",
                "custom",
            ],
        ),
        (drawHeaderIcon, ["vdbbench"]),
    ],
)
def test_navigation_links_keep_base_path(monkeypatch, base_path, prefix, renderer, pages):
    rendered = []

    def get_option(name):
        assert name == "server.baseUrlPath"
        return base_path

    monkeypatch.setattr(welcomePrams, "get_image_as_base64", lambda path: "image")
    renderer(
        SimpleNamespace(
            get_option=get_option,
            html=rendered.append,
            markdown=lambda text, **kwargs: rendered.append(text),
            title=lambda text: None,
        )
    )

    parser = LinkParser()
    parser.feed(rendered[0])
    assert parser.links == [prefix + page for page in pages]
