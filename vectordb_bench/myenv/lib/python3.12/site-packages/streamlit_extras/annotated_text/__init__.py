from annotated_text import annotated_text

from .. import extra

annotated_text = extra(annotated_text)


def example_1():
    from annotated_text import annotated_text

    annotated_text(
        "This ",
        ("is", "verb", "#8ef"),
        " some ",
        ("annotated", "adj", "#faa"),
        ("text", "noun", "#afa"),
        " for those of ",
        ("you", "pronoun", "#fea"),
        " who ",
        ("like", "verb", "#8ef"),
        " this sort of ",
        ("thing", "noun", "#afa"),
    )


def example_2():
    from annotated_text import annotated_text, annotation

    annotated_text(
        "Hello ",
        annotation("world!", "noun", color="#8ef", border="1px dashed red"),
    )


__title__ = "Annotated text"
__desc__ = "A simple way to display annotated text in Streamlit apps"
__icon__ = "🖊️"
__examples__ = [example_1, example_2]
__author__ = "tvst"
__github_repo__ = "tvst/st-annotated-text"
__pypi_name__ = "st-annotated-text"
__package_name__ = "annotated_text"
