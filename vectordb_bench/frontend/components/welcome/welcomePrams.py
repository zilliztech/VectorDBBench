import base64
from PIL import Image
from io import BytesIO
import os

from vectordb_bench.frontend.components.welcome.pagestyle import pagestyle


def get_image_as_base64(image_path):
    try:
        if image_path.startswith("http"):
            return image_path

        path = os.path.expanduser(image_path)
        img = Image.open(path)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    except Exception as e:
        raise (f"wrong loading: {e}")


def welcomePrams(st):
    st.title("Welcome to VectorDB Benchmark!")
    options = [
        {
            "title": "Results",
            "description": (
                "<span style='font-size: 17px;'>"
                "Select a specific run or compare all results side by side to view the results of previous tests."
                "</span>"
            ),
            "image": "/Users/zilliz/static/results.png",
            "link": "results",
        },
        {
            "title": "Quries Per Dollar",
            "description": (
                "<span style='font-size: 17px;'>"
                "To view the results of quries per dollar.<br> "
                "(similar to qps in Results) "
                "</span>"
            ),
            "image": "/Users/zilliz/static/qpd.png",
            "link": "quries_per_dollar",
        },
        {
            "title": "Tables",
            "description": (
                "<span style='font-size: 17px;'>" "To view the results of differnt datasets in tables." "</span>"
            ),
            "image": "/Users/zilliz/static/tables.png",
            "link": "tables",
        },
        {
            "title": "Concurrent Performance",
            "description": (
                "<span style='font-size: 17px;'>"
                "To view the variation of qps with latency under different concurrent."
                "</span>"
            ),
            "image": "/Users/zilliz/static/concurrent.png",
            "link": "concurrent",
        },
        {
            "title": "Label Filter",
            "description": (
                "<span style='font-size: 17px;'>"
                "To view the perfomance of datasets under different filter ratios "
                "</span>"
            ),
            "image": "/Users/zilliz/static/label_filter.png",
            "link": "label_filter",
        },
        {
            "title": "Streaming Performance",
            "description": (
                "<span style='font-size: 17px;'>"
                "To view the perfomance of datasets under different search stages and insertion rates. "
                "</span>"
            ),
            "image": "/Users/zilliz/static/streaming.png",
            "link": "streaming",
        },
        {
            "title": "Run Test",
            "description": (
                "<span style='font-size: 17px;'>"
                "Select the databases and cases to test.<br>"
                "The test results will be displayed in Results."
                "</span>"
            ),
            "image": "/Users/zilliz/static/run_test.png",
            "link": "run_test",
        },
        {
            "title": "Custom Dataset",
            "description": (
                "<span style='font-size: 17px;'>"
                "Define users' own datasets with detailed descriptions of setting each parameter."
                "</span>"
            ),
            "image": "/Users/zilliz/static/custom.png",
            "link": "custom",
        },
    ]

    html_content = pagestyle()

    for option in options:
        option["image"] = get_image_as_base64(option["image"])

    for i, option in enumerate(options[:6]):
        html_content += f"""
        <a href="/{option['link']}" target="_self" style="text-decoration: none;">
            <div class="section-card">
                <img src="{option['image']}" class="section-image" alt="{option['title']}">
                <div class="section-title">{option['title']}</div>
                <div class="section-description">{option['description']}</div>
            </div>
        </a>
        """

    html_content += """
    </div>
    <div class="title-row">
        <h2>Set And Run</h2>
    </div>
    <div class="last-row">
    """

    for option in options[6:8]:
        html_content += f"""
        <a href="/{option['link']}" target="_self" style="text-decoration: none;">
            <div class="section-card">
                <img src="{option['image']}" class="section-image" alt="{option['title']}">
                <div class="section-title">{option['title']}</div>
                <div class="section-description">{option['description']}</div>
            </div>
        </a>
        """

    html_content += """
    </div>
    """

    st.html(html_content)
