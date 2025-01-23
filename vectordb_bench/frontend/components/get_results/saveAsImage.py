import requests
import streamlit as st
import streamlit.components.v1 as components

HTML_2_CANVAS_URL = "https://unpkg.com/html2canvas@1.4.1/dist/html2canvas.js"


@st.cache_data
def load_unpkg(src: str) -> str:
    return requests.get(src).text


def getResults(container, pageName="vectordb_bench"):
    container.subheader("Get results")
    saveAsImage(container, pageName)


def saveAsImage(container, pageName):
    html2canvasJS = load_unpkg(HTML_2_CANVAS_URL)
    container.write()
    buttonText = "Save as Image"
    savePDFButton = container.button(buttonText)
    if savePDFButton:
        components.html(
            f"""
<script>{html2canvasJS}</script>

<script>
const html2canvas = window.html2canvas

const streamlitDoc = window.parent.document;
const stApp = streamlitDoc.querySelector('.main > .block-container');

const buttons = Array.from(streamlitDoc.querySelectorAll('.stButton > button'));
const imgButton = buttons.find(el => el.innerText === '{buttonText}');

if (imgButton)
    imgButton.innerText = 'Creating Image...';

html2canvas(stApp, {{ allowTaint: false, useCORS: true }}).then(function (canvas) {{
    a = document.createElement('a');
    a.href = canvas.toDataURL("image/jpeg", 1.0).replace("image/jpeg", "image/octet-stream");
    a.download = '{pageName}.png';
    a.click();
    
    if (imgButton)
        imgButton.innerText = '{buttonText}';
}})
</script>""",
            height=0,
            width=0,
        )
