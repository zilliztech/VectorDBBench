import random
import string

passwordKeys = ["password", "api_key"]


def inputIsPassword(key: str) -> bool:
    return key.lower() in passwordKeys


def addHorizontalLine(st):
    st.markdown(
        "<div style='border: 1px solid #cccccc60; margin-bottom: 24px;'></div>",
        unsafe_allow_html=True,
    )


def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    result = "".join(random.choice(letters) for _ in range(length))
    return result


def get_page_url_prefix(base_path: str) -> str:
    base_path = base_path.strip("/")
    return f"/{base_path}/" if base_path else "/"
