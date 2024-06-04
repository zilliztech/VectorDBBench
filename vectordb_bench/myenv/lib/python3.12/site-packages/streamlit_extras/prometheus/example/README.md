# Streamlit Prometheus example

This repo has a simple example of a Streamlit app that exposes prometheus metrics via `streamlit_extras.prometheus`.

It demonstrates the approach of creating metrics in a separate file and importing them into your main app file
for object persistence across runs.

## Running the example

```sh
pip install streamlit-extras
streamlit run app.py
```
