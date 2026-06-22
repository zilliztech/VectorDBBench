import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from vectordb_bench import config
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import NavToPages
from vectordb_bench.frontend.config.styles import FAVICON


RESULT_DIR = config.RESULTS_LOCAL_DIR / "FullTextSearch"
DATASET_ORDER = [
    "MS MARCO Small",
    "MS MARCO Medium",
    "MS MARCO Large",
    "HotpotQA Small",
    "HotpotQA Medium",
    "HotpotQA Large",
]
BACKEND_ORDER = ["Milvus", "ElasticSearch", "Vespa"]
SIZE_ORDER = ["Small", "Medium", "Large"]


def _normalize_backend(db: str, result_file: Path) -> str:
    if db == "ElasticCloud":
        return "ElasticSearch"
    if db:
        return db
    return result_file.parent.name


def _dataset_parts(dataset_label: str) -> tuple[str, str, str]:
    if dataset_label.startswith("MS MARCO"):
        family = "MS MARCO"
    elif dataset_label.startswith("HotpotQA"):
        family = "HotpotQA"
    else:
        family = dataset_label.split(" ", 1)[0]

    size = next((name for name in SIZE_ORDER if name in dataset_label), "")
    return family, size, f"{family} {size}".strip()


def _run_context(task_label: str) -> str:
    if "mathgt" in task_label:
        return "Math GT"
    return "Recorded"


def _parse_result_file(result_file: Path) -> list[dict[str, Any]]:
    with result_file.open() as f:
        test_result = json.load(f)

    task_label = test_result.get("task_label") or result_file.stem
    rows = []
    for case_result in test_result.get("results", []):
        metrics = case_result.get("metrics", {})
        task_config = case_result.get("task_config", {})
        case_config = task_config.get("case_config", {})
        custom_case = case_config.get("custom_case") or {}
        dataset_label = custom_case.get("dataset_with_size_type", "")
        dataset_family, dataset_size, dataset_key = _dataset_parts(dataset_label)
        backend = _normalize_backend(task_config.get("db", ""), result_file)
        payload = metrics.get("payload_profile") or custom_case.get("payload_profile") or "ids_only"

        rows.append(
            {
                "backend": backend,
                "dataset_family": dataset_family,
                "dataset_size": dataset_size,
                "dataset": dataset_key,
                "payload": payload,
                "context": _run_context(task_label),
                "task_label": task_label,
                "load_s": metrics.get("load_duration", 0.0),
                "qps": metrics.get("qps", 0.0),
                "recall": metrics.get("recall", 0.0),
                "p95_s": metrics.get("serial_latency_p95", 0.0),
                "p99_s": metrics.get("serial_latency_p99", 0.0),
                "concurrency": metrics.get("conc_num_list") or [],
                "concurrent_qps": metrics.get("conc_qps_list") or [],
            }
        )

    return rows


def load_full_text_search_rows(result_dir: Path = RESULT_DIR) -> pd.DataFrame:
    if not result_dir.exists():
        return pd.DataFrame()

    rows = []
    for result_file in sorted(result_dir.rglob("result_*.json")):
        rows.extend(_parse_result_file(result_file))

    data = pd.DataFrame(rows)
    if data.empty:
        return data

    data["dataset"] = pd.Categorical(data["dataset"], DATASET_ORDER, ordered=True)
    data["backend"] = pd.Categorical(data["backend"], BACKEND_ORDER, ordered=True)
    data["dataset_size"] = pd.Categorical(data["dataset_size"], SIZE_ORDER, ordered=True)
    return data.sort_values(["dataset", "backend", "payload"]).reset_index(drop=True)


def _filter_data(st, data: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        selected_datasets = st.multiselect(
            "Dataset",
            [dataset for dataset in DATASET_ORDER if dataset in set(data["dataset"].astype(str))],
            default=[dataset for dataset in DATASET_ORDER if dataset in set(data["dataset"].astype(str))],
        )
        selected_backends = st.multiselect(
            "Backend",
            [backend for backend in BACKEND_ORDER if backend in set(data["backend"].astype(str))],
            default=[backend for backend in BACKEND_ORDER if backend in set(data["backend"].astype(str))],
        )
        payloads = sorted(data["payload"].dropna().unique().tolist())
        selected_payloads = st.multiselect("Payload", payloads, default=payloads)

    return data[
        data["dataset"].astype(str).isin(selected_datasets)
        & data["backend"].astype(str).isin(selected_backends)
        & data["payload"].isin(selected_payloads)
    ].copy()


def _draw_summary_table(st, data: pd.DataFrame) -> None:
    columns = [
        "dataset",
        "backend",
        "payload",
        "context",
        "load_s",
        "qps",
        "recall",
        "p95_s",
        "p99_s",
    ]
    st.dataframe(
        data[columns],
        hide_index=True,
        use_container_width=True,
        column_config={
            "load_s": st.column_config.NumberColumn("Load s", format="%.4f"),
            "qps": st.column_config.NumberColumn("QPS", format="%.4f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
            "p95_s": st.column_config.NumberColumn("p95 s", format="%.4f"),
            "p99_s": st.column_config.NumberColumn("p99 s", format="%.4f"),
        },
    )


def _draw_metric_chart(st, data: pd.DataFrame, metric: str, title: str) -> None:
    fig = px.bar(
        data,
        x="dataset",
        y=metric,
        color="backend",
        pattern_shape="payload",
        barmode="group",
        category_orders={"dataset": DATASET_ORDER, "backend": BACKEND_ORDER},
        hover_data=["payload", "context", "task_label"],
        text_auto=".4g",
        title=title,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=48, b=12, pad=8),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, title=""),
        xaxis_title="",
    )
    st.plotly_chart(fig, width="stretch", key=f"fts-{metric}")


def _concurrency_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in data.to_dict("records"):
        for concurrency, qps in zip(row["concurrency"], row["concurrent_qps"]):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "backend": row["backend"],
                    "payload": row["payload"],
                    "context": row["context"],
                    "concurrency": concurrency,
                    "qps": qps,
                    "task_label": row["task_label"],
                }
            )
    return pd.DataFrame(rows)


def _draw_concurrency_chart(st, data: pd.DataFrame) -> None:
    concurrency_data = _concurrency_rows(data)
    if concurrency_data.empty:
        return

    fig = px.line(
        concurrency_data,
        x="concurrency",
        y="qps",
        color="backend",
        line_dash="dataset",
        symbol="payload",
        markers=True,
        category_orders={"dataset": DATASET_ORDER, "backend": BACKEND_ORDER},
        hover_data=["payload", "context", "task_label"],
        title="Concurrent Search QPS",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=48, b=12, pad=8),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, title=""),
    )
    fig.update_xaxes(title_text="Concurrency")
    fig.update_yaxes(title_text="QPS")
    st.plotly_chart(fig, width="stretch", key="fts-concurrency-qps")


def main():
    st.set_page_config(
        page_title="Full Text Search",
        page_icon=FAVICON,
        layout="wide",
    )

    drawHeaderIcon(st)
    NavToPages(st)

    st.title("Full Text Search")
    st.caption("Math-GT benchmark results from committed `vectordb_bench/results/FullTextSearch` JSONs.")

    data = load_full_text_search_rows()
    if data.empty:
        st.warning("No FullTextSearch result JSONs found.")
        footer(st.container())
        return

    shown_data = _filter_data(st, data)
    if shown_data.empty:
        st.warning("No rows match the selected filters.")
        footer(st.container())
        return

    _draw_summary_table(st, shown_data)
    chart_tabs = st.tabs(["QPS", "Recall", "Load", "Concurrency"])
    with chart_tabs[0]:
        _draw_metric_chart(st, shown_data, "qps", "Search QPS")
    with chart_tabs[1]:
        _draw_metric_chart(st, shown_data, "recall", "Math-GT Recall")
    with chart_tabs[2]:
        _draw_metric_chart(st, shown_data, "load_s", "Load Duration")
    with chart_tabs[3]:
        _draw_concurrency_chart(st, shown_data)

    footer(st.container())


if __name__ == "__main__":
    main()
