import json
import re
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
# Published FTS results currently cover this cloud/service backend subset.
BACKEND_ORDER = ["ZillizCloud", "ElasticSearch", "OSSOpenSearch", "TurboPuffer"]
BACKEND_COLORS = {
    "ZillizCloud": "#0D6EFD",
    "ElasticSearch": "#04D6C8",
    "OSSOpenSearch": "#61D790",
    "TurboPuffer": "#FF6B2C",
}
SIZE_ORDER = ["Small", "Medium", "Large"]
FILTER_RATE_LABEL_ORDER = ["50%", "75%", "90%", "95%", "99%"]
CHART_TABS = ["QPS", "Recall", "NDCG", "MRR", "Load", "Filtered Search"]
FILTERED_TAB = "Filtered Search"


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


def _dataset_doc_count(dataset_label: str) -> str:
    match = re.search(r"\(([^)]+)\)", dataset_label)
    return match.group(1) if match else ""


def _dataset_axis_label(dataset: str, doc_count: str) -> str:
    return f"{dataset}<br>{doc_count}" if doc_count else dataset


def _dataset_axis_order(data: pd.DataFrame) -> list[str]:
    labels = []
    for dataset in DATASET_ORDER:
        matches = data[data["dataset"].astype(str) == dataset]
        if not matches.empty:
            labels.append(matches["dataset_axis_label"].iloc[0])
    return labels


def _filter_rate_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Unfiltered"
    return f"{float(value):.0%}"


def _backend_metric_order(data: pd.DataFrame, metric: str, ascending: bool) -> list[str]:
    if data.empty or metric not in data:
        return BACKEND_ORDER

    metric_data = data[["backend", metric]].dropna().copy()
    if metric_data.empty:
        return BACKEND_ORDER

    metric_data["backend"] = metric_data["backend"].astype(str)
    scores = metric_data.groupby("backend")[metric].mean()
    ordered = scores.sort_values(ascending=ascending).index.tolist()
    return ordered + [backend for backend in BACKEND_ORDER if backend not in ordered]


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
        additional_parameters = metrics.get("additional_parameters") or {}
        fts_filter = additional_parameters.get("fts_filter") or {}
        filter_rate = fts_filter.get("filter_rate", custom_case.get("filter_rate"))
        dataset_label = custom_case.get("dataset_with_size_type", "")
        dataset_family, dataset_size, dataset_key = _dataset_parts(dataset_label)
        dataset_doc_count = _dataset_doc_count(dataset_label)
        dataset_axis_label = _dataset_axis_label(dataset_key, dataset_doc_count)
        backend = _normalize_backend(task_config.get("db", ""), result_file)
        payload = metrics.get("payload_profile") or custom_case.get("payload_profile") or "ids_only"
        row_task_label = task_config.get("task_label") or task_label

        rows.append(
            {
                "backend": backend,
                "dataset_family": dataset_family,
                "dataset_size": dataset_size,
                "dataset": dataset_key,
                "dataset_doc_count": dataset_doc_count,
                "dataset_axis_label": dataset_axis_label,
                "payload": payload,
                "context": _run_context(task_label),
                "task_label": row_task_label,
                "filter_rate": filter_rate,
                "filter_rate_label": _filter_rate_label(filter_rate),
                "is_filtered": filter_rate is not None,
                "load_s": metrics.get("load_duration", 0.0),
                "qps": metrics.get("qps", 0.0),
                "recall": metrics.get("recall", 0.0),
                "ndcg": metrics.get("ndcg", 0.0),
                "mrr": metrics.get("mrr", 0.0),
                "p95_s": metrics.get("serial_latency_p95", 0.0),
                "p99_s": metrics.get("serial_latency_p99", 0.0),
                "concurrency": metrics.get("conc_num_list") or [],
                "concurrent_qps": metrics.get("conc_qps_list") or [],
            }
        )

    return rows


def _result_files(result_dir: Path) -> list[Path]:
    return sorted(result_dir.rglob("result_*.json"))


def load_full_text_search_rows(result_dir: Path = RESULT_DIR) -> pd.DataFrame:
    if not result_dir.exists():
        return pd.DataFrame()

    rows = []
    for result_file in _result_files(result_dir):
        rows.extend(_parse_result_file(result_file))

    data = pd.DataFrame(rows)
    if data.empty:
        return data

    data = data[data["backend"] != "Milvus"].copy()
    if data.empty:
        return data

    data["dataset"] = pd.Categorical(data["dataset"], DATASET_ORDER, ordered=True)
    data["backend"] = pd.Categorical(data["backend"], BACKEND_ORDER, ordered=True)
    data["dataset_size"] = pd.Categorical(data["dataset_size"], SIZE_ORDER, ordered=True)
    data["filter_rate"] = pd.to_numeric(data["filter_rate"], errors="coerce")
    return data.sort_values(["is_filtered", "dataset", "filter_rate", "backend", "payload"]).reset_index(drop=True)


def _filter_data(st: Any, data: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        selected_datasets = st.multiselect(
            "Dataset",
            [dataset for dataset in DATASET_ORDER if dataset in set(data["dataset"].astype(str))],
            default=[dataset for dataset in DATASET_ORDER if dataset in set(data["dataset"].astype(str))],
            key="fts-standard-datasets",
        )
        backend_options = [backend for backend in BACKEND_ORDER if backend in set(data["backend"].astype(str))]
        selected_backends = st.multiselect("Backend", backend_options, default=backend_options, key="fts-standard-backends")
        payloads = sorted(data["payload"].dropna().unique().tolist())
        default_payloads = ["ids_only"] if "ids_only" in payloads else payloads
        selected_payloads = st.multiselect("Payload", payloads, default=default_payloads, key="fts-standard-payloads")

    filters = (
        data["dataset"].astype(str).isin(selected_datasets)
        & data["backend"].astype(str).isin(selected_backends)
        & data["payload"].isin(selected_payloads)
    )

    return data[filters].copy()


def _filter_filtered_data(st: Any, data: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        dataset_options = [
            family for family in ["MS MARCO", "HotpotQA"] if family in set(data["dataset_family"].astype(str))
        ]
        selected_dataset = st.selectbox("Dataset", dataset_options, key="fts-filtered-dataset-family")
        backend_options = [backend for backend in BACKEND_ORDER if backend in set(data["backend"].astype(str))]
        selected_backends = st.multiselect(
            "Backend",
            backend_options,
            default=backend_options,
            key="fts-filtered-backends",
        )

    filters = data["dataset_family"].astype(str).eq(selected_dataset) & data["backend"].astype(str).isin(
        selected_backends
    )
    return data[filters].copy()


def _draw_summary_table(st: Any, data: pd.DataFrame) -> None:
    columns = [
        "dataset",
        "backend",
        "payload",
        "load_s",
        "qps",
        "recall",
        "ndcg",
        "mrr",
        "p95_s",
        "p99_s",
    ]
    st.dataframe(
        data[columns],
        hide_index=True,
        width="stretch",
        column_config={
            "load_s": st.column_config.NumberColumn("Load s", format="%.4f"),
            "qps": st.column_config.NumberColumn("QPS", format="%.4f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
            "ndcg": st.column_config.NumberColumn("NDCG", format="%.4f"),
            "mrr": st.column_config.NumberColumn("MRR", format="%.4f"),
            "p95_s": st.column_config.NumberColumn("p95 s", format="%.4f"),
            "p99_s": st.column_config.NumberColumn("p99 s", format="%.4f"),
        },
    )


def _draw_filtered_summary_table(st: Any, data: pd.DataFrame) -> None:
    columns = ["dataset", "backend", "filter_rate_label", "qps", "concurrency", "task_label"]
    st.dataframe(
        data[columns],
        hide_index=True,
        width="stretch",
        column_config={
            "filter_rate_label": "Filter",
            "qps": st.column_config.NumberColumn("QPS", format="%.4f"),
        },
    )


def _chart_label(value: Any, metric: str) -> str:
    if pd.isna(value):
        return ""

    value = float(value)
    abs_value = abs(value)
    if metric in {"recall", "ndcg", "mrr"}:
        return f"{value:.3f}"
    if metric == "qps":
        return f"{value / 1000:.1f}k" if abs_value >= 1000 else f"{value:.0f}"
    if metric == "load_s":
        if abs_value >= 3600:
            return f"{value / 3600:.1f}h"
        if abs_value >= 60:
            return f"{value / 60:.1f}m"
        return f"{value:.1f}s"
    return f"{value:.1f}"


def _draw_metric_chart(
    st: Any,
    data: pd.DataFrame,
    metric: str,
    title: str,
    backend_order: list[str] | None = None,
) -> None:
    if backend_order is None:
        backend_order = BACKEND_ORDER

    show_text = data["payload"].nunique() <= 1
    chart_data = data.copy()
    if show_text:
        chart_data["_chart_label"] = chart_data[metric].apply(lambda value: _chart_label(value, metric))

    fig = px.bar(
        chart_data,
        x="dataset_axis_label",
        y=metric,
        color="backend",
        pattern_shape="payload",
        barmode="group",
        category_orders={"dataset_axis_label": _dataset_axis_order(data), "backend": backend_order},
        color_discrete_map=BACKEND_COLORS,
        hover_data=["dataset_doc_count", "payload", "context", "task_label"],
        text="_chart_label" if show_text else None,
        title=title,
    )
    if show_text:
        fig.update_traces(
            texttemplate="%{text}",
            textposition="outside",
            textangle=0,
            textfont={"size": 10},
            cliponaxis=False,
        )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 56, "b": 12, "pad": 8},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1, "xanchor": "right", "x": 1, "title": ""},
        xaxis_title="",
        xaxis={"tickfont": {"size": 12}},
        uniformtext={"minsize": 10, "mode": "show"},
    )
    st.plotly_chart(fig, width="stretch", key=f"fts-{metric}")


def _concurrency_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in data.to_dict("records"):
        for concurrency, qps in zip(row["concurrency"], row["concurrent_qps"], strict=True):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "dataset_axis_label": row["dataset_axis_label"],
                    "dataset_doc_count": row["dataset_doc_count"],
                    "backend": row["backend"],
                    "payload": row["payload"],
                    "context": row["context"],
                    "concurrency": concurrency,
                    "qps": qps,
                    "task_label": row["task_label"],
                }
            )
    return pd.DataFrame(rows)


def _draw_concurrency_chart(st: Any, data: pd.DataFrame) -> None:
    concurrency_data = _concurrency_rows(data)
    if concurrency_data.empty:
        return

    fig = px.line(
        concurrency_data,
        x="concurrency",
        y="qps",
        color="backend",
        line_dash="dataset_axis_label",
        symbol="payload",
        markers=True,
        category_orders={"dataset_axis_label": _dataset_axis_order(data), "backend": BACKEND_ORDER},
        color_discrete_map=BACKEND_COLORS,
        hover_data=["dataset", "dataset_doc_count", "payload", "context", "task_label"],
        title="Concurrent Search QPS",
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 48, "b": 12, "pad": 8},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1, "xanchor": "right", "x": 1, "title": ""},
    )
    fig.update_xaxes(title_text="Concurrency")
    fig.update_yaxes(title_text="QPS")
    st.plotly_chart(fig, width="stretch", key="fts-concurrency-qps")


def _draw_filtered_search_tab(st: Any, data: pd.DataFrame) -> None:
    filtered_data = data[(data["payload"] == "ids_only") & data["filter_rate"].notna()].copy()
    if filtered_data.empty:
        st.info("No filtered FTS result rows found.")
        return

    chart_data = filtered_data.copy()
    selected_family = str(chart_data["dataset_family"].iloc[0])
    chart_data["filter_rate_label"] = chart_data["filter_rate"].apply(_filter_rate_label)
    chart_data["_chart_label"] = chart_data["qps"].apply(lambda value: _chart_label(value, "qps"))
    backend_order = _backend_metric_order(chart_data, "qps", ascending=False)
    filter_rate_order = [label for label in FILTER_RATE_LABEL_ORDER if label in set(chart_data["filter_rate_label"])]

    fig = px.bar(
        chart_data,
        x="filter_rate_label",
        y="qps",
        color="backend",
        barmode="group",
        category_orders={"filter_rate_label": FILTER_RATE_LABEL_ORDER, "backend": backend_order},
        color_discrete_map=BACKEND_COLORS,
        hover_data=["dataset", "dataset_doc_count", "payload", "concurrency", "task_label"],
        text="_chart_label",
        title=f"{selected_family} Filtered Search QPS",
    )
    fig.update_traces(
        texttemplate="%{text}",
        textposition="outside",
        textangle=0,
        textfont={"size": 10},
        cliponaxis=False,
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 56, "b": 12, "pad": 8},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1, "xanchor": "right", "x": 1, "title": ""},
        xaxis_title="Filter rate",
        xaxis={
            "tickfont": {"size": 12},
            "type": "category",
            "categoryorder": "array",
            "categoryarray": filter_rate_order,
        },
        yaxis_title="QPS",
        uniformtext={"minsize": 10, "mode": "show"},
    )
    st.plotly_chart(fig, width="stretch", key=f"fts-filtered-qps-{selected_family}")


def main():
    st.set_page_config(
        page_title="Full Text Search Cloud Results",
        page_icon=FAVICON,
        layout="wide",
    )

    drawHeaderIcon(st)
    NavToPages(st)

    st.title("Full Text Search Cloud Results")
    st.caption("Published FTS results for Zilliz Cloud, ElasticSearch, OpenSearch, and TurboPuffer.")

    data = load_full_text_search_rows()
    if data.empty:
        st.warning("No FullTextSearch result JSONs found.")
        footer(st.container())
        return

    normal_data = data[~data["is_filtered"]].copy()
    filtered_data = data[data["is_filtered"]].copy()
    active_tab = st.session_state.get("fts-chart-tabs", CHART_TABS[0])
    if active_tab == FILTERED_TAB:
        shown_data = normal_data
        shown_filtered_data = _filter_filtered_data(st, filtered_data) if not filtered_data.empty else filtered_data
    else:
        shown_data = _filter_data(st, normal_data) if not normal_data.empty else normal_data
        shown_filtered_data = filtered_data

    if shown_data.empty and shown_filtered_data.empty:
        st.warning("No rows match the selected filters.")
        footer(st.container())
        return

    if active_tab == FILTERED_TAB and not shown_filtered_data.empty:
        _draw_filtered_summary_table(st, shown_filtered_data)
    elif not shown_data.empty:
        _draw_summary_table(st, shown_data)

    chart_tabs = st.tabs(CHART_TABS, key="fts-chart-tabs", on_change="rerun")
    with chart_tabs[0]:
        qps_data = shown_data
        if qps_data.empty:
            st.info("No standard FTS rows match the selected filters.")
        else:
            _draw_metric_chart(
                st,
                qps_data,
                "qps",
                "Search QPS",
                _backend_metric_order(qps_data, "qps", ascending=False),
            )
    with chart_tabs[1]:
        recall_data = shown_data[shown_data["payload"] == "ids_only"]
        _draw_metric_chart(
            st,
            recall_data,
            "recall",
            "Semantic Recall",
            _backend_metric_order(recall_data, "recall", ascending=False),
        )
    with chart_tabs[2]:
        ndcg_data = shown_data[shown_data["payload"] == "ids_only"]
        _draw_metric_chart(
            st,
            ndcg_data,
            "ndcg",
            "NDCG",
            _backend_metric_order(ndcg_data, "ndcg", ascending=False),
        )
    with chart_tabs[3]:
        mrr_data = shown_data[shown_data["payload"] == "ids_only"]
        _draw_metric_chart(
            st,
            mrr_data,
            "mrr",
            "MRR",
            _backend_metric_order(mrr_data, "mrr", ascending=False),
        )
    with chart_tabs[4]:
        load_data = shown_data[shown_data["payload"] == "ids_only"]
        _draw_metric_chart(
            st,
            load_data,
            "load_s",
            "Load Duration",
            _backend_metric_order(load_data, "load_s", ascending=True),
        )
    with chart_tabs[5]:
        _draw_filtered_search_tab(st, shown_filtered_data)

    footer(st.container())


if __name__ == "__main__":
    main()
