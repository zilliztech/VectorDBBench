from typing import List, NamedTuple

from prometheus_client import CollectorRegistry
from prometheus_client.openmetrics.exposition import generate_latest
from streamlit.runtime.stats import CacheStatsProvider

from .. import extra


class CustomStat(NamedTuple):
    metric_str: str = ""

    def to_metric_str(self) -> str:
        return self.metric_str

    def marshall_metric_proto(self, metric) -> None:
        """Custom OpenMetrics collected via protobuf is not currently supported."""
        # Included to be compatible with the RequestHandler's _stats_to_proto() method:
        # https://github.com/streamlit/streamlit/blob/1.29.0/lib/streamlit/web/server/stats_request_handler.py#L73
        # Fill in dummy values so protobuf format isn't broken
        label = metric.labels.add()
        label.name = "cache_type"
        label.value = "custom_metrics"

        label = metric.labels.add()
        label.name = "cache"
        label.value = "not_implemented"

        metric_point = metric.metric_points.add()
        metric_point.gauge_value.int_value = 0


class PrometheusMetricsProvider(CacheStatsProvider):
    def __init__(self, registry: CollectorRegistry):
        self.registry = registry

    def get_stats(self) -> List[CustomStat]:
        """
        Use generate_latest() method provided by prometheus to produce the
        appropriately formatted OpenMetrics text encoding for all the stored metrics.

        Then do a bit of string manipulation to package it in the format expected
        by Streamlit's stats handler, so the final output looks the way we expect.
        """
        DUPLICATE_SUFFIX = "\n# EOF\n"
        output_str = generate_latest(self.registry).decode(encoding="utf-8")
        if not output_str.endswith(DUPLICATE_SUFFIX):
            raise ValueError("Unexpected output from OpenMetrics text encoding")
        output = CustomStat(metric_str=output_str[: -len(DUPLICATE_SUFFIX)])
        return [output]


@extra
def streamlit_registry() -> CollectorRegistry:
    """
    Expose Prometheus metrics (https://prometheus.io) from your Streamlit app.

    Create and use Prometheus metrics in your app with `registry=streamlit_registry()`.
    The metrics will be exposed at Streamlit's existing `/_stcore/metrics` route.

    **Note:** This extra works best with Streamlit >= 1.31. There are known bugs with
    some earlier Streamlit versions, especially 1.30.0.

    See more example metrics in the Prometheus Python docs:
    https://prometheus.github.io/client_python/

    To produce accurate metrics, you are responsible to ensure that unique metric
    objects are shared across app runs and sessions. We recommend either 1) initialize
    metrics in a separate file and import them in the main app script, or 2) initialize
    metrics in a cached function (and ensure the cache is not cleared during execution).

    For an app running locally you can view the output with
    `curl localhost:8501/_stcore/metrics` or equivalent.
    """
    from streamlit import runtime

    stats = runtime.get_instance().stats_mgr

    # Did we already register it elsewhere? If so, return that copy
    for prv in stats._cache_stats_providers:
        if isinstance(prv, PrometheusMetricsProvider):
            return prv.registry

    # This is the function was called, so create the registry
    # and hook it into Streamlit stats
    registry = CollectorRegistry(auto_describe=True)
    prv = PrometheusMetricsProvider(registry=registry)
    stats.register_provider(prv)
    return registry


def example():
    import streamlit as st
    from prometheus_client import Counter

    @st.cache_resource
    def get_metric():
        registry = streamlit_registry()
        return Counter(
            name="my_counter",
            documentation="A cool counter",
            labelnames=("app_name",),
            registry=registry,  # important!!
        )

    SLIDER_COUNT = get_metric()

    app_name = st.text_input("App name", "prometheus_app")
    latest = st.slider("Latest value", 0, 20, 3)
    if st.button("Submit"):
        SLIDER_COUNT.labels(app_name).inc(latest)

    st.write(
        """
        View a fuller example that uses the (safer) import metrics method at:
        https://github.com/arnaudmiribel/streamlit-extras/tree/main/src/streamlit_extras/prometheus/example
        """
    )

    st.write(
        """
        ### Example output at `{host:port}/_stcore/metrics`
        ```
        # TYPE my_counter counter
        # HELP my_counter A cool counter
        my_counter_total{app_name="prometheus_app"} 14.0
        my_counter_created{app_name="prometheus_app"} 1.7042185907557938e+09
        ```
        """
    )


__title__ = "Prometheus"
__desc__ = "Expose Prometheus metrics (https://prometheus.io) from your Streamlit app."
__icon__ = "📊"
__examples__ = [example]
__author__ = "Joshua Carroll"
__experimental_playground__ = False
__stlite__ = False
