from unittest.mock import MagicMock, patch

import pytest

from vectordb_bench.frontend.components.int_filter import charts as int_charts
from vectordb_bench.frontend.components.label_filter import charts as label_charts

NONE_FILTER_DATA = [
    {
        "filter_rate": None,
        "qps": 10.0,
        "recall": 0.9,
        "db_name": "A",
        "dataset_name": "custom",
    },
    {
        "filter_rate": 0.99,
        "qps": 20.0,
        "recall": 0.8,
        "db_name": "B",
        "dataset_name": "custom",
    },
]


@pytest.mark.parametrize("charts", [label_charts, int_charts])
def test_get_range_coerces_none_filter_rate(charts):
    data = [{"filter_rate": None}, {"filter_rate": 0.5}]
    xrange = charts.getRange("filter_rate", data, [0.05, 0.1])
    assert xrange[0] <= 0
    assert xrange[1] >= 0.5


@pytest.mark.parametrize("charts", [label_charts, int_charts])
def test_draw_chart_does_not_crash_when_filter_rate_is_none(charts):
    st = MagicMock()
    with patch.object(charts, "px") as px:
        px.line.return_value = MagicMock()
        charts.drawChart(st, list(NONE_FILTER_DATA), "qps")
    px.line.assert_called_once()
    st.plotly_chart.assert_called_once()
