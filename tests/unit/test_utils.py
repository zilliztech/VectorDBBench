import pytest
import logging

from vectordb_bench.backend import utils
from vectordb_bench import metric
from vectordb_bench.metric import calc_ndcg, calc_recall, get_ideal_dcg

log = logging.getLogger(__name__)


class NoLinearContains(list):
    def __contains__(self, item):
        raise AssertionError("recall must not linearly scan ground truth")


class TestUtils:
    @pytest.mark.parametrize(
        "testcases",
        [
            (1, "1"),
            (10, "10"),
            (100, "100"),
            (1000, "1K"),
            (2000, "2K"),
            (30_000, "30K"),
            (400_000, "400K"),
            (5_000_000, "5M"),
            (60_000_000, "60M"),
            (1_000_000_000, "1B"),
            (1_000_000_000_000, "1000B"),
        ],
    )
    def test_numerize(self, testcases):
        t_in, expected = testcases
        assert expected == utils.numerize(t_in)

    @pytest.mark.parametrize(
        "got_expected",
        [
            ([1, 3, 5, 7, 9, 10], 1.0),
            ([11, 12, 13, 14, 15, 16], 0.0),
            ([1, 3, 5, 11, 12, 13], 0.5),
            ([1, 3, 5], 0.5),
        ],
    )
    def test_recall(self, got_expected):
        got, expected = got_expected
        ground_truth = [1, 3, 5, 7, 9, 10]
        res = calc_recall(6, ground_truth, got)
        log.info(f"recall: {res}, expected: {expected}")
        assert res == expected

    def test_recall_does_not_count_duplicate_backend_ids_twice(self):
        assert calc_recall(4, [1, 2, 3, 4], [1, 1, 2, 9]) == 0.5

    def test_recall_does_not_use_linear_list_membership(self):
        ground_truth = NoLinearContains([1, 2, 3, 4])

        assert calc_recall(4, ground_truth, [1, 2, 8, 9]) == 0.5

    def test_ndcg_preserves_rank_based_semantics_and_deduplicates_results(self):
        ground_truth = [10, 20, 30, 40]
        got = [30, 10, 30, 99]
        expected = (1 / 2 + 1) / get_ideal_dcg(4)

        assert calc_ndcg(ground_truth, got, get_ideal_dcg(4)) == pytest.approx(expected)

    def test_calc_vector_metrics_reports_prefix_recall(self):
        assert hasattr(metric, "calc_vector_metrics")

        recall, ndcg, recall_at = metric.calc_vector_metrics(
            4,
            [10, 20, 30, 40],
            [10, 30, 20, 99],
            recall_cutoffs=(2, 4, 8),
        )

        assert recall == 0.75
        assert ndcg == pytest.approx(calc_ndcg([10, 20, 30, 40], [10, 30, 20, 99], get_ideal_dcg(4)))
        assert recall_at == {2: 0.5, 4: 0.75}


class TestGetFiles:
    @pytest.mark.parametrize(
        "train_count",
        [
            1,
            10,
            50,
            100,
        ],
    )
    def test_train_count(self, train_count):
        files = utils.compose_train_files(train_count, True)
        log.info(files)

        assert len(files) == train_count

    @pytest.mark.parametrize("use_shuffled", [True, False])
    def test_use_shuffled(self, use_shuffled):
        files = utils.compose_train_files(1, use_shuffled)
        log.info(files)

        trains = [f for f in files if "train" in f]
        if use_shuffled:
            for t in trains:
                assert "shuffle_train" in t
        else:
            for t in trains:
                assert "shuffle" not in t
                assert "train" in t
