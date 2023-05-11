import pytest

from falcon_mark.backend import utils

class TestUtils:
    @pytest.mark.parametrize("testcases", [
        (1, '1'),
        (10, '10'),
        (100, '100'),
        (1000, '1K'),
        (2000, '2K'),
        (30_000, '30K'),
        (400_000, '400K'),
        (5_000_000, '5M'),
        (60_000_000, '60M'),
        (1_000_000_000, '1B'),
        (1_000_000_000_000, '1000B'),
    ])
    def test_numerize(self, testcases):
        t_in, expected = testcases
        assert expected == utils.numerize(t_in)
