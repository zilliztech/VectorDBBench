import pytest
import logging
from falcon_mark.backend import cases

log  = logging.getLogger(__name__)
class TestCases:
    @pytest.mark.skip()
    def test_init_LoadCase(self):
        c = cases.LoadSDimCase(run_id=1)
        log.debug(f"c: {c}")


    def test_case_type(self):
        from falcon_mark.models import CaseType
        log.debug(f"{CaseType.LoadLDim}")
