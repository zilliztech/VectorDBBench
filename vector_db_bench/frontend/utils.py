from vector_db_bench.models import CaseType

passwordKeys = ["password", "api_key"]
def inputIsPassword(key: str) -> bool:
    return key.lower() in passwordKeys


caseTextMap = {
    CaseType.LoadLDim.value: "Capacity Test (Large-dim)",
    CaseType.LoadSDim.value: "Capacity Test (Small-dim)",
    CaseType.PerformanceLZero.value: "Search Performance Test (Large Dataset)",
    CaseType.PerformanceMZero.value: "Search Performance Test (Medium Dataset)",
    CaseType.PerformanceSZero.value: "Search Performance Test (Small Dataset)",
    CaseType.PerformanceLLow.value: (
        "Filtering Search Performance Test (Large Dataset, Low Filtering Rate)"
    ),
    CaseType.PerformanceMLow.value: (
        "Filtering Search Performance Test (Medium Dataset, Low Filtering Rate)"
    ),
    CaseType.PerformanceSLow.value: (
        "Filtering Search Performance Test (Small Dataset, Low Filtering Rate)"
    ),
    CaseType.PerformanceLHigh.value: (
        "Filtering Search Performance Test (Large Dataset, High Filtering Rate)"
    ),
    CaseType.PerformanceMHigh.value: (
        "Filtering Search Performance Test (Medium Dataset, High Filtering Rate)"
    ),
    CaseType.PerformanceSHigh.value: (
        "Filtering Search Performance Test (Small Dataset, High Filtering Rate)"
    ),
    CaseType.Performance100M.value: "Search Performance Test (XLarge Dataset)",
}


def displayCaseText(case):
    return caseTextMap.get(case, case)
