from vectordb_bench.models import CaseType

passwordKeys = ["password", "api_key", "key"]
def inputIsPassword(key: str) -> bool:
    return key.lower() in passwordKeys

