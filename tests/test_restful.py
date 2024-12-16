import requests
from loguru import logger
import time

base_url = "http://10.100.36.26:5000"


def test_get(path: str, data: dict = None):
    url = base_url + path
    logger.info(f"[GET] {path}")
    response = requests.get(url, params=data)
    logger.info(f"[GET] {path} - status_code: {response.status_code}")
    return response.json()


def test_post(path: str, data: dict = None):
    url = base_url + path
    logger.info(f"[POST] {path}")
    response = requests.post(url, json=data)
    logger.info(f"[POST] {path} - status_code: {response.status_code}")
    return response.json()


def test_run(task_label: str, tasks: list[dict], use_aliyun: bool = True):
    data = dict(task_label=task_label, tasks=tasks, use_aliyun=use_aliyun)
    res = test_post("/run", data)
    logger.info(f"test_run: {res}")
    return res


def test_stop():
    res = test_get("/stop")
    logger.info(f"test_stop: {res}")


def test_get_status():
    res = test_get("/get_status")
    logger.info(f"test_get_status: {res}")


def test_get_res(task_label: str = "standard"):
    res = test_get("/get_res", dict(task_label=task_label))
    logger.info(f"get_res: {res}")


if __name__ == "__main__":
    # 示例调用
    task_label = "example"
    use_aliyun = True
    tasks = [
        dict(
            db="Milvus",
            case_config=dict(case_id=50),  # Performance1536D50K
            db_config=dict(),
            db_case_config=dict(index="HNSW", M=16, efConstruction=100, ef=100),
            stages=["search_serial"],
        ),
        dict(
            db="Milvus",
            case_config=dict(case_id=50),  # Performance1536D50K
            db_config=dict(),
            db_case_config=dict(index="HNSW", M=16, efConstruction=100, ef=512),
            stages=["search_concurrent"],
        ),
        dict(
            db="Milvus",
            case_config=dict(
                case_id=50,
                k=200,
                concurrency_search_config=dict(
                    num_concurrency=[5], concurrency_duration=10
                ),
            ),  # Performance1536D50K
            db_config=dict(),
            db_case_config=dict(index="HNSW", M=16, efConstruction=100, ef=512),
            stages=["search_concurrent"],
        ),
    ]
    test_run(task_label, tasks, use_aliyun=use_aliyun)

    # time.sleep(10)
    # test_stop()
