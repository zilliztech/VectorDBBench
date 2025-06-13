import json

from pydantic import BaseModel

from vectordb_bench import config


class CustomDatasetConfig(BaseModel):
    name: str = "custom_dataset"
    dir: str = ""
    size: int = 0
    dim: int = 0
    metric_type: str = "L2"
    file_count: int = 1
    use_shuffled: bool = False
    with_gt: bool = True
    train_name: str = "train"
    test_name: str = "test"
    gt_name: str = "neighbors"
    train_id_name: str = "id"
    train_col_name: str = "emb"
    test_col_name: str = "emb"
    gt_col_name: str = "neighbors_id"
    scalar_labels_name: str = "scalar_labels"
    label_percentages: list[str] = []
    with_label_percentages: list[float] = [0.001, 0.02, 0.5]


class CustomCaseConfig(BaseModel):
    name: str = "custom_dataset (Performace Case)"
    description: str = ""
    load_timeout: int = 36000
    optimize_timeout: int = 36000
    dataset_config: CustomDatasetConfig = CustomDatasetConfig()


def get_custom_configs():
    with open(config.CUSTOM_CONFIG_DIR, "r") as f:
        custom_configs = json.load(f)
        return [CustomCaseConfig(**custom_config) for custom_config in custom_configs]


def save_custom_configs(custom_configs: list[CustomDatasetConfig]):
    with open(config.CUSTOM_CONFIG_DIR, "w") as f:
        json.dump([custom_config.dict() for custom_config in custom_configs], f, indent=4)


def generate_custom_case():
    return CustomCaseConfig()
