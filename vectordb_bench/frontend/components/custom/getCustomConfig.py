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
