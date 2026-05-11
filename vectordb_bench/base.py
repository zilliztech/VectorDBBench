from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
