from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel, arbitrary_types_allowed=True):
    pass
