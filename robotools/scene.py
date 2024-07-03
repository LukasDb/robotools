import robotools
import importlib
from robotools import Entity
from typing import TypeVar, Union, Callable

BaseEntityType = TypeVar("BaseEntityType", bound="Entity")

# BaseModelType = TypeVar("BaseModelType", bound="BaseModel")


class Scene:
    def __init__(self):
        self._entities: dict[str, Entity] = {}

    def add_entity(self, entity: BaseEntityType) -> BaseEntityType:
        self._entities[entity.name] = entity
        return entity

    def to_config(self):
        output = {}
        for name, entity in self._entities.items():
            cls_name = str(entity.__class__).split("'")[1]

            num_present = len(
                [x for x in self._entities.values() if isinstance(x, entity.__class__)]
            )
            output[cls_name + f".{num_present:03}"] = entity.to_config()

        return output

    def from_config(self, config: dict[str, str]):
        for cls_name, config in config.items():
            # instanciate cls from cls name
            module = ".".join(cls_name.split(".")[:-2])
            cls: type[Entity] = getattr(importlib.import_module(module), cls_name.split(".")[-2])
            name = config.get('name', cls_name)  # Default to cls_name if no name is provided
            
            entity = cls()
        
            entity.from_config(config)
            self.add_entity(entity)
