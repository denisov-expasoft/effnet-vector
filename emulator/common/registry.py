from os import environ
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from emulator.common.exceptions import EmulatorRegistryKeyError
from emulator.common.misc_utils import normalize_to_list


def _is_strict() -> bool:
    return environ.get('EMULATOR_FLEXIBLE_REGISTRIES') is None


class Registry(dict):

    def __init__(self, key_type: Optional[type] = str, value_type: type = None):
        self._key_type = key_type
        self._value_type = value_type
        super().__init__()

    def _check_key_type(self, key: Any) -> None:
        if self._key_type is None or isinstance(key, self._key_type):
            return

        raise TypeError(f'registry key must be of type {self._key_type}, got {type(key)}')

    def _check_value_type(self, value: Any) -> None:
        if self._value_type is None or isinstance(value, self._value_type):
            return

        raise TypeError(f'registry value must be of type {self._value_type}, got {type(value)}')

    def __getitem__(self, key: Any) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError as exc:
            raise EmulatorRegistryKeyError(f'item with key "{key}" does not exist') from exc

    def __setitem__(self, key: Any, value: Any) -> None:
        self._check_key_type(key)
        self._check_value_type(value)
        if key in self and _is_strict():
            raise ValueError(f'item with key "{key}" already exists')

        super().__setitem__(key, value)

    def add_item_decorator(self, aliases: Union[Any, List[Any]]) -> Callable[[Any], Any]:
        aliases = normalize_to_list(aliases)

        def add_item(item):
            for alias in aliases:
                self.__setitem__(alias, item)

            return item

        return add_item
