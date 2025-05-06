"""Common utilities."""
from enum import Enum
from typing import Any


def flatten_dict(d: dict[str, Any], parent_key: str = "",
                 sep: str = ".") -> dict[str, Any]:
    """{"a": {"b": 1}, "c": Enum.NAME} -> {"a.b": 1, "c": "Value"}"""
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))

        else:
            if isinstance(v, Enum):
                items[new_key] = v.value
            else:
                items[new_key] = v

    return items
