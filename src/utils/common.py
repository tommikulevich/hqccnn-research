"""Common utilities."""
from enum import Enum
from typing import Any


def flatten_dict(d: dict[str, Any], parent_key: str = "",
                 sep: str = ".") -> dict[str, Any]:
    """{"a": {"b": 1}, "c": [ {"x":2}, 3 ]}
       -> {"a.b": 1, "c.0.x": 2, "c.1": 3}"""
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))

        elif isinstance(v, (list, tuple)):
            for idx, item in enumerate(v):
                item_key = f"{new_key}{sep}{idx}"
                if isinstance(item, dict):
                    items.update(flatten_dict(item, item_key, sep=sep))
                else:
                    items[item_key] = item

        else:
            if isinstance(v, Enum):
                items[new_key] = v.value
            else:
                items[new_key] = v

    return items
