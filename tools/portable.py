"""Keep artifact identity and command flags without publishing local directories."""
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]


def portable(value: Any, locations: Mapping[Path | str, str] | None = None) -> Any:
    substitutions = {str(ROOT): "."}
    # Never replace relative words: a directory named "a" must not edit a hash.
    substitutions.update({str(Path(path).resolve()): name for path, name in (locations or {}).items()})
    substitutions = sorted(substitutions.items(), key=lambda item: len(item[0]), reverse=True)

    def convert(item: Any) -> Any:
        if isinstance(item, str):
            for path, name in substitutions:
                item = item.replace(path, name)
            return item
        if isinstance(item, dict):
            return {key: convert(child) for key, child in item.items()}
        if isinstance(item, list):
            return [convert(child) for child in item]
        return item

    return convert(value)
