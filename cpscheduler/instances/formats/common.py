"""Common methods for general parsing."""

from typing import Any, TextIO


def read_metadata(f: TextIO) -> dict[str, Any]:
    """Parse key: value pairs after a '---' separator in the current file."""
    for line in f:
        if line.strip().startswith("---"):
            break
    else:
        return {}

    metadata: dict[str, Any] = {}

    for line in f:
        line = line.strip()

        if ":" in line:
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()

            if val.lower() == "true":
                metadata[key] = True

            elif val.lower() == "false":
                metadata[key] = False

            else:
                try:
                    metadata[key] = int(val)
                except ValueError:
                    try:
                        metadata[key] = float(val)
                    except ValueError:
                        metadata[key] = val

    return metadata
