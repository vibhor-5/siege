"""Workaround for Unsloth's use of ``inspect.getsource(BitsAndBytesConfig.__init__)``.

Unsloth patches ``transformers`` by parsing ``__init__`` source. In some runtimes
(e.g. certain studio sandboxes) ``inspect.getsource`` raises ``OSError: could not
get source code`` on methods from an installed ``transformers`` wheel even when the
``quantization_config.py`` file is readable. We re-extract the method from that file
with :func:`ast.get_source_segment` and apply the same patch to ``inspect.getsource`` idempotently
before the first ``import unsloth``.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

_patched = False


def _bitsandbytes_config_init_source() -> str:
    spec = importlib.util.find_spec("transformers.utils.quantization_config")
    if not spec or not spec.origin or not str(spec.origin).endswith(".py"):
        msg = f"Could not find transformers quantization_config.py (spec={spec!r})"
        raise OSError(msg)
    text = Path(spec.origin).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "BitsAndBytesConfig":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                seg = ast.get_source_segment(text, item)
                if seg is not None:
                    return seg
    raise OSError("BitsAndBytesConfig.__init__ not found in quantization_config.py")


def apply_unsloth_inspect_patch() -> None:
    """Idempotently wrap ``inspect.getsource`` to recover BitsAndBytesConfig source when needed."""
    global _patched
    if _patched:
        return

    _real = inspect.getsource

    def getsource(object):  # noqa: ANN001
        try:
            return _real(object)
        except OSError:
            if getattr(object, "__qualname__", None) == "BitsAndBytesConfig.__init__":
                return _bitsandbytes_config_init_source()
            raise

    inspect.getsource = getsource  # type: ignore[assignment]
    _patched = True
