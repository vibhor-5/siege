"""FastAPI server entry point for Interpretability Arena."""

import os
import sys

# OpenEnv runs sync reset() in a thread pool; the first `import transformer_lens` happens there.
# Loading TL here in the main import path ensures: (1) the same venv is used as training, and
# (2) the server process exits immediately if this stack is broken, instead of failing on first
# WebSocket reset with a cryptic remote error.
try:
    import torch  # noqa: F401 — transformer-lens / HF expect torch imported first
    from transformers import BertForPreTraining  # noqa: F401
    import transformer_lens  # noqa: F401
except Exception as e:
    sys.exit(
        "Arena server: could not import torch + transformer-lens (same stack as the training client). "
        "The process running uvicorn must use the project venv: `uv run uvicorn server.app:app --host 0.0.0.0 --port 8000`. "
        "If the error mentions Bert, reinstall pins: `uv sync` or `pip install -r server/requirements.txt --force-reinstall`, "
        "then stop any other process on that port and start again. "
        f"Original: {e}"
    )

from openenv.core.env_server import create_fastapi_app

from models import InterpArenaAction, InterpArenaObservation
from server.interp_arena_environment import InterpArenaEnvironment
from server.web_playground import create_arena_web_interface_app

# OpenEnv expects the environment *class* (or factory), not an instance—
# the HTTP server instantiates it per its own lifecycle.
if os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() == "true":
    app = create_arena_web_interface_app(
        InterpArenaEnvironment, InterpArenaAction, InterpArenaObservation
    )
else:
    app = create_fastapi_app(InterpArenaEnvironment, InterpArenaAction, InterpArenaObservation)


def main() -> None:
    """Console entry for OpenEnv multi-mode / `openenv validate` and `uv run server`."""
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
