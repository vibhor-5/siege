"""FastAPI server entry point for Interpretability Arena."""

import os
import sys

# transformer-lens imports this at load time; a mismatched transformers (e.g. 5.x) fails on first
# reset. Fail here with a clear fix if the *uvicorn* interpreter is not the project venv.
try:
    from transformers import BertForPreTraining  # noqa: F401
except Exception as e:
    sys.exit(
        "Arena server: could not import transformers' BertForPreTraining (required by transformer-lens). "
        "The process running uvicorn must use the same env as `uv run` (see README section 3). "
        "Try:  uv run uvicorn server.app:app --host 0.0.0.0 --port 8000  "
        "or:  pip install -r server/requirements.txt --force-reinstall  then restart. "
        f"Original: {e}"
    )

from openenv.core.env_server import create_fastapi_app, create_web_interface_app

from models import InterpArenaAction, InterpArenaObservation
from server.interp_arena_environment import InterpArenaEnvironment

# OpenEnv expects the environment *class* (or factory), not an instance—
# the HTTP server instantiates it per its own lifecycle.
if os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() == "true":
    app = create_web_interface_app(InterpArenaEnvironment, InterpArenaAction, InterpArenaObservation)
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
