"""FastAPI server entry point for Interpretability Arena."""

import os

from openenv.core.env_server import create_fastapi_app, create_web_interface_app

from models import InterpArenaAction, InterpArenaObservation
from server.interp_arena_environment import InterpArenaEnvironment

env = InterpArenaEnvironment()

if os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() == "true":
    app = create_web_interface_app(env, InterpArenaAction, InterpArenaObservation)
else:
    app = create_fastapi_app(env, InterpArenaAction, InterpArenaObservation)
