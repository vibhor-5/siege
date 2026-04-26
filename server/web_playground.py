"""Custom Gradio playground for Interpretability Arena.

OpenEnv's default UI labels fields by snake_case → Title Case, which reads as
vague ("Red Type"). We use Pydantic ``Field(title=...)`` on ``InterpArenaAction``
and build the form from those titles, with Red / Blue section headers.

One ``env.step()`` always carries **both** agents: every field in
``InterpArenaAction`` is sent together (Red + Blue in the same payload).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Type

import gradio as gr
from fastapi import Body, FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse

from openenv.core.env_server.gradio_theme import OPENENV_GRADIO_CSS, OPENENV_GRADIO_THEME
from openenv.core.env_server.gradio_ui import get_gradio_display_title, _format_observation
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation
from openenv.core.env_server.web_interface import (
    WebInterfaceManager,
    _extract_action_fields,
    _is_chat_env,
    get_quick_start_markdown,
    load_environment_metadata,
)


def _arena_enrich_action_fields(action_cls: Type[Action]) -> List[Dict[str, Any]]:
    """Like OpenEnv's extractor, plus JSON Schema ``title`` / ``description``."""
    fields = _extract_action_fields(action_cls)
    props = action_cls.model_json_schema().get("properties", {})
    for f in fields:
        name = f["name"]
        fi = props.get(name, {})
        f["title"] = fi.get("title")
        f["schema_description"] = (fi.get("description") or "").strip()
    return fields


def _coerce_web_field(name: str, val: Any, field_type: str) -> Any:
    """Map Gradio widget values → JSON types ``InterpArenaAction`` expects."""
    if val is None or val == "":
        return None
    if field_type == "checkbox":
        return bool(val)
    int_like = (
        name.endswith("_layer")
        or name.endswith("_head")
        or name.endswith("_position")
        or name.endswith("_token_ids")
        or name.endswith("_ids")
    )
    if field_type == "number" and int_like and isinstance(val, float):
        if val == int(val):
            return int(val)
    if name in ("red_target_token_ids", "blue_prohibited_token_ids") and isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            if s.startswith("["):
                return json.loads(s)
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        except (ValueError, json.JSONDecodeError):
            return val
    return val


def build_arena_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str = "OpenEnv Environment",
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    """Gradio Blocks mirroring OpenEnv's ``build_gradio_app`` with clearer labels."""
    from openenv.core.env_server.gradio_ui import _readme_section

    readme_content = _readme_section(metadata)
    display_title = get_gradio_display_title(metadata, fallback=title)

    async def reset_env():
        try:
            data = await web_manager.reset_environment()
            obs_md = _format_observation(data)
            return (
                obs_md,
                json.dumps(data, indent=2),
                "Environment reset successfully.",
            )
        except Exception as e:
            return ("", "", f"Error: {e}")

    def _step_with_action(action_data: Dict[str, Any]):
        async def _run():
            try:
                data = await web_manager.step_environment(action_data)
                obs_md = _format_observation(data)
                return (
                    obs_md,
                    json.dumps(data, indent=2),
                    "Step complete.",
                )
            except Exception as e:
                return ("", "", f"Error: {e}")

        return _run

    async def step_chat(message: str):
        if not (message or str(message).strip()):
            return ("", "", "Please enter an action message.")
        action = {"message": str(message).strip()}
        return await _step_with_action(action)()

    def get_state_sync():
        try:
            data = web_manager.get_state()
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error: {e}"

    with gr.Blocks(title=display_title) as demo:
        with gr.Row():
            with gr.Column(scale=1, elem_classes="col-left"):
                if quick_start_md:
                    with gr.Accordion("Quick Start", open=True):
                        gr.Markdown(quick_start_md)
                with gr.Accordion("README", open=False):
                    gr.Markdown(readme_content)

            with gr.Column(scale=2, elem_classes="col-right"):
                obs_display = gr.Markdown(
                    value=(
                        "# Playground\n\n"
                        "Click **Reset** to start a new episode.\n\n"
                        "**Each Step sends Red and Blue together** — fill both sides "
                        "(or leave Blue on *noop*). One combined action → one model forward."
                    ),
                )
                with gr.Group():
                    if is_chat_env:
                        action_input = gr.Textbox(
                            label="Action message",
                            placeholder="e.g. Enter your message...",
                        )
                        step_inputs = [action_input]
                        step_fn = step_chat
                    else:
                        step_inputs = []
                        seen_red = False
                        seen_blue = False
                        for field in action_fields:
                            name = field["name"]
                            if name.startswith("red_") and not seen_red:
                                gr.Markdown(
                                    "#### Red (attacker)\n"
                                    "Pick an intervention type, then only the parameters that action uses "
                                    "(others can stay empty)."
                                )
                                seen_red = True
                            if name.startswith("blue_") and not seen_blue:
                                gr.Markdown(
                                    "#### Blue (defender)\n"
                                    "Pick how Blue intervenes on the **same** forward pass as Red."
                                )
                                seen_blue = True

                            field_type = field.get("type", "text")
                            label = field.get("title") or name.replace("_", " ").title()
                            placeholder = field.get("placeholder", "")
                            info = (field.get("schema_description") or "")[:500] or None

                            common_kw: dict[str, Any] = {"label": label}
                            if info:
                                common_kw["info"] = info

                            if field_type == "checkbox":
                                inp = gr.Checkbox(**common_kw)
                            elif field_type == "number":
                                inp = gr.Number(**common_kw)
                            elif field_type == "select":
                                choices = field.get("choices") or []
                                inp = gr.Dropdown(
                                    choices=choices,
                                    allow_custom_value=False,
                                    **common_kw,
                                )
                            elif field_type in ("textarea", "tensor"):
                                inp = gr.Textbox(
                                    placeholder=placeholder,
                                    lines=3,
                                    **common_kw,
                                )
                            else:
                                inp = gr.Textbox(placeholder=placeholder, **common_kw)
                            step_inputs.append(inp)

                        async def step_form(*values):
                            if not action_fields:
                                return await _step_with_action({})()
                            action_data: Dict[str, Any] = {}
                            for i, field in enumerate(action_fields):
                                if i >= len(values):
                                    break
                                fname = field["name"]
                                raw = values[i]
                                ft = field.get("type", "text")
                                coerced = _coerce_web_field(fname, raw, ft)
                                if coerced is not None and coerced != "":
                                    action_data[fname] = coerced
                            return await _step_with_action(action_data)()

                        step_fn = step_form

                    with gr.Row():
                        step_btn = gr.Button("Step", variant="primary")
                        reset_btn = gr.Button("Reset", variant="secondary")
                        state_btn = gr.Button("Get state", variant="secondary")
                    with gr.Row():
                        status = gr.Textbox(label="Status", interactive=False)
                    raw_json = gr.Code(
                        label="Raw JSON response",
                        language="json",
                        interactive=False,
                    )

        reset_btn.click(fn=reset_env, outputs=[obs_display, raw_json, status])
        step_btn.click(fn=step_fn, inputs=step_inputs, outputs=[obs_display, raw_json, status])
        if is_chat_env:
            action_input.submit(fn=step_fn, inputs=step_inputs, outputs=[obs_display, raw_json, status])
        state_btn.click(fn=get_state_sync, outputs=[raw_json])

    return demo


def create_arena_web_interface_app(
    env: Environment | Type[Environment],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[Any] = None,
) -> FastAPI:
    """Same as OpenEnv ``create_web_interface_app`` but with arena-specific Gradio UI."""
    app = create_fastapi_app(
        env, action_cls, observation_cls, max_concurrent_envs, concurrency_config
    )

    metadata = load_environment_metadata(env, env_name)
    web_manager = WebInterfaceManager(env, action_cls, observation_cls, metadata)

    @app.get("/", include_in_schema=False)
    async def web_root():
        return RedirectResponse(url="/web/")

    @app.get("/web", include_in_schema=False)
    async def web_root_no_slash():
        return RedirectResponse(url="/web/")

    @app.get("/web/metadata")
    async def web_metadata():
        return web_manager.metadata.model_dump()

    @app.websocket("/ws/ui")
    async def websocket_ui_endpoint(websocket: WebSocket):
        await web_manager.connect_websocket(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await web_manager.disconnect_websocket(websocket)

    @app.post("/web/reset")
    async def web_reset(request: Optional[Dict[str, Any]] = Body(default=None)):
        return await web_manager.reset_environment(request)

    @app.post("/web/step")
    async def web_step(request: Dict[str, Any]):
        if "message" in request:
            message = request["message"]
            if hasattr(web_manager.env, "message_to_action"):
                action = web_manager.env.message_to_action(message)
                if hasattr(action, "tokens"):
                    action_data = {"tokens": action.tokens.tolist()}
                else:
                    action_data = action.model_dump(exclude={"metadata"})
            else:
                action_data = {"message": message}
        else:
            action_data = request.get("action", {})

        return await web_manager.step_environment(action_data)

    @app.get("/web/state")
    async def web_state():
        try:
            return web_manager.get_state()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    action_fields = _arena_enrich_action_fields(action_cls)
    is_chat_env = _is_chat_env(action_cls)
    quick_start_md = get_quick_start_markdown(metadata, action_cls, observation_cls)

    gradio_blocks = build_arena_gradio_app(
        web_manager,
        action_fields,
        metadata,
        is_chat_env,
        title=metadata.name,
        quick_start_md=quick_start_md,
    )
    return gr.mount_gradio_app(
        app,
        gradio_blocks,
        path="/web",
        theme=OPENENV_GRADIO_THEME,
        css=OPENENV_GRADIO_CSS,
    )
