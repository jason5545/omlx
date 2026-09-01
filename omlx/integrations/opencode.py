"""OpenCode integration."""

from __future__ import annotations

import copy
import os
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext
from omlx.integrations.pi import PiIntegration
from omlx.utils.install import get_cli_command_prefix

# Reasoning variants for Qwen3.8-style chat templates. The only reliable
# thinking on/off channel is chat_template_kwargs.enable_thinking;
# reasoning_effort is a soft system-prompt hint limited to the template's
# vocabulary (xhigh/medium/low). OpenCode passes arbitrary variant keys
# through to the request body, and oMLX strips unsupported template kwargs,
# so these are safe for non-Qwen reasoning models too.
_REASONING_VARIANTS = {
    "off": {"chat_template_kwargs": {"enable_thinking": False}},
    "low": {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}
    },
    "medium": {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}
    },
    "xhigh": {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}
    },
}

# Model-entry keys preserved across re-launches (hand-tuned in opencode.json).
_PRESERVED_MODEL_KEYS = ("reasoning", "interleaved", "variants", "options")


class OpenCodeIntegration(Integration):
    """OpenCode integration that writes ~/.config/opencode/opencode.json."""

    CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"

    def __init__(self):
        super().__init__(
            name="opencode",
            display_name="OpenCode",
            type="config_file",
            install_check="opencode",
            install_hint="curl -fsSL https://opencode.ai/install | bash",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        return (
            f"{get_cli_command_prefix()} "
            f"launch opencode --model {ctx.model or 'select-a-model'}"
        )

    @staticmethod
    def _modalities_for_model(model_type: str | None) -> dict[str, list[str]]:
        """Build OpenCode modality metadata for the selected oMLX model."""
        input_modalities = ["text"]
        if model_type == "vlm":
            input_modalities.append("image")
        return {
            "input": input_modalities,
            "output": ["text"],
        }

    def configure(self, ctx: IntegrationContext) -> None:
        def updater(config: dict) -> None:
            config.setdefault("provider", {})
            provider_config = {
                "npm": "@ai-sdk/openai-compatible",
                "name": "oMLX",
                "options": {
                    "baseURL": ctx.openai_base_url,
                },
            }
            if ctx.api_key:
                provider_config["options"]["apiKey"] = ctx.api_key
            if ctx.model:
                model_entry: dict = {
                    "name": ctx.model,
                    "modalities": self._modalities_for_model(ctx.model_type),
                }
                if ctx.supports_images:
                    model_entry["attachment"] = True
                if ctx.context_window:
                    model_entry["limit"] = {
                        "context": ctx.context_window,
                        "output": ctx.max_tokens or ctx.context_window,
                    }
                # Keep hand-tuned keys (variants, reasoning, ...) for the
                # same model id across re-launches.
                existing_entry = (
                    (config.get("provider", {}).get("omlx", {}).get("models", {}) or {})
                    .get(ctx.model)
                    or {}
                )
                for key in _PRESERVED_MODEL_KEYS:
                    if key in existing_entry:
                        model_entry[key] = existing_entry[key]
                reasoning = (
                    bool(ctx.reasoning)
                    if ctx.reasoning is not None
                    else PiIntegration._is_reasoning_model(ctx.model)
                )
                if reasoning and "variants" not in model_entry:
                    model_entry["reasoning"] = True
                    model_entry["interleaved"] = {"field": "reasoning_content"}
                    model_entry["variants"] = copy.deepcopy(_REASONING_VARIANTS)
                provider_config["models"] = {ctx.model: model_entry}
            config["provider"]["omlx"] = provider_config

            # Set as default model
            if ctx.model:
                config["model"] = f"omlx/{ctx.model}"

        self._write_json_config(self.CONFIG_PATH, updater)

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        args = ["opencode", *ctx.extra_args]

        os.execvpe("opencode", args, env)
