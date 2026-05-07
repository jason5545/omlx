# SPDX-License-Identifier: Apache-2.0
"""Tests for context window validation feature."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from omlx.model_settings import ModelSettings


class TestGetMaxContextWindow:
    """Tests for get_max_context_window() priority logic."""

    def _make_server_state(self, global_max_ctx=32768):
        """Create a mock server state with given global max_context_window."""
        from omlx.server import SamplingDefaults

        state = MagicMock()
        state.sampling = SamplingDefaults(max_context_window=global_max_ctx)
        state.settings_manager = None
        return state

    def test_returns_global_default(self):
        """Test returns global default when no model settings."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=32768)
        with patch("omlx.server._server_state", state):
            result = get_max_context_window()
            assert result == 32768

    def test_model_setting_overrides_global(self):
        """Test model-specific setting takes priority over global."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=32768)
        mock_manager = MagicMock()
        mock_manager.get_settings.return_value = ModelSettings(
            max_context_window=4096
        )
        state.settings_manager = mock_manager

        with patch("omlx.server._server_state", state):
            result = get_max_context_window("test-model")
            assert result == 4096

    def test_falls_back_to_global_when_model_not_set(self):
        """Test falls back to global when model has no max_context_window."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=65536)
        mock_manager = MagicMock()
        mock_manager.get_settings.return_value = ModelSettings(
            max_context_window=None
        )
        state.settings_manager = mock_manager

        with patch("omlx.server._server_state", state):
            result = get_max_context_window("test-model")
            assert result == 65536

    def test_no_model_id_returns_global(self):
        """Test returns global when model_id is None."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=16384)
        with patch("omlx.server._server_state", state):
            result = get_max_context_window(None)
            assert result == 16384

    def test_request_policy_can_shrink_context_window(self):
        """Test request policy caps the effective context window."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=32768)
        with patch("omlx.server._server_state", state):
            result = get_max_context_window(
                "test-model",
                request_policy={"max_context_window": 8192},
            )
            assert result == 8192

    def test_request_policy_does_not_expand_context_window(self):
        """Test request policy cannot exceed the model/global context limit."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=4096)
        with patch("omlx.server._server_state", state):
            result = get_max_context_window(
                "test-model",
                request_policy={"max_context_window": 8192},
            )
            assert result == 4096


class TestValidateContextWindow:
    """Tests for validate_context_window()."""

    def _make_server_state(self, global_max_ctx=32768):
        from omlx.server import SamplingDefaults

        state = MagicMock()
        state.sampling = SamplingDefaults(max_context_window=global_max_ctx)
        state.settings_manager = None
        return state

    def test_passes_when_under_limit(self):
        """Test no exception when token count is under limit."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=1000)
        with patch("omlx.server._server_state", state):
            # Should not raise
            validate_context_window(500)

    def test_passes_at_exact_limit(self):
        """Test no exception when token count equals limit."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=1000)
        with patch("omlx.server._server_state", state):
            # Should not raise (equal is OK)
            validate_context_window(1000)

    def test_raises_when_over_limit(self):
        """Test HTTPException raised when token count exceeds limit."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=1000)
        with patch("omlx.server._server_state", state):
            with pytest.raises(HTTPException) as exc_info:
                validate_context_window(1001)
            assert exc_info.value.status_code == 400
            assert "1001 tokens" in exc_info.value.detail
            assert "1000 tokens" in exc_info.value.detail

    def test_raises_with_model_specific_limit(self):
        """Test uses model-specific limit when available."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=32768)
        mock_manager = MagicMock()
        mock_manager.get_settings.return_value = ModelSettings(
            max_context_window=100
        )
        state.settings_manager = mock_manager

        with patch("omlx.server._server_state", state):
            with pytest.raises(HTTPException) as exc_info:
                validate_context_window(200, "test-model")
            assert exc_info.value.status_code == 400
            assert "200 tokens" in exc_info.value.detail
            assert "100 tokens" in exc_info.value.detail

    def test_raises_with_request_policy_limit(self):
        """Test request policy limit is used during validation."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=32768)
        with patch("omlx.server._server_state", state):
            with pytest.raises(HTTPException) as exc_info:
                validate_context_window(
                    9000,
                    "test-model",
                    request_policy={"max_context_window": 8192},
                )
            assert exc_info.value.status_code == 400
            assert "8192 tokens" in exc_info.value.detail


class TestRequestPolicy:
    """Tests for per-client request policy resolution."""

    def test_voco_sub_key_uses_default_policy(self):
        from omlx.server import _build_request_policy
        from omlx.settings import SubKeyEntry

        request = MagicMock()
        request.headers = {}
        request.state = SimpleNamespace(
            omlx_api_key_name="voco",
            omlx_sub_key=SubKeyEntry(key="secret", name="voco"),
        )

        policy = _build_request_policy(request)
        assert policy["client"] == "voco"
        assert policy["source"] == "api-sub-key"
        assert policy["max_context_window"] == 16384
        assert policy["enable_thinking"] is False

    def test_sub_key_explicit_policy_overrides_voco_default(self):
        from omlx.server import _build_request_policy
        from omlx.settings import SubKeyEntry

        request = MagicMock()
        request.headers = {}
        request.state = SimpleNamespace(
            omlx_api_key_name="voco",
            omlx_sub_key=SubKeyEntry(
                key="secret",
                name="voco",
                max_context_window=4096,
                enable_thinking=True,
            ),
        )

        policy = _build_request_policy(request)
        assert policy["max_context_window"] == 4096
        assert policy["enable_thinking"] is True

    def test_policy_disables_thinking_template_override(self):
        from omlx.server import _apply_request_chat_template_overrides

        merged = {"preserve_thinking": True}
        _apply_request_chat_template_overrides(
            merged,
            {"enable_thinking": False},
        )
        assert merged == {"enable_thinking": False}


class TestCountChatTokens:
    """Tests for BatchedEngine.count_chat_tokens()."""

    def test_count_chat_tokens(self):
        """Test token counting with mocked tokenizer."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        engine._tokenizer = mock_tokenizer

        # Mock model (not gpt_oss)
        engine._model = MagicMock(spec=[])
        engine._enable_thinking = None

        messages = [{"role": "user", "content": "Hello"}]
        count = engine.count_chat_tokens(messages)

        assert count == 5
        mock_tokenizer.apply_chat_template.assert_called_once()
        mock_tokenizer.encode.assert_called_once_with("formatted prompt")

    def test_count_chat_tokens_with_tools(self):
        """Test token counting includes tools in template."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt with tools"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7]
        engine._tokenizer = mock_tokenizer
        engine._model = MagicMock(spec=[])
        engine._enable_thinking = None

        messages = [{"role": "user", "content": "Call a tool"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        count = engine.count_chat_tokens(messages, tools)

        assert count == 7
