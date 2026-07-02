"""Anthropic provider: talks to the real Claude API, rotating across keys.

Configure ``ANTHROPIC_API_KEY`` with one key, or several comma-separated keys
(e.g. ``"sk-ant-aaa,sk-ant-bbb,sk-ant-ccc"``). Whenever the key currently in
use comes back rate-limited (HTTP 429) or overloaded (HTTP 529), the provider
transparently retries the same request with the next key in the list before
surfacing an error. Combine this with ``MODEL_OPUS`` / ``MODEL_SONNET`` /
``MODEL_HAIKU`` pointing at ``anthropic/claude-...`` to use it as your
top-tier route, or as the last resort once the free providers above it are
exhausted.
"""

from __future__ import annotations

from typing import Any

import httpx

from providers.base import ProviderConfig
from providers.defaults import ANTHROPIC_DEFAULT_BASE
from providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
    NativeMessagesRequestPolicy,
    build_native_messages_request_body,
)
from providers.transports.http import maybe_await_aclose

_ANTHROPIC_VERSION = "2023-06-01"
# 429 = rate limited, 529 = Anthropic-side overloaded. Both are worth
# retrying on a different key; anything else (400, 401, 403...) is a real
# error and should surface immediately instead of burning through keys.
_ROTATE_ON_STATUS_CODES = frozenset({429, 529})
_REQUEST_POLICY = NativeMessagesRequestPolicy(provider_name="ANTHROPIC")


class AnthropicProvider(AnthropicMessagesTransport):
    """Real Anthropic Claude API, with automatic multi-key rotation."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="ANTHROPIC",
            default_base_url=ANTHROPIC_DEFAULT_BASE,
        )
        keys = tuple(key.strip() for key in config.api_key.split(",") if key.strip())
        self._keys: tuple[str, ...] = keys or (config.api_key,)
        self._current_key_index = 0

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_native_messages_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )

    def _headers_for_key(self, api_key: str) -> dict[str, str]:
        return {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _request_headers(self) -> dict[str, str]:
        return self._headers_for_key(self._keys[self._current_key_index])

    def _model_list_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._keys[self._current_key_index],
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        """Send the request, rotating to the next key on 429/529 responses."""
        response: httpx.Response | None = None
        attempts = len(self._keys)
        for attempt in range(attempts):
            api_key = self._keys[self._current_key_index]
            request = self._client.build_request(
                "POST",
                "/messages",
                json=body,
                headers=self._headers_for_key(api_key),
            )
            response = await self._client.send(request, stream=True)
            if response.status_code not in _ROTATE_ON_STATUS_CODES:
                return response
            is_last_attempt = attempt == attempts - 1
            if is_last_attempt:
                return response
            # This key is exhausted/overloaded: close its response, advance
            # to the next key, and try the same request again.
            await maybe_await_aclose(response)
            self._current_key_index = (self._current_key_index + 1) % attempts
        assert response is not None  # attempts >= 1, loop always sets response
        return response
