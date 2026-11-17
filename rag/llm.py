"""Provider-agnostic chat client.

Only urllib is used, so OpenAI-compatible providers (OpenRouter, Groq, Gemini,
custom endpoints) and Ollama's native API all work without extra SDKs.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator

from .config import Settings


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    usage: dict = field(default_factory=dict)
    cached: bool = False
    backend: str = ""
    model: str = ""


class BaseBackend:
    api_style = "?"

    def __init__(self, cfg: dict, timeout: int = 120):
        self.cfg = cfg
        self.timeout = timeout
        self.base_url = (cfg.get("base_url") or "").rstrip("/")
        if not self.base_url:
            raise LLMError(
                f"No base_url configured for backend '{cfg['name']}'. "
                "Set RAG_BASE_URL or a provider preset."
            )

    def _post(self, path: str, payload: dict, headers: dict | None = None, stream: bool = False):
        url = self.base_url + path
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:800]
            raise LLMError(f"{exc.code} {exc.reason} from {url}: {body}") from exc
        except TimeoutError as exc:
            raise LLMError(
                f"Request to {url} timed out after {self.timeout}s. "
                "For slow local models try --timeout 600, or --no-think if the "
                "model is a thinking model."
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Could not reach {url}: {exc.reason}") from exc

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> LLMResponse:
        raise NotImplementedError

    def chat_stream(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> Iterator[str]:
        raise NotImplementedError

    def embeddings(self, texts: list[str], model: str) -> list[list[float]]:
        raise NotImplementedError


class OllamaBackend(BaseBackend):
    api_style = "ollama"

    def _payload(self, messages, temperature, max_tokens, stream):
        payload = {
            "model": self.cfg["model"],
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        # Thinking models (e.g. lfm3) can consume the entire token budget on
        # private reasoning and return an empty 'content' unless this is sent.
        payload["think"] = bool(self.cfg.get("think", False))
        return payload

    def chat(self, messages, temperature, max_tokens):
        with self._post("/api/chat", self._payload(messages, temperature, max_tokens, False)) as response:
            data = json.loads(response.read().decode("utf-8"))
        message = data.get("message", {})
        text = message.get("content", "")
        if not text and message.get("thinking"):
            raise LLMError(
                "Ollama returned only thinking tokens and no answer. "
                "Re-run with --no-think (or RAG_THINK=0), or raise --max-tokens."
            )
        return LLMResponse(
            text=text,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_duration_ms": round(data.get("total_duration", 0) / 1_000_000),
            },
            backend=self.cfg["name"],
            model=self.cfg["model"],
        )

    def chat_stream(self, messages, temperature, max_tokens):
        with self._post("/api/chat", self._payload(messages, temperature, max_tokens, True)) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                data = json.loads(line)
                delta = data.get("message", {}).get("content", "")
                if delta:
                    yield delta

    def embeddings(self, texts: list[str], model: str):
        raise LLMError(
            "This Ollama server does not expose /api/embed (it was started without "
            "--embeddings). Use --embed-backend fastembed, or restart Ollama with "
            "`ollama serve --embeddings` and pull an embedding model."
        )


class OpenAICompatibleBackend(BaseBackend):
    api_style = "openai"

    def _headers(self) -> dict:
        headers = {}
        if self.cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {self.cfg['api_key']}"
        return headers

    def _payload(self, messages, temperature, max_tokens, stream):
        return {
            "model": self.cfg["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def chat(self, messages, temperature, max_tokens):
        with self._post(
            "/chat/completions",
            self._payload(messages, temperature, max_tokens, False),
            self._headers(),
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected chat response: {str(data)[:500]}") from exc
        return LLMResponse(
            text=text,
            usage=dict(data.get("usage") or {}),
            backend=self.cfg["name"],
            model=self.cfg["model"],
        )

    def chat_stream(self, messages, temperature, max_tokens):
        with self._post(
            "/chat/completions",
            self._payload(messages, temperature, max_tokens, True),
            self._headers(),
        ) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                try:
                    delta = data["choices"][0].get("delta", {}).get("content") or ""
                except (KeyError, IndexError):
                    continue
                if delta:
                    yield delta

    def embeddings(self, texts: list[str], model: str) -> list[list[float]]:
        with self._post(
            "/embeddings",
            {"model": model, "input": texts},
            self._headers(),
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
        try:
            return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected embeddings response: {str(data)[:500]}") from exc


def get_backend(settings: Settings) -> BaseBackend:
    cfg = settings.resolve_llm()
    cls = OllamaBackend if cfg["api_style"] == "ollama" else OpenAICompatibleBackend
    return cls(cfg, timeout=settings.request_timeout)  # type: ignore[return-value]


def _cache_key(settings: Settings, messages: list[dict]) -> str:
    cfg = settings.resolve_llm()
    payload = {
        "backend": cfg["name"],
        "base_url": cfg["base_url"],
        "model": cfg["model"],
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "think": settings.think,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _cache_file(settings: Settings, key: str):
    directory = settings.cache_dir / "llm"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.json"


def _read_cache(settings: Settings, key: str) -> LLMResponse | None:
    path = _cache_file(settings, key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse(
            text=data["text"],
            usage=data.get("usage", {}),
            cached=True,
            backend=data.get("backend", ""),
            model=data.get("model", ""),
        )
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def _write_cache(settings: Settings, key: str, response: LLMResponse) -> None:
    path = _cache_file(settings, key)
    path.write_text(
        json.dumps(
            {"text": response.text, "usage": response.usage, "backend": response.backend, "model": response.model},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _log_usage(settings: Settings, response: LLMResponse) -> None:
    path = settings.cache_dir / "usage.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "backend": response.backend,
                    "model": response.model,
                    "cached": response.cached,
                    "think": settings.think,
                    **response.usage,
                }
            )
            + "\n"
        )


def chat(settings: Settings, messages: list[dict], stream: bool = False):
    """Return an LLMResponse, or a generator of text deltas when stream=True."""
    cfg = settings.resolve_llm()
    key = _cache_key(settings, messages)
    cached = _read_cache(settings, key)
    if cached and not stream:
        return cached
    backend = get_backend(settings)

    if stream:
        def generator() -> Iterator[str]:
            if cached:
                yield cached.text
                return
            parts: list[str] = []
            for delta in backend.chat_stream(messages, settings.temperature, settings.max_tokens):
                parts.append(delta)
                yield delta
            response = LLMResponse("".join(parts), backend=cfg["name"], model=str(cfg["model"]))
            _write_cache(settings, key, response)
            _log_usage(settings, response)

        return generator()

    if cached:
        return cached
    response = backend.chat(messages, settings.temperature, settings.max_tokens)
    _write_cache(settings, key, response)
    _log_usage(settings, response)
    return response


def embed_texts(settings: Settings, texts: list[str]) -> list[list[float]]:
    """Cloud embeddings through the configured OpenAI-compatible endpoint."""
    backend = get_backend(settings)
    if not isinstance(backend, OpenAICompatibleBackend):
        raise LLMError("Cloud embeddings require an OpenAI-compatible backend.")
    model = settings.dense_model_for_openai
    out: list[list[float]] = []
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        out.extend(backend.embeddings(batch, model))
    return out
