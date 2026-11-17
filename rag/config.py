"""Configuration for the RAG terminal app.

Non-secret defaults may live in ~/.config/edu_rag/config.json.
Secrets are read from environment variables only and are never written to disk
by this program.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"
EVAL_DIR = ROOT / "eval"

CONFIG_PATH = Path.home() / ".config" / "edu_rag" / "config.json"

# Portable default: download the book yourself and place it here, or pass
# --pdf PATH / set RAG_PDF_PATH. The file is not shipped with this repo.
DEFAULT_PDF = ROOT / "books" / "noble-warrior.pdf"

# api_style decides which wire format the chat client uses.
PROVIDER_PRESETS: dict[str, dict[str, str | None]] = {
    "ollama": {
        "api_style": "ollama",
        "base_url": "http://localhost:11434",
        "model": "qwen2.5:3b",
    },
    "openrouter": {
        "api_style": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "model": None,
    },
    "groq": {
        "api_style": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "model": None,
    },
    "gemini": {
        "api_style": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash",
    },
    "custom": {
        "api_style": "openai",
        "base_url": None,
        "model": None,
    },
}

_KEY_ENV = {
    "openrouter": ("RAG_API_KEY", "OPENROUTER_API_KEY"),
    "groq": ("RAG_API_KEY", "GROQ_API_KEY"),
    "gemini": ("RAG_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    pdf_path: Path = DEFAULT_PDF
    data_dir: Path = DATA_DIR
    chunk_size: int = 1200
    overlap: int = 200
    top_k: int = 4
    llm_backend: str = "ollama"
    llm_model: str = "qwen2.5:3b"
    base_url: str | None = None
    api_key: str | None = None
    embed_backend: str = "tfidf"  # tfidf | fastembed | openai
    embed_model: str = "BAAI/bge-small-en-v1.5"
    dense_model_for_openai: str = "text-embedding-3-small"
    hybrid_weight: float = 0.5  # 0 = only lexical, 1 = only dense
    temperature: float = 0.2
    max_tokens: int = 700
    show_context: bool = False
    # Ollama "thinking" models can spend the whole token budget on internal
    # reasoning and return an empty answer, so it is off by default.
    think: bool = False
    request_timeout: int = 300

    # ---- derived paths -------------------------------------------------
    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def pages_path(self) -> Path:
        return self.index_dir / "pages.json"

    @property
    def lexical_path(self) -> Path:
        return self.index_dir / "lexical.joblib"

    def dense_path(self) -> Path:
        safe = f"{self.embed_backend}-{self.embed_model}".replace("/", "_")
        return self.index_dir / f"dense-{safe}.npy"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.index_dir, self.cache_dir, EVAL_DIR):
            path.mkdir(parents=True, exist_ok=True)

    # ---- loading -------------------------------------------------------
    @classmethod
    def load(cls) -> "Settings":
        raw: dict = {}
        if CONFIG_PATH.exists():
            try:
                raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Could not read {CONFIG_PATH}: {exc}") from exc

        def pick(name: str, env_names: tuple[str, ...]):
            for env_name in env_names:
                value = os.environ.get(env_name)
                if value:
                    return value
            return raw.get(name)

        backend = str(pick("llm_backend", ("RAG_LLM_BACKEND",)) or "ollama")
        preset = PROVIDER_PRESETS.get(backend, PROVIDER_PRESETS["custom"])
        model = pick("llm_model", ("RAG_LLM_MODEL",)) or preset.get("model") or "gpt-4o-mini"
        base_url = pick("base_url", ("RAG_BASE_URL",)) or preset.get("base_url")
        key_names = _KEY_ENV.get(backend, ("RAG_API_KEY",))
        api_key = pick("api_key", key_names)

        defaults = dict(
            pdf_path=os.environ.get("RAG_PDF_PATH") or raw.get("pdf_path") or DEFAULT_PDF,
            llm_backend=backend,
            llm_model=str(model),
            base_url=base_url,
            api_key=api_key,
        )
        for name in (
            "chunk_size",
            "overlap",
            "top_k",
            "embed_backend",
            "embed_model",
            "hybrid_weight",
            "temperature",
            "max_tokens",
            "think",
            "request_timeout",
        ):
            env_value = os.environ.get("RAG_" + name.upper())
            if env_value is not None:
                defaults[name] = env_value
            elif name in raw:
                defaults[name] = raw[name]

        settings = cls()
        for name, value in defaults.items():
            field_type = type(getattr(settings, name))
            if field_type is Path:
                value = Path(str(value)).expanduser()
            elif field_type is int:
                value = int(value)
            elif field_type is float:
                value = float(value)
            elif field_type is bool:
                value = _as_bool(value)
            elif field_type is str or value is None:
                value = None if value is None else str(value)
            setattr(settings, name, value)

        settings.ensure_dirs()
        return settings

    def overrides(self, **changes) -> "Settings":
        """Return a copy with explicit CLI overrides applied."""
        clean = {key: value for key, value in changes.items() if value is not None}
        updated = replace(self, **clean)
        updated.ensure_dirs()
        return updated

    def resolve_llm(self) -> dict[str, str | bool | None]:
        preset = PROVIDER_PRESETS.get(self.llm_backend, PROVIDER_PRESETS["custom"])
        return {
            "name": self.llm_backend,
            "api_style": str(preset["api_style"]),
            "base_url": self.base_url or preset.get("base_url"),
            "model": self.llm_model,
            "api_key": self.api_key,
            "think": self.think,
        }

    def public_dict(self) -> dict:
        data = asdict(self)
        data.pop("api_key", None)
        # convert Paths for display
        return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


def save_public_config(settings: Settings, path: Path = CONFIG_PATH) -> Path:
    """Save non-secret defaults only."""
    data = settings.public_dict()
    for key in ("api_key", "base_url"):
        data.pop(key, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
