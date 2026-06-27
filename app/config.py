from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    data_dir: Path = Path("data")
    docs_dir: Path = Path("data/docs")
    chroma_dir: Path = Path("data/chroma")
    collection_name: str = "documents"

    # SQLAlchemy URL. SQLite by default; swap to postgresql+psycopg://... later.
    database_url: str = "sqlite:///./data/app.db"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = 800
    chunk_overlap: int = 120

    default_top_k: int = 5

    # Confidence calibration (tuned for sentence-transformers/all-MiniLM-L6-v2).
    # Raw cosine at/above `full_score` maps to 100% confidence; the two thresholds
    # split the resulting 0..1 confidence into high / medium / low labels.
    confidence_full_score: float = 0.60
    confidence_high_threshold: float = 0.70
    confidence_medium_threshold: float = 0.40

    # Google Drive sync (public folders; server-side API key only).
    google_api_key: str = ""
    drive_request_timeout_seconds: int = 60

    templates_dir: Path = Path("app/templates")
    static_dir: Path = Path("app/static")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
