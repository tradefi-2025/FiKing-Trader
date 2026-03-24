"""Shared environment loader for repo-root .env."""

from pathlib import Path
from dotenv import load_dotenv


def load_env() -> None:
    repo_root = None
    for parent in Path(__file__).resolve().parents:
        if (parent / "src").is_dir():
            repo_root = parent
            break

    if repo_root:
        env_path = repo_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return

    load_dotenv()
