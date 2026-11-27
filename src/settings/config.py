from pathlib import Path

from pydantic_settings import SettingsConfigDict

from src.settings.ai import AISettings
from src.settings.agent import AgentSettings
from src.settings.evaluation import EvaluationSettings
from src.settings.mail import MailSettings
from src.settings.rag import RAGSettings
from src.settings.database import DatabaseSettings
from src.settings.auth import AuthSettings
from src.shared import host_platform

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(
    AISettings,
    AgentSettings,
    EvaluationSettings,
    MailSettings,
    RAGSettings,
    DatabaseSettings,
    AuthSettings,
):
    model_config = SettingsConfigDict(
        # SETTINGS_ENV_FILE lets tests point away from this checkout's real .env (2 Nov: a test passed in a worktree and
        # failed in the live checkout because it read the live APP_URL).
        # Windows: .env is in %LOCALAPPDATA%\\AskWorkspace (host_platform.env_file), kept across reinstalls.
        env_file=host_platform.settings_env_file(),  # SETTINGS_ENV_FILE, else host_platform.env_file()
        env_file_encoding="utf-8",
        extra="ignore",
        # An empty line in .env ("AGENT_MAX_TOKENS=") means "use the default". Without this, a fresh install made from
        # .env.example crash-looped on start (found by the installer test, 30 Oct).
        env_ignore_empty=True,
    )


# Native Windows: .env values the code reads with os.environ.get (HF_HUB_OFFLINE, WORKSPACE_*) reach the environment
# for every entry point that imports settings (alembic, the pack CLI). main.py already did this first; it never
# overrides, so calling it twice changes nothing. Linux: does nothing (compose's env_file: does it).
host_platform.load_env_into_environ()
settings = Settings() #type: ignore
