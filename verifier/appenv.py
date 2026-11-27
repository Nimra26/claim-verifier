"""Import this before anything from `src/`.

The app's settings (src/settings/config.py) are built at import time and need the app's secret
env vars. The verifier never uses them, so give them harmless dummy values, and point the app's
env-file lookup (SETTINGS_ENV_FILE) at a file that does not exist so the app's .env is never read.
"""
import os
from pathlib import Path

NO_ENV_FILE = Path(__file__).resolve().parent / ".no-app-env-file"  # never created

os.environ["SETTINGS_ENV_FILE"] = str(NO_ENV_FILE)
for _key, _value in {
    "ACCESS_TOKEN_SECRET_KEY": "verifier-dummy",
    "REFRESH_TOKEN_SECRET_KEY": "verifier-dummy",
    "RESET_PASSWORD_TOKEN_SECRET": "verifier-dummy",
    "VERIFICATION_TOKEN_SECRET": "verifier-dummy",
    # Not the verifier database (that is VERIFIER_DATABASE_URL): a dummy nothing connects to.
    "DATABASE_URL": "postgresql+asyncpg://unused:unused@localhost:1/unused",
}.items():
    os.environ.setdefault(_key, _value)
