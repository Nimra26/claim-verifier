import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _clean_env():
    keep = {k: os.environ[k] for k in ("PATH", "HOME", "LANG") if k in os.environ}
    keep["PYTHONPATH"] = str(REPO)
    return keep


def test_cli_imports_with_clean_env():
    res = subprocess.run([sys.executable, "-c", "import verifier.__main__"], cwd=REPO, env=_clean_env(),
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]


def test_app_env_file_is_never_read():
    code = ("import os, verifier.__main__\n"
            "from src.shared import host_platform\n"
            "p = host_platform.settings_env_file()\n"
            "assert not p.exists(), p\n"
            "assert p.name != '.env' and p.name != 'some-other.env', p\n"
            "print(os.environ['DATABASE_URL'])\n")
    env = _clean_env()
    env["SETTINGS_ENV_FILE"] = str(REPO / "some-other.env")  # a preset value is overridden too
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]
    assert "localhost:1/" in res.stdout
