"""The operating system the app runs on, and every machine fact that differs between Linux and Windows (2 Nov).

Why: clients will run the app natively on their own Windows laptop (Setup.exe, no Docker, no WSL; decision 2 Nov
22:32, docs/decisions/2026-11-02-client-distribution-options.md). The app used to read Linux files directly
(/proc/meminfo, /proc/stat, /proc/pressure/memory, /sys/class/power_supply, /proc/net/route, /etc/resolv.conf). On
Windows those paths do not exist, and worse, "/proc/..." there means C:\\proc\\..., a folder any local user may create,
so a planted file could fake the numbers. Everything OS-specific now lives here, and Windows never opens a Linux path.

What each OS gives (None = not measured; the pages say so rather than show a made-up 0):

  fact                 Linux                         Windows
  RAM total/available  /proc/meminfo                 GlobalMemoryStatusEx (kernel32)
  swap total/free      /proc/meminfo                 None: Windows has a page file, not swap. The commit charge
                                                     (RAM + page file promised to programs) is given as
                                                     commit_total / commit_available instead.
  memory pressure      /proc/pressure/memory (PSI)   None: Windows has no equivalent counter
  CPU working %        /proc/stat                    GetSystemTimes (kernel32)
  waiting for disk %   /proc/stat iowait             None: Windows does not count it
  swap pages per s     /proc/vmstat                  None (a performance counter would be needed; not used)
  load average         /proc/loadavg                 None: Windows has no load average
  plugged in, battery  /sys/class/power_supply       GetSystemPowerStatus (kernel32)
  power mode           platform_profile / EPP        "power-saver" when Battery saver is on, else "unknown"
  file lock            fcntl.flock                   msvcrt.locking
  links                is_symlink                    is_symlink or is_junction (a junction is a Windows folder link)

The Windows calls are documented Win32 functions reached through ctypes (standard library): psutil is not a
dependency (checked 2 Nov: not in the app image), so nothing new is installed.

WORKSPACE_PLATFORM=windows|linux overrides what is detected, but only off real Windows (on Windows it is ignored). It
is for tests: it changes which paths, messages and sources the app picks, so the Windows branches can be exercised on
Linux with the Win32 calls replaced by fakes.
It never makes Linux call kernel32: with no kernel32 the Windows sources answer None (unmeasured), never a guess.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import sys
import time
from pathlib import Path

try:  # Linux / macOS
    import fcntl
except ImportError:  # Windows
    fcntl = None
try:  # Windows
    import msvcrt
except ImportError:  # Linux / macOS
    msvcrt = None

logger = logging.getLogger(__name__)

PLATFORM_ENV = "WORKSPACE_PLATFORM"
DATA_ENV = "WORKSPACE_DATA_DIR"
APP_FOLDER = "AskWorkspace"  # %LOCALAPPDATA%\AskWorkspace: the data folder on Windows (as Setup, the launcher and
# the uninstaller name it: windows/AskTheWorkspace.iss DataDirName; one name everywhere, review of 3 Nov)
ROOT = Path(__file__).resolve().parents[2]  # the app folder (main.py lives here)
NOT_ON_WINDOWS = "not available on Windows"
# subprocess.CREATE_NO_WINDOW (process creation flag, documented value). Spelled out so this module never imports
# subprocess: tests/unit/test_no_outbound_code.py counts that as a way to run other programs.
CREATE_NO_WINDOW = 0x08000000

_ALIASES = {"windows": "windows", "win32": "windows", "win": "windows", "nt": "windows",
            "linux": "linux", "mac": "mac", "macos": "mac", "darwin": "mac"}
_warned: set[str] = set()


# --- which OS -------------------------------------------------------------------------------------------------------

def name() -> str:
    """windows | linux | mac | other, read on every call. Real Windows is always "windows": WORKSPACE_PLATFORM is
    obeyed only off Windows, where the tests need it. On a Windows laptop WORKSPACE_PLATFORM=linux would otherwise
    bring back the /etc/resolv.conf and /proc/net/route reads, i.e. C:\\etc and C:\\proc, folders any local user may
    create, and a planted resolv.conf could make the Docker names "db" and "ollama" count as this machine (review
    must-fix, 2 Nov)."""
    if sys.platform == "win32":
        return "windows"
    raw = os.environ.get(PLATFORM_ENV, "").strip().lower()
    if raw in _ALIASES:
        return _ALIASES[raw]
    if raw and "platform" not in _warned:
        _warned.add("platform")
        logger.warning("%s=%r is not windows/linux/mac, so the detected platform is used.", PLATFORM_ENV, raw[:20])
    if sys.platform.startswith("linux"):
        return "linux"
    return "mac" if sys.platform == "darwin" else "other"


def is_windows() -> bool:
    return name() == "windows"


def is_linux() -> bool:
    return name() == "linux"


# --- where the app keeps its data -----------------------------------------------------------------------------------

def data_dir() -> Path:
    """The folder the app writes to. WORKSPACE_DATA_DIR when set (relative = under the app folder); on Windows
    %LOCALAPPDATA%\\AskWorkspace (per user, no admin rights; windows/AskTheWorkspace.iss's uninstaller keeps it
    unless the person says yes to deleting it, default No); elsewhere <app>/data, as
    before. On Windows the install folder holds only the program, so a reinstall or an uninstall never touches data."""
    raw = os.environ.get(DATA_ENV, "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else ROOT / p
    if is_windows():
        base = os.environ.get("LOCALAPPDATA", "").strip()
        return (Path(base) if base else Path.home() / "AppData" / "Local") / APP_FOLDER
    return ROOT / "data"


def data_path(name_in_data: str, linux_default: str | os.PathLike) -> Path:
    """A file or folder the app keeps: under data_dir() on Windows or when WORKSPACE_DATA_DIR is set, otherwise the
    Linux default unchanged (so existing installs and their backups keep working)."""
    if is_windows() or os.environ.get(DATA_ENV, "").strip():
        return data_dir() / name_in_data
    return Path(linux_default)


def env_file() -> Path:
    """Where .env is read from when SETTINGS_ENV_FILE is not set: the data folder on Windows (it holds the token keys
    made at install and must survive a reinstall), the app folder elsewhere."""
    return data_dir() / ".env" if is_windows() else ROOT / ".env"


def settings_env_file() -> Path:
    """The .env the app actually reads: SETTINGS_ENV_FILE when set, else env_file(). config.py and
    load_env_into_environ() both use this, so they never read two different files."""
    raw = os.environ.get("SETTINGS_ENV_FILE", "").strip()
    return Path(raw) if raw else env_file()


def load_env_into_environ(environ=None) -> list[str]:
    """Windows only: copy the .env values into the process environment, never overriding a value already set there.
    Returns the names it set (never the values: .env holds the database password and token keys).

    Why (review must-fix, 2 Nov): only pydantic Settings reads .env. Many settings are read with os.environ.get
    instead: HF_HUB_OFFLINE (huggingface_hub, at import, and egress_policy), WORKSPACE_STRICT_MODE,
    WORKSPACE_NETWORK_LOCK, WORKSPACE_DATA_DIR / PACKS_DIR / ACCESS_FILE / FILES_DIR, GIT_SHA. On Linux docker compose
    (`env_file:`) puts .env into the environment; native Windows has no compose, so a value written only to .env was
    silently ignored, and with HF_HUB_OFFLINE ignored the local reranker could download from huggingface.co. main.py
    calls this before anything else is imported. Linux and macOS: does nothing (compose already did it).
    WORKSPACE_DATA_DIR cannot move the .env it is read from: set that one in the launcher's environment."""
    if not is_windows():
        return []
    environ = os.environ if environ is None else environ
    path = settings_env_file()
    if not path.is_file():
        return []
    try:
        from dotenv import dotenv_values  # python-dotenv: already installed, pydantic-settings reads .env with it
        values = dotenv_values(path, encoding="utf-8")
    except (OSError, UnicodeDecodeError, ImportError) as exc:
        logger.warning("Could not read %s into the environment (%s).", path, type(exc).__name__)
        return []
    added = []
    for key, value in values.items():
        if value is None or value.strip() == "" or key in environ:
            continue  # empty = "use the default", like env_ignore_empty in config.py; the environment always wins
        environ[key] = value
        added.append(key)
    return added


# --- files ----------------------------------------------------------------------------------------------------------

def is_link(path: Path) -> bool:
    """A symlink, or on Windows a junction (a folder link any user can make without admin rights, which
    Path.is_symlink() does not report). Code that refuses links calls this, so a junction is refused too."""
    try:
        return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())
    except OSError:
        return False


def remove_link(path: Path) -> None:
    """Remove a link itself, never what it points at: a junction with rmdir (unlink refuses a folder on Windows),
    a symlink with unlink."""
    if getattr(path, "is_junction", lambda: False)():
        os.rmdir(path)
    else:
        path.unlink()


def replace(src, dst, *, tries: int = 5, wait_s: float = 0.1) -> None:
    """os.replace, retried briefly on Windows: there a rename onto a file fails with PermissionError while another
    program holds it open (Defender scanning it, a reader); that clears in well under a second. Linux: one try."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if not is_windows() or i == tries - 1:
                raise
            time.sleep(wait_s * (i + 1))


@contextlib.contextmanager
def exclusive_lock(fh):
    """Hold an exclusive lock on an open file while the block runs. fcntl.flock on Linux (waits); msvcrt.locking on
    Windows (one byte at offset 0; Windows retries for about 10 s, then we go on without the lock and log it). Best
    effort, like the caller asks: the writes it guards are atomic replaces anyway, so no lock never means a torn file."""
    if fcntl is not None:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield  # closing the file releases it
        return
    if msvcrt is not None:
        fd = fh.fileno()
        try:
            fh.seek(0)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        except OSError as exc:
            logger.warning("Could not lock %s within 10 s (%s); going on without the lock.", getattr(fh, "name", "?"),
                           exc.strerror or exc)
            yield
            return
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fh.seek(0)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    yield


def no_console_window() -> dict:
    """subprocess keyword arguments that keep a child from flashing a console window when the app runs without one
    (pythonw.exe from the desktop icon). Empty off real Windows: POSIX refuses creationflags."""
    if sys.platform == "win32":
        return {"creationflags": CREATE_NO_WINDOW}
    return {}


# --- Linux sources --------------------------------------------------------------------------------------------------

def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return None


def _linux_meminfo(path: str = "/proc/meminfo") -> dict:
    out = {}
    for line in (_read(path) or "").splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"} and rest.split():
            with contextlib.suppress(ValueError):
                out[key] = int(rest.split()[0]) * 1024
    return out


def linux_pressure(path: str = "/proc/pressure/memory") -> float | None:
    """'full avg10': share of the last 10 s in which every task was stalled waiting for memory."""
    try:
        for line in (_read(path) or "").splitlines():
            if line.startswith("full"):
                return float(dict(kv.split("=") for kv in line.split()[1:])["avg10"])
    except (KeyError, ValueError):
        pass
    return None


# --- Windows sources (documented Win32 calls through ctypes) --------------------------------------------------------

class MEMORYSTATUSEX(ctypes.Structure):  # https://learn.microsoft.com/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64)]


class FILETIME(ctypes.Structure):  # 100-nanosecond ticks, split in two 32-bit halves
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]

    def ticks(self) -> int:
        return (self.dwHighDateTime << 32) | self.dwLowDateTime


class SYSTEM_POWER_STATUS(ctypes.Structure):  # https://learn.microsoft.com/windows/win32/api/winbase/ns-winbase-system_power_status
    _fields_ = [("ACLineStatus", ctypes.c_uint8), ("BatteryFlag", ctypes.c_uint8),
                ("BatteryLifePercent", ctypes.c_uint8), ("SystemStatusFlag", ctypes.c_uint8),
                ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]


def _kernel32():
    """kernel32.dll, or None off Windows. Tests replace this with a fake that fills the structures."""
    windll = getattr(ctypes, "WinDLL", None)
    if windll is None:
        return None
    try:
        return windll("kernel32", use_last_error=True)
    except OSError:
        return None


def _win_memory() -> dict:
    k = _kernel32()
    if k is None:
        return {}
    s = MEMORYSTATUSEX()
    s.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not k.GlobalMemoryStatusEx(ctypes.byref(s)):
        return {}
    return {"MemTotal": s.ullTotalPhys, "MemAvailable": s.ullAvailPhys,
            "CommitTotal": s.ullTotalPageFile, "CommitAvailable": s.ullAvailPageFile}


def _win_cpu() -> tuple[int, int, int | None] | None:
    k = _kernel32()
    if k is None:
        return None
    idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
    if not k.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    return kernel.ticks() + user.ticks(), idle.ticks(), None  # kernel time already includes idle time


def _win_power() -> dict:
    k = _kernel32()
    if k is None:
        return {"plugged_in": None, "battery_percent": None, "power_mode": "unknown"}
    s = SYSTEM_POWER_STATUS()
    if not k.GetSystemPowerStatus(ctypes.byref(s)):
        return {"plugged_in": None, "battery_percent": None, "power_mode": "unknown"}
    plugged = {0: False, 1: True}.get(s.ACLineStatus)  # 255 = unknown
    no_battery = s.BatteryFlag == 128 or s.BatteryFlag == 255
    battery = None if no_battery or s.BatteryLifePercent > 100 else int(s.BatteryLifePercent)
    # SystemStatusFlag 1 = Battery saver is on (Windows 10+). Balanced vs best performance is not in this call.
    mode = "power-saver" if s.SystemStatusFlag == 1 else "unknown"
    return {"plugged_in": plugged, "battery_percent": battery, "power_mode": mode}


# --- what the Monitor and the health bar read -----------------------------------------------------------------------

def memory() -> dict:
    """mem_total, mem_available, swap_total, swap_free in bytes (None = not measured here); on Windows also
    commit_total / commit_available (the commit limit: RAM plus page file)."""
    if is_windows():
        m = _win_memory()
        return {"mem_total": m.get("MemTotal"), "mem_available": m.get("MemAvailable"), "swap_total": None,
                "swap_free": None, "commit_total": m.get("CommitTotal"), "commit_available": m.get("CommitAvailable")}
    m = _linux_meminfo()
    return {"mem_total": m.get("MemTotal"), "mem_available": m.get("MemAvailable"),
            "swap_total": m.get("SwapTotal"), "swap_free": m.get("SwapFree")}


def memory_pressure() -> float | None:
    """Linux PSI 'full avg10'. None on Windows (no such counter) and wherever PSI is off."""
    return None if is_windows() else linux_pressure()


def cpu_times() -> tuple[int, int, int | None] | None:
    """Cumulative (total, idle, waiting-for-disk) ticks; the Monitor turns two samples into percentages. Waiting is
    None on Windows. Linux: the first 8 fields of the /proc/stat cpu line (guest time is already in user time)."""
    if is_windows():
        return _win_cpu()
    line = (_read("/proc/stat") or "").splitlines()[:1]
    try:
        f = [int(x) for x in line[0].split()[1:9]] if line else []
    except ValueError:
        return None
    if len(f) < 5:
        return None
    return sum(f), f[3], f[4]


def swap_pages() -> int | None:
    """Pages swapped in + out since boot (Linux /proc/vmstat). None on Windows."""
    if is_windows():
        return None
    raw = _read("/proc/vmstat")
    if raw is None:
        return None
    vm = dict(l.split()[:2] for l in raw.splitlines() if l.startswith(("pswpin ", "pswpout ")))
    try:
        return int(vm.get("pswpin", 0)) + int(vm.get("pswpout", 0))
    except ValueError:
        return None


def load_average() -> list[float] | None:
    """1, 5 and 15 minute load (Linux /proc/loadavg). None on Windows, which has no load average: never 0 0 0."""
    if is_windows():
        return None
    raw = (_read("/proc/loadavg") or "").split()[:3]
    try:
        return [float(x) for x in raw] if len(raw) == 3 else None
    except ValueError:
        return None


def power(linux_power_mode) -> dict:
    """plugged_in, battery_percent, power_mode. `linux_power_mode` is monitor.power_mode (kept there: its tests)."""
    if is_windows():
        return _win_power()
    plugged, battery = None, None
    try:
        for d in Path("/sys/class/power_supply").iterdir():
            kind = _read(str(d / "type"))
            if kind == "Mains":
                plugged = (_read(str(d / "online")) or "0") == "1"
            elif kind == "Battery":
                cap = _read(str(d / "capacity"))
                battery = int(cap) if cap and cap.isdigit() else None
    except OSError:
        pass
    cpufreq = "/sys/devices/system/cpu/cpu0/cpufreq/"
    mode = linux_power_mode(_read("/sys/firmware/acpi/platform_profile"),
                            _read(cpufreq + "energy_performance_preference"),
                            _read(cpufreq + "scaling_governor"), _read(cpufreq + "scaling_driver"))
    return {"plugged_in": plugged, "battery_percent": battery, "power_mode": mode}


def unmeasured() -> list[str]:
    """Machine figures this OS does not give the app, as the Monitor names them."""
    if is_windows():
        return ["swap", "memory pressure", "waiting for disk", "pages swapped", "load average", "power mode"]
    return []


# --- what the pages tell people to do -------------------------------------------------------------------------------

def answers_down_text(dev: bool = False) -> str:
    """What is down, in plain words (19 Nov, owner: no internals for a client); the engine's name only on our own
    machines (dev = roles.dev_pages_on())."""
    return "The part of the app that writes answers isn't reachable" + (" (Ollama)." if dev else ".")


def ollama_start() -> dict:
    """How to start the model server here: {say, command} (command None when there is nothing to type)."""
    if is_windows():
        return {"say": "Close Ask the Workspace and open it again from its desktop icon; that starts it again",
                "command": None}
    return {"say": "Start it with", "command": "sudo systemctl start ollama"}


def ollama_start_hint() -> str:
    s = ollama_start()
    return f"{s['say']}: {s['command']}" if s["command"] else f"{s['say']}."


def setup_code_where(linux_text: str) -> str:
    """Where the first-run setup code is, in the words the setup page uses: it follows "It is " and also stands alone
    as a hint. Windows (messages audit 10 Nov): how to open the data folder without knowing what an environment
    variable is, then the easier way (the launcher opens this page with a one-time link that needs no code:
    askws_launcher setup_link). A data folder moved elsewhere (WORKSPACE_DATA_DIR / WORKSPACE_SETUP_CODE_FILE, 19 Nov)
    is named by its real path; the same file in another spelling (Windows ignores case) keeps the easy words."""
    if is_windows():
        easier = ("Easier: close Ask the Workspace's window and open it again from its desktop icon, and it opens this "
                  "page without the code")
        raw = os.environ.get("WORKSPACE_SETUP_CODE_FILE", "").strip()
        actual = Path(raw) if raw else data_dir() / "setup-code"
        local = os.environ.get("LOCALAPPDATA", "").strip()
        if local and str(actual).lower() != str(Path(local) / APP_FOLDER / "setup-code").lower():
            return f"in the file {actual} (open that folder in File Explorer). {easier}"
        return ("in the file setup-code in Ask the Workspace's data folder: press Windows+R, type "
                f"%LOCALAPPDATA%\\AskWorkspace and press Enter. {easier}")
    return linux_text


# Messages batch 3 (docs/research/messages-audit-2026-11-10.md): every text that tells a person to do something on the
# machine picks its words HERE, by OS, so a Windows laptop never shows a Docker or Linux command and Linux keeps its
# own.

# The Windows kit's own locked-out-admin step: the launcher runs scripts/reset_admin.py (shipped in the kit) with the
# app's settings, while Ask the Workspace runs (windows/launcher/askws_launcher.py --reset-admin).
WINDOWS_RESET_COMMAND = ('"%LOCALAPPDATA%\\Programs\\AskWorkspace\\python\\python.exe" -I -B -X utf8 -m askws_launcher '
                         "--reset-admin <admin email>")
LINUX_RESET_COMMAND = "docker exec agentic_rag_app /app/.venv/bin/python scripts/reset_admin.py --email <admin email>"


def admin_reset_how() -> str:
    """How IT makes a set-password link for a locked-out admin on this machine (a full sentence, ends with a dot)."""
    if is_windows():
        return ("while Ask the Workspace is running and the person who installed it is signed in to Windows, IT "
                "opens Command Prompt and runs " + WINDOWS_RESET_COMMAND + " (it shows a one-time set-password link).")
    return "IT runs " + LINUX_RESET_COMMAND + ' on this computer (docs/INSTALL.md, "Locked out").'


def model_missing_hint(model: str) -> str:
    """What to do when the default model is not on this machine."""
    if is_windows():
        return ("Close Ask the Workspace's window and open it again from its desktop icon: it checks the AI models "
                "when it starts and says what to do.")
    return f"Run: ollama pull {model}"


def describe() -> dict:
    """One line for the start-up log: which OS, where data goes, what is not measured, UTF-8 mode."""
    return {"platform": name(), "data_dir": str(data_dir()), "unmeasured": unmeasured(),
            "utf8_mode": bool(sys.flags.utf8_mode)}


def check_at_start() -> dict:
    """Logged once by main.py. On Windows, warns when Python is not in UTF-8 mode: our own files are always opened as
    UTF-8, but a library that opens a file without an encoding would read cp1252 there and garble Urdu."""
    d = describe()
    logger.info("Platform: %s; data folder %s; not measured here: %s.", d["platform"], d["data_dir"],
                ", ".join(d["unmeasured"]) or "nothing")
    if is_windows() and not d["utf8_mode"]:
        logger.warning("Python is not in UTF-8 mode. Start it with PYTHONUTF8=1 (the launcher does) so libraries "
                       "read Urdu file names and text correctly.")
    return d
