"""
Configuration module.

Loads settings from environment variables / .env file using pydantic-settings.
Keep this simple - a single Settings object used across the app.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "Linux Copilot XAI"
    app_env: str = "development"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "sqlite:///./linux_copilot.db"

    # CORS - comma separated string, parsed into a list.
    # Includes every port this project's own docs tell someone to serve the
    # frontend on: python3 -m http.server default doc'd port (5500), the
    # common Vite/CRA dev ports (5173/3000), and 8000/8080 in case the
    # frontend is ever served from the same box as the API. Widen further
    # (or set to "*") via the CORS_ORIGINS env var if your demo setup uses
    # a different host/port.
    cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5500,http://127.0.0.1:5500,"
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:8080,http://127.0.0.1:8080"
    )

    # Logging
    log_level: str = "INFO"

    # Ollama (local LLM) - used by the AI Ops Assistant
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: int = 140  # read timeout; was 60 - CPU demo target is a 5-10s reply
    ollama_connect_timeout_seconds: int = 5  # fail fast if Ollama isn't up at all
    ollama_temperature: float = 0.2

    # CPU-demo speed tuning (see ollama_client.chat):
    ollama_think: bool = False  # disable "thinking" traces on reasoning-capable models (e.g. Qwen3)
    ollama_num_predict: int = 400  # cap generated tokens - long tail of a rambling reply is the #1 latency cost
    ollama_num_ctx: int = 4096  # context window in tokens (prompt + generation share this budget).
    # Was 2048. Root-cause fix: DRIVER_HEALTH context can include up to 50
    # loaded kernel modules (driver_monitor.get_loaded_kernel_modules) plus
    # the system prompt plus a full ~400-token generation - at 2048 total,
    # the prompt alone could consume nearly the whole window, leaving too
    # little room for the model to write a real explanation. That's what
    # produced "just recommends lsmod" / empty-explanation replies even
    # after the num_predict fix: the model ran out of context budget, not
    # generation budget. 4096 gives comfortable headroom for the largest
    # real context (driver inventory) plus a full reply.
    ollama_keep_alive: str = "30m"  # keep the model loaded between requests instead of reloading each call

    # Local Ollama has no request concurrency here (one model, CPU-only) -
    # every chat() call, interactive or background, ultimately queues on
    # the same backend. The AI Ops Assistant's live chat/ops queries use
    # `ollama_timeout_seconds` (140s) above since a real user is waiting
    # on that specific answer. Fix-engine's background issue diagnosis
    # (app/services/fix_engine.py: _ai_diagnose, polled by the frontend
    # every ~30s) is different: it already degrades gracefully to a
    # deterministic fallback diagnosis on any failure (see
    # _fallback_diagnosis), so there is no reason for one slow/stuck
    # background diagnosis call to be able to hold the single shared
    # Ollama connection for anywhere near 140s and starve a concurrent,
    # actually-waited-on chat/ops request behind it - which is exactly
    # what produced the observed "Ollama did not respond within 140s" on
    # a live chat question competing with a burst of new-issue background
    # diagnoses right after backend startup. 20s comfortably covers the
    # ~19s a num_predict=128-capped reply takes on this CPU-only setup
    # (see ollama_client._DEMO_MAX_NUM_PREDICT), with a few seconds of
    # margin, while bounding worst-case background Ollama occupancy per
    # diagnosis to a small fraction of the interactive timeout.
    ollama_background_timeout_seconds: int = 20

    # AI Assistant behavior
    assistant_history_turns: int = 2  # number of past turns fed back into the prompt
    assistant_max_history_stored: int = 50  # per session, in the DB

    # Safe Command Execution behavior
    execution_timeout_seconds: int = 30  # max seconds a confirmed command may run before being killed
    # systemctl/service start|restart|reload|try-restart commands legitimately take
    # longer than a normal command: e.g. NetworkManager-wait-online.service is a
    # oneshot unit that blocks on `nm-online` for up to ~60s, and `systemctl
    # restart` waits for that ExecStart to finish before returning. These get a
    # longer budget instead of being killed as if they'd hung.
    service_restart_timeout_seconds: int = 90
    execution_max_output_chars: int = 20000  # stdout/stderr truncated beyond this, per stream
    execution_max_history_stored: int = 100  # command executions kept per session, in the DB

    # --- Shared system thresholds ---
    # Used by the AI One-Click Fix Engine (app/services/fix_engine.py) for
    # its High CPU / High Memory / Disk Almost Full detectors.
    alert_cpu_threshold_percent: float = 90.0
    alert_memory_threshold_percent: float = 90.0
    alert_disk_threshold_percent: float = 90.0

    # --- Sprint 8: AI One-Click Fix Engine ---
    fix_apache_service_name: str = "apache2"  # systemd unit basename checked by the "Apache Down" detector

    # --- Autonomous Fault Detection & Self-Healing additions ---
    # Failure prediction (app/services/failure_predictor.py): trend-based
    # prediction over a rolling window of CPU/memory/disk samples.
    predictor_window_size: int = 20  # samples kept per metric for the trend fit
    predictor_min_samples: int = 5  # samples needed before a prediction is attempted
    predictor_lookahead_minutes: float = 30.0  # only alert if a breach is projected within this window

    # Hardware health monitoring (app/services/hardware_monitor.py):
    # temperature thresholds and smartctl timeout.
    hardware_temp_warning_celsius: float = 80.0
    hardware_temp_critical_celsius: float = 95.0
    hardware_smartctl_timeout_seconds: int = 10

    # Driver anomaly monitoring (app/services/driver_monitor.py): how much
    # of the kernel ring buffer to scan per pass, and how many example
    # log lines to keep per detected anomaly category.
    driver_monitor_log_lines: int = 500
    driver_monitor_max_examples: int = 5

    # Server-side background detection scan (see fix_engine.
    # run_background_detection_loop, wired up in main.py). Root-cause fix:
    # previously `fix_engine.detect_all_issues()` only ever ran when the
    # frontend dashboard polled GET /api/fixes/detect (every ~30s while a
    # browser tab is open). The AI Assistant's driver/hardware/failure
    # chat answers are grounded in `issue_alert_store`, which is only as
    # fresh as the last scan - so with the dashboard closed (chat-only /
    # API-only usage), those questions were silently answered from a
    # stale or entirely empty store. Running the exact same scan on a
    # fixed server-side schedule, independent of any frontend, keeps the
    # store fresh no matter how the assistant is used. Same cadence as
    # the frontend's own polling by default.
    background_detection_interval_seconds: float = 30.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a clean list of strings."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def sqlite_db_path(self) -> str:
        """Extract the filesystem path from the sqlite DATABASE_URL."""
        return self.database_url.replace("sqlite:///", "", 1)


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance so we don't re-read the .env file every call."""
    return Settings()