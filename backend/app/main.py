"""
Linux Copilot XAI - FastAPI application entrypoint.

A clean, simple Proof of Concept backend for the Linux Copilot XAI
hackathon project: a base API skeleton, an Ubuntu System Monitor, an AI
Linux Operations Assistant (natural language -> explanation + recommended
commands), and Safe Command Execution (natural language or a typed
command -> AI preview/explanation -> deterministic safety analysis ->
explicit user confirmation -> execution -> logged output). Both are
powered by a local Ollama LLM where an LLM is involved. Nothing in this
project executes anything without an explicit, separate user confirmation
step, and a fixed set of genuinely dangerous commands (rm -rf /, mkfs, dd
to a whole disk, shutdown/reboot, disk formatting, fork bombs, ...) can
never be executed at all.
"""

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.logger import get_logger, setup_logging
from app.routes import assistant, commands, fixes, health, system
from app.services import fix_engine

# Sprint 8 adds the AI One-Click Fix Engine (app/routes/fixes.py,
# app/services/fix_engine.py): detects High CPU, High Memory, Disk Almost
# Full, Apache Down, Docker Container Stopped, and Failed Service, then
# reuses the existing Safe Command Execution pipeline unchanged to preview
# and (only on explicit confirmation) run a fix.

# --- Logging must be configured before anything else logs ---
setup_logging()
logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="AI-powered Linux Operations Assistant for Ubuntu (PoC).",
    version="0.1.0",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---
app.include_router(health.router)
app.include_router(system.router)
app.include_router(assistant.router)
app.include_router(commands.router)
app.include_router(fixes.router)


@app.get("/", tags=["Root"])
def read_root():
    """Simple welcome route confirming the API is reachable."""
    return {
        "message": f"{settings.app_name} API is running",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.on_event("startup")
def on_startup():
    """Run once when the application starts."""
    logger.info("Starting %s (env=%s, debug=%s)", settings.app_name, settings.app_env, settings.debug)
    init_db()

    # Root-cause fix (see fix_engine.run_background_detection_loop): keep
    # the active-issue store fresh on a fixed server-side schedule instead
    # of relying entirely on the frontend dashboard polling GET
    # /api/fixes/detect. This guarantees driver/hardware/failure-prediction
    # chat questions are grounded in current data even when no dashboard
    # tab is open.
    app.state.background_detection_task = asyncio.create_task(
        fix_engine.run_background_detection_loop(settings.background_detection_interval_seconds)
    )


@app.on_event("shutdown")
async def on_shutdown():
    """Run once when the application shuts down."""
    task = getattr(app.state, "background_detection_task", None)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    logger.info("Shutting down %s", settings.app_name)
