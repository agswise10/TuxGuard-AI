"""
Ollama client.

Thin wrapper around the local Ollama HTTP API (https://ollama.com), used to
run an open-weight model such as Qwen2.5 or Llama3.1 fully locally - no
data leaves the machine.

Kept deliberately dumb: this module knows nothing about intents, prompts,
or conversation history. It only knows how to send a list of chat messages
to Ollama and get a raw text reply back, raising a clear, catchable error
if Ollama isn't installed/running/reachable.
"""

import httpx

from app.config import get_settings
from app.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# NOTE: there used to be a `_DEMO_MAX_NUM_PREDICT = 128` hard floor applied
# here on top of `settings.ollama_num_predict` (400), intended as a latency
# fix. It was the root cause of the "response could not be parsed as
# structured JSON" / "no reasoning was provided" / "did not provide an
# explanation" failures on PERFORMANCE, DRIVER_HEALTH, and GENERAL queries:
# this app's own required JSON contract is
# {explanation, recommended_commands: [...], confidence_score, reasoning} -
# a real answer plus a commands array plus reasoning routinely runs
# 150-250+ tokens, so a 128-token hard cap cut generation off mid-object on
# any question that needed more than a one-line answer. Ollama stops with
# `done_reason == "length"` in that case and returns whatever partial text
# it had generated, which is exactly the "valid-looking prefix, then
# nothing" raw output that broke `llm_json.parse_json_object()` (or, when
# the cut happened to land on a valid closing brace, produced valid JSON
# missing whichever field(s) hadn't been generated yet).
#
# Short, single-schema replies (e.g. DISK_USAGE's tool-grounded `summary`)
# stayed under 128 tokens most of the time, which is why only some intents
# appeared broken.
#
# Fix: trust the configured `ollama_num_predict` (400) instead of
# re-capping it here. 400 is still a real ceiling against runaway replies -
# it just isn't smaller than the JSON contract the app itself requires.


class OllamaUnavailableError(Exception):
    """Raised when Ollama cannot be reached or returns an error."""


async def chat(
    messages: list[dict],
    model: str | None = None,
    timeout_seconds: int | None = None,
) -> str:
    """Send a chat-style request to Ollama and return the raw text reply.

    Args:
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        model: overrides the configured default model if provided.
        timeout_seconds: overrides the configured read timeout
            (`settings.ollama_timeout_seconds`) if provided. Used by
            background callers (fix_engine's issue diagnosis) that need a
            much shorter budget than an interactive chat request, since
            Ollama has no concurrency here and every call - interactive or
            background - queues on the same single backend; see
            `settings.ollama_background_timeout_seconds`.

    Raises:
        OllamaUnavailableError: if Ollama is unreachable, times out, or
            returns a non-2xx / malformed response.
    """
    read_timeout = timeout_seconds if timeout_seconds is not None else settings.ollama_timeout_seconds
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model or settings.ollama_model,
        "messages": messages,
        "stream": False,
        # "think" must be a top-level field (not under "options") - this is
        # what actually turns off the long <think>...</think> preamble that
        # reasoning-capable models (Qwen3, DeepSeek-R1, etc.) emit before
        # every reply. It's a no-op on non-reasoning models, so it's always
        # safe to send.
        "think": settings.ollama_think,
        # NOTE: deliberately NOT setting "format": "json" here.
        #
        # Ollama's grammar-constrained JSON mode (format="json") forces
        # token-by-token GBNF grammar masking, which is a well-documented
        # Ollama performance issue - often 10x+ slower than unconstrained
        # sampling, especially on CPU-only inference (see e.g.
        # ollama/ollama#3154, #3851, ollama-python#79). That constraint was
        # the actual root cause of AI Assistant queries timing out at the
        # configured `ollama_timeout_seconds` even though a plain manual
        # `ollama run` / `/api/generate` call against the same model
        # responded quickly.
        #
        # JSON output reliability is instead handled the way it always was
        # meant to be defended: SYSTEM_PROMPT / OPS_SYSTEM_PROMPT explicitly
        # instruct the model to reply with ONLY a JSON object (see
        # prompts.py / ops_prompts.py), and llm_json.parse_json_object() +
        # ai_assistant._parse_llm_response() / ops_assistant._parse_ops_llm_
        # response() already robustly handle a malformed/non-JSON reply by
        # degrading to a low-confidence explanation instead of raising - so
        # no downstream behavior changes if a reply is occasionally messy.
        #
        # Keep the model resident in memory between requests. Without this,
        # Ollama can unload the model after each call and pay the (multi-
        # second, CPU-bound) load cost again on the very next question.
        "keep_alive": settings.ollama_keep_alive,
        "options": {
            "temperature": settings.ollama_temperature,
            # Bounds how much the model can generate. Still the single
            # biggest lever for CPU latency, but the cap has to stay large
            # enough to fit this app's own required JSON contract
            # (explanation + a recommended_commands array + reasoning) -
            # see the note above `chat()` for why a too-small cap here
            # silently broke JSON parsing rather than just running long.
            "num_predict": settings.ollama_num_predict,
            # Smaller context window -> less prompt to process per token on
            # CPU. 2048 is comfortable for the small, intent-scoped prompts
            # this app sends (see prompts.py / context_builder.py).
            # NOT reduced further here - out of scope for this change.
            "num_ctx": settings.ollama_num_ctx,
        },
    }

    # --- TEMPORARY DIAGNOSTIC LOGGING (see below in this function for the
    # matching post-response log) ---
    call_kind = "background" if timeout_seconds is not None else "interactive"
    logger.info(
        "OLLAMA DIAG: sending call_kind=%s model=%s num_predict=%s num_ctx=%s think=%s "
        "read_timeout=%ss prompt_chars=%d",
        call_kind,
        payload["model"],
        payload["options"]["num_predict"],
        payload["options"]["num_ctx"],
        payload["think"],
        read_timeout,
        sum(len(m.get("content", "")) for m in messages),
    )
    # --- END TEMPORARY DIAGNOSTIC LOGGING ---

    timeout = httpx.Timeout(
        connect=settings.ollama_connect_timeout_seconds,
        read=read_timeout,
        write=settings.ollama_connect_timeout_seconds,
        pool=settings.ollama_connect_timeout_seconds,
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
    except httpx.ConnectError as exc:
        raise OllamaUnavailableError(
            f"Could not connect to Ollama at {settings.ollama_base_url}. "
            "Is Ollama installed and running? (`ollama serve`)"
        ) from exc
    except httpx.TimeoutException as exc:
        raise OllamaUnavailableError(
            f"Ollama did not respond within {read_timeout}s."
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaUnavailableError(f"Unexpected error contacting Ollama: {exc}") from exc

    if response.status_code == 404:
        raise OllamaUnavailableError(
            f"Model '{payload['model']}' not found in Ollama. "
            f"Pull it first with: ollama pull {payload['model']}"
        )
    if response.status_code != 200:
        raise OllamaUnavailableError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        data = response.json()
        content = data["message"]["content"]
    except (ValueError, KeyError, TypeError) as exc:
        raise OllamaUnavailableError(f"Malformed response from Ollama: {exc}") from exc

    # --- TEMPORARY DIAGNOSTIC LOGGING (see ollama_client.py TODO) ---
    # Added to trace the "valid JSON but missing expected keys" failure
    # mode reported on the live demo box. Safe to leave in (INFO level,
    # no secrets, content truncated) but should be removed once the root
    # cause is confirmed and fixed.
    # `done_reason` == "length" means Ollama stopped ONLY because it hit
    # num_predict, not because the model naturally finished - the single
    # most direct signal for "was this cut off by our token cap".
    logger.info(
        "OLLAMA DIAG: done_reason=%s eval_count=%s(generated) prompt_eval_count=%s(prompt) "
        "content_len=%d content_repr=%r",
        data.get("done_reason"),
        data.get("eval_count"),
        data.get("prompt_eval_count"),
        len(content),
        content[:1500],
    )
    # --- END TEMPORARY DIAGNOSTIC LOGGING ---

    if not content or not content.strip():
        raise OllamaUnavailableError("Ollama returned an empty response.")

    return content


async def check_health() -> dict:
    """Check whether Ollama is reachable and whether the configured model is pulled."""
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(url)
        response.raise_for_status()
        models = [m.get("name") for m in response.json().get("models", [])]
        model_available = any(
            m == settings.ollama_model or (m or "").startswith(settings.ollama_model.split(":")[0])
            for m in models
        )
        return {
            "reachable": True,
            "configured_model": settings.ollama_model,
            "model_available": model_available,
            "installed_models": models,
        }
    except httpx.HTTPError as exc:
        return {
            "reachable": False,
            "configured_model": settings.ollama_model,
            "model_available": False,
            "installed_models": [],
            "error": str(exc),
        }