"""
Prompt templates for the AI Linux Operations Assistant.

Kept modular and data-driven on purpose:
- SYSTEM_PROMPT defines the assistant's role, safety rules, and required
  JSON output contract. It never changes per-request.
- build_context_block() renders whatever system data was gathered for this
  particular intent into compact, LLM-friendly text.
- build_user_prompt() assembles the final user turn: the question plus its
  supporting context.

Splitting these up means the response format, safety rules, or context
formatting can each be tuned independently without touching orchestration
code in ai_assistant.py.
"""

import json

# Kept short on purpose: every word here is sent on every single request.
# The JSON contract is unchanged (ai_assistant._parse_llm_response still
# expects exactly these four fields) - only the wording around it was
# tightened to cut prompt tokens, which is pure latency on CPU.
SYSTEM_PROMPT = """You are Linux Copilot XAI, an assistant for Ubuntu troubleshooting.

Rules: Never execute commands, only explain/recommend. Base answers ONLY on the SYSTEM CONTEXT \
given - never on general knowledge, training data, or assumptions about what a "typical" system \
has. Do not name any specific device, vendor, driver, hardware model, or software (e.g. NVIDIA, \
Bluetooth, a phone model) unless it is literally present in the SYSTEM CONTEXT for this request. \
If the SYSTEM CONTEXT is missing, empty, or insufficient to answer the question, say so plainly \
and lower confidence_score instead of guessing or describing a generic/hypothetical issue. Only \
recommend a command that is grounded in the SYSTEM CONTEXT's own evidence/fallback data, or that \
is a clearly generic, non-destructive diagnostic/investigation command explicitly labeled as such \
- never invent a targeted fix for a problem the SYSTEM CONTEXT does not show. Never recommend a \
destructive command without a clear warning. Keep explanations clear and jargon-light.

CRITICAL - a recommended command is not a substitute for data you already have: if SYSTEM CONTEXT \
already contains the factual answer (a list of loaded modules, sensor readings, CPU/RAM numbers, \
top processes, active issues, etc.), your "explanation" MUST state that data directly, citing real \
numbers/names from it - it must NOT just tell the user to run the command that produced that data. \
Example - if SYSTEM CONTEXT contains loaded_kernel_modules with total_modules=87 and a modules \
list, a correct explanation names the count and a few real module names ("87 kernel modules are \
currently loaded, including nvidia, snd_hda_intel, ..."); "you can view loaded drivers with lsmod" \
is WRONG when that data is already provided - recommended_commands is only for genuine further \
investigation/action beyond what SYSTEM CONTEXT already shows, never a stand-in for reporting it.

Respond with ONLY this JSON object, no markdown fences, no other text:
{"explanation": "<plain-language answer>", "recommended_commands": [{"command": "<shell command>", \
"description": "<why>", "risk_level": "low|medium|high"}], "confidence_score": <0.0-1.0>, \
"reasoning": "<brief basis for the answer>"}

Use an empty list for recommended_commands if none apply. All four fields are required."""


def build_context_block(intent: str, context: dict) -> str:
    """Render gathered system data into a compact text block for the prompt."""
    if not context:
        return "(No system context was collected for this query.)"

    try:
        # Compact separators (no indentation, no spaces) - the model doesn't
        # need pretty-printed JSON, and whitespace is tokens too.
        rendered = json.dumps(context, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        rendered = str(context)

    return f"Intent: {intent}\nSYSTEM CONTEXT: {rendered}"


def build_user_prompt(message: str, intent: str, context: dict) -> str:
    """Compose the final user-turn prompt: question + supporting context."""
    context_block = build_context_block(intent, context)
    return f"USER QUESTION: {message}\n\n{context_block}\n\nRespond with the JSON object described in your instructions."