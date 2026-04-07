"""
╔══════════════════════════════════════════════════════════════╗
║  SilentAssist — Hybrid Intent Decoder                        ║
║  ──────────────────────────────────────────────────────────  ║
║  Two-tier decoding pipeline:                                 ║
║                                                              ║
║  Tier 1 — Agentic LLM (Ollama)                              ║
║    Pipes the raw CTC text into a local LLM to extract        ║
║    semantic intent. Handles garbled homophene noise,          ║
║    scales to arbitrary commands, and understands context.     ║
║                                                              ║
║  Tier 2 — Fuzzy String Matching (fallback)                   ║
║    If Ollama is unavailable, falls back to thefuzz            ║
║    token_sort_ratio against the static command palette.       ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional
from thefuzz import fuzz, process

# ══════════════════════════════════════════════════════════════
#  Predefined Command Palette
# ══════════════════════════════════════════════════════════════
COMMAND_PALETTE: list[str] = [
    "Turn on the lights",
    "Turn off the lights",
    "Call for help",
    "Send emergency text",
    "Lock the doors",
    "Unlock the doors",
    "Open the window",
    "Close the window",
    "Play some music",
    "Stop the music",
    "Set an alarm",
    "Cancel the alarm",
    "Take a screenshot",
    "Read my messages",
    "Start recording",
    "Stop recording",
    "Call an ambulance",
    "Increase the volume",
    "Decrease the volume",
    "Navigate home",
]

# Minimum confidence score (0–100) required to accept a match.
CONFIDENCE_THRESHOLD: int = 70

# Ollama connection settings
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "llama3.2:1b"   # Lightweight, fast for intent parsing
OLLAMA_TIMEOUT: int = 15            # seconds


# ══════════════════════════════════════════════════════════════
#  Match Result Container
# ══════════════════════════════════════════════════════════════
@dataclass
class MatchResult:
    """Structured result from the decoder."""

    raw_input: str                   # What the VSR network produced
    matched_command: str             # Best-matching command (or fallback)
    confidence: int                  # 0-100 score
    is_recognised: bool              # True iff confidence >= threshold
    all_candidates: list[tuple]      # Top-5 (command, score) pairs
    decode_method: str = "fuzzy"     # "llm" or "fuzzy"
    llm_reasoning: str = ""         # LLM's reasoning (if used)


# ══════════════════════════════════════════════════════════════
#  Tier 1 — Agentic LLM Intent Parser (Ollama)
# ══════════════════════════════════════════════════════════════
_LLM_SYSTEM_PROMPT = """You are an intent-extraction agent for a Visual Speech Recognition (VSR) system called SilentAssist.

The VSR model reads lip movements and produces text via CTC decoding. This text is often garbled due to:
- Homophenes: words that look identical on the lips (e.g., "bat"/"pat"/"mat")
- Character-level noise from CTC collapse artifacts
- Missing or swapped characters

Your task: Given the raw VSR text, determine the user's intended command.

Available commands:
{commands}

RULES:
1. Respond ONLY with valid JSON in this exact format:
   {{"command": "<matched command or null>", "confidence": <0-100>, "reasoning": "<brief explanation>"}}
2. If the VSR text clearly maps to one of the available commands, return it with high confidence.
3. If the text is ambiguous between 2-3 commands, pick the most likely and explain why.
4. If the text doesn't match any command, set command to null and confidence to 0.
5. Be forgiving of noise — "trn on te lits" should map to "Turn on the lights".
6. Do NOT invent new commands. Only use commands from the list above."""


def _check_ollama_available() -> bool:
    """Check if Ollama server is running and reachable."""
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _query_ollama(
    raw_text: str,
    commands: list[str],
    model: str = OLLAMA_MODEL,
    timeout: int = OLLAMA_TIMEOUT,
) -> Optional[dict]:
    """
    Send the raw VSR text to Ollama for semantic intent extraction.

    Args:
        raw_text:  Garbled CTC output from the VSR model.
        commands:  List of valid commands to match against.
        model:     Ollama model name to use.
        timeout:   Request timeout in seconds.

    Returns:
        Parsed JSON dict with keys: command, confidence, reasoning.
        None on any failure (timeout, parse error, Ollama down).
    """
    commands_str = "\n".join(f"  - {cmd}" for cmd in commands)
    system_prompt = _LLM_SYSTEM_PROMPT.format(commands=commands_str)

    user_prompt = (
        f"Raw VSR output: \"{raw_text}\"\n\n"
        "Extract the intended command. Respond with JSON only."
    )

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,         # Low temp for deterministic output
            "num_predict": 200,         # Short response (JSON only)
        },
        "format": "json",               # Force JSON output format
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))

        # Extract the message content
        content = response_data.get("message", {}).get("content", "")
        result = json.loads(content)

        # Validate structure
        if "command" in result and "confidence" in result:
            return result
        return None

    except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError) as e:
        print(f"[SilentAssist] Ollama query failed: {e}")
        return None


def llm_decode(
    raw_text: str,
    commands: Optional[list[str]] = None,
    threshold: int = CONFIDENCE_THRESHOLD,
) -> Optional[MatchResult]:
    """
    Attempt to decode the raw VSR text using Ollama LLM.

    Returns:
        MatchResult if LLM successfully parsed the intent, else None.
    """
    if commands is None:
        commands = COMMAND_PALETTE

    cleaned = raw_text.strip()
    if not cleaned:
        return None

    # Check if Ollama is available
    if not _check_ollama_available():
        print("[SilentAssist] Ollama not available, falling back to fuzzy matching.")
        return None

    result = _query_ollama(cleaned, commands)
    if result is None:
        return None

    matched_cmd = result.get("command")
    confidence = int(result.get("confidence", 0))
    reasoning = result.get("reasoning", "")

    # Validate the matched command exists in our palette
    if matched_cmd and matched_cmd not in commands:
        # LLM returned a command not in our list — try fuzzy matching it
        best = process.extractOne(matched_cmd, commands, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= 80:
            matched_cmd = best[0]
        else:
            matched_cmd = None
            confidence = 0

    is_recognised = matched_cmd is not None and confidence >= threshold

    # Build candidate list from fuzzy scores for display
    fuzzy_candidates = process.extractBests(
        cleaned, commands, scorer=fuzz.token_sort_ratio, limit=5,
    )

    return MatchResult(
        raw_input=raw_text,
        matched_command=matched_cmd if is_recognised else "Command not recognized.",
        confidence=confidence,
        is_recognised=is_recognised,
        all_candidates=fuzzy_candidates,
        decode_method="llm",
        llm_reasoning=reasoning,
    )


# ══════════════════════════════════════════════════════════════
#  Tier 2 — Fuzzy String Matching (Fallback)
# ══════════════════════════════════════════════════════════════
def fuzzy_decode(
    raw_text: str,
    commands: Optional[list[str]] = None,
    threshold: int = CONFIDENCE_THRESHOLD,
    top_n: int = 5,
) -> MatchResult:
    """
    Match *raw_text* against the command palette using fuzzy logic.

    Uses ``thefuzz.process.extractBests`` with ``token_sort_ratio``
    as the scorer — this normalises word order and is resilient to
    the kind of character-level noise that CTC decoding produces.

    Args:
        raw_text:   Raw string from the VSR model / CTC decoder.
        commands:   Override the default COMMAND_PALETTE if desired.
        threshold:  Minimum score to accept the best match.
        top_n:      Number of candidates to return for debugging.

    Returns:
        MatchResult with the best match details.
    """
    if commands is None:
        commands = COMMAND_PALETTE

    # Handle empty / whitespace-only input gracefully
    cleaned = raw_text.strip()
    if not cleaned:
        return MatchResult(
            raw_input=raw_text,
            matched_command="Command not recognized.",
            confidence=0,
            is_recognised=False,
            all_candidates=[],
            decode_method="fuzzy",
        )

    # Extract top-N candidates using token_sort_ratio
    candidates = process.extractBests(
        cleaned,
        commands,
        scorer=fuzz.token_sort_ratio,
        limit=top_n,
    )

    best_command, best_score = candidates[0] if candidates else ("", 0)

    is_recognised = best_score >= threshold
    matched = best_command if is_recognised else "Command not recognized."

    return MatchResult(
        raw_input=raw_text,
        matched_command=matched,
        confidence=best_score,
        is_recognised=is_recognised,
        all_candidates=candidates,
        decode_method="fuzzy",
    )


# ══════════════════════════════════════════════════════════════
#  Hybrid Decoder — LLM first, fuzzy fallback
# ══════════════════════════════════════════════════════════════
def hybrid_decode(
    raw_text: str,
    commands: Optional[list[str]] = None,
    threshold: int = CONFIDENCE_THRESHOLD,
    use_llm: bool = True,
) -> MatchResult:
    """
    Two-tier decoding pipeline:
        1. Try Ollama LLM for semantic intent extraction
        2. Fall back to fuzzy string matching if LLM unavailable

    Args:
        raw_text:   Raw VSR/CTC output text.
        commands:   Command palette to match against.
        threshold:  Minimum confidence to accept.
        use_llm:    Whether to attempt LLM decoding first.

    Returns:
        MatchResult from whichever tier succeeded.
    """
    if use_llm:
        llm_result = llm_decode(raw_text, commands, threshold)
        if llm_result is not None:
            return llm_result

    # Fallback to fuzzy matching
    return fuzzy_decode(raw_text, commands, threshold)
