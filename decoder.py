"""
╔══════════════════════════════════════════════════════════════╗
║  SilentAssist — Agentic Intent Parser                        ║
║  ──────────────────────────────────────────────────────────  ║
║  Pipes raw CTC phoneme/character output directly into a      ║
║  local LLM (Ollama) to extract semantic intent, mapping      ║
║  noisy homophenes to strict executable tool payloads.        ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

# Ollama connection settings
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "llama3.2:1b"   # Lightweight, fast agent
OLLAMA_TIMEOUT: int = 15            # seconds

@dataclass
class IntentResult:
    """Structured Agent output determining which Tool to call."""
    raw_input: str
    tool_name: Optional[str]
    tool_args: dict
    reasoning: str


# System Prompt instructing the LLM on Tool Calling
_AGENT_SYSTEM_PROMPT = """You are the core intelligence for SilentAssist, a Visual Speech Recognition platform.
The system reads lip movements and outputs raw text via CTC decoding. This text is typically highly garbled due to:
- Homophenes (words looking identical on lips)
- Missing or swapped characters

Your sole task is to map this noisy, garbled input text to an executable system tool.

AVAILABLE TOOLS:
1. "set_volume" -- Sets system volume to a specific percentage. Argument: {"level": <int>}
2. "increase_volume" -- Turns up the sound. No arguments.
3. "decrease_volume" -- Turns down the sound. No arguments.
4. "toggle_media" -- Plays or pauses music/media. No arguments.
5. "lock_screen" -- Locks the computer. No arguments.
6. "open_application" -- Opens a macOS app. Argument: {"app_name": "<string>"}
7. "emergency_protocol" -- Triggers an emergency sequence or text. No arguments.

RULES:
1. You MUST respond with ONLY a valid, parseable JSON object.
2. Format: {"tool": "<tool_name or null>", "args": {<arguments if any>}, "reasoning": "<brief explanation of why this garbled text maps to this tool>"}
3. If the input is completely unrecognizable, set tool to null.
4. Tolerate heavy typos: "trn op apul msc" -> open_application(app_name="Music")
5. "stahhp th msic" -> toggle_media()
6. "lck scn" -> lock_screen()
"""

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


def decode_intent(raw_text: str, timeout: int = OLLAMA_TIMEOUT) -> IntentResult:
    """
    Send the raw VSR text to Ollama for Agentic tool extraction.
    """
    cleaned = raw_text.strip()
    
    if not cleaned:
        return IntentResult(
            raw_input=raw_text,
            tool_name=None,
            tool_args={},
            reasoning="Empty input string."
        )

    if not _check_ollama_available():
        # FALLBACK: Execute string matching if Ollama is not accessible (e.g. Hugging Face Spaces)
        from thefuzz import process
        
        COMMAND_PALETTE = {
            "increase_volume": ["turn up", "increase volume", "louder", "raise volume", "turn on the lights"],
            "decrease_volume": ["turn down", "decrease volume", "quieter", "lower volume"],
            "toggle_media": ["play music", "pause music", "stop music", "toggle media", "play some music"],
            "lock_screen": ["lock screen", "sleep", "lock the computer", "lock the doors"],
            "emergency_protocol": ["help", "emergency", "send message", "call for help", "send emergency text"],
            "open_application": ["open the window", "set an alarm"]
        }
        
        best_match_tool = None
        best_score = 0
        
        for tool, phrases in COMMAND_PALETTE.items():
            match, score = process.extractOne(cleaned, phrases)
            if score > best_score:
                best_score = score
                best_match_tool = tool
                
        if best_score >= 60:
            return IntentResult(
                raw_input=raw_text,
                tool_name=best_match_tool,
                tool_args={},
                reasoning=f"Agent offline. Fallback matched via Fuzzy String Logic (Score: {best_score})"
            )
        else:
            return IntentResult(
                raw_input=raw_text,
                tool_name=None,
                tool_args={},
                reasoning="Ollama LLM server offline. Fallback fuzzy matcher failed to find confident match."
            )

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Garbled VSR Output: \"{cleaned}\""},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,  # Strongly deterministic
            "num_predict": 250,
        },
        "format": "json",        # Force JSON constraint
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

        content = response_data.get("message", {}).get("content", "")
        result_json = json.loads(content)

        return IntentResult(
            raw_input=raw_text,
            tool_name=result_json.get("tool"),
            tool_args=result_json.get("args", {}),
            reasoning=result_json.get("reasoning", "Parsed successfully via Agent LLM.")
        )

    except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError) as e:
         return IntentResult(
            raw_input=raw_text,
            tool_name=None,
            tool_args={},
            reasoning=f"LLM parsing failed: {str(e)}"
        )
