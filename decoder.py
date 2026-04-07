"""
╔══════════════════════════════════════════════════════════════╗
║  SilentAssist — Fuzzy Command Decoder ("Hackathon Magic")   ║
║  ──────────────────────────────────────────────────────────  ║
║  VSR output is noisy because many phonemes share identical   ║
║  mouth shapes (homophenes: "bat" ↔ "pat" ↔ "mat").           ║
║                                                              ║
║  This module acts as a safety-net: it fuzzy-matches the raw  ║
║  (often garbled) VSR text against a fixed command palette    ║
║  using thefuzz's token_sort_ratio, which is robust to word   ║
║  order and minor spelling errors.                            ║
╚══════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass
from typing import Optional
from thefuzz import fuzz, process

# ══════════════════════════════════════════════════════════════
#  Predefined Command Palette
# ══════════════════════════════════════════════════════════════
#  Extend this list freely — the fuzzy matcher scales well to
#  dozens of entries.

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


# ══════════════════════════════════════════════════════════════
#  Match Result Container
# ══════════════════════════════════════════════════════════════
@dataclass
class MatchResult:
    """Structured result from the fuzzy decoder."""

    raw_input: str              # What the VSR network produced
    matched_command: str        # Best-matching command (or fallback)
    confidence: int             # 0-100 score
    is_recognised: bool         # True iff confidence >= threshold
    all_candidates: list[tuple] # Top-5 (command, score) pairs


# ══════════════════════════════════════════════════════════════
#  Core Decoding Function
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
    )
