"""
╔══════════════════════════════════════════════════════════════╗
║  SilentAssist — Execution Engine                             ║
║  ──────────────────────────────────────────────────────────  ║
║  Interfaces natively with macOS OS-level tools and APIs      ║
║  to physically execute commands inferred by the Agentic LLM. ║
╚══════════════════════════════════════════════════════════════╝
"""

import subprocess
import json
import logging

def _run_osascript(script: str) -> bool:
    """Helper to run AppleScript via osascript."""
    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"AppleScript execution failed: {e.stderr.decode()}")
        return False


def set_volume(level: int) -> dict:
    """Set the system volume to a specific level (0-100)."""
    # AppleScript volume is 0 to 100
    level = max(0, min(100, level))
    success = _run_osascript(f"set volume output volume {level}")
    return {
        "status": "success" if success else "error",
        "message": f"System volume set to {level}%" if success else "Failed to set system volume",
        "action": "volume_adjust"
    }


def change_volume(direction: str, amount: int = 15) -> dict:
    """Increase or decrease system volume."""
    try:
        # Get current volume
        result = subprocess.run(["osascript", "-e", "output volume of (get volume settings)"], 
                                capture_output=True, text=True, check=True)
        current = int(result.stdout.strip())
        
        if direction == "up":
            return set_volume(current + amount)
        else:
            return set_volume(current - amount)
    except Exception as e:
        return {"status": "error", "message": f"Failed to change volume: {str(e)}"}


def toggle_media() -> dict:
    """Play or pause the current media (Spotify, Music, etc) using media keys."""
    script = """
    tell application "System Events"
        key code 100
    end tell
    """
    success = _run_osascript(script)
    return {
        "status": "success" if success else "error",
        "message": "Toggled media playback" if success else "Failed to toggle media",
        "action": "media_toggle"
    }


def lock_screen() -> dict:
    """Lock the macOS screen."""
    script = """
    tell application "System Events"
        keystroke "q" using {control down, command down}
    end tell
    """
    success = _run_osascript(script)
    return {
        "status": "success" if success else "error",
        "message": "Screen locked successfully" if success else "Failed to lock screen",
        "action": "lock_screen"
    }


def open_application(app_name: str) -> dict:
    """Open a macOS application by name."""
    success = _run_osascript(f'tell application "{app_name}" to activate')
    return {
        "status": "success" if success else "error",
        "message": f"Opened application: {app_name}" if success else f"Failed to open application: {app_name}",
        "action": "open_app"
    }


def execute_tool_call(tool_name: str, args: dict) -> dict:
    """
    Dispatch a tool call from the LLM to the actual python function.
    """
    logging.info(f"Executing tool: {tool_name} with args: {args}")
    
    try:
        if tool_name == "set_volume":
            return set_volume(args.get("level", 50))
        elif tool_name == "increase_volume":
            return change_volume("up")
        elif tool_name == "decrease_volume":
            return change_volume("down")
        elif tool_name == "toggle_media":
            return toggle_media()
        elif tool_name == "lock_screen":
            return lock_screen()
        elif tool_name == "open_application":
            return open_application(args.get("app_name", "Safari"))
        elif tool_name == "emergency_protocol":
            open_application("Messages")
            return {"status": "success", "message": "Emergency protocol activated: Messages opened", "action": "emergency"}
        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    except Exception as e:
        return {"status": "error", "message": f"Execution failed: {str(e)}"}
