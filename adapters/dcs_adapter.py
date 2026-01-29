"""
Empty-shell DCS adapter for UDP-based overlay messaging.

Compatible with Scripts/Hooks/VRHilite.lua which accepts plain text:
- "pnt_XXX" to highlight the cockpit element
- "HILITE pnt_XXX" alternative syntax
- "CLEAR" to remove highlight

Extensible: can map abstract targets (via ui_map) to pnt codes before sending.
"""

from __future__ import annotations

import socket
from typing import Dict, List, Optional


class DcsAdapter:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7778,
        capabilities: Optional[List[str]] = None,
        timeout: float = 0.5,
    ):
        self.server = (host, port)
        self.capabilities = capabilities or ["overlay"]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

    def negotiate(self) -> None:
        """
        Placeholder for future capability handshake.
        VRHilite.lua does not reply, so this is a no-op for now.
        """
        return None

    def highlight(self, pnt_code: str) -> None:
        """Send highlight command for a DCS cockpit point code."""
        msg = f"HILITE {pnt_code}".encode("utf-8")
        self.sock.sendto(msg, self.server)

    def clear(self) -> None:
        """Clear current highlight."""
        self.sock.sendto(b"CLEAR", self.server)

    def send_overlay_intent(self, intent: Dict, expect_reply: bool = False) -> Optional[Dict]:
        """
        Translate overlay intent to VRHilite wire format.
        intent: {"intent": "highlight"|"clear", "element_id": "pnt_331"}
        """
        action = intent.get("intent")
        element = intent.get("element_id")
        if action == "clear":
            self.clear()
        elif action == "highlight" and element:
            self.highlight(element)
        else:
            raise ValueError("Invalid overlay intent")

        if expect_reply:
            try:
                data, _ = self.sock.recvfrom(4096)
                return {"reply": data.decode("utf-8")}
            except socket.timeout:
                return None
        return None

    def close(self):
        self.sock.close()
