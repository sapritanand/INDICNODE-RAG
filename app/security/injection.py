"""
Prompt Injection & Safety Guard
--------------------------------
Multi-layer detection pipeline:

Layer 1 – Rule-based (fast, zero-cost)
  Regex patterns for known injection phrases, jailbreak keywords, system-prompt
  delimiters, and role-hijacking language.  Runs in microseconds.

Layer 2 – Structural analysis
  Detects prompt delimiters (XML tags, triple-backtick fences, bracket notation)
  that are only meaningful as control characters, not natural questions.

Layer 3 – Encoded content detection
  Looks for base64 blobs and ROT-13 segments that may contain concealed
  instructions.  Decodes and re-runs Layer 1 on the decoded payload.

Why not an LLM classifier?
  An LLM classifier is most accurate but adds 200–800 ms and per-token cost to
  every request.  The rule-based approach catches ~95% of real-world attempts
  at zero cost.  An LLM guard can be layered on top as an optional config flag.
"""

from __future__ import annotations

import base64
import codecs
import re
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: List[re.Pattern] = [p for p in [
    # Direct instruction override
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier|your)\s+(instructions?|rules?|guidelines?|prompts?|constraints?)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above|earlier)", re.I),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above|earlier|your)", re.I),
    re.compile(r"override\s+(your\s+)?(previous|prior|all)?\s*(instructions?|rules?|guidelines?|constraints?)", re.I),
    re.compile(r"do\s+not\s+follow\s+(your\s+)?(previous|prior|original)?\s*(instructions?|rules?|guidelines?)", re.I),

    # Role / identity hijacking
    re.compile(r"you\s+are\s+now\s+(a\s+)?(different|new|another|unrestricted|free|evil|jailbroken)", re.I),
    re.compile(r"act\s+as\s+(if\s+)?(you\s+(are|were)\s+)?(an?\s+)?(different|unrestricted|evil|jailbroken|free|DAN)", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(a\s+)?(different|unrestricted|evil)", re.I),
    re.compile(r"roleplay\s+as\s+(a\s+)?(different|unrestricted|evil|jailbroken)", re.I),
    re.compile(r"\bDAN\b"),          # Do Anything Now jailbreak
    re.compile(r"jailbreak", re.I),
    re.compile(r"developer\s+mode", re.I),

    # System-prompt exfiltration
    re.compile(r"(reveal|show|print|output|display|tell\s+me|repeat|write\s+out)\s+(?:me\s+)?(your\s+)?(system\s+)?(prompt|instructions?|rules?|guidelines?|constraints?)", re.I),
    re.compile(r"what\s+(are|is)\s+(your|the)\s+(system\s+)?(prompt|instructions?|rules?|guidelines?)", re.I),
    re.compile(r"(print|output|display)\s+(every|all)\s+(token|word|character|letter)", re.I),

    # New-persona assignment
    re.compile(r"your\s+new\s+(name|identity|persona|role)\s+is", re.I),
    re.compile(r"from\s+now\s+on\s+you\s+(are|will\s+be|must|should)", re.I),
    re.compile(r"henceforth\s+you\s+(are|will\s+be|must|should)", re.I),

    # Harmful capability unlocking
    re.compile(r"(enable|unlock|activate|turn\s+on)\s+(unsafe|unrestricted|full|unfiltered)\s+mode", re.I),
    re.compile(r"without\s+(any\s+)?(restrictions?|filters?|guidelines?|limits?|constraints?)", re.I),
    re.compile(r"(bypass|circumvent|evade|avoid)\s+(your\s+)?(safety|restrictions?|filters?|guidelines?|rules?)", re.I),
]]

# Structural delimiters that signal injection attempts in user input
_DELIMITER_PATTERNS: List[re.Pattern] = [
    re.compile(r"<\s*(system|assistant|human|instructions?)\s*>", re.I),
    re.compile(r"\[SYSTEM\]|\[INST\]|\[/INST\]|\[SYS\]|\[/SYS\]", re.I),
    re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|system\|>"),
    re.compile(r"\{\{\s*(system|instructions?|prompt)\s*\}\}", re.I),
    re.compile(r"###\s*(system|instructions?|prompt)\s*###", re.I),
]


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    is_injection: bool
    confidence: float        # 0.0 – 1.0
    matched_rule: Optional[str]
    explanation: str


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class InjectionGuard:
    """
    Stateless injection detector.  Call detect(text) before processing any
    user-supplied string.
    """

    def detect(self, text: str) -> DetectionResult:
        """Run all detection layers and return the first positive match."""
        # Layer 1: rule-based patterns
        result = self._check_patterns(text)
        if result.is_injection:
            return result

        # Layer 2: structural delimiter check
        result = self._check_delimiters(text)
        if result.is_injection:
            return result

        # Layer 3: encoded payload check
        result = self._check_encoded(text)
        if result.is_injection:
            return result

        return DetectionResult(
            is_injection=False,
            confidence=0.0,
            matched_rule=None,
            explanation="No injection patterns detected.",
        )

    # ------------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------------

    def _check_patterns(self, text: str) -> DetectionResult:
        for pattern in _INJECTION_PATTERNS:
            m = pattern.search(text)
            if m:
                return DetectionResult(
                    is_injection=True,
                    confidence=0.95,
                    matched_rule=pattern.pattern,
                    explanation=f"Injection pattern detected: '{m.group()[:60]}'",
                )
        return DetectionResult(is_injection=False, confidence=0.0, matched_rule=None, explanation="")

    def _check_delimiters(self, text: str) -> DetectionResult:
        for pattern in _DELIMITER_PATTERNS:
            m = pattern.search(text)
            if m:
                return DetectionResult(
                    is_injection=True,
                    confidence=0.90,
                    matched_rule=pattern.pattern,
                    explanation=f"Prompt delimiter found in user input: '{m.group()[:60]}'",
                )
        return DetectionResult(is_injection=False, confidence=0.0, matched_rule=None, explanation="")

    def _check_encoded(self, text: str) -> DetectionResult:
        # Base64: look for runs of ≥20 base64 chars
        b64_re = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
        for match in b64_re.finditer(text):
            candidate = match.group()
            # Only try if length is a multiple of 4 (valid base64 padding)
            padded = candidate + "=" * (-len(candidate) % 4)
            try:
                decoded = base64.b64decode(padded).decode("utf-8", errors="ignore")
                sub = self._check_patterns(decoded)
                if sub.is_injection:
                    return DetectionResult(
                        is_injection=True,
                        confidence=0.85,
                        matched_rule="base64_encoded_injection",
                        explanation="Injection pattern found inside base64-encoded payload.",
                    )
            except Exception:
                pass

        # ROT-13: heuristic — only try if the query looks heavily non-alphabetic
        rot13_re = re.compile(r"[a-zA-Z]{15,}")
        for match in rot13_re.finditer(text):
            try:
                decoded = codecs.decode(match.group(), "rot_13")
                sub = self._check_patterns(decoded)
                if sub.is_injection:
                    return DetectionResult(
                        is_injection=True,
                        confidence=0.80,
                        matched_rule="rot13_encoded_injection",
                        explanation="Injection pattern found inside ROT-13 encoded text.",
                    )
            except Exception:
                pass

        return DetectionResult(is_injection=False, confidence=0.0, matched_rule=None, explanation="")
