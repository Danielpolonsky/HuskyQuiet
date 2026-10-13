"""Leakage-aware features from the free-text ``description`` field.

The audit found that SDCI staff append dated administrative notes to descriptions after
intake (e.g. ``"1/10/17 MD.CANCELLED IAW SBC 106.6.9."`` or ``"EXTENSION GRANTED"``).
These notes reveal outcomes, so the text is sanitized before any feature is derived:
every segment containing an administrative keyword or an embedded date is dropped, and
only a whitelist of project-scope keyword flags is computed from what remains.
Description length and raw tokens are deliberately *not* used.
"""

from __future__ import annotations

import re

import pandas as pd

ADMIN_PATTERN = re.compile(
    r"CANCEL|WITHDR|EXPIR|\bVOID|EXTEN[DS]|REINSTAT|REISSU|RE-ISSU|ON HOLD|\bHOLD\b|\bIAW\b"
    r"|PER APPLICANT REQUEST|ABANDON|DENIED|DENIAL|\bCLOSED\b|FINAL(?:ED)? INSPECTION|SUPERSEDED",
)
DATE_PATTERN = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
_SEGMENT_SPLIT = re.compile(r"(?<=[.;!?])\s+|\s+--+\s*|\s*\|\s*")

SCOPE_KEYWORDS: dict[str, str] = {
    "kw_new_construction": r"\bCONSTRUCT(?:ION)? (?:OF )?(?:A )?NEW\b"
    r"|\bNEW (?:ONE|SINGLE|MULTI|TOWN|ROW|APARTMENT|BUILDING|MIXED|COMMERCIAL|\d)|\bERECT",
    "kw_demolition": r"DEMOLISH|DEMOLITION|DECONSTRUCT",
    "kw_adu": r"\bA?DADU\b|\bADU\b|ACCESSORY DWELLING|BACKYARD COTTAGE",
    "kw_townhouse": r"TOWNHOUSE|TOWNHOME|ROWHOUSE|ROW HOUSE",
    "kw_multifamily": r"APARTMENT|MULTI-?FAMILY|MIXED[- ]USE|\bCONDO",
    "kw_tenant_improvement": r"TENANT IMPROVEMENT|\bT\.?I\.?\b",
    "kw_stfi": r"SUBJECT TO FIELD INSPECTION|\bSTFI\b",
    "kw_change_of_use": r"CHANGE (?:OF )?USE",
    "kw_establish_use": r"ESTABLISH (?:THE )?USE",
    "kw_earthwork": r"GRADING|EXCAVAT|SHORING|RETAINING WALL",
    "kw_seismic": r"SEISMIC|RETROFIT",
    "kw_solar": r"SOLAR|PHOTOVOLTAIC|\bPV\b",
    "kw_deck_porch": r"\bDECKS?\b|\bPORCH",
    "kw_basement": r"BASEMENT",
    "kw_addition": r"\bADDITIONS?\b",
}
_SCOPE_COMPILED = {k: re.compile(v) for k, v in SCOPE_KEYWORDS.items()}
KEYWORD_FEATURES: tuple[str, ...] = tuple(SCOPE_KEYWORDS)


def sanitize_description(text: str | None) -> str:
    """Remove administrative/outcome annotations; return upper-cased scope text."""
    if text is None or (isinstance(text, float) and pd.isna(text)) or text is pd.NA:
        return ""
    upper = str(text).upper()
    kept = [
        seg
        for seg in _SEGMENT_SPLIT.split(upper)
        if seg and not ADMIN_PATTERN.search(seg) and not DATE_PATTERN.search(seg)
    ]
    return " ".join(kept)


def keyword_features(descriptions: pd.Series) -> pd.DataFrame:
    """Binary scope-keyword flags computed on sanitized descriptions."""
    clean = descriptions.map(sanitize_description).astype("string")
    out = pd.DataFrame(index=descriptions.index)
    for name, pattern in _SCOPE_COMPILED.items():
        out[name] = clean.str.contains(pattern, regex=True).fillna(False).astype("int8")
    out["desc_missing"] = descriptions.isna().astype("int8")
    return out
