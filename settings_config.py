"""Canonical backend settings: settings.json overrides these defaults.

An unmarked older file keeps the historical legacy-vault grading thresholds
until a user saves settings through the API. This preserves existing behavior.
"""
import json
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().with_name("settings.json")
DEFAULTS = {
    "FETCH_K": 30, "PROCESS_K": 10, "CHUNK_WORD_LIMIT": 300,
    "CORRECT_THRESHOLD": 0.55, "AMBIGUOUS_THRESHOLD": 0.25,
    "BENDY_THRESHOLD": 0.10, "MAX_VERIFIED_CHUNKS": 2,
    "SOFTMAX_MASS_CUTOFF": 0.90, "DISTANCE_THRESHOLD": 0.70,
    "TOTAL_DOCS": 60,
}
LEGACY_GRADING = {"CORRECT_THRESHOLD": 0.52, "AMBIGUOUS_THRESHOLD": 0.22}


def load_settings(vault_schema: str | None = None) -> dict:
    try:
        with SETTINGS_PATH.open(encoding="utf-8") as file:
            stored = json.load(file)
    except FileNotFoundError:
        stored = {}
    settings = DEFAULTS | {key: stored[key] for key in DEFAULTS if key in stored}
    if vault_schema is not None and vault_schema != "new" and stored.get("_threshold_profile") != "custom":
        settings.update(LEGACY_GRADING)
    return settings


def save_settings(values: dict) -> None:
    with SETTINGS_PATH.open("w", encoding="utf-8") as file:
        json.dump(values | {"_threshold_profile": "custom"}, file, indent=2)
