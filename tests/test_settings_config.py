import json

import settings_config


def test_vault_defaults_and_custom_settings(tmp_path, monkeypatch):
    config = tmp_path / "settings.json"
    monkeypatch.setattr(settings_config, "SETTINGS_PATH", config)
    assert settings_config.load_settings("new")["CORRECT_THRESHOLD"] == 0.55
    assert settings_config.load_settings("legacy")["CORRECT_THRESHOLD"] == 0.52
    config.write_text(json.dumps({"CORRECT_THRESHOLD": 0.55, "AMBIGUOUS_THRESHOLD": 0.25,
                                  "DISTANCE_THRESHOLD": 0.70}))
    assert settings_config.load_settings("legacy")["AMBIGUOUS_THRESHOLD"] == 0.22
    settings_config.save_settings({"CORRECT_THRESHOLD": 0.61, "AMBIGUOUS_THRESHOLD": 0.31,
                                   "DISTANCE_THRESHOLD": 0.42})
    assert settings_config.load_settings("legacy")["CORRECT_THRESHOLD"] == 0.61
    assert settings_config.load_settings("new")["DISTANCE_THRESHOLD"] == 0.42
    assert config.read_text().count("_threshold_profile") == 1


def test_api_uses_canonical_settings_file(tmp_path, monkeypatch):
    import api

    config = tmp_path / "settings.json"
    monkeypatch.setattr(settings_config, "SETTINGS_PATH", config)
    api._save_settings({"CORRECT_THRESHOLD": 0.63})
    assert api._load_settings()["CORRECT_THRESHOLD"] == 0.63
    assert api._load_settings()["DISTANCE_THRESHOLD"] == 0.70
