"""Unit tests for the deterministic core: types, config invariants, path
translation, and the cause/label canonicalization. Pure logic, no I/O."""
from __future__ import annotations

from cardiag import config
from cardiag.types import Band, Cause, Diagnosis, Segment, TriageResult, Verdict


# ----------------------------------------------------------------- types
def test_segment_duration_and_dict():
    s = Segment(start=1.0, end=3.5, flatness=0.12)
    assert s.duration == 2.5
    assert s.to_dict()["duration"] == 2.5


def test_diagnosis_to_dict_schema():
    d = Diagnosis(
        file="x.wav", verdict=Verdict.FAULT, fault_probability=0.812345,
        engine_knock_probability=0.1, causes=[Cause("brakes", 0.34, "note")],
        segments=[Segment(0.0, 2.0)],
    )
    out = d.to_dict()
    assert set(out) == {"file", "verdict", "fault_probability", "regions",
                        "engine_knock_probability", "causes", "segments", "note"}
    assert out["verdict"] == "fault"
    assert out["fault_probability"] == 0.812          # rounded to 3
    assert out["causes"][0] == {"part": "brakes", "p": 0.34, "note": "note"}
    assert out["regions"] == []                       # none supplied -> empty list


def test_triage_result_dict():
    t = TriageResult(file="x.wav", triage="engine", label="ENGINE", confidence=0.91,
                     band=Band.HIGH, band_gloss="g", probabilities={"engine": 0.91})
    out = t.to_dict()
    assert out["band"] == "high" and out["triage"] == "engine"


def test_verdict_and_band_are_str_enums():
    assert Verdict.FAULT.value == "fault"
    assert Band.ABSTAIN.value == "abstain"


# ----------------------------------------------------------------- config
def test_config_threshold_invariants():
    # documented ordering relationships that the gating logic relies on
    assert config.MECH_REJECT_BELOW < config.MECH_CONFIRM_MIN
    assert 0 < config.NORMAL_MARGIN < 1
    assert config.SR_CHEAP < config.SR_CLAP
    assert 0 < config.SPECTRAL_FLATNESS_MAX <= 1


def test_config_prompt_sets_nonempty():
    for name in ("L1_PROMPTS", "FAULT_PROMPTS", "TOOL_PROMPTS",
                 "CONFIRM_KEEP", "CONFIRM_DROP", "FAULT_QUERIES"):
        assert len(getattr(config, name)) > 0
    assert config.L1_NORMAL in " ".join(config.L1_PROMPTS)


# ----------------------------------------------------------------- paths
def test_resolve_clip_prefix_translation():
    # import inside the test so CARDIAG_DATA monkeypatching is honored elsewhere
    from cardiag import paths
    r = paths.resolve_clip("youtube/data/clips/abc/clip_00.wav")
    assert str(r).endswith("youtube/clips/abc/clip_00.wav")
    assert str(paths.DATA) in str(r)


def test_resolve_clip_absolute_passthrough():
    from cardiag import paths
    assert str(paths.resolve_clip("/tmp/x.wav")) == "/tmp/x.wav"


def test_data_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CARDIAG_DATA", str(tmp_path))
    import importlib

    import cardiag.paths as p
    importlib.reload(p)
    try:
        assert p.DATA == tmp_path.resolve()
    finally:
        monkeypatch.delenv("CARDIAG_DATA", raising=False)
        importlib.reload(p)


# ------------------------------------------------------- cause canonicalization
def test_canonical_cause_maps_known_parts():
    from cardiag.training.prep import causes
    assert causes.canonical_cause("wheel bearing") == "wheel_bearing"
    assert causes.canonical_cause("brake pad") == "brakes"
    assert causes.canonical_cause("serpentine belt") == "belt"


def test_canonical_cause_tail_falls_to_other():
    from cardiag.training.prep import causes
    assert causes.canonical_cause("completely unknown gizmo 1234") == "other"


# --------------------------------------------------------- reddit parser (pure)
def test_reddit_media_id_dedups_reposts():
    from cardiag.ingest.reddit import scrape
    a = scrape.media_id("https://v.redd.it/abc123/DASH_720.mp4")
    b = scrape.media_id("https://v.redd.it/abc123")
    assert a == b == "abc123"


def test_reddit_is_video_domain():
    from cardiag.ingest.reddit import scrape
    assert scrape.is_video({"domain": "v.redd.it"})
    assert not scrape.is_video({"domain": "self.MechanicAdvice"})
