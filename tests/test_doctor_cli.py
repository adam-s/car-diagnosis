"""Smoke tests for doctor + the onboarding/version CLI surface."""
from __future__ import annotations

from typer.testing import CliRunner

from cardiag.cli import app

runner = CliRunner()


def test_doctor_runs_and_returns_int():
    from cardiag import doctor
    n = doctor.run()                      # prints a report; returns # hard failures
    assert isinstance(n, int) and n >= 0


def test_doctor_check_helpers():
    from cardiag import doctor
    # each check returns (status, detail, fix) without raising
    for name, fn in doctor.CHECKS:
        status, detail, fix = fn()
        assert status in (doctor.OK, doctor.WARN, doctor.BAD)
        assert isinstance(detail, str)


def test_cli_doctor_command():
    res = runner.invoke(app, ["doctor"])
    assert "preflight" in res.stdout.lower() or "cardiag doctor" in res.stdout.lower()


def test_cli_start_runs(tmp_path, monkeypatch):
    # point at an empty data dir so 'start' trains fixtures, then prints next steps
    monkeypatch.setenv("CARDIAG_DATA", str(tmp_path))
    res = runner.invoke(app, ["start"])
    # exits 0 (ready) or 1 (a hard dep like yt-dlp missing in CI), never crashes;
    # either way it produced a guided report
    assert res.exit_code in (0, 1)
    assert "diagnose" in res.stdout or "Fix the" in res.stdout
