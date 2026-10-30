"""Solver output: quiet by default, the solver's banner and log with verbose=True.

capfd captures file descriptors 1 and 2, where HiGHS writes its banner from C (sys.stdout capture misses it).
"""

import os

import pytest

from ghg_framework import Case, run
from ghg_framework.cli import main
from ghg_framework.solve import quiet_stdout

BANNER = "Running HiGHS"


def test_default_solve_prints_no_banner(week, capfd):
    print("before")
    quiet = run(week, Case(window="day", lam=2.0))
    print("after")
    out, err = capfd.readouterr()
    assert quiet.totals["all_optimal"]
    assert out == "before\nafter\n"
    assert BANNER not in err


def test_verbose_solve_prints_banner_and_log_with_the_same_result(week, capfd):
    case = Case(window="all", lam=2.0)
    quiet = run(week, case)
    assert capfd.readouterr().out == ""
    loud = run(week, case, verbose=True)
    out = capfd.readouterr().out
    assert out.count(BANNER) == 1 and "MIP" in out                  # one window: one banner, then the log
    assert loud.totals["reduction_t"] == quiet.totals["reduction_t"]
    assert loud.totals["adjusted_hours"] == quiet.totals["adjusted_hours"]


def test_quiet_stdout_restores_stdout_and_leaves_stderr(capfd):
    with pytest.raises(RuntimeError, match="boom"):
        with quiet_stdout():
            print("python-level, discarded")
            os.write(1, b"fd-level, discarded\n")
            os.write(2, b"stderr kept\n")
            raise RuntimeError("boom")
    print("visible")
    out, err = capfd.readouterr()
    assert out == "visible\n"
    assert err == "stderr kept\n"


def test_cli_quiet_by_default_and_verbose_flag(tmp_path, capfd):
    assert main(["synthetic", str(tmp_path), "--hours", "336"]) == 0
    capfd.readouterr()
    assert main(["run", str(tmp_path), "--region", "Windy", "--window", "week"]) == 0
    quiet = capfd.readouterr().out
    assert BANNER not in quiet and '"all_optimal": true' in quiet
    assert main(["run", str(tmp_path), "--region", "Windy", "--window", "week", "--verbose"]) == 0
    assert BANNER in capfd.readouterr().out
