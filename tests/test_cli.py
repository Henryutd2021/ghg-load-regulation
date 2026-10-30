import pandas as pd

from ghg_framework.cli import main


def test_cli_synthetic_run_metrics(tmp_path, capsys):
    inputs = tmp_path / "inputs"
    assert main(["synthetic", str(inputs), "--hours", "336"]) == 0
    out = tmp_path / "out"
    assert main(["run", str(inputs), "--region", "Windy", "--window", "week", "--lam", "1", "--out-dir", str(out)]) == 0
    totals = pd.read_csv(out / "totals.csv")
    assert totals.loc[0, "region"] == "Windy" and totals.loc[0, "window"] == "week"
    assert bool(totals.loc[0, "all_optimal"])
    assert len(pd.read_csv(out / "hourly.csv")) == 336
    assert main(["metrics", str(inputs), "--out", str(tmp_path / "m.csv")]) == 0
    assert set(pd.read_csv(tmp_path / "m.csv")["region"]) == {"Sunny", "Windy", "Mixed", "Thermal"}
    assert main(["run", str(inputs), "--region", "Nowhere"]) == 2
    capsys.readouterr()
