import numpy as np
import pandas as pd
import pytest

from ghg_framework import FUELS, classify, diversity_index, intensity_fluctuation, make_region, region_metrics, renewable_share, synthetic_inputs


def _flat_region(shares):
    time = pd.date_range("2025-01-01", periods=24, freq="h")
    gen = {f: np.full(24, 100.0 * shares.get(f, 0.0)) for f in FUELS}
    return make_region("flat", time, np.full(24, 100.0), np.linspace(100, 500, 24), gen)


def test_diversity_index_eq8():
    equal = _flat_region({f: 1.0 for f in FUELS})
    assert diversity_index(equal) == pytest.approx(np.log(len(FUELS)))
    single = _flat_region({"gas": 1.0})
    assert diversity_index(single) == pytest.approx(0.0)
    two = _flat_region({"gas": 0.25, "wind": 0.75})
    assert diversity_index(two) == pytest.approx(-(0.25 * np.log(0.25) + 0.75 * np.log(0.75)))


def test_renewable_share():
    r = _flat_region({"gas": 0.5, "solar": 0.2, "wind": 0.2, "hydro": 0.1})
    assert renewable_share(r) == pytest.approx(0.5)


def test_fluctuation_statistics():
    f = intensity_fluctuation(_flat_region({"gas": 1.0}))
    e = np.linspace(100, 500, 24)
    assert f["range"] == pytest.approx(400.0)
    assert f["iqr"] == pytest.approx(np.percentile(e, 75) - np.percentile(e, 25))
    assert f["median"] == pytest.approx(300.0)


def test_metrics_table_and_classification():
    m = region_metrics(synthetic_inputs(hours=24 * 14))
    assert list(m.index) == ["Sunny", "Windy", "Mixed", "Thermal"]
    assert ((m["renewable_share"] > 0) & (m["renewable_share"] < 1)).all()
    groups = classify(m)
    assert set(groups.columns) == {"penetration", "diversity", "fluctuation"}
    for col in groups:
        assert (groups[col] == "high").sum() == 2                      # median split of four regions
    with pytest.raises(ValueError):
        classify(m, "kurtosis")
