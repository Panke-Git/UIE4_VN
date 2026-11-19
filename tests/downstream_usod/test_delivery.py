import tarfile

import pytest

from src.downstream_usod.export import export_results
from src.downstream_usod.pipeline import phase_methods


def test_phase_menu_keeps_primary_protocol():
    assert phase_methods("p0") == ("raw", "icar_uie")
    assert phase_methods("p1", ["marine_net"]) == ("raw", "icar_uie", "marine_net")
    assert phase_methods("p2") == ("raw", "icar_uie", "icar_uie_uieb")
    with pytest.raises(ValueError, match="unique"):
        phase_methods("p1", ["waternet", "waternet"])


def test_export_is_lightweight_and_exclusive(tmp_path):
    root = tmp_path / "fixture"
    files = ["reports/p0_summary.json", "reports/p0_summary.md", "metrics/p0/overall.csv",
             "metrics/p0/paired_bootstrap.csv", "metrics/p0/paired_bootstrap.json",
             "predictions/raw/00002.png", "logs/inference.log"]
    for filename in files:
        file = root / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fixture")
    archive = tmp_path / "report.tar.gz"
    export_results(root, archive, ["p0"])
    with tarfile.open(archive) as result:
        assert set(result.getnames()) == {*files[:5], "SHA256SUMS"}
        assert len(result.extractfile(result.getmember("SHA256SUMS")).read().splitlines()) == 5
    with pytest.raises(FileExistsError):
        export_results(root, archive, ["p0"])
