import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts" / "models" / "production" / "model.joblib"
CLEAN = ROOT / "data" / "interim" / "permits_clean.parquet"

pytestmark = pytest.mark.integration


@pytest.fixture
def app():
    if not MODEL.exists() or not CLEAN.exists():
        pytest.skip("production artifacts not built")
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=180).run()


def _text(at) -> str:
    parts = [block.value for block in at.markdown]
    parts += [block.value for block in at.caption]
    parts += [block.value for block in at.subheader]
    return " ".join(parts)


def _search_coordinates(at, lat, lon):
    at.text_input(key="explore_lat").input(lat)
    at.text_input(key="explore_lon").input(lon)
    return at.button(key="explore_search").click().run()


def test_explore_starts_empty_and_names_the_product(app):
    assert not app.exception
    assert app.header[0].value == "Explore"
    text = _text(app)
    assert "HuskyQuiet · Seattle pending-permit explorer" in text
    assert "See pending development permits near a Seattle address" in text
    assert "choose **Search**" in app.info[0].value
    assert len(app.metric) == 0


def test_coordinate_search_shows_scoped_results_and_a_download(app):
    at = _search_coordinates(app, "47.6615", "-122.3130")
    assert not at.exception
    assert not at.error
    assert any("within 500 m of 47.66150, -122.31300" in block.value for block in at.subheader)
    labels = [metric.label for metric in at.metric]
    assert labels[:3] == ["Pending permits", "Expected permits issued within 90 days", "Nearest permit"]
    assert at.metric[1].proto.help.startswith("Adds the issuance probabilities")
    assert any("(CSV)" in button.proto.label for button in at.get("download_button"))


def test_changed_inputs_are_flagged_until_searched_again(app):
    at = _search_coordinates(app, "47.6615", "-122.3130")
    at.segmented_control(key="explore_radius").set_value("1 km").run()
    assert "previous search" in at.warning[0].value
    assert any("within 500 m" in block.value for block in at.subheader)
    at = at.button(key="explore_search").click().run()
    assert not at.warning
    assert any("within 1,000 m" in block.value for block in at.subheader)


def test_failed_search_clears_the_old_results(app):
    at = _search_coordinates(app, "47.6615", "-122.3130")
    assert len(at.metric) == 3
    at = _search_coordinates(at, "40.7128", "-74.0060")
    assert "outside Seattle" in at.error[0].value
    assert len(at.metric) == 0
    assert not any("Pending permits within" in block.value for block in at.subheader)


def test_swapped_coordinates_are_explained(app):
    at = _search_coordinates(app, "-122.3130", "47.6615")
    assert "swapped" in at.error[0].value


def test_filters_that_match_nothing_say_so(app):
    at = _search_coordinates(app, "47.6615", "-122.3130")
    at = at.slider(key="explore_minimum").set_value(100).run()
    assert not at.exception
    assert at.metric[0].value == "0"
    assert any("No permits match these filters" in block.value for block in at.info)
    assert any("Filters are on" in block.value for block in at.caption)


def test_area_mode_uses_one_scope_for_every_figure(app):
    at = app.segmented_control(key="explore_mode").set_value("An area").run()
    assert not at.exception
    assert any("inside the University District extent" in block.value for block in at.subheader)
    total = int(at.metric[0].value.replace(",", ""))
    at = at.multiselect(key="explore_classes").set_value(["Commercial"]).run()
    filtered = int(at.metric[0].value.replace(",", ""))
    assert filtered < total
    assert any(f"showing {filtered:,} of {total:,}" in block.value for block in at.caption)
    at = at.pills(key="explore_area").set_value("University Village (shopping center)").run()
    assert any("not a neighborhood" in block.value for block in at.caption)


def test_compare_identical_coordinates(app):
    at = app.switch_page("app_pages/compare.py").run()
    assert at.header[0].value == "Compare locations"
    for index in range(2):
        at.text_input(key=f"compare_lat_{index}").input("47.6615")
        at.text_input(key=f"compare_lon_{index}").input("-122.3130")
    at = at.button(key="compare_submit").click().run()
    assert not at.exception
    assert any("same place" in block.value for block in at.info)
    assert len(at.metric) == 6
    assert at.metric[0].value == at.metric[3].value
    bullets = [block.value.lower() for block in at.markdown if block.value.startswith("- ")]
    assert bullets and bullets[0].startswith("- both entries resolve to the same point")
    assert not any(word in bullet for bullet in bullets for word in ("quieter", "safer", "better"))


def test_compare_areas_without_network(app):
    at = app.switch_page("app_pages/compare.py").run()
    at = at.segmented_control(key="compare_mode").set_value("Two areas").run()
    assert not at.exception
    assert any("No permit filters apply here" in block.value for block in at.caption)


def test_how_it_works_labels_the_evaluated_model_and_period(app):
    at = app.switch_page("app_pages/how_it_works.py").run()
    assert not at.exception
    values = {metric.label: metric.value for metric in at.metric}
    assert values["ROC-AUC"] == "0.76"
    assert values["Baseline ROC-AUC"] == "0.69"
    text = _text(at)
    assert "Evaluation copy, forecast dates Aug 1, 2025 to Jul 1, 2026." in text
    assert "| Deployed model | Calibration | Aug 2025 to Jul 2026 |" in text
    assert "ROC-AUC is not accuracy" in text
    assert "evaluation copy" in text


_STALE_PROCESS = """
import sys
sys.path.insert(0, {src!r})
import huskyquiet, huskyquiet.analysis
del huskyquiet.IMPORTED_AT_NS, huskyquiet.analysis.ISSUANCE_NOTE
from streamlit.testing.v1 import AppTest
at = AppTest.from_file({app!r}, default_timeout=180).run()
assert not at.exception, at.exception
assert hasattr(sys.modules["huskyquiet.analysis"], "ISSUANCE_NOTE")
"""


def test_app_reloads_a_package_left_over_from_the_previous_deploy():
    if not MODEL.exists() or not CLEAN.exists():
        pytest.skip("production artifacts not built")
    script = _STALE_PROCESS.format(src=str(ROOT / "src"), app=str(ROOT / "streamlit_app.py"))
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
