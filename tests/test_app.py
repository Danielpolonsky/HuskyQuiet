from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    paths = ROOT
    model = paths / "artifacts" / "models" / "production" / "model.joblib"
    clean = paths / "data" / "interim" / "permits_clean.parquet"
    if not model.exists() or not clean.exists():
        pytest.skip("production artifacts not built")
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=180).run()


@pytest.mark.integration
def test_explore_page_renders(app):
    assert not app.exception
    assert app.header[0].value == "Explore map"


@pytest.mark.integration
def test_address_and_compare_pages_render_forms(app):
    analyze = app.switch_page("app_pages/analyze.py").run()
    assert not analyze.exception
    assert analyze.header[0].value == "Analyze address"
    compared = analyze.switch_page("app_pages/compare.py").run()
    assert not compared.exception
    assert compared.header[0].value == "Compare apartments"


@pytest.mark.integration
def test_performance_page_matches_the_saved_report(app):
    page = app.switch_page("app_pages/performance.py").run()
    assert not page.exception
    values = " ".join(str(metric.value) for metric in page.metric)
    assert "0.765" in values
    assert "0.686" in values


@pytest.mark.integration
def test_about_page_renders(app):
    page = app.switch_page("app_pages/about.py").run()
    assert not page.exception
    text = " ".join(block.value for block in page.markdown)
    assert "not the day construction starts" in text
